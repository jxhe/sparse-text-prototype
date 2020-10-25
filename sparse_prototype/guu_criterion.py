# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import LegacyFairseqCriterion, FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    """compute labeled smoothed nll loss
    Returns:
        loss: the actual loss to be optimized (after smoothing), with
            shape (batch) if reduce is true else (batch, seq_len)
        nll_loss: the NLL loss with shape (batch) if reduce is true else
            (batch, seq_len)
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)

    nll_loss = nll_loss.squeeze(-1)
    smooth_loss = smooth_loss.squeeze(-1)

    # (batch, seq_len) --> (batch)
    if reduce:
        nll_loss = nll_loss.sum(-1)
        smooth_loss = smooth_loss.sum(-1)
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('guu_elbo')
class GuuELBO(LegacyFairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, data_len, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model.guu_forward(**sample['net_input'], data_len=data_len)
        loss, neg_elbo, recon_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        nsentences = sample['target'].size(0)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'neg_elbo': utils.item(neg_elbo.data) if reduce else neg_elbo.data,
            'recon_loss': utils.item(recon_loss.data) if reduce else recon_loss.data,
            'ntokens': sample['ntokens'] / model.infer_ns,
            'nsentences': sample['target'].size(0) / model.infer_ns,
            'sample_size': sample_size / model.infer_ns,
        }

        return loss, sample_size, logging_output

    # compute the ELBO loss, involving reinforcement learning
    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output['recon_out'], log_probs=True)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        smoothed_nll_loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        revert_order = sample['net_input']['revert_order']

        nll_loss = nll_loss.index_select(0, revert_order)
        smoothed_nll_loss = smoothed_nll_loss.index_select(0, revert_order)


        nsentences = sample['target'].size(0) / model.infer_ns

        loss = smoothed_nll_loss.view(-1, model.infer_ns).mean(1).sum()

        with torch.no_grad():
            neg_elbo = nll_loss.view(-1, model.infer_ns).mean(1).sum() + \
                       math.log(model.num_prototypes / model.infer_ns) * nsentences

        return loss, neg_elbo, nll_loss.view(-1, model.infer_ns).mean(1).sum()

    def iw_eval_new(self, model, sample, data_len, iw_nsample, retrieve_dataset, reduce=True):
        """Compute the importance-weighted loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model.guu_forward(**sample['net_input'], data_len=data_len)
        nll_iw = self.compute_loss_iw(model, net_output, sample, reduce=reduce)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        nsentences = sample['target'].size(0) / model.infer_ns

        logging_output = {
            'nll_iw': utils.item(nll_iw.data) if reduce else nll_iw.data,
            'ntokens': sample['ntokens'] / model.infer_ns,
            'nsentences': nsentences,
            'sample_size': sample_size / model.infer_ns,
        }

        return nll_iw, sample_size, logging_output

    def compute_loss_iw(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output['recon_out'], log_probs=True)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        smoothed_nll_loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        revert_order = sample['net_input']['revert_order']

        nll_loss = nll_loss.index_select(0, revert_order)


        nsentences = sample['target'].size(0) / model.infer_ns

        nll_iw = (torch.logsumexp(-nll_loss.view(-1, model.infer_ns), dim=1) + \
                   math.log(1.0 / model.num_class)).sum()

        nll_iw = -nll_iw

        return nll_iw

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        neg_elbo_sum = sum(log.get('neg_elbo', 0) for log in logging_outputs)
        recon_loss_sum = sum(log.get('recon_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)

        if 'nll_iw' in logging_outputs[0]:
            nll_iw_sum = sum(log.get('nll_iw', 0) for log in logging_outputs)
            metrics.log_scalar('nll_iw_s', nll_iw_sum / nsentences,
                nsentences, round=3, priority=4)
            metrics.log_scalar('nll_iw_t', nll_iw_sum / ntokens / math.log(2),
                ntokens, round=3, priority=5)
            metrics.log_derived('ppl_iw', lambda meters: utils.get_perplexity(meters['nll_iw_t'].avg), priority=6)
        else:
            metrics.log_scalar('loss', loss_sum / sample_size / math.log(2),
                sample_size, round=3, priority=3)

            metrics.log_scalar('neg_elbo_s', neg_elbo_sum / nsentences,
                nsentences, round=3, priority=4)
            metrics.log_scalar('recon_loss_s', recon_loss_sum / nsentences,
                nsentences, round=3, priority=4)

            metrics.log_scalar('neg_elbo_t', neg_elbo_sum / ntokens / math.log(2),
                ntokens, round=3, priority=5)
            metrics.log_scalar('recon_loss_t', recon_loss_sum / ntokens / math.log(2),
                ntokens, round=3, priority=5)

            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['neg_elbo_t'].avg), priority=6)
            metrics.log_derived('recon_ppl', lambda meters: utils.get_perplexity(meters['recon_loss_t'].avg), priority=7)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
