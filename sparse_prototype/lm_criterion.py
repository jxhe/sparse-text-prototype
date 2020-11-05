# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
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

def write_loss(ll_batch, sample, infer_ns, fout):
    # revert_order = sample['net_input']['revert_order']
    id_list = sample['id']
    length_list = sample['net_input']['src_lengths']

    for id_, ntoken, ll in zip(id_list, length_list, ll_batch):
        fout.write('{} {} {}\n'.format(id_.item(), ntoken.item(), ll.item()))


@register_criterion('lm_baseline')
class LMBaseline(LegacyFairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

        if args.write_loss_path is not None:
            self.f_loss = open(os.path.join(args.save_dir, args.write_loss_path), 'w')
        else:
            self.f_loss = None

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
        net_output = model.lm_forward(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        nsentences = sample['target'].size(0)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
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

        loss = smoothed_nll_loss.sum()

        if self.f_loss is not None:
            # revert_order = sample['net_input']['revert_order']
            # nll_loss_reorder = nll_loss.index_select(0, revert_order).view(-1, model.infer_ns).mean(1)
            write_loss(-nll_loss, sample, 1, self.f_loss)


        return loss, nll_loss.sum()

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2),
            sample_size, round=3, priority=3)

        metrics.log_scalar('nll_loss_s', nll_loss_sum / nsentences,
            nsentences, round=3, priority=4)

        metrics.log_scalar('nll_loss_t', nll_loss_sum / ntokens / math.log(2),
            ntokens, round=3, priority=5)

        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss_t'].avg), priority=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
