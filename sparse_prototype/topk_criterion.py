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


@register_criterion('topk_elbo')
class TopkELBO(LegacyFairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.free_bits = args.free_bits

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
        net_output = model.topk_forward(**sample['net_input'], data_len=data_len)
        loss, neg_elbo, recon_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        nsentences = sample['target'].size(0) / model.infer_ns
        lambda_stats = model.measure_lambda_sparsity()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'neg_elbo': utils.item(neg_elbo.data) if reduce else neg_elbo.data,
            'recon_loss': utils.item(recon_loss.data) if reduce else recon_loss.data,
            'ntokens': sample['ntokens'] / model.infer_ns,
            'nsentences': nsentences,
            'sample_size': sample_size / model.infer_ns,
            'KLz': utils.item(net_output['KLz'].sum().data / model.infer_ns),
            'KLt': utils.item(net_output['KLt'].sum().data),
            'KLtheta': utils.item(net_output['KLtheta'] * nsentences)
        }

        logging_output.update(lambda_stats)
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

        KLz = net_output['KLz']
        KLt = net_output['KLt']
        KLtheta = net_output['KLtheta']
        logits_topk = net_output['logits_topk']
        # logq = net_output['logq']

        nll_loss = nll_loss.index_select(0, revert_order)
        smoothed_nll_loss = smoothed_nll_loss.index_select(0, revert_order)
        KLz = KLz.index_select(0, revert_order)

        nsentences = sample['target'].size(0) / model.infer_ns

        if self.free_bits > 0:
            lower_bound = KLt.new_full(KLt.size(), self.free_bits)
            KLt_fake, _ = torch.stack((KLt, lower_bound)).max(dim=0)  
        else:
            KLt_fake = KLt

        if model.training:
            loss = (((smoothed_nll_loss + KLz).view(-1, model.infer_ns) * F.softmax(logits_topk, dim=1)).sum(1) 
                    + KLt_fake).sum() + KLtheta * nsentences
        else:
            loss = ((nll_loss + KLz).view(-1, model.infer_ns).mean(1) + KLt).sum() + KLtheta * nsentences

        return loss, loss, nll_loss.view(-1, model.infer_ns).mean(1).sum()

    def iw_eval(self, model, sample, data_len, iw_nsample, reduce=True):
        """Compute the importance-weighted loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """   

        tmp = []
        for _ in range(int(iw_nsample / model.infer_ns)):
            net_output = model.iw_forward(**sample['net_input'], data_len=data_len)

            # log [p(x, t, z) / q(t, z |x)]
            # (batch, infer_ns)
            log_ratio = self._compulte_iw_loss(model, net_output, sample, reduce=reduce)
            tmp.append(log_ratio)

        # (batch)
        ll_iw = torch.logsumexp(torch.cat(tmp, dim=-1), dim=-1) - math.log(iw_nsample)
        ll_iw = -ll_iw.sum()

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        nsentences = sample['target'].size(0) / model.infer_ns

        logging_output = {
            'nll_iw': utils.item(ll_iw.data) if reduce else ll_iw.data,
            'ntokens': sample['ntokens'] / model.infer_ns,
            'nsentences': nsentences,
            'sample_size': sample_size / model.infer_ns,
        }

        return ll_iw, sample_size, logging_output

    def _compulte_iw_loss(self, model, net_output, sample, reduce=True):
        """compute the importance weighted loss
        """
        lprobs = model.get_normalized_probs(net_output['recon_out'], log_probs=True)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        smoothed_nll_loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        revert_order = sample['net_input']['revert_order']

        # (batch, infer_ns)
        log_pxtz = -nll_loss.index_select(0, revert_order).view(-1, model.infer_ns)

        log_ratio = net_output['log_pz'] + net_output['log_pt'] + log_pxtz \
                    - net_output['log_qz'] - net_output['log_qt'] 

        return log_ratio

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        neg_elbo_sum = sum(log.get('neg_elbo', 0) for log in logging_outputs)
        recon_loss_sum = sum(log.get('recon_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        KLz_sum = sum(log.get('KLz', 0) for log in logging_outputs)
        KLt_sum = sum(log.get('KLt', 0) for log in logging_outputs)
        KLtheta_sum = sum(log.get('KLtheta', 0) for log in logging_outputs)

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

            metrics.log_scalar('KLz', KLz_sum / nsentences, nsentences, round=1, priority=8)
            metrics.log_scalar('KLt', KLt_sum / nsentences, nsentences, round=1, priority=8)
            metrics.log_scalar('KLtheta', KLtheta_sum / nsentences, nsentences, round=1, priority=8)

            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['neg_elbo_t'].avg), priority=6)
            metrics.log_derived('recon_ppl', lambda meters: utils.get_perplexity(meters['recon_loss_t'].avg), priority=7)

        if 'active' in logging_outputs[0]:
            metrics.log_scalar('active', logging_outputs[0]['active'], weight=0, round=1, priority=10)
            metrics.log_scalar('percent', logging_outputs[0]['percent'], weight=0, round=2, priority=10)
        # metrics.log_scalar('nlow', logging_outputs[0]['nlow'], weight=0, priority=10)
        # metrics.log_scalar('nhigh', logging_outputs[0]['nhigh'], weight=0, priority=10)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
