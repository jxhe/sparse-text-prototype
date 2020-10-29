import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy
import warnings

from fairseq import options, utils
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax
from fairseq.models.lstm import LSTMDecoder

from scipy.special import digamma,loggamma
from datasets import load_dataset

from .vae import VAEEncoder
from .inv_editor import GuuInvEditor, LevenshteinInvEditor
from .retriever import PrecomputeEmbedRetriever, BertRetriever


@register_model('sparse_prototype')
class TemplateModel(BaseFairseqModel):
    def __init__(self, classifier, editor,
        decoder, alpha, cuda, grad_lambda, args):
        super().__init__()

        self.args = args
        self.classifier = classifier

        self.classifier_ahead = self.classifier


        self.editor = editor

        self.decoder = decoder

        if args.criterion == 'lm_baseline':
            self.num_class = 1
        else:
            template_group = load_dataset('csv',
                                          data_files=f'{args.emb_dataset_file}.template.csv.gz',
                                          cache_dir='hf_dataset_cache')

            template_group = template_group['train']
            self.num_class = len(template_group)

        self.device = torch.device('cuda' if cuda else 'cpu')

        # Dirichlet vae_prior
        self.alpha = torch.tensor([alpha] * self.num_class).to(self.device)
        # self.alpha = Parameter(self.alpha)
        # self.alpha.requires_grad = False

        # Dirichlet posterior
        self.lambda_ = nn.Parameter(torch.ones(self.num_class).fill_(alpha))
        self.grad_lambda = grad_lambda

        if not grad_lambda:
            self.lambda_.requires_grad = False

        self.alpha_stats = self.digamma_stats1(self.alpha)
        # self.alpha_stats = self.scipy_digamma_stats1(self.alpha)

        self.prune_index = None
        self.prune_lambda = None
        self._lambda_t = 1

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-freeze-embed', action='store_true',
                            help='freeze encoder embeddings')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-freeze-embed', action='store_true',
                            help='freeze decoder embeddings')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')


        # latent parameters
        parser.add_argument('--latent-dim', type=int, metavar='N',
                            help='latent edit vector dimension')
        parser.add_argument('--vae-prior', type=str, metavar='STR',
                            choices=['vmf'],
                            help='the prior distribution over edit vector')
        parser.add_argument('--vae-nsamples', type=int, metavar='N',
                            help='number of samples during vae training')
        parser.add_argument('--vmf-kappa', type=int, metavar='N',
                            help='number of samples during vae training')
        parser.add_argument('--alpha', type=float, default=0.5, metavar='D',
                            help='the (flat) Dirichlet prior parameter')

        # inference parameters
        parser.add_argument('--inveditor-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--retriever', type=str, metavar='STR',
                            choices=['precompute_emb', 'bert', 'sentbert'],
                            help='the retrieve module')
        parser.add_argument('--grad-lambda', type=str, metavar='BOOL',
                            help='if update lambda with gradient descent')
        parser.add_argument('--freeze-retriever', type=str, metavar='BOOL',
                            help='if update the retriever')

        # retriever options
        parser.add_argument('--linear-bias', type=str, metavar='BOOL',
                help='if includes a final bias term')
        parser.add_argument('--stop-bert-grad', type=str, metavar='BOOL',
                            help='if stop updating BERT retriver')
        parser.add_argument('--emb-dataset-file', type=str, metavar='STR',
                            help='the hdf5 dataset file')
        parser.add_argument('--embed-init-rescale', type=float, metavar='D',
                            help='rescaling factor when initializing sent embeddings')
        parser.add_argument('--embed-transform-nlayer', type=int, default=0, metavar='N',
                            help='middle non-linear layers when transforming embeddings')

        # inference training method
        parser.add_argument('--reinforce', type=str, metavar='BOOL',
                help='if use reinforce samples, else use sum to approximate')
        parser.add_argument('--infer-ns', type=int, metavar='N',
                help='number of samples in reinforce, or top K sum when reinforce is False')
        parser.add_argument('--reinforce-temperature', type=float, default=1.0, metavar='D',
                help='temperature to sample using reinforce')

        # inverse editor parameters
        parser.add_argument('--inv-editor', type=str, metavar='STR',
                            choices=['guu', 'levenshtein'],
                            help='the inverse editor module')
        parser.add_argument('--edit-embed-dim', type=int, metavar='N',
                            help='embed dim for edit diff sequences')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        # fmt: on

        # others
        parser.add_argument('--decoder-copy', type=str, metavar='BOOL',
                            help='use copy mechanism in decoder')
        parser.add_argument('--sparse-threshold', type=float, default=0.9, metavar='D',
                            help='threshold for pruning')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False


        pretrained_inveditor_embed = None
        if args.inveditor_embed_path:
            pretrained_inveditor_embed = load_pretrained_embedding_from_file(
                args.inveditor_embed_path, task.source_dictionary, args.encoder_embed_dim)


        cuda = torch.cuda.is_available() and not args.cpu
        # this is a seq2seq model from templates to observed sentences
        # the Neural Editor p(x|t, z)
        encoder = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
        )

    
        decoder = LSTMSkipDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=False if args.criterion == 'lm_baseline' else options.eval_bool(args.decoder_attention),
            encoder_output_units=0 if args.criterion == 'lm_baseline' else encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
        )

        if args.criterion == 'lm_baseline':
            return cls(None, None,
                decoder, args.alpha, cuda,
                options.eval_bool(args.grad_lambda), args)

        if args.retriever == 'bert' or args.retriever == 'sentbert':
            retriever = BertRetriever(
                dictionary=task.target_dictionary,
                emb_dataset_path=args.emb_dataset_file,
                rescale=args.embed_init_rescale,
                linear_bias=options.eval_bool(args.linear_bias),
                stop_grad=options.eval_bool(args.stop_bert_grad),
                freeze=options.eval_bool(args.freeze_retriever),
                cuda=cuda,
                sentbert=False if args.retriever == 'bert' else True,
                )
        elif args.retriever == 'precompute_emb':
            retriever = PrecomputeEmbedRetriever(
                dictionary=task.target_dictionary,
                emb_dataset_path=args.emb_dataset_file,
                rescale=args.embed_init_rescale,
                linear_bias=options.eval_bool(args.linear_bias),
                freeze=options.eval_bool(args.freeze_retriever),
                nlayers=args.embed_transform_nlayer,
                )
        else:
            raise ValueError('retriever {} is not supported'.format(args.retriever))

        editor = FairseqEncoderDecoderModel(encoder, decoder)

        return cls(retriever, editor,
            decoder, args.alpha, cuda,
            options.eval_bool(args.grad_lambda), args)

    def measure_lambda_sparsity(self):
        if self.training:
            lambda_stats = {}
        else:
            with torch.no_grad():
                prob = self.lambda_ / (self.lambda_.sum())
                sorted_prob, _ = torch.sort(prob, descending=True)
                sum_ = 0.

                for i in range(self.lambda_.size(0)):
                    sum_ += sorted_prob[i].item()
                    if sum_ >= self.args.sparse_threshold:
                        break

                lambda_stats = {
                    'active': i,
                    'percent': i / self.lambda_.size(0)
                    # 'max': max_,
                    # 'min': min_,
                    # 'nhigh': (prob > (base * 10)).sum().item(),
                    # 'nlow': (prob < (base / 10)).sum().item()
                }
        return lambda_stats

    def set_prune_index(self, k):
        """select top k and the rest is pruned
        """
        _, index = torch.sort(self.lambda_, descending=True)

        # # self.prune_index = index[k:]
        # self.classifier.set_prune_index(self.prune_index)
        self.classifier.set_prune_index(index[:k])
        self.prune_lambda = self.lambda_[index[:k]]

        index_map = {i: index[i] for i in range(k)}
        return index_map

    def reset_prune_index(self):
        """select top k and the rest is pruned
        """

        self.prune_index = None
        self.classifier.reset_prune_index()

        self.prune_lambda = None

        return

    def digamma_stats1(self, alpha: torch.tensor):
        return torch.lgamma(alpha.sum()) - torch.lgamma(alpha).sum()


    def digamma_stats2(self, alpha1, alpha2):
        return ((alpha1 - 1.) * (torch.digamma(alpha2) - torch.digamma(alpha2.sum()))).sum()

    def scipy_digamma_stats1(self, alpha):
        return loggamma(alpha * self.num_class) - self.num_class * loggamma(alpha)

    def scipy_digamma_stats2(self, alpha1, alpha2):
        return ((alpha1 - 1.) * (digamma(alpha2) - digamma(alpha2.sum()))).sum()

    def update_lambda(self, value: torch.tensor):
        self.lambda_.copy_(value)

    def set_lambda_t(self, value: float):
        self._lambda_t = value

    def get_alpha(self) -> float:
        return self.alpha

    def get_lambda(self):
        return self.lambda_

    def get_prototypes(self, num):
        _, indices = torch.topk(self.lambda_, num)

        return indices.tolist()

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.editor.forward_decoder(prev_output_tokens, **kwargs)

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.editor.max_decoder_positions()

    def set_num_samples(self, val):
        self.args.infer_ns = val

    @property
    def infer_ns(self):
        return self.args.infer_ns

    @property
    def encoder(self):
        return self.editor.encoder

    @property
    def decoder(self):
        return self.editor.decoder

    @property
    def lambda_t(self):
        return self._lambda_t


    @property
    def num_prototypes(self):
        return self.num_class

    @property
    def cont_params(self):
        return list(self.editor.parameters()) + list(self.vae_encoder.parameters()) \
                + list(self.decoder.parameters()) + [self.lambda_]

    @property
    def discrete_params(self):
        return self.classifier_ahead.parameters()

    def sample_from_uniform_sphere(self, ns):
        """sampling from the uniform vmf prior
        reference:
        https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
        """

        mean = torch.zeros((ns, self.args.latent_dim), device=self.device)
        std = torch.ones((ns, self.args.latent_dim), device=self.device)

        gaussians = torch.normal(mean=mean, std=std)

        return gaussians / ((gaussians * gaussians).sum(dim=1).unsqueeze(1).sqrt())


    def forward(self, src_tokens, src_lengths, temp_tokens, temp_lengths,
                temp_ids, logits, prev_output_tokens, **kwargs) -> dict:
        """
        Args:
            src_tokens (LongTensor): the src tokens, (batch * nsample, seq_length)
            temp_tokens (LongTensor): the template tokens, (batch * nsample, seq_length)
            temp_lengths:  (batch * nsample)
            temp_ids (LongTensor): the index of selected templates, (batch * nsample)
            logits (tensor): the output logits of classification network,
                             shape of (batch, num_class)
        """

        if self.training:
            # E_{q(t|x)q(z|x,t)}[log p(x|st,z)] =
            # \sum_i\sum_n [prob(t_ni=1)][digamma(\lambda_i) - digamma(\sum_j\lambda_j)]
            term2 = F.softmax(logits, dim=1) * ((torch.digamma(self.lambda_) - torch.digamma(self.lambda_.sum())).unsqueeze(0))

            # (batch, num_class) -> (batch)
            term2 = term2.sum(1)
        else:
            if self.prune_lambda is None:
                theta = self.lambda_ / self.lambda_.sum()
            else:
                # lambda_prune = self.lambda_.index_fill(0, self.prune_index, self.lambda_.min().item() * 1e-3)
                # theta = lambda_prune / lambda_prune.sum()
                theta = self.prune_lambda / self.prune_lambda.sum()
            term2 = F.softmax(logits, dim=1) * torch.log(theta)
            term2 = term2.sum(1)


        if self.training:
            # torch.lgamma(x) seems to have numeric erros when x is larger than 1e4
            # E_{q(\theta; \lambda)}[log p(\theta) - \log q(\theta)] / (number of all training samples)
            term1 = self.alpha_stats + self.digamma_stats2(self.alpha, self.lambda_) \
                    - (self.digamma_stats1(self.lambda_) +
                        self.digamma_stats2(self.lambda_, self.lambda_))
            # term1 = self.alpha_stats + self.digamma_stats2(self.alpha, self.lambda_) \
            #         - (loggamma(self.lambda_.sum().item()) - loggamma(self.lambda_.cpu().numpy()).sum() +
            #             self.digamma_stats2(self.lambda_, self.lambda_))

            term1 = term1 / kwargs['data_len']
        else:
            term1 = 0.

        # (batch * nsample, tgt_len, vocab)
        recon_out = self.editor(temp_tokens, temp_lengths, prev_output_tokens)

        # neg_entropy: (batch)
        tmp = F.log_softmax(logits, dim=1)
        prob = torch.exp(tmp)
        neg_entropy = (prob * tmp).sum(dim=1)

        temp_ids = temp_ids.index_select(0, kwargs['revert_order'])
        # (batch, nsample)
        temp_ids_reshape = temp_ids.view(-1, self.args.infer_ns)

        # (batch, nsample)
        logq = torch.gather(F.log_softmax(logits, dim=1), dim=1, index=temp_ids_reshape)


        res = {"recon_out": recon_out,
               "logits": logits,
               "KLtheta": -term1,
               "KLt": neg_entropy - term2,
               "logq": logq,
               }

        return res

    def entropy_forward(self, src_tokens, src_lengths, temp_tokens, temp_lengths,
                temp_ids, logits, prev_output_tokens, **kwargs) -> dict:
        """
        Args:
            src_tokens (LongTensor): the src tokens, (batch * nsample, seq_length)
            temp_tokens (LongTensor): the template tokens, (batch * nsample, seq_length)
            temp_lengths:  (batch * nsample)
            temp_ids (LongTensor): the index of selected templates, (batch * nsample)
            logits (tensor): the output logits of classification network,
                             shape of (batch, num_class)
        """


        # neg_entropy: (batch)
        tmp = F.log_softmax(logits, dim=1)
        prob = torch.exp(tmp)
        entropy = -(prob * tmp).sum(dim=1)

        res = {"entropy": entropy,
               "logits": logits,
               }

        return res

    def topk_forward(self, src_tokens, src_lengths, temp_tokens, temp_lengths,
                temp_ids, logits, logits_topk, prev_output_tokens, **kwargs) -> dict:
        """
        Args:
            src_tokens (LongTensor): the src tokens, (batch * nsample, seq_length)
            temp_tokens (LongTensor): the template tokens, (batch * nsample, seq_length)
            temp_lengths:  (batch * nsample)
            temp_ids (LongTensor): the index of selected templates, (batch * nsample)
            logits (tensor): the output logits of classification network,
                             shape of (batch, num_class)
        """

        if self.training:
            digamma_vec = torch.digamma(self.lambda_) - torch.digamma(self.lambda_.sum())
            digamma_select = torch.index_select(digamma_vec, dim=0, index=temp_ids)
            digamma_select = digamma_select.index_select(0, kwargs['revert_order'])

            # (batch, nsample)
            digamma_select = digamma_select.view(-1, self.infer_ns)
            term2 = F.softmax(logits_topk, dim=1) *  digamma_select
            # E_{q(t|x)q(z|x,t)}[log p(x|t,z)] =
            # \sum_i\sum_n [prob(t_ni=1)][digamma(\lambda_i) - digamma(\sum_j\lambda_j)]
            # term2 = F.softmax(logits, dim=1) * ((torch.digamma(self.lambda_) - torch.digamma(self.lambda_.sum())).unsqueeze(0))

            # (batch, num_class) -> (batch)
            term2 = term2.sum(1)
        # at evaluation time we should use unbiased estimator of expectations
        else:
            if self.prune_lambda is None:
                theta = self.lambda_ / self.lambda_.sum()
            else:
                # lambda_prune = self.lambda_.index_fill(0, self.prune_index, self.lambda_.min().item() * 1e-3)
                # theta = lambda_prune / lambda_prune.sum()
                theta = self.prune_lambda / self.prune_lambda.sum()
            term2 = F.softmax(logits, dim=1) * torch.log(theta)
            term2 = term2.sum(1)


        if self.training:
            # torch.lgamma(x) seems to have numeric erros when x is larger than 1e4
            # E_{q(\theta; \lambda)}[log p(\theta) - \log q(\theta)] / (number of all training samples)
            term1 = self.alpha_stats + self.digamma_stats2(self.alpha, self.lambda_) \
                    - (self.digamma_stats1(self.lambda_) +
                        self.digamma_stats2(self.lambda_, self.lambda_))
            # term1 = self.alpha_stats + self.digamma_stats2(self.alpha, self.lambda_) \
            #         - (loggamma(self.lambda_.sum().item()) - loggamma(self.lambda_.cpu().numpy()).sum() +
            #             self.digamma_stats2(self.lambda_, self.lambda_))

            term1 = term1 / kwargs['data_len']
        else:
            term1 = 0.

        # z: (batch * nsample, nsz, nz)
        # KLz: (batch * nsample)
        z, KLz, _ = self.vae_encoder(src_tokens, src_lengths, temp_tokens, temp_lengths, **kwargs)
        z = z.squeeze(1)

        # (batch * nsample, tgt_len, vocab)
        recon_out = self.editor(temp_tokens, temp_lengths, prev_output_tokens,
            edit_vector=z, src_t=temp_tokens, src_l=temp_lengths)

        # neg_entropy: (batch)
        if self.training:
            tmp = F.log_softmax(logits_topk, dim=1)
            prob = torch.exp(tmp)
            neg_entropy = (prob * tmp).sum(dim=1)
        else:
            tmp = F.log_softmax(logits, dim=1)
            prob = torch.exp(tmp)
            neg_entropy = (prob * tmp).sum(dim=1)

        temp_ids = temp_ids.index_select(0, kwargs['revert_order'])
        # (batch, nsample)
        temp_ids_reshape = temp_ids.view(-1, self.args.infer_ns)

        # (batch, nsample)
        logq = torch.gather(F.log_softmax(logits, dim=1), dim=1, index=temp_ids_reshape)


        res = {"KLz": KLz,
               "recon_out": recon_out,
               "logits": logits,
               "logits_topk": logits_topk,
               "KLtheta": -term1,
               "KLt": neg_entropy - term2,
               "logq": logq,
               }

        return res


    def guu_forward(self, src_tokens, src_lengths, temp_tokens, temp_lengths,
                temp_ids, logits, prev_output_tokens, **kwargs) -> dict:
        """
        Args:
            src_tokens (LongTensor): the src tokens, (batch * nsample, seq_length)
            temp_tokens (LongTensor): the template tokens, (batch * nsample, seq_length)
            temp_lengths:  (batch * nsample)
            temp_ids (LongTensor): the index of selected templates, (batch * nsample)
            logits (tensor): the output logits of classification network,
                             shape of (batch, num_class)
        """

        # z: (batch * nsample, nsz, nz)
        # KLz: (batch * nsample)
        # z, KLz, _ = self.vae_encoder(src_tokens, src_lengths, temp_tokens, temp_lengths, **kwargs)
        # z = z.squeeze(1)

        # (batch * nsample, tgt_len, vocab)
        recon_out = self.editor(temp_tokens, temp_lengths, prev_output_tokens,
             src_t=temp_tokens, src_l=temp_lengths, edit_vector=None)


        res = {
               "recon_out": recon_out,
               "logits": logits,
               # "KLz": KLz
               }

        return res

    def lm_forward(self, prev_output_tokens, **kwargs) -> dict:
        """
        Args:
            src_tokens (LongTensor): the src tokens, (batch * nsample, seq_length)
            temp_tokens (LongTensor): the template tokens, (batch * nsample, seq_length)
            temp_lengths:  (batch * nsample)
            temp_ids (LongTensor): the index of selected templates, (batch * nsample)
            logits (tensor): the output logits of classification network,
                             shape of (batch, num_class)
        """

        # (batch * nsample, tgt_len, vocab)
        recon_out = self.decoder(prev_output_tokens)


        res = {
               "recon_out": recon_out,
               }

        return res

    def iw_forward(self, src_tokens, src_lengths, temp_tokens, temp_lengths,
                temp_ids, logits, prev_output_tokens, **kwargs) -> dict:
        """compute quantities required by importance-weighted evaluation
        Args:
            src_tokens (LongTensor): the src tokens, (batch * nsample, seq_length)
            temp_tokens (LongTensor): the template tokens, (batch * nsample, seq_length)
            temp_lengths:  (batch * nsample)
            temp_ids (LongTensor): the index of selected templates, (batch * nsample)
            logits (tensor): the output logits of classification network,
                             shape of (batch, num_class)
        """

        temp_ids = temp_ids.index_select(0, kwargs['revert_order'])
        # (batch, nsample)
        temp_ids_reshape = temp_ids.view(-1, self.infer_ns)

        if self.prune_lambda is None:
            theta = self.lambda_ / self.lambda_.sum()
        else:
            # lambda_prune = self.lambda_.index_fill(0, self.prune_index, self.lambda_.min().item() * 1e-3)
            # theta = lambda_prune / lambda_prune.sum()
            theta = self.prune_lambda / self.prune_lambda.sum()

        # (batch, nsample)
        log_pt = torch.index_select(torch.log(theta), dim=0, index=temp_ids).view(-1, self.infer_ns)
        # log_qt = torch.gather(F.log_softmax(logits, dim=1),
        #     dim=1, index=temp_ids_reshape)

        # (batch * infer_ns, 1, nz) --> (batch * infer_ns, nz)
        # z, KLz, param_dict = self.vae_encoder(src_tokens, src_lengths, temp_tokens, temp_lengths, **kwargs)
        # z = z.squeeze(1)

        # log_pz = self.vae_encoder.log_prior_vmf_density(z)
        # log_qz = self.vae_encoder.log_vmf_density(z, param_dict['mu'])

        # # (batch, nsample)
        # log_pz = log_pz.index_select(0, kwargs['revert_order']).view(-1, self.infer_ns)
        # log_qz = log_qz.index_select(0, kwargs['revert_order']).view(-1, self.infer_ns)


        # (batch * nsample, tgt_len, vocab)
        recon_out = self.editor(temp_tokens, temp_lengths, prev_output_tokens)


        res = {"recon_out": recon_out,
               "log_pt": log_pt,
               }

        return res




class LSTMEncoder(FairseqEncoder):
    """LSTM encoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        left_pad=True, pretrained_embed=None, padding_idx=None,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths, **kwargs):
        if self.left_pad:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # nn.utils.rnn.pack_padded_sequence requires right-padding;
                # convert left-padding to right-padding
                src_tokens = utils.convert_padding_direction(
                    src_tokens,
                    self.padding_idx,
                    left_to_right=True,
                )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_idx)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        # x: bsz x source_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores, torch.cat((x, input), dim=1)



class LSTMSkipDecoder(LSTMDecoder):
    """LSTMDecoder with skip connections between layers"""
    def __init__(self, **kwargs):
        super(LSTMSkipDecoder, self).__init__(**kwargs)

    def extract_features(
        self, prev_output_tokens, encoder_out, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """
        if encoder_out is not None:
            encoder_padding_mask = encoder_out['encoder_padding_mask']
            encoder_out = encoder_out['encoder_out']
        else:
            encoder_padding_mask = None
            encoder_out = None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
            srclen = encoder_outs.size(0)
        else:
            srclen = None

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        elif encoder_out is not None:
            # setup recurrent cells
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)
        else:
            # setup zero cells, since there is no encoder
            num_layers = len(self.layers)
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(num_layers)]
            prev_cells = [zero_state for i in range(num_layers)]
            input_feed = None

        assert srclen is not None or self.attention is None, \
            "attention is not supported if there are no encoder outputs"
        attn_scores = x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input = x[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                pre_input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # skip connection
                if i == 0:
                    input = pre_input
                    out = hidden
                else:
                    input = pre_input + input
                    out = hidden + input

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                out, attn_scores[:, j, :] = self.attention(out, encoder_outs, encoder_padding_mask)
            else:
                out = out
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn and self.attention is not None:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None
        return x, attn_scores



def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

@register_model_architecture('sparse_prototype', 'sparse_prototype')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 300)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 400)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', True)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 300)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 400)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 300)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')

    args.latent_dim = getattr(args, 'latent_dim', 50)
    args.vae_prior = getattr(args, 'vae_prior', 'vmf')
    args.vae_nsamples = getattr(args, 'vae_nsamples', 1)
    args.vmf_kappa = getattr(args, 'vmf_kappa', args.vmf_kappa)
    args.alpha = getattr(args, 'alpha', args.alpha)

    args.inveditor_embed_path = getattr(args, 'inveditor_embed_path', None)

    args.retriever = getattr(args, 'retriever', args.retriever)

    args.grad_lambda = getattr(args, 'grad_lambda', 0)

@register_model_architecture('sparse_prototype', 'yelp')
def yelp_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 300)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 300)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 256)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', args.decoder_hidden_size)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.)

    args.latent_dim = getattr(args, 'latent_dim', 128)
    base_architecture(args)

@register_model_architecture('sparse_prototype', 'yelp_large')
def yelp_large_architecture(args):
    yelp_architecture(args)

# small
@register_model_architecture('sparse_prototype', 'yelp_large_s')
def yelp_large_s_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 300)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 300)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 256)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', args.decoder_hidden_size)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.)

    base_architecture(args)

# xs
@register_model_architecture('sparse_prototype', 'yelp_large_xs')
def yelp_large_xs_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 300)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 128)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 300)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 128)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', args.decoder_hidden_size)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.)

    base_architecture(args)


@register_model_architecture('sparse_prototype', 'coco40k')
def coco40k_architecture(args):
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.3)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.3)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.5)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.5)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    base_architecture(args)
