# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import math
import copy

import torch
import torch.nn.functional as F

from fairseq import options, utils, metrics, search, tokenizer
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationTask

from sparse_prototype.retrieve_prototype_dataset import RetrievePrototypeDataset
from sparse_prototype.language_pair_map_dataset import LanguagePairMapDataset


# ported from UnsupervisedMT
def parse_lambda_config(x):
    """
    Parse the configuration of lambda coefficient (for scheduling).
    x = "3"                  # lambda will be a constant equal to x
    x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease
                             # to 0 during the first 1000 iterations
    x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000
                             # iterations, then will linearly increase to 1 until iteration 2000
    """
    split = x.split(',')
    if len(split) == 1:
        return float(x), None
    else:
        split = [s.split(os.pathsep) for s in split]
        assert all(len(s) == 2 for s in split)
        assert all(k.isdigit() for k, _ in split)
        assert all(int(split[i][0]) < int(split[i + 1][0]) for i in range(len(split) - 1))
        return float(split[0][1]), [(int(k), float(v)) for k, v in split]



@register_task('sparse_prototype')
class SparsePrototypeTask(TranslationTask):
    """A task for training multiple translation models simultaneously.
    We iterate round-robin over batches from multiple language pairs, ordered
    according to the `--lang-pairs` argument.
    The training loop is roughly:
        for i in range(len(epoch)):
            for lang_pair in args.lang_pairs:
                batch = next_batch_for_lang_pair(lang_pair)
                loss = criterion(model_for_lang_pair(lang_pair), batch)
                loss.backward()
            optimizer.step()
    In practice, `next_batch_for_lang_pair` is abstracted in a FairseqDataset
    (e.g., `RoundRobinZipDatasets`) and `model_for_lang_pair` is a model that
    implements the `FairseqMultiModel` interface.
    During inference it is required to specify a single `--source-lang` and
    `--target-lang`, instead of `--lang-pairs`.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--forget-rate', type=float, default=0.9, metavar='D',
                            help='rho = (t + decay)^{-forget}')
        parser.add_argument('--decay-rate', type=float, default=1., metavar='D',
                            help='rho = (t + decay)^{-forget}')
        parser.add_argument('--retrieve-split', type=str, default='train',
                            help='the retrieve pool')

        parser.add_argument('--dec-opt-freq', type=int, default=1,
                            help='the relative update freq of decoder')
        parser.add_argument('--enc-opt-freq', type=int, default=1,
                            help='the relative update freq of encoder')

        parser.add_argument('--iw-nsamples', type=int, default=1000,
                            help='number of importance-weighted samples')
        parser.add_argument('--eval-mode', type=str, default='none',
                            choices=['iw', 'entropy', 'gen_sample', 'gen_reconstruction',
                            'time', 'none', 'from_file', 'gen_interpolation'],
                            help='evaluation modes')
        parser.add_argument('--eval-gen-file', type=str, default=None,
                            help='read in prototypes and edit vectors')
        parser.add_argument('--eval-gen-edit-vec', action='store_true', default=False,
                            help='write edit vectors in the generation file')

        parser.add_argument('--prune-num', type=int, default=-1,
                            help='perform evaluation based on top prune_num templates only')
        # parser.add_argument('--prune-num-offline', type=int, default=-1,
        #                     help='perform evaluation based on top prune_num templates only (offline version)')

        parser.add_argument('--free-bits', type=float, default=0,
                            help='the free bits param to regularize KLt, 0 to disable')
        parser.add_argument('--lambda-t-config', default="1.0", type=str, metavar='CONFIG',
                            help='KLt coefficient '
                                 'use fixed weight during training if set to floating point number. '
                                 'use piecewise linear function over number of updates to schedule the '
                                 'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--gen-nz', type=int, default=10,
                            help='number of edit vector samples to draw from the prior')
        parser.add_argument('--gen-np', type=int, default=200,
                        help='number of top prototypes')
        parser.add_argument('--write-loss-path', type=str, default=None,
                            help='write out loss at evaluation time for interpolation exp')

    def __init__(self, args, src_dict, edit_dict):
        super().__init__(args, src_dict, src_dict)
        self.forget_rate = args.forget_rate
        self.decay_rate = args.decay_rate
        self.dictionary = src_dict
        self.retrieve_fn = None
        self.retrieve_pool = None

        self.edit_dict = edit_dict

        self.lambda_t, self.lambda_t_steps = parse_lambda_config(args.lambda_t_config)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = args.data.split(os.pathsep)
        assert len(paths) > 0
        # find language pair automatically
        # if args.source_lang is None or args.target_lang is None:
        #     args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        # if args.source_lang is None or args.target_lang is None:
        #     raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.txt'))

        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))

        if args.inv_editor == 'levenshtein':
            edit_dict = RetrievePrototypeDataset.get_edit_dict()
        else:
            edit_dict = None

        if edit_dict is not None:
            print('| [edit] dictionary: {} types'.format(len(edit_dict)))

        return cls(args, src_dict, edit_dict)


    def load_dataset(self, split, epoch=0, **kwargs):

        if self.retrieve_fn is None:
            self.build_model(self.args)
            # raise ValueError(
            #     "retrieve_fn is None !"
            # )

        retrieve_dataset = None
        if self.retrieve_pool is None:
            paths = self.args.data.split(os.pathsep)
            assert len(paths) > 0
            data_path = paths[epoch % len(paths)]
            split_path = os.path.join(data_path, split)

            dataset = data_utils.load_indexed_dataset(
                split_path, self.dictionary, self.args.dataset_impl
            )

            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )

            lang_pair_dataset = LanguagePairDataset(
                dataset,
                dataset.sizes,
                self.src_dict,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                )

            if split == self.args.retrieve_split:
                print("split {} is used as the retrieve_pool".format(split))
                retrieve_dataset = lang_pair_dataset
            else:
                print("loading the retrieve split {}".format(self.args.retrieve_split))

                split_path = os.path.join(self.args.data, self.args.retrieve_split)
                dataset = data_utils.load_indexed_dataset(
                    split_path, self.dictionary, self.args.dataset_impl
                )

                if dataset is None:
                    raise FileNotFoundError(
                        "Dataset not found: {} ({})".format(
                            self.args.retrieve_split, split_path)
                    )

                if self.args.prune_num > 0:
                    retrieve_dataset = LanguagePairMapDataset(
                        dataset,
                        dataset.sizes,
                        self.src_dict,
                        left_pad_source=self.args.left_pad_source,
                        left_pad_target=self.args.left_pad_target,
                        )
                else:
                    retrieve_dataset = LanguagePairDataset(
                        dataset,
                        dataset.sizes,
                        self.src_dict,
                        left_pad_source=self.args.left_pad_source,
                        left_pad_target=self.args.left_pad_target,
                        )

            self.retrieve_pool = retrieve_dataset

        elif split == self.args.retrieve_split:
            print("skip reading split {} since it is used as the retrieve_pool"
                .format(split))
            lang_pair_dataset = self.retrieve_pool

        else:
            paths = self.args.data.split(os.pathsep)
            assert len(paths) > 0
            data_path = paths[epoch % len(paths)]
            split_path = os.path.join(data_path, split)

            dataset = data_utils.load_indexed_dataset(
                split_path, self.dictionary, self.args.dataset_impl
            )

            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )

            lang_pair_dataset = LanguagePairDataset(
                dataset,
                dataset.sizes,
                self.src_dict,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                )


        # always use unbiased estimator at test time
        # Avoid selecting self as templates at training time
        if 'train' not in split and self.args.criterion != 'guu_elbo':
            sampling = True
            masks = None
        else:
            def read_mask(fpath):
                with open(fpath) as fin:
                    return [int(x.rstrip()) for x in fin]

            sampling = options.eval_bool(self.args.reinforce)

            if os.path.exists(os.path.join(self.args.data, 'mask_id.txt')):
                masks = read_mask(os.path.join(self.args.data, 'mask_id.txt'))
            else:
                masks = None

        self.datasets[split] = RetrievePrototypeDataset(lang_pair_dataset,
            self.src_dict,
            retrieve_dataset=self.retrieve_pool,
            retrieve_fn=self.retrieve_fn,
            cuda=not self.args.cpu,
            num_samples=self.args.infer_ns,
            temperature=self.args.reinforce_temperature,
            sampling=sampling,
            edit_dict=self.edit_dict,
            split=split,
            masks=masks,
            )


    def build_model(self, args):
        if self.retrieve_fn is None:
            from fairseq import models
            model = models.build_model(args, self)

            def retrieve_fn(samples, split, model_=model):
                return model_.classifier(samples, split)

            self.retrieve_fn = retrieve_fn
            self.model = model

        return self.model

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False, update_lambda=True):

        model.train()
        model.set_num_updates(update_num)

        model.set_lambda_t(self.lambda_t)

        if update_num > 0:
            self.update_step(update_num)

        loss, sample_size, logging_output = criterion(model, sample, len(self.datasets['train']))
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)


        # following only supports single GPU training
        # if update_lambda and not ignore_grad:
        #     if not model.grad_lambda:
        #         with torch.no_grad():
        #             forget = math.pow(update_num + self.decay_rate, -self.forget_rate)
        #             new_lambda = model.get_alpha() + \
        #                 F.softmax(sample['net_input']['logits'], dim=1).mean(dim=0) * len(self.datasets['train'])
        #             # new_lambda = new_lambda.cpu()

        #         # update the Dirichlet posterior
        #         model.update_lambda((1. - forget) * model.get_lambda() +
        #             forget * new_lambda)
            # else:
            #     raise ValueError('There are bugs to be fixed when dealing with KL theta')

        return loss, sample_size, logging_output

    # to support distributed training
    def collect_lambda_stats(self, model, sample):
        return F.softmax(sample['net_input']['logits'], dim=1).sum(0)

    def distributed_update_lambda(self, model, lambda_stats_sum, nsentences, update_num, ignore_grad=False):
        if ignore_grad:
            return

        forget = math.pow(update_num + self.decay_rate, -self.forget_rate)
        new_lambda = model.get_alpha() + lambda_stats_sum / nsentences * len(self.datasets['train'])

        # update the Dirichlet posterior
        model.update_lambda((1. - forget) * model.get_lambda() +
            forget * new_lambda)


    def update_step(self, num_updates):
        def lambda_step_func(config, n_iter):
            """
            Update a lambda value according to its schedule configuration.
            """
            ranges = [i for i in range(len(config) - 1) if config[i][0] <= n_iter < config[i + 1][0]]
            if len(ranges) == 0:
                assert n_iter >= config[-1][0]
                return config[-1][1]
            assert len(ranges) == 1
            i = ranges[0]
            x_a, y_a = config[i]
            x_b, y_b = config[i + 1]
            return y_a + (n_iter - x_a) * float(y_b - y_a) / float(x_b - x_a)

        if self.lambda_t_steps is not None:
            self.lambda_t = lambda_step_func(self.lambda_t_steps, num_updates)

    def valid_step(self, sample, model, criterion, split='valid'):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample, len(self.datasets[split]))

        return loss, sample_size, logging_output

    def valid_iw_step(self, sample, model, criterion, mode='iw'):
        model.eval()
        with torch.no_grad():
            if mode == 'iw':
                loss, sample_size, logging_output = criterion.iw_eval_new(model, sample, 0, self.args.iw_nsamples,
                    retrieve_dataset=self.datasets[self.args.valid_subset])
            elif mode == 'entropy':
                loss, sample_size, logging_output = criterion.entropy_eval(model, sample, 0)
            else:
                raise ValueError("mode {} is not supported".format(mode))

        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None, mode='gen_sample'):
        if mode == "gen_sample":
            sample_z = models[0].sample_from_uniform_sphere(ns=sample['net_input']['src_tokens'].size(0))
            sample['net_input'].update({'edit_vecs': sample_z})
        elif mode == "gen_interpolation":
            def slerp(val, low, high):
                """spherical linear interpolation
                from: https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
                """
                low_norm = low/torch.norm(low, dim=1, keepdim=True)
                high_norm = high/torch.norm(high, dim=1, keepdim=True)
                omega = torch.acos((low_norm*high_norm).sum(1))
                so = torch.sin(omega)
                res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
                return res
            sample_z = models[0].sample_from_uniform_sphere(ns=sample['net_input']['src_tokens'].size(0) * 2 // self.args.gen_nz)
            new_sample_z = []
            for i in range(0, sample_z.size(0), 2):
                start = sample_z[i].unsqueeze(0)
                end = sample_z[i+1].unsqueeze(0)
                for weight in torch.arange(0, 1, 1/self.args.gen_nz):
                    new_sample_z.append(slerp(weight.item(), start, end))

            new_sample_z = torch.cat(new_sample_z, dim=0)
            assert new_sample_z.size(0) == sample['net_input']['src_tokens'].size(0)
            new_sample_z = new_sample_z.index_select(0, sample['net_input']['sort_order'])
            sample['net_input'].update({'edit_vecs': new_sample_z})


        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)

    def set_index_map(self, index_map):
        self.retrieve_pool.set_index_map(index_map)

    def reset_index_map(self):
        self.retrieve_pool.reset_index_map()

    def write_lambda(self, fout, model):
        fout.write('-------------top templates by q(theta)-------------\n')
        lambda_ = model.lambda_
        prob = lambda_ / lambda_.sum()

        max_, index_ = torch.topk(lambda_, 1000)
        for val, id_ in zip(max_, index_):
            sent = self.dictionary.string(self.retrieve_pool[id_.item()]['source'])
            fout.write('{}\t{}\t{}\n'.format(id_.item(), prob[id_].item(), sent))

        fout.write('\n---------------------------------\n')

    def write_template(self, sample, model, fout):
        if sample is None or len(sample) == 0:
            return

        net_input = sample['net_input']
        index = sample['id']

        revert_order = net_input['revert_order']
        src_tokens = net_input['src_tokens']
        temp_tokens = net_input['temp_tokens']

        src_tokens = src_tokens.index_select(0, revert_order).view(-1, model.infer_ns, src_tokens.size(-1))
        temp_tokens = temp_tokens.index_select(0, revert_order).view(-1, model.infer_ns, temp_tokens.size(-1))
        index = index.index_select(0, revert_order).view(-1, model.infer_ns)

        logits = net_input['logits']
        prob = F.softmax(logits, dim=1)
        max_, max_id = torch.topk(prob, 10, dim=1)

        bs = src_tokens.size(0)
        assert bs == logits.size(0)
        for i in range(bs):
            # if src_tokens.size(1) > 1:
            #     assert src_tokens[i, 0] == src_tokens[i, 1]
            sent = self.dictionary.string(utils.strip_pad(src_tokens[i, 0], self.dictionary.pad()))
            fout.write('\nsrc: {}\n'.format(sent))
            fout.write('-----------retrieved templates------------\n')
            for j in range(model.infer_ns):
                id_ = index[i, j].item()
                string_ = self.dictionary.string(utils.strip_pad(temp_tokens[i, j], self.dictionary.pad()))
                fout.write('{}\t{}\t{}\n'.format(id_, prob[i, id_].item(),
                    string_))

            fout.write('\n-----------top K templates------------\n')
            for val, id_ in zip(max_[i], max_id[i]):
                sent = self.dictionary.string(utils.strip_pad(
                    self.retrieve_pool[id_.item()]['source'], self.dictionary.pad()))
                fout.write('{}\t{}\t{}\n'.format(id_.item(), val.item(), sent))

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
    ):


        # Recreate epoch iterator every epoch cause the underlying
        # datasets are dynamic due to sampling.
        self.dataset_to_epoch_iter = {}
        epoch_iter = super().get_batch_iterator(
            dataset, max_tokens, max_sentences, max_positions,
            ignore_invalid_inputs, required_batch_size_multiple,
            seed, num_shards, shard_id, num_workers, epoch,
        )
        self.dataset_to_epoch_iter = {}
        return epoch_iter


    def build_generator(self, args):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        if getattr(args, "print_alignment", False):
            seq_gen_cls = SequenceGeneratorWithAlignment
        else:
            seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
        )

