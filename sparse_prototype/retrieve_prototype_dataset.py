# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from fairseq import utils

from fairseq.data import data_utils, FairseqDataset, Dictionary



def output_collate(
    samples, logits, logits_topk, sample_orig, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if  len(samples) == 0:
            return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            print("| alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        tgt_id = torch.LongTensor([s['tgt_id'] for s in samples])
        tgt_id = tgt_id.index_select(0, sort_order)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    _, revert_order = sort_order.sort()
    batch = {
        'id': id,
        'tgt_id': tgt_id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'sample_orig': sample_orig,
        'net_input': {
            'temp_ids': id,
            'temp_tokens': src_tokens,
            'temp_lengths': src_lengths,
            'src_tokens': target,
            'src_lengths': tgt_lengths,
            'logits_topk': logits_topk,
            'logits': logits,
            'revert_order': revert_order,
        },
        'target': target,
    }

    # aligned pairs and edit path are required
    if samples[0].get('src_aligned', None) is not None:
        src_aligned = merge('src_aligned', left_pad=False).index_select(0, sort_order)
        aligned_length = torch.LongTensor([s['src_aligned'].numel() for s in samples]).index_select(0, sort_order)

        tgt_aligned = merge('tgt_aligned', left_pad=False).index_select(0, sort_order)
        edit_aligned = merge('edit_aligned', left_pad=False).index_select(0, sort_order)

        assert src_aligned.size(1) == edit_aligned.size(1)
        batch['net_input']['src_aligned'] = src_aligned
        batch['net_input']['tgt_aligned'] = tgt_aligned
        batch['net_input']['edit_aligned'] = edit_aligned
        batch['net_input']['aligned_length'] = aligned_length



    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch

def lang_pair_collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    _, revert_order = sort_order.sort()

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'revert_order': revert_order,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch


class RetrievePrototypeDataset(FairseqDataset):
    """
    Sets up a prototype dataset which takes a tgt batch, generates
    the prototype id with the classification function, and returns
    the corresponding `{generated prototype, input tgt}` batch.

    Args:
        tgt_dataset (~fairseq.data.FairseqDataset): the input dataset to be
            classified.
        tgt_dict (~fairseq.data.Dictionary): the dictionary of sentences.
        retrieve_fn (callable, optional): function to call to generate
            prototype ids. This is typically the `forward` method of a
            classification network. Pass in None when it is not available at initialization time, and
            use set_retrieve_fn function to set it when available.
        output_collater (callable, optional): function to call on the
            backtranslated samples to create the final batch
            (default: ``tgt_dataset.collater``).
        cuda: use GPU for generation
    """

    # this is class attribute
    # should be of type fairseq.data.FairseqDataset

    def __init__(
        self,
        tgt_dataset,
        tgt_dict,
        retrieve_dataset=None,
        retrieve_fn=None,
        cuda=True,
        num_samples=1,
        temperature=1,
        sampling=True,
        edit_dict=None,
        split=None,
        masks=None,
        **kwargs
    ):
        self.tgt_dataset = tgt_dataset
        self.retrieve_fn = retrieve_fn
        self.cuda = cuda if torch.cuda.is_available() else False
        self.tgt_dict = tgt_dict
        self.num_samples = num_samples
        self.temperature = temperature
        self.tgt_dict = tgt_dict
        self.sampling = sampling

        self.retrieve_dataset = retrieve_dataset
        self.edit_align = (edit_dict is not None)

        self.edit_dict = edit_dict
        self.split = split

        self.masks = masks

    @classmethod
    def get_edit_dict(self):
        # unchanged, substitute, delete, add
        tag_list = ['=', 'X', 'D', 'I']
        edit_dict = Dictionary()
        for tag in tag_list:
            edit_dict.add_symbol(tag)

        return edit_dict


    def __getitem__(self, index):
        """
        Returns a single sample from *tgt_dataset*.
        """
        return self.tgt_dataset[index]

    def __len__(self):
        return len(self.tgt_dataset)

    def get_string(self, index):
        return self.tgt_dict.string(self.tgt_dataset[index])

    def set_retrieve_fn(self, retrieve_fn):
        self.retrieve_fn = retrieve_fn

    def set_sampling(self, val):
        self.sampling = val

    def set_num_samples(self, val):
        self.num_samples = val

    def wrap_collate(self, samples):
        return lang_pair_collate(
            samples, pad_idx=self.tgt_dict.pad(),
            eos_idx=self.tgt_dict.eos(), left_pad_source=self.tgt_dataset.left_pad_source,
            left_pad_target=self.tgt_dataset.left_pad_target,
            input_feeding=self.tgt_dataset.input_feeding,
        )

    def retrieve_prototypes(self, samples, dataset, collate_fn,
                            retrieve_fn, num_samples=1,
                            temperature=1, cuda=True,
                            sampling=True,
                            edit_align=False,
                            edit_dict=None):
        """retrieve a list of samples.

        Given an input (*samples*) of the form:

            [{'id': source_id, 'source': 'hallo world'}]

        this will return:

            [{'id': prototype_id, 'source': *prototype*, 'target': 'hallo world'}]

        Args:
            samples (List[dict]): Individual samples are expected to have a 'source' key,
                which will become the 'target' after retrieving.
            dataset (~fairseq.data.FairseqDataset): the dataset to be used for indexing. Only
                the source side of this dataset will be used. After retrieving, the source
                sentences in this dataset will still be returns as source prototypes.
            collate_fn (callable): function to collate samples into a mini-batch ready to input
                to retrieve_fn.
            generate_fn (callable): function to generate classfication logits.
            cuda (bool): use GPU for generation (default: ``True``)

        Returns:
            dict: contains `logits` and `samples`, which are an updated list of samples
                with a retrieved prototype source
        """

        assert dataset is not None

        collated_samples = collate_fn(samples)

        collated_samples = utils.move_to_cuda(collated_samples) if cuda else collated_samples


        logits = retrieve_fn(collated_samples, self.split)
        logits = logits.index_select(0, collated_samples['net_input']['revert_order'])

        # logits = logits / temperature

        # avoid selecting self as templates at training time
        if self.masks is not None:
            logits_min, min_index = torch.min(logits, 1)
            mask_ids = [self.masks[s['id']] if self.masks[s['id']] != -1 else min_index[i].item() for i, s in enumerate(samples)]
            mask_ids = torch.LongTensor(mask_ids)
            if cuda:
                mask_ids = mask_ids.cuda()

            logits.index_fill_(1, mask_ids, logits_min.min())


        bs = logits.size(0)

        # (batch, nsample) -> (batch * nsample)
        if sampling:
            prototype_ids = torch.multinomial(F.softmax(logits / temperature, dim=1),
                num_samples, replacement=True).view(-1)
            logits_topk = None
            # prototype_ids = torch.multinomial(F.softmax(logits, dim=1),
            #     num_samples, replacement=True).view(-1)
        else:
            logits_topk, prototype_ids = torch.topk(logits, num_samples, dim=1)
            prototype_ids = prototype_ids.view(-1)

        samples_expand = []
        for i in range(bs):
            samples_expand.extend([samples[i]] * num_samples)


        # List[dict]
        prototypes = [dataset[id_.item()] for id_ in prototype_ids]
        assert prototypes[0]['id'] == prototype_ids[0].item()
        # s = utils.move_to_cuda(collated_samples) if cuda else collated_samples
        # generated_sources = generate_fn(s)

        # find the minimum edit path from source to target
        if edit_align:
            import edlib

            def flat_cigar(cigar):
                r = []
                pointer = 0

                while pointer < len(cigar):
                    num = []
                    while cigar[pointer].isdigit():
                        num.append(cigar[pointer])
                        pointer += 1
                    num = int(''.join(num))

                    r.extend([cigar[pointer]] * num)
                    pointer += 1

                return r

            src_aligned_l = []
            tgt_aligned_l = []
            edit_aligned_l = []
            for prototype_s, tgt_s in zip(prototypes, samples_expand):
                query, answer = prototype_s['source'], tgt_s['source']
                query = [x.item() for x in query]
                answer = [x.item() for x in answer]
                res = edlib.align(answer, query, task='path')

                _edit_aligned = flat_cigar(res['cigar'])

                _edit_aligned_l = []
                _src_aligned_l = []
                _tgt_aligned_l = []
                src_cur = tgt_cur = 0

                for edit in _edit_aligned:
                    if edit == '=' or edit == 'X':
                        _src_aligned_l.append(query[src_cur])
                        _tgt_aligned_l.append(answer[tgt_cur])
                        src_cur += 1
                        tgt_cur += 1
                    elif edit == 'I':
                        _src_aligned_l.append(self.tgt_dict.unk_index)
                        _tgt_aligned_l.append(answer[tgt_cur])
                        tgt_cur += 1
                    elif edit == 'D':
                        _src_aligned_l.append(query[src_cur])
                        _tgt_aligned_l.append(self.tgt_dict.unk_index)
                        src_cur += 1
                    else:
                        raise ValueError('{} edit operation is invalid!'.format(edit))

                    _edit_aligned_l.append(edit_dict.index(edit))

                assert len(_src_aligned_l) == len(_tgt_aligned_l) == len(_edit_aligned_l)
                src_aligned_l.append(torch.LongTensor(_src_aligned_l))
                tgt_aligned_l.append(torch.LongTensor(_tgt_aligned_l))
                edit_aligned_l.append(torch.LongTensor(_edit_aligned_l))


        if not edit_align:
            # Note that the 'id' here is the prototype id instead of the input target ids
            return {
                'logits': logits,
                'logits_topk': logits_topk,
                'sample_orig': samples, # to compute importance weighted likelihood
                'samples': [
                {'id': prototype_s['id'], 'tgt_id': tgt_s['id'], 'source': prototype_s['source'], 'target': tgt_s['source']}
                for tgt_s, prototype_s in zip(samples_expand, prototypes)
                ]
            }
        else:
             return {
                'logits': logits,
                'logits_topk': logits_topk,
                'sample_orig': samples, # to compute importance weighted likelihood
                'samples': [
                {'id': prototype_s['id'], 'tgt_id': tgt_s['id'], 'source': prototype_s['source'], 'target': tgt_s['source'],
                'src_aligned': src_a, 'tgt_aligned': tgt_a, 'edit_aligned': edit_a}
                for tgt_s, prototype_s, src_a, tgt_a, edit_a in zip(samples_expand, prototypes, src_aligned_l, tgt_aligned_l, edit_aligned_l)
                ]
            }


    def collater(self, samples):
        """Merge and backtranslate a list of samples to form a mini-batch.

        Using the samples from *tgt_dataset*, load a collated target sample to
        feed to the retrieve function. Then sample indexes, index the samples from
        *tgt_dataset* as prototypes,

        Note: we expect *tgt_dataset* to provide a function `collater()` that
        will collate samples into the format expected by *retrieve_fn*.
        After retrieving and indexing, we will feed the new list of samples (i.e., the
        `(retrieved source, original target)` pairs) to *output_collater*
        and return the result.

        Args:
            samples (List[dict]): samples to classifiy and collate

        Returns:
            dict: a mini-batch with keys coming from *output_collater*
        """
        if len(samples) == 0:
            return {}


        if samples[0].get('is_dummy', False):
            return samples
        samples = self.retrieve_prototypes(
            samples=samples,
            dataset=self.retrieve_dataset,
            collate_fn=self.wrap_collate,
            retrieve_fn=(
                lambda net_input, split: self.retrieve_fn(net_input, split)
            ),
            num_samples=self.num_samples,
            temperature=self.temperature,
            cuda=self.cuda,
            sampling=self.sampling,
            edit_align=self.edit_align,
            edit_dict=self.edit_dict,
        )

        return output_collate(
            samples['samples'], samples['logits'], logits_topk=samples['logits_topk'], sample_orig=samples['sample_orig'],
            pad_idx=self.tgt_dict.pad(),
            eos_idx=self.tgt_dict.eos(), left_pad_source=self.tgt_dataset.left_pad_source,
            left_pad_target=self.tgt_dataset.left_pad_target,
            input_feeding=self.tgt_dataset.input_feeding,
        )

    def num_tokens(self, index):
        """Just use the tgt dataset num_tokens"""
        return self.tgt_dataset.num_tokens(index)

    def ordered_indices(self):
        """Just use the tgt dataset ordered_indices"""
        return self.tgt_dataset.ordered_indices()

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used
        when filtering a dataset with ``--max-positions``.

        Note: we use *tgt_dataset* to approximate the length of the source
        sentence, since we do not know the actual length until after
        backtranslation.
        """
        tgt_size = self.tgt_dataset.size(index)[0]
        return (tgt_size, tgt_size)

    @property
    def supports_prefetch(self):
        return getattr(self.tgt_dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        return self.tgt_dataset.prefetch(indices)
