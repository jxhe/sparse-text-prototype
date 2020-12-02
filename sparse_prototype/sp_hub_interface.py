import argparse
import copy
import logging
import os
from typing import List, Dict, Iterator, Tuple, Any

import torch
from torch import nn

from fairseq import utils
from fairseq import hub_utils
from fairseq.data import encoders


# logger = logging.getLogger(__name__)


class TemplateHubInterface(hub_utils.GeneratorHubInterface):
    """The hub interface
    """
    
    def __init__(self, args, task, models):
        super().__init__(args, task, models)
        
    def generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        beam: int = 5,
        verbose: bool = False,
        skip_invalid_size_inputs=False,
        inference_step_args=None,
        **kwargs,
    ) -> List[List[Dict[str, torch.Tensor]]]:
        if torch.is_tensor(tokenized_sentences) and tokenized_sentences.dim() == 1:
            return self.generate(
                tokenized_sentences.unsqueeze(0), beam=beam, verbose=verbose, **kwargs
            )[0]

        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator(gen_args)

        results = []
        for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(generator, self.models, batch, **inference_step_args)
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

        if verbose:

            def getarg(name, default):
                return getattr(gen_args, name, getattr(self.args, name, default))

            for source_tokens, target_hypotheses in zip(tokenized_sentences, outputs):
                src_str_with_unk = self.string(source_tokens)
                logger.info('S\t{}'.format(src_str_with_unk))
                for hypo in target_hypotheses:
                    hypo_str = self.decode(hypo['tokens'])
                    logger.info('H\t{}\t{}'.format(hypo['score'], hypo_str))
                    logger.info('P\t{}'.format(
                        ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                    ))
                    if hypo['alignment'] is not None and getarg('print_alignment', False):
                        logger.info('A\t{}'.format(
                            ' '.join(map(lambda x: str(utils.item(x)), hypo['alignment'].int().cpu()))
                        ))
        return outputs
