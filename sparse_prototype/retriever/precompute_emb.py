import os
import json
import subprocess
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import chain
from fairseq import utils
# from datasets import load_dataset

def get_file_len(file):
    proc = subprocess.run(['wc', '-l', file], capture_output=True)
    return int(proc.stdout.decode('utf-8').split()[0])

class PrecomputeEmbedRetriever(nn.Module):
    """the retriever module based on pretrained embeddings"""
    def __init__(self, args, dictionary, emb_dataset_path, rescale=1.,
        linear_bias=False, freeze=False, nlayers=0, emb_size=768):

        super(PrecomputeEmbedRetriever, self).__init__()

        self.dict = dictionary

        self.dataset = {}
        split_list = ['template', 'train', 'valid', 'test']
        # for split in split_list:
        #     if os.path.isfile(f'{emb_dataset_path}.{split}.csv.gz'):
        #         self.dataset[split] = load_dataset('csv',
        #                                     data_files=f'{emb_dataset_path}.{split}.csv.gz',
        #                                     cache_dir='hf_dataset_cache')

        infer_dataset = args.data.split('/')[-1]
        for split in split_list:
            if os.path.isfile(f'{emb_dataset_path}.{split}.npy'):

                # emb_size is defaulted too 768 (BERT)
                # here we use float16 since the embeddings are presaved in float16
                example_size = get_file_len(f'datasets/{infer_dataset}/{split}.txt')
                self.dataset[split] = np.memmap(f'{emb_dataset_path}.{split}.npy', 
                    dtype='float16', mode='r', shape=(example_size, emb_size))

        template_weight = self.dataset['template']
        num_template = len(template_weight)
        # template_weight = []

        # for i in range(num_template):
        #     template_weight.append(json.loads(template_group[i]['embedding']))

        template_weight = torch.tensor(template_weight)

        print('read template embeddings complete!')

        nfeat = template_weight.size(1)

        self.linear1 = nn.Linear(nfeat, nfeat, bias=False)

        modules = []
        for _ in range(nlayers):
            linear_tmp = nn.Linear(nfeat, nfeat)
            with torch.no_grad():
                nn.init.eye_(linear_tmp.weight)
                nn.init.zeros_(linear_tmp.bias)

            modules.extend([linear_tmp, nn.ReLU()])

        self.middle = nn.Sequential(*modules)

        # output layer
        self.linear2 = nn.Linear(nfeat, num_template, bias=linear_bias)


        with torch.no_grad():
            nn.init.eye_(self.linear1.weight)
            self.linear1.weight.data = self.linear1.weight.data / rescale
            self.linear2.weight.copy_(template_weight)

            if linear_bias:
                self.linear2.bias.zero_()

        self.linear2.weight.requires_grad = False

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        self.prune_index = None
        self.prune_linear2_weight = None

    def forward(self, samples, split, key='id'):
        """
        Args:
            samples (dict): input dict with keys 'net_input', 'id', etc.


        Returns:
            logits (tensor): shape (B, class_num)
        """

        id_ = samples[key]

        # embeddings = [json.loads(self.dataset[split]['train'][i.item()]['embedding']) for i in id_]
        embeddings = [self.dataset[split][i.item()] for i in id_]
        embeddings = self.linear1.weight.new_tensor(embeddings)

        if self.prune_linear2_weight is None:
            logits = self.linear2(self.middle(self.linear1(embeddings)))
        else:
            logits = F.linear(self.middle(self.linear1(embeddings)), self.prune_linear2_weight)

        # mask itself only during training to mitigate overfitting
        # while this is pretty rough, but should be enough considering both time and memory efficiency

        # if split == 'train':
        #     with torch.no_grad():
        #         mask = (self.linear2(embeddings) - (embeddings * embeddings).sum(1).unsqueeze(1)).abs() < 1e-5

        #     logits = logits.masked_fill(mask, logits.min().item())

        # prune at test time
        if self.prune_index is not None:
            logits.index_fill_(1, self.prune_index, logits.min().item() - 1e3)

        # mask itself
        # mask = logits.new_zeros(logits.size(), dtype=torch.bool)
        # for i, id_ in enumerate(samples['id']):
        #     mask[i, id_.item()] = 1
        # logits = logits.masked_fill(mask, logits.min().item())

        return logits

    def set_prune_index(self, index):
        # self.prune_index = index
        self.prune_linear2_weight = self.linear2.weight[index]


    def reset_prune_index(self):
        # self.prune_index = None

        self.prune_linear2_weight = None

