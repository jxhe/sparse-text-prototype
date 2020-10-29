# import h5py
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import *
from fairseq import utils
from datasets import load_dataset

class BertRetriever(nn.Module):
    """the retriever module based on pretrained sentence-Bert embeddings"""
    def __init__(self, dictionary, emb_dataset_path,
        rescale=1., linear_bias=False, stop_grad=False, 
        freeze=False, cuda=True, sentbert=False):

        super(BertRetriever, self).__init__()

        self.dict = dictionary
        self.stop_grad = stop_grad
        self.device = torch.device("cuda" if cuda else "cpu")

        template_group = load_dataset('csv',
                                      data_files=f'{emb_dataset_path}.template.csv.gz',
                                      cache_dir='hf_dataset_cache')

        template_group = template_group['train']

        num_template = len(template_group)
        template_weight = []

        for i in range(num_template):
            template_weight.append(json.loads(template_group[i]['embedding']))

        template_weight = torch.tensor(template_weight)

        print('read template embeddings complete!')

        nfeat = template_weight.size(1)

        self.linear1 = nn.Linear(nfeat, nfeat, bias=False)
        self.linear2 = nn.Linear(nfeat, num_template, bias=linear_bias)

        # this should be consistent with pre-saved template embeddings
        model_name = 'bert-base-uncased' if not sentbert else 'sentence-transformers/bert-base-nli-mean-tokens'

        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if stop_grad:
            for param in self.encoder.parameters():
                param.requires_grad = False

        with torch.no_grad():
            nn.init.eye_(self.linear1.weight)
            self.linear1.weight.data = self.linear1.weight.data / rescale
            self.linear2.weight.copy_(template_weight)

            if linear_bias:
                self.linear2.bias.zero_()

        self.linear2.weight.requires_grad = False

        # self.linear1.weight.requires_grad = False

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        self.prune_index = None
        self.prune_linear2_weight = None


    def encode(self, batches, maxlen=500):
        features = self.tokenizer.batch_encode_plus(batches, padding=True,
            return_attention_mask=True, return_token_type_ids=True, 
            truncation=True, max_length=maxlen, return_tensors='pt')
        attention_mask = features['attention_mask'].to(self.device)
        input_ids = features['input_ids'].to(self.device)
        token_type_ids= features['token_type_ids'].to(self.device)

        # (batch, seq_len, nfeature)
        token_embeddings = self.encoder(input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)[0]

        # mean of context embeddings as sentence embeddings
        embeddings = (attention_mask.unsqueeze(-1) * token_embeddings).sum(1) / attention_mask.sum(1).unsqueeze(-1)

        return embeddings

    def forward(self, samples, split=None, key=None):
        """
        Args:
            samples (dict): input dict with keys 'net_input', 'id', etc.


        Returns:
            logits (tensor): shape (B, num_template)
        """

        net_input = samples['net_input']
        x = net_input['src_tokens']
        bs = x.size(0)
        sent_strings = []
        for sent in x:
            sent = utils.strip_pad(sent, self.dict.pad())
            sent_strings.append(self.dict.string(sent).strip('\n'))

        # (bs x nfeats)
        if self.stop_grad:
            self.eval()
            with torch.no_grad():
                embeddings = self.encode(sent_strings)
                                                 
        else:
            embeddings = self.encode(sent_strings)
                                             

        if self.prune_linear2_weight is None:
            logits = self.linear2(self.linear1(embeddings))
        else:
            logits = F.linear(self.linear1(embeddings), self.prune_linear2_weight)

        # mask itself only during training to mitigate overfitting
        # while this is pretty rough, but should be enough considering both time and memory efficiency
        
        # if split == 'train':
        #     with torch.no_grad():
        #         mask = (self.linear2(embeddings) - (embeddings * embeddings).sum(1).unsqueeze(1)).abs() < 1e-5

        #     logits = logits.masked_fill(mask, logits.min().item())
        # mask itself
        # mask = logits.new_zeros(logits.size(), dtype=torch.bool)
        # for i, id_ in enumerate(samples['id']):
        #     mask[i, id_.item()] = 1
        # logits = logits.masked_fill(mask, logits.min().item())

        # prune at test time
        if self.prune_index is not None:
            logits.index_fill_(1, self.prune_index, logits.min().item() - 1e3)

        return logits

    def set_prune_index(self, index):
        # self.prune_index = index
        self.prune_linear2_weight = self.linear2.weight[index]


    def reset_prune_index(self):
        # self.prune_index = None

        self.prune_linear2_weight = None
