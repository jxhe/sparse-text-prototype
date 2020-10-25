import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from sentence_transformers import SentenceTransformer

class SentBert(nn.Module):
    """the retriever module based on pretrained sentence-Bert embeddings"""
    def __init__(self, class_num, dictionary, retrieve_embed,
        linear_bias=False, stop_grad=False, freeze=False):

        super(SentBert, self).__init__()

        self.dict = dictionary
        self.stop_grad = stop_grad

        sent_embed = []
        with h5py.File(retrieve_embed, 'r') as fin:
            for i in range(class_num):
                sent_embed.append(fin[str(i)].value)

        print('read Bert embed from {} complete!'.format(retrieve_embed))

        sent_embed = torch.tensor(sent_embed)
        nfeat = sent_embed.size(1)

        self.linear1 = nn.Linear(nfeat, nfeat, bias=False)
        self.linear2 = nn.Linear(nfeat, class_num, bias=linear_bias)

        self.encoder = SentenceTransformer('bert-base-nli-mean-tokens')

        if stop_grad:
            for param in self.encoder.parameters():
                param.requires_grad = False

        with torch.no_grad():
            nn.init.eye_(self.linear1.weight)
            self.linear2.weight.copy_(sent_embed)

            if linear_bias:
                self.linear2.bias.zero_()

        self.linear2.weight.requires_grad = False

        # self.linear1.weight.requires_grad = False

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, samples, split=None):
        """
        Args:
            samples (dict): input dict with keys 'net_input', 'id', etc.


        Returns:
            logits (tensor): shape (B, class_num)
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
            with torch.no_grad():
                embeddings = self.encoder.encode(sent_strings,
                                                 batch_size=bs,
                                                 online=True,
                                                 show_progress_bar=False,
                                                 )
        else:
            embeddings = self.encoder.encode(sent_strings,
                                             batch_size=bs,
                                             online=True,
                                             show_progress_bar=False,
                                             )

        logits = self.linear2(self.linear1(embeddings))

        # mask itself only during training to mitigate overfitting
        # while this is pretty rough, but should be enough considering both time and memory efficiency
        
        if split == 'train':
            with torch.no_grad():
                mask = (self.linear2(embeddings) - (embeddings * embeddings).sum(1).unsqueeze(1)).abs() < 1e-5

            logits = logits.masked_fill(mask, logits.min().item())
        # mask itself
        # mask = logits.new_zeros(logits.size(), dtype=torch.bool)
        # for i, id_ in enumerate(samples['id']):
        #     mask[i, id_.item()] = 1
        # logits = logits.masked_fill(mask, logits.min().item())

        return logits

