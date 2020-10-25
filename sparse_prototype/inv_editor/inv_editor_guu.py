import torch
import torch.nn as nn

from .inv_editor import InvEditorBase, Embedding



class GuuInvEditor(InvEditorBase):
    """the inverse editor from https://arxiv.org/abs/1709.08878"""
    def __init__(self, embed_dim, dictionary, pretrained_embed=None, cuda=True):
        super(GuuInvEditor, self).__init__(embed_dim)

        self.embed_dim = embed_dim
        self.padding_idx = dictionary.pad()
        num_embeddings = len(dictionary)
        self.device = torch.device('cuda' if cuda else 'cpu')
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed


    def forward(self, src_tokens, temp_tokens, **kwargs):
        """
        Args: 
            src_tokens (LongTensor): (batch, seq_len)
            temp_tokens (LongTensor): (batch, seq_len)

        Returns: Tensor1
            Tensor1: the representation with shape [batch, embed_dim]
        """

        res = []

        for src_tokens_, temp_tokens_ in zip(src_tokens, temp_tokens):
            src_token_list, temp_token_list = src_tokens_.tolist(), temp_tokens_.tolist()
            delete_words = set(src_token_list) - set(temp_token_list) - set([self.padding_idx])
            insert_words = set(temp_token_list) - set(src_token_list) - set([self.padding_idx])

            delete_words_t = torch.tensor(list(delete_words), dtype=torch.long, device=self.device)
            insert_words_t = torch.tensor(list(insert_words), dtype=torch.long, device=self.device)

            res.append(torch.cat((self.embed_tokens(delete_words_t).sum(0), 
                self.embed_tokens(insert_words_t).sum(0)), dim=0).unsqueeze(0))

        return torch.cat(res, dim=0)

    @property
    def output_units(self):
        return 2 * self.embed_dim
            