import torch
import torch.nn as nn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

class InvEditorBase(nn.Module):
    """Base class for Inverse Editor p(z|t, x)"""
    def __init__(self, embed_dim):
        super(InvEditorBase, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, src_tokens, temp_tokens, **kwargs):
        """
        Args: 
            src_tokens (LongTensor): (batch, seq_len)
            temp_tokens (LongTensor): (batch, seq_len)

        Returns: Tensor1
            Tensor1: the representation with shape [batch, embed_dim]
        """

        raise NotImplementedError