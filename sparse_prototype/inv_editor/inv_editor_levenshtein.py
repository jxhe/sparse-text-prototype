import torch
import torch.nn as nn

from .inv_editor import InvEditorBase, Embedding



class LevenshteinInvEditor(InvEditorBase):
    """the inverse editor from https://arxiv.org/abs/1709.08878"""
    def __init__(self, token_embed_dim, edit_embed_dim, hidden_size, 
        tgt_dict, edit_dict, num_layers=1, pretrained_token_embed=None):
        super(LevenshteinInvEditor, self).__init__(hidden_size)


        self.hidden_size = hidden_size
        self.padding_idx = tgt_dict.pad()
        num_token_embeddings = len(tgt_dict)
        num_edit_embeddings = len(edit_dict)

        if pretrained_token_embed is None:
            self.embed_tokens = Embedding(num_token_embeddings, token_embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_token_embed

        self.embed_edit = Embedding(num_edit_embeddings, edit_embed_dim, self.padding_idx)
        self.num_layers=num_layers

        self.lstm = nn.LSTM(
            input_size=token_embed_dim * 2 + edit_embed_dim,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
        )


    def forward(self, src_aligned, tgt_aligned, edit_aligned, aligned_length, **kwargs):
        """
        Args: 
            src_aligned (LongTensor): (batch, seq_len)
            tgt_aligned (LongTensor): (batch, seq_len)

        Returns: Tensor1
            Tensor1: the representation with shape [batch, embed_dim]
        """

        bsz, seqlen = src_aligned.size()

        edit_embed = self.embed_edit(edit_aligned)
        src_embed = self.embed_tokens(src_aligned)
        tgt_embed = self.embed_tokens(tgt_aligned)

        x = torch.cat((edit_embed, src_embed, tgt_embed), -1)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        packed_x = nn.utils.rnn.pack_padded_sequence(x, aligned_length.data.tolist(), enforce_sorted=False)
        state_size = 2 * self.num_layers, bsz, self.hidden_size

        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)

        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_idx)

        def combine_bidir(outs):
            out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
            return out.view(self.num_layers, bsz, -1)
        

        return combine_bidir(final_hiddens)[-1]

    @property
    def output_units(self):
        return 2 * self.hidden_size