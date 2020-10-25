import torch
import torch.nn as nn

from vae import VAEBase

class VMFVAE(VAEBase):
    """VAE base class"""
    def __init__(self, encoder, kappa,):
        super(VMFVAE, self).__init__(encoder)

        # self.args = args

        # loc = torch.zeros(self.nz, device=args.device)
        # scale = torch.ones(self.nz, device=args.device)

        # self.prior = torch.distributions.normal.Normal(loc, scale)

    def encode(self, src_tokens, src_lengths, nsamples=1, **kwargs=None):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """
        
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
