import math
import torch
import torch.nn as nn

from .distribution import vMF

class VAEEncoder(nn.Module):
    """VAEEncoder class"""
    def __init__(self, encoder, hidden_dim, latent_dim, 
        nsamples=1, dist="vmf", kappa=1, cuda=True
    ):
        super(VAEEncoder, self).__init__()
        self.encoder = encoder

        # self.args = args

        self.nsamples = nsamples
        self.kappa = kappa

        if dist == "vmf":
            self.dist = vMF(hidden_dim, latent_dim, kappa, cuda=cuda)
        else:
            raise ValueError("the {} distribution is not supported".format(dist))


        # loc = torch.zeros(self.nz, device=args.device)
        # scale = torch.ones(self.nz, device=args.device)

        # self.prior = torch.distributions.normal.Normal(loc, scale)

    def forward(self, src_tokens, src_lengths, temp_tokens, temp_lengths, **kwargs):
        """
        Args: 
            src_tokens (LongTensor): (batch, seq_len)
            temp_tokens (LongTensor): (batch, seq_len)

        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """
        bs, seq_len = src_tokens.size()

        # (batch, hidden_dim)
        hidden_code = self.encoder(src_tokens=src_tokens, 
            src_lengths=src_lengths, 
            temp_tokens=temp_tokens, 
            temp_lengths=temp_lengths,
            **kwargs)

        # ret_dict: a dict that contains parameters of the approx posterior
        # KL: (batch)
        # z: (batch, nsamples, nz)
        ret_dict, KL, z = self.dist.build_bow_rep(hidden_code, self.nsamples)

        return z, KL, ret_dict

    def log_prior_vmf_density(self, x):
        """compute log density under the uniform vmf prior
        Args:
            x: tensor with shape (batch, *, latent_dim)
        """


        return self.dist.log_density(0, x)

    def log_vmf_density(self, x, mu):

        return self.dist.log_density(self.kappa, x, mu)



    def encode_stats(self, x):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the mean of latent z with shape [batch, nz]
            Tensor2: the logvar of latent z with shape [batch, nz]
        """

        return self.encoder(x)

    def decode(self, z, strategy, K=5):
        """generate samples from z given strategy
        Args:
            z: [batch, nsamples, nz]
            strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter
        Returns: List1
            List1: a list of decoded word sequence
        """

        if strategy == "beam":
            return self.decoder.beam_search_decode(z, K)
        elif strategy == "greedy":
            return self.decoder.greedy_decode(z)
        elif strategy == "sample":
            return self.decoder.sample_decode(z)
        else:
            raise ValueError("the decoding strategy is not supported")

    def reconstruct(self, x, decoding_strategy="greedy", K=5):
        """reconstruct from input x
        Args:
            x: (batch, *)
            decoding_strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter (if applicable)
        Returns: List1
            List1: a list of decoded word sequence
        """
        z = self.sample_from_inference(x).squeeze(1)

        return self.decode(z, decoding_strategy, K)


    def loss(self, x, kl_weight, nsamples=1):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list
        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: total loss [batch]
            Tensor2: reconstruction loss shape [batch]
            Tensor3: KL loss shape [batch]
        """

        z, KL = self.encode(x, nsamples)

        # (batch)
        reconstruct_err = self.decoder.reconstruct_error(x, z).mean(dim=1)


        return reconstruct_err + kl_weight * KL, reconstruct_err, KL

    def nll_iw(self, x, nsamples, ns=100):
        """compute the importance weighting estimate of the log-likelihood
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list
            nsamples: Int
                the number of samples required to estimate marginal data likelihood
        Returns: Tensor1
            Tensor1: the estimate of log p(x), shape [batch]
        """

        # compute iw every ns samples to address the memory issue
        # nsamples = 500, ns = 100
        # nsamples = 500, ns = 10
        tmp = []
        for _ in range(int(nsamples / ns)):
            # [batch, ns, nz]
            # param is the parameters required to evaluate q(z|x)
            z, param = self.encoder.sample(x, ns)

            # [batch, ns]
            log_comp_ll = self.eval_complete_ll(x, z)
            log_infer_ll = self.eval_inference_dist(x, z, param)

            tmp.append(log_comp_ll - log_infer_ll)

        ll_iw = torch.logsumexp(torch.cat(tmp, dim=-1), dim=-1) - math.log(nsamples)

        return -ll_iw

    def KL(self, x):
        _, KL = self.encode(x, 1)

        return KL

    def eval_prior_dist(self, zrange):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/space
        """

        # (k^2)
        return self.prior.log_prob(zrange).sum(dim=-1)

    def eval_complete_ll(self, x, z):
        """compute log p(z,x)
        Args:
            x: Tensor
                input with shape [batch, seq_len]
            z: Tensor
                evaluation points with shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log p(z,x) Tensor with shape [batch, nsamples]
        """

        # [batch, nsamples]
        log_prior = self.eval_prior_dist(z)
        log_gen = self.eval_cond_ll(x, z)

        return log_prior + log_gen

    def eval_cond_ll(self, x, z):
        """compute log p(x|z)
        """

        return self.decoder.log_probability(x, z)

    def eval_log_model_posterior(self, x, grid_z):
        """perform grid search to calculate the true posterior
         this function computes p(z|x)
        Args:
            grid_z: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/pace
        Returns: Tensor
            Tensor: the log posterior distribution log p(z|x) with
                    shape [batch_size, K^2]
        """
        try:
            batch_size = x.size(0)
        except:
            batch_size = x[0].size(0)

        # (batch_size, k^2, nz)
        grid_z = grid_z.unsqueeze(0).expand(batch_size, *grid_z.size()).contiguous()

        # (batch_size, k^2)
        log_comp = self.eval_complete_ll(x, grid_z)

        # normalize to posterior
        log_posterior = log_comp - torch.logsumexp(log_comp, dim=1, keepdim=True)

        return log_posterior

    def sample_from_prior(self, nsamples):
        """sampling from prior distribution
        Returns: Tensor
            Tensor: samples from prior with shape (nsamples, nz)
        """
        return self.prior.sample((nsamples,))


    def sample_from_inference(self, x, nsamples=1):
        """perform sampling from inference net
        Returns: Tensor
            Tensor: samples from infernece nets with
                shape (batch_size, nsamples, nz)
        """
        z, _ = self.encoder.sample(x, nsamples)

        return z


    def sample_from_posterior(self, x, nsamples):
        """perform MH sampling from model posterior
        Returns: Tensor
            Tensor: samples from model posterior with
                shape (batch_size, nsamples, nz)
        """

        # use the samples from inference net as initial points
        # for MCMC sampling. [batch_size, nsamples, nz]
        cur = self.encoder.sample_from_inference(x, 1)
        cur_ll = self.eval_complete_ll(x, cur)
        total_iter = self.args.mh_burn_in + nsamples * self.args.mh_thin
        samples = []
        for iter_ in range(total_iter):
            next = torch.normal(mean=cur,
                std=cur.new_full(size=cur.size(), fill_value=self.args.mh_std))
            # [batch_size, 1]
            next_ll = self.eval_complete_ll(x, next)
            ratio = next_ll - cur_ll

            accept_prob = torch.min(ratio.exp(), ratio.new_ones(ratio.size()))

            uniform_t = accept_prob.new_empty(accept_prob.size()).uniform_()

            # [batch_size, 1]
            mask = (uniform_t < accept_prob).float()

            mask_ = mask.unsqueeze(2)

            cur = mask_ * next + (1 - mask_) * cur
            cur_ll = mask * next_ll + (1 - mask) * cur_ll

            if iter_ >= self.args.mh_burn_in and (iter_ - self.args.mh_burn_in) % self.args.mh_thin == 0:
                samples.append(cur.unsqueeze(1))


        return torch.cat(samples, dim=1)
