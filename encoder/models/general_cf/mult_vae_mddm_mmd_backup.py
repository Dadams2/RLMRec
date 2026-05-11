import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
import numpy as np

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class DeepKernel(nn.Module):
    """
    Deep kernel network for learning adaptive kernel functions.
    This enables the MMD to capture high-order distribution mismatches.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(DeepKernel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.output_dim = hidden_dim // 2
        
    def forward(self, x):
        """Transform input to kernel feature space"""
        return self.network(x)


class mult_vae_MDDM_MMD(BaseModel):
    """
    MultVAE with Mixing Divergence using Maximum Mean Discrepancy (MMD).
    
    Advantages over KL-based MDDM:
    - Can detect high-order distribution mismatches through learned kernels
    - Avoids Gaussian distribution assumptions
    - More stable when distributions have low overlap
    - Non-parametric distribution comparison
    """
    def __init__(self, data_handler):
        super(mult_vae_MDDM_MMD, self).__init__(data_handler)

        self.beta = self.hyper_config['beta']
        self.data_handler = data_handler

        # According to the original paper, the default structure is [item_num, 600, 200, 600, item_num]
        self.p_dims = [200, 600, self.item_num]
        self.q_dims = [self.item_num, 600, 200]

        # Compute the mean and variance in parallel.
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]

        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])]
        )

        # Load the representations of user (or item) profiles.
        self.usrprf_embeds = torch.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = torch.tensor(configs['itmprf_embeds']).float().cuda()  # [item_num, 1536]

        self.mlp = nn.Sequential(
            nn.Linear(self.itmprf_embeds.shape[1], 600),
            nn.Tanh(),
            nn.Linear(600, 400)
        )

        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])]
        )

        self.drop = nn.Dropout(self.hyper_config['dropout'])

        # Deep kernel networks for MMD computation
        # Kernel for collaborative space distribution q_phi
        self.kernel_q = DeepKernel(self.q_dims[-1], hidden_dim=128)
        # Kernel for language space distribution p_phi
        self.kernel_p = DeepKernel(self.q_dims[-1], hidden_dim=128)
        
        # MMD kernel bandwidth (learnable or fixed)
        self.use_learnable_bandwidth = self.hyper_config.get('learnable_bandwidth', True)
        if self.use_learnable_bandwidth:
            self.log_bandwidth = nn.Parameter(torch.tensor(0.0))
        else:
            self.bandwidth = self.hyper_config.get('mmd_bandwidth', 1.0)

        self.final_embeds = None
        self.is_training = False

    def encode(self, x, user_emb):
        h = self.drop(x)
        # [batch, item_num] * [item_num, dim] = [batch, dim]
        hidden = torch.matmul(h, self.itmprf_embeds) + user_emb
        hidden = self.mlp(hidden)

        mu_llm = hidden[:, :200]
        logvar_llm = hidden[:, 200:]

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]

        return mu, mu_llm, logvar, logvar_llm

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def reparameterize(self, mu, logvar):
        if self.is_training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std) + mu
        else:
            z = mu
        return z

    def compute_mmd(self, z_src, z_tgt, kernel_z_src=None, kernel_z_tgt=None):
        """
        Compute Maximum Mean Discrepancy (MMD) with deep kernels.
        
        MMD^2(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
        where x ~ P, y ~ Q, and k is the kernel function.
        
        Args:
            z_src: samples from source distribution (collaborative space)
            z_tgt: samples from target distribution (language space)
            kernel_z_src: deep kernel for source distribution
            kernel_z_tgt: deep kernel for target distribution
            
        Returns:
            MMD value (scalar)
        """
        # Apply deep kernel transformations
        if kernel_z_src is not None:
            phi_src = kernel_z_src(z_src)
        else:
            phi_src = z_src
            
        if kernel_z_tgt is not None:
            phi_tgt = kernel_z_tgt(z_tgt)
        else:
            phi_tgt = z_tgt
        
        # Get bandwidth
        if self.use_learnable_bandwidth:
            bandwidth = torch.exp(self.log_bandwidth)
        else:
            bandwidth = self.bandwidth
        
        # Compute RBF kernel: k(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))
        def rbf_kernel(x, y, bandwidth):
            # x: [batch_size, feature_dim]
            # y: [batch_size, feature_dim]
            # Compute pairwise squared distances
            xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [batch_size, 1]
            yy = torch.sum(y ** 2, dim=1, keepdim=True)  # [batch_size, 1]
            xy = torch.mm(x, y.t())  # [batch_size, batch_size]
            
            # ||x - y||^2 = ||x||^2 - 2<x,y> + ||y||^2
            distances = xx - 2 * xy + yy.t()
            return torch.exp(-distances / (2 * bandwidth ** 2))
        
        # Compute kernel matrices
        k_src_src = rbf_kernel(phi_src, phi_src, bandwidth)
        k_tgt_tgt = rbf_kernel(phi_tgt, phi_tgt, bandwidth)
        k_src_tgt = rbf_kernel(phi_src, phi_tgt, bandwidth)
        
        # Unbiased MMD estimator (excluding diagonal elements)
        batch_size = z_src.size(0)
        
        # E[k(x, x')] - exclude diagonal
        mask = 1 - torch.eye(batch_size, device=z_src.device)
        term1 = (k_src_src * mask).sum() / (batch_size * (batch_size - 1))
        
        # E[k(y, y')] - exclude diagonal
        term2 = (k_tgt_tgt * mask).sum() / (batch_size * (batch_size - 1))
        
        # -2 E[k(x, y)]
        term3 = -2 * k_src_tgt.mean()
        
        mmd_squared = term1 + term2 + term3
        
        # Return absolute value to avoid numerical issues
        return torch.clamp(mmd_squared, min=0.0)

    def compute_kl_standard(self, mu, logvar):
        """
        Standard KL divergence to standard Gaussian prior: KL(q_phi || N(0,I))
        """
        kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return kld

    def cal_loss(self, user, batch_data):
        self.is_training = True

        user_emb = self.usrprf_embeds[user]

        mu_src, mu_llm, logvar_src, logvar_llm = self.encode(batch_data, user_emb)

        # Combined distributions (following MDDM strategy)
        mu = mu_src + mu_llm
        logvar = logvar_src + logvar_llm

        # Sample from collaborative space distribution
        z_collab = self.reparameterize(mu, logvar)
        
        # Sample from language space distribution
        z_lang = self.reparameterize(mu_llm, logvar_llm)
        
        # Reconstruction
        recon_x = self.decode(z_collab)
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * batch_data, -1))

        # Mixing Divergence with MMD:
        # L_MDDM_MMD = β · KL(q_phi || p_z) + (1-β) · MMD(q_phi || p_phi)
        
        # Term 1: Standard KL to Gaussian prior (regularization)
        KLD_standard = self.compute_kl_standard(mu, logvar)
        
        # Term 2: MMD between collaborative and language space (matching)
        MMD_matching = self.compute_mmd(z_collab, z_lang, 
                                       kernel_z_src=self.kernel_q,
                                       kernel_z_tgt=self.kernel_p)
        
        # Mixing divergence
        mixing_divergence = self.beta * KLD_standard + (1 - self.beta) * MMD_matching
        
        loss = BCE + mixing_divergence
        losses = {
            'rec_loss': BCE, 
            'reg_loss': mixing_divergence,
            'kld_standard': KLD_standard,
            'mmd_matching': MMD_matching
        }
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        batch_data = self.data_handler.train_data[pck_users.cpu()]
        data = torch.FloatTensor(batch_data.toarray()).to(configs['device'])
        user_emb = self.usrprf_embeds[pck_users]

        mu, mu_llm, logvar, logvar_llm = self.encode(data, user_emb)

        # Combined distributions
        mu = mu + mu_llm
        logvar = logvar + logvar_llm

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        full_preds = self._mask_predict(recon_x, train_mask)
        return full_preds
