import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
import numpy as np

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class mult_vae_MDDM_MMD(BaseModel):
    """
    MultVAE with Mixing Divergence using Maximum Mean Discrepancy (MMD).
    
    This is an improved implementation of MDDM that replaces the KL-based distribution
    matching with MMD-based matching. Key improvements:
    
    1. Multi-scale MMD: Uses multiple bandwidths to capture distribution mismatches
       at different scales
    2. Adaptive bandwidth learning: Learns optimal bandwidth parameters during training
    3. Simplified architecture: Removes complex deep kernels for better performance
    4. Better numerical stability: Uses unbiased MMD estimator with proper normalization
    
    Advantages over KL-based MDDM:
    - Can detect high-order distribution mismatches
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

        # MMD parameters
        self.use_multi_scale = self.hyper_config.get('use_multi_scale', True)
        self.mmd_weight = self.hyper_config.get('mmd_weight', 10.0)
        
        # Multi-scale bandwidths for robust distribution matching
        if self.use_multi_scale:
            # Cover different scales: fine-grained to coarse-grained
            self.bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0]
            # Learnable scaling factors for each bandwidth
            self.log_bandwidth_scales = nn.Parameter(torch.zeros(len(self.bandwidths)))
        else:
            self.bandwidths = [self.hyper_config.get('mmd_bandwidth', 1.0)]
            self.log_bandwidth_scales = nn.Parameter(torch.zeros(1))

        self.init_weights()

    def init_weights(self):
        for layer in self.q_layers:
            # Weights
            init(layer.weight)
            # Biases
            layer.bias.data.normal_(0, 0.001)
        for layer in self.p_layers:
            init(layer.weight)
            layer.bias.data.normal_(0, 0.001)

    def encode(self, x, user_emb):
        h = self.drop(x)
        # Language space encoding
        # [batch, item_num] * [item_num, dim] = [batch, dim]
        hidden = torch.matmul(h, self.itmprf_embeds) + user_emb
        hidden = self.mlp(hidden)
        
        mu_llm = hidden[:, :200]
        logvar_llm = hidden[:, 200:]

        # Collaborative space encoding
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]

        return mu, mu_llm, logvar, logvar_llm

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std) + mu
        else:
            z = mu
        return z

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def compute_mmd_rbf(self, z_src, z_tgt):
        """
        Compute multi-scale Maximum Mean Discrepancy (MMD) using RBF kernels.
        
        MMD^2(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
        where x ~ P, y ~ Q, and k is the RBF kernel.
        
        Multi-scale version sums MMD values across multiple bandwidths to capture
        distribution mismatches at different granularities.
        
        Args:
            z_src: samples from source distribution (collaborative space) [batch_size, latent_dim]
            z_tgt: samples from target distribution (language space) [batch_size, latent_dim]
            
        Returns:
            MMD value (scalar)
        """
        
        def rbf_kernel(x, y, bandwidth):
            """
            RBF (Gaussian) kernel: k(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))
            """
            # Compute pairwise squared distances efficiently
            xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [batch_size, 1]
            yy = torch.sum(y ** 2, dim=1, keepdim=True)  # [batch_size, 1]
            xy = torch.mm(x, y.t())  # [batch_size, batch_size]
            
            # ||x - y||^2 = ||x||^2 - 2<x,y> + ||y||^2
            distances = xx - 2 * xy + yy.t()
            # Clamp for numerical stability
            distances = torch.clamp(distances, min=0.0)
            return torch.exp(-distances / (2 * bandwidth ** 2 + 1e-8))
        
        batch_size = z_src.size(0)
        
        # For unbiased estimator, exclude diagonal elements
        mask = 1 - torch.eye(batch_size, device=z_src.device)
        
        mmd_total = 0.0
        
        # Multi-scale MMD: sum over different bandwidths
        for idx, base_bandwidth in enumerate(self.bandwidths):
            # Apply learnable scaling to bandwidth
            if self.use_multi_scale:
                scale = torch.exp(self.log_bandwidth_scales[idx])
            else:
                scale = torch.exp(self.log_bandwidth_scales[0])
            bandwidth = base_bandwidth * scale
            
            # Compute kernel matrices
            k_src_src = rbf_kernel(z_src, z_src, bandwidth)
            k_tgt_tgt = rbf_kernel(z_tgt, z_tgt, bandwidth)
            k_src_tgt = rbf_kernel(z_src, z_tgt, bandwidth)
            
            # Unbiased MMD^2 estimator
            # E[k(x, x')] where x != x'
            term1 = (k_src_src * mask).sum() / (batch_size * (batch_size - 1) + 1e-8)
            
            # E[k(y, y')] where y != y'
            term2 = (k_tgt_tgt * mask).sum() / (batch_size * (batch_size - 1) + 1e-8)
            
            # -2 * E[k(x, y)]
            term3 = -2 * k_src_tgt.mean()
            
            # MMD^2 for this bandwidth
            mmd_squared = term1 + term2 + term3
            
            # Clamp to avoid negative values due to numerical errors
            mmd_total += torch.clamp(mmd_squared, min=0.0)
        
        # Average across bandwidths and apply weighting
        # The mmd_weight parameter scales MMD to be comparable with KL divergence
        return (mmd_total / len(self.bandwidths)) * self.mmd_weight

    def forward(self, rating_matrix, user_emb):
        # Encode to get mean and variance
        mu, mu_llm, logvar, logvar_llm = self.encode(rating_matrix, user_emb)
        
        # Following MDDM: combine collaborative and language distributions
        mu_combined = mu + mu_llm
        logvar_combined = logvar + logvar_llm
        
        # Reparameterization trick
        z = self.reparameterize(mu_combined, logvar_combined)
        
        # Decode
        x_pred = self.decode(z)
        
        # Also sample from language space for MMD computation
        z_lang = self.reparameterize(mu_llm, logvar_llm)
        
        return x_pred, mu_combined, logvar_combined, z, z_lang

    def cal_loss(self, user, batch_data):
        # Get user embeddings
        user_emb = self.usrprf_embeds[user]
        
        # Forward pass
        x_pred, mu, logvar, z_collab, z_lang = self.forward(batch_data, user_emb)
        
        # Reconstruction loss (negative log-likelihood)
        BCE = -torch.mean(torch.sum(F.log_softmax(x_pred, 1) * batch_data, -1))
        
        # Mixing Divergence with MMD:
        # L = BCE + β·KL(q||N(0,I)) + (1-β)·MMD(q_collab||q_lang)
        
        # Term 1: Standard KL divergence to Gaussian prior (regularization)
        # KL(N(μ, σ²) || N(0, I)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        KLD_standard = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        
        # Term 2: MMD between collaborative and language distributions (matching)
        # This replaces the KL matching term in original MDDM
        MMD_matching = self.compute_mmd_rbf(z_collab, z_lang)
        
        # Mixing divergence: balance regularization and matching
        # β controls the trade-off:
        # - High β (e.g., 0.6-0.8): more Gaussian regularization
        # - Low β (e.g., 0.2-0.4): more distribution matching via MMD
        mixing_divergence = self.beta * KLD_standard + (1 - self.beta) * MMD_matching
        
        # Total loss
        loss = BCE + mixing_divergence
        
        # Return detailed losses for monitoring
        losses = {
            'rec_loss': BCE,
            'reg_loss': mixing_divergence,
            'kld_standard': KLD_standard,
            'mmd_matching': MMD_matching
        }
        return loss, losses

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        batch_data = self.data_handler.train_data[pck_users.cpu()]
        data = torch.FloatTensor(batch_data.toarray()).to(configs['device'])
        user_emb = self.usrprf_embeds[pck_users]

        # Encode
        mu, mu_llm, logvar, logvar_llm = self.encode(data, user_emb)
        
        # Combined distributions
        mu_combined = mu + mu_llm
        logvar_combined = logvar + logvar_llm
        
        # Use mean during inference (no sampling)
        z = mu_combined
        
        # Decode
        x_pred = self.decode(z)
        
        full_preds = self._mask_predict(x_pred, train_mask)
        return full_preds
