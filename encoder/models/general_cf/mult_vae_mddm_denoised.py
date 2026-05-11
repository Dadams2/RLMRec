"""
Phase 2: Mult-VAE MDDM with Learnable Semantic Denoising

Based on semantic noise analysis showing:
- 100% consensus on signal dimensions across all models
- 100% consensus on noise dimensions across all models
- 20 consensus signal dimensions, 20 consensus noise dimensions

This model adds a learnable denoising layer that:
1. Estimates and removes noise from LLM embeddings
2. Amplifies signal dimensions
3. Learns end-to-end from recommendation loss
"""

import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.base_model import BaseModel
import numpy as np

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class SemanticDenoiser(nn.Module):
    """
    Learnable denoising layer for LLM embeddings.
    
    Learns to:
    1. Identify and suppress noise dimensions
    2. Amplify signal dimensions
    3. Refine embeddings based on recommendation loss gradients
    """
    def __init__(self, embedding_dim=1536, hidden_dim=512, 
                 signal_dims=None, noise_dims=None, use_prior=True):
        super(SemanticDenoiser, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.signal_dims = signal_dims if signal_dims is not None else []
        self.noise_dims = noise_dims if noise_dims is not None else []
        self.use_prior = use_prior
        
        # Noise estimation network
        self.noise_estimator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()  # Bounded noise estimation
        )
        
        # Dimension-wise attention (learns importance weights)
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Sigmoid()  # 0-1 weights per dimension
        )
        
        # Refinement network
        self.refiner = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Initialize with prior knowledge from analysis (optional)
        if self.use_prior and len(self.signal_dims) > 0 and len(self.noise_dims) > 0:
            self._init_with_prior()
        else:
            # Random initialization - let the model learn from scratch
            self._init_random()
    
    def _init_with_prior(self):
        """Initialize attention with prior knowledge from semantic analysis."""
        with torch.no_grad():
            # Initialize attention to favor signal dims, suppress noise dims
            # This provides a good starting point for learning
            if hasattr(self.attention[2], 'bias'):
                # Start with slight positive bias for signal dims
                for dim in self.signal_dims:
                    self.attention[2].bias[dim] += 0.5
                # Start with slight negative bias for noise dims
                for dim in self.noise_dims:
                    self.attention[2].bias[dim] -= 0.5
    
    def _init_random(self):
        """Random initialization - learn signal/noise dimensions from scratch."""
        # Use Xavier/Glorot initialization (already default for Linear layers)
        # No bias towards any specific dimensions
        # The model will learn which dimensions are signal vs noise during training
        pass  # Default initialization is already random
    
    def forward(self, embeddings, return_components=False):
        """
        Args:
            embeddings: [batch, embedding_dim] LLM embeddings
            return_components: If True, return intermediate results for analysis
            
        Returns:
            denoised: [batch, embedding_dim] denoised embeddings
            components (optional): dict with noise, attention, etc.
        """
        # Step 1: Estimate noise
        noise_estimate = self.noise_estimator(embeddings)
        
        # Step 2: Compute attention weights (per-dimension importance)
        attention_weights = self.attention(embeddings)
        
        # Step 3: Remove noise and apply attention
        denoised = embeddings - noise_estimate
        denoised = denoised * attention_weights
        
        # Step 4: Refinement
        refined = self.refiner(denoised)
        output = denoised + refined  # Residual connection
        
        if return_components:
            components = {
                'noise_estimate': noise_estimate,
                'attention_weights': attention_weights,
                'denoised': denoised,
                'refined': refined
            }
            return output, components
        
        return output


class mult_vae_MDDM_Denoised(BaseModel):
    def __init__(self, data_handler):
        super(mult_vae_MDDM_Denoised, self).__init__(data_handler)

        self.beta = self.hyper_config['beta']
        self.denoise_lambda = self.hyper_config.get('denoise_lambda', 0.01)  # Regularization weight
        self.use_prior = self.hyper_config.get('use_prior', False)  # Whether to use prior from analysis
        
        self.data_handler = data_handler

        # VAE structure: [item_num, 600, 200, 600, item_num]
        self.p_dims = [200, 600, self.item_num]
        self.q_dims = [self.item_num, 600, 200]

        # Compute mean and variance in parallel
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]

        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])]
        )

        # Load LLM embeddings
        self.usrprf_embeds_raw = torch.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds_raw = torch.tensor(configs['itmprf_embeds']).float().cuda()
        
        # Consensus dimensions from semantic analysis (Amazon dataset)
        # Only used if use_prior=True
        self.user_signal_dims = [57, 89, 310, 378, 402, 408, 413, 422, 508, 654, 
                                  701, 738, 781, 830, 861, 920, 979, 997, 1166, 1330]
        self.user_noise_dims = [0, 137, 194, 271, 393, 470, 481, 489, 574, 699, 
                                 750, 994, 1031, 1055, 1205, 1387, 1486, 1522, 1524, 1532]
        self.item_signal_dims = [57, 310, 378, 654, 738, 830, 861, 979, 997, 1330]
        self.item_noise_dims = [194, 228, 518, 652, 954, 1120, 1240, 1246, 1348, 1487]
        
        # Learnable denoisers (Phase 2!)
        # Can use prior from analysis OR learn from scratch
        self.user_denoiser = SemanticDenoiser(
            embedding_dim=self.usrprf_embeds_raw.shape[1],
            hidden_dim=512,
            signal_dims=self.user_signal_dims if self.use_prior else None,
            noise_dims=self.user_noise_dims if self.use_prior else None,
            use_prior=self.use_prior
        )
        
        self.item_denoiser = SemanticDenoiser(
            embedding_dim=self.itmprf_embeds_raw.shape[1],
            hidden_dim=512,
            signal_dims=self.item_signal_dims if self.use_prior else None,
            noise_dims=self.item_noise_dims if self.use_prior else None,
            use_prior=self.use_prior
        )
        
        # Denoise embeddings (will be updated during training)
        self.usrprf_embeds = None
        self.itmprf_embeds = None

        # MLP for processing semantic information
        self.mlp = nn.Sequential(
            nn.Linear(self.itmprf_embeds_raw.shape[1], 600),
            nn.Tanh(),
            nn.Linear(600, 400)
        )

        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])]
        )

        self.drop = nn.Dropout(self.hyper_config['dropout'])

        self.final_embeds = None
        self.is_training = False

    def denoise_embeddings(self):
        """Apply learned denoising to raw embeddings."""
        self.usrprf_embeds = self.user_denoiser(self.usrprf_embeds_raw)
        self.itmprf_embeds = self.item_denoiser(self.itmprf_embeds_raw)

    def encode(self, x, user_emb):
        h = self.drop(x)
        
        # Use denoised item embeddings
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
    
    def compute_denoising_loss(self):
        """
        Regularization loss to encourage meaningful denoising.
        
        Encourages:
        1. Sparsity in noise estimation (most dims should not be noisy)
        2. Attention weights to be decisive (not all 0.5)
        3. Consistency between denoised and original (don't remove too much)
        """
        # Get denoising components for a sample
        user_sample = self.usrprf_embeds_raw[:100]  # Sample for efficiency
        item_sample = self.itmprf_embeds_raw[:100]
        
        user_denoised, user_comp = self.user_denoiser(user_sample, return_components=True)
        item_denoised, item_comp = self.item_denoiser(item_sample, return_components=True)
        
        # L1 sparsity on noise (encourage most noise to be zero)
        noise_sparsity = (torch.abs(user_comp['noise_estimate']).mean() + 
                          torch.abs(item_comp['noise_estimate']).mean())
        
        # Encourage attention to be decisive (close to 0 or 1, not 0.5)
        attention_entropy = -(user_comp['attention_weights'] * 
                             torch.log(user_comp['attention_weights'] + 1e-8) +
                             (1 - user_comp['attention_weights']) * 
                             torch.log(1 - user_comp['attention_weights'] + 1e-8)).mean()
        
        # Consistency: denoised should not be too different from original
        user_consistency = F.cosine_similarity(user_sample, user_denoised).mean()
        item_consistency = F.cosine_similarity(item_sample, item_denoised).mean()
        consistency_loss = 2.0 - user_consistency - item_consistency  # Want close to 0
        
        total_denoise_loss = (0.5 * noise_sparsity + 
                              0.3 * attention_entropy + 
                              0.2 * consistency_loss)
        
        return total_denoise_loss

    def cal_loss(self, user, batch_data):
        self.is_training = True
        
        # Denoise embeddings first (this is where gradients flow!)
        self.denoise_embeddings()

        user_emb = self.usrprf_embeds[user]

        mu_src, mu_llm, logvar_src, logvar_llm = self.encode(batch_data, user_emb)

        # Combine distributions (MDDM strategy)
        mu = mu_src + mu_llm
        logvar = logvar_src + logvar_llm

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        # Reconstruction loss
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * batch_data, -1))

        # KL divergence with mixing (MDDM)
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        KLD_llm = -0.5 * torch.mean(torch.sum(
            1 + torch.log(logvar.exp()/(logvar_llm.exp() + 1e-8) + 1e-8) -
            (mu - mu_llm).pow(2)/(logvar_llm.exp() + 1e-8) - 
            logvar.exp()/(logvar_llm.exp() + 1e-8), dim=1))

        KLD = self.beta * KLD + (1 - self.beta) * KLD_llm
        
        # NEW: Denoising regularization loss
        denoise_loss = self.compute_denoising_loss()

        loss = BCE + KLD + self.denoise_lambda * denoise_loss
        
        losses = {
            'rec_loss': BCE, 
            'reg_loss': KLD,
            'denoise_loss': denoise_loss
        }
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        # Denoise embeddings for inference
        self.denoise_embeddings()

        batch_data = self.data_handler.train_data[pck_users.cpu()]
        data = torch.FloatTensor(batch_data.toarray()).to(configs['device'])
        user_emb = self.usrprf_embeds[pck_users]

        mu, mu_llm, logvar, logvar_llm = self.encode(data, user_emb)

        mu = mu + mu_llm
        logvar = logvar + logvar_llm

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        full_preds = self._mask_predict(recon_x, train_mask)
        return full_preds
