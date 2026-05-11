"""
Mult-VAE MDDM with Diffusion-based Meta-Learner

The standard MLP meta-learner (2 FC layers with Tanh) may be insufficient to capture
the complex shape and dynamics of LLM embedding distributions. This model replaces
the MLP with a diffusion-based architecture that:

1. Uses iterative denoising to refine distribution parameters
2. Learns the score function (gradient of log-probability) of the target distribution
3. Better captures multi-modal and complex distribution shapes
4. Provides a more principled approach to distribution matching

The diffusion meta-learner transforms LLM embeddings through a learned denoising process:
    Input (LLM embedding space) → Noisy latent → Iterative denoising → (μ, σ) in collaborative space

This is mathematically motivated by:
- Score matching connects to distribution matching objectives
- Diffusion models can approximate complex distributions
- The iterative refinement allows learning multi-scale features
"""

import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.base_model import BaseModel
import numpy as np
import math

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Sinusoidal timestep embeddings (from DDPM/Transformer).
    
    Args:
        timesteps: [batch_size] tensor of timesteps
        embedding_dim: dimension of the embedding
        
    Returns:
        [batch_size, embedding_dim] tensor of embeddings
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class DiffusionBlock(nn.Module):
    """
    A single block of the diffusion denoising network.
    Uses residual connections and layer normalization for stable training.
    """
    def __init__(self, hidden_dim, time_emb_dim, dropout=0.1):
        super(DiffusionBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, t_emb):
        """
        Args:
            x: [batch, hidden_dim] input features
            t_emb: [batch, time_emb_dim] timestep embedding
            
        Returns:
            [batch, hidden_dim] output features
        """
        # First sub-layer with time conditioning
        h = self.norm1(x)
        h = self.linear1(h)
        h = h + self.time_proj(t_emb)  # Add time information
        h = F.silu(h)  # SiLU/Swish activation (common in diffusion models)
        h = self.dropout(h)
        
        # Second sub-layer
        h = self.norm2(h)
        h = self.linear2(h)
        h = F.silu(h)
        h = self.dropout(h)
        
        # Residual connection
        return x + h


class DiffusionMetaLearner(nn.Module):
    """
    Diffusion-based Meta-Learner for transforming LLM embeddings to 
    distribution parameters (μ, Σ) in the collaborative space.
    
    Instead of a simple MLP: input → FC → Tanh → FC → (μ, σ)
    
    This uses a diffusion process:
    1. Project LLM embedding to hidden space
    2. Add noise at various levels
    3. Learn to denoise iteratively, conditioned on the original embedding
    4. Output refined (μ, σ) parameters
    
    The key insight is that the denoising process learns the score function
    (gradient of log probability), which directly relates to distribution matching.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=512, 
                 num_diffusion_steps=10, num_blocks=3, dropout=0.1):
        """
        Args:
            input_dim: Dimension of LLM embeddings (e.g., 1536)
            output_dim: Dimension of output (μ, σ) = 2 * latent_dim (e.g., 400)
            hidden_dim: Hidden dimension of the diffusion network
            num_diffusion_steps: Number of denoising steps (T)
            num_blocks: Number of diffusion blocks in the network
            dropout: Dropout rate
        """
        super(DiffusionMetaLearner, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_diffusion_steps
        self.time_emb_dim = hidden_dim // 4
        
        # Noise schedule (linear beta schedule)
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, num_diffusion_steps))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
        # Input projection (LLM embedding → hidden)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Condition embedding (original LLM embedding as condition)
        self.condition_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.time_emb_dim)
        )
        
        # Diffusion denoising blocks
        self.blocks = nn.ModuleList([
            DiffusionBlock(hidden_dim, self.time_emb_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Output projection (hidden → output distribution parameters)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # For direct mapping (bypass diffusion for comparison/ablation)
        self.direct_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.use_diffusion = True  # Can be toggled for ablation
        
    def _denoise_step(self, x_t, t, condition):
        """
        Single denoising step.
        
        Args:
            x_t: [batch, hidden_dim] noisy latent at timestep t
            t: [batch] timestep indices
            condition: [batch, hidden_dim] conditioning from original input
            
        Returns:
            [batch, hidden_dim] predicted noise (or directly denoised x)
        """
        # Get timestep embedding
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Combine noisy input with condition
        h = x_t + condition  # Additive conditioning
        
        # Pass through diffusion blocks
        for block in self.blocks:
            h = block(h, t_emb)
            
        return h
    
    def forward(self, llm_embedding, num_inference_steps=None):
        """
        Transform LLM embedding to (μ, σ) using iterative diffusion.
        
        During training: Use full diffusion process with noise
        During inference: Use learned denoising process
        
        Args:
            llm_embedding: [batch, input_dim] LLM embedding
            num_inference_steps: Override default steps for inference
            
        Returns:
            [batch, output_dim] distribution parameters (first half μ, second half logvar)
        """
        if not self.use_diffusion:
            # Ablation: use direct MLP instead
            return self.direct_mlp(llm_embedding)
        
        batch_size = llm_embedding.shape[0]
        device = llm_embedding.device
        
        # Project input to hidden space
        x = self.input_proj(llm_embedding)
        
        # Get condition from original embedding
        condition = self.condition_proj(llm_embedding)
        
        if self.training:
            # Training: Single-step denoising loss (DDPM-style)
            # Sample random timesteps
            t = torch.randint(0, self.num_steps, (batch_size,), device=device)
            
            # Add noise to the hidden representation
            noise = torch.randn_like(x)
            sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            x_noisy = sqrt_alpha * x + sqrt_one_minus_alpha * noise
            
            # Predict denoised output
            x_denoised = self._denoise_step(x_noisy, t, condition)
            
            # For training, we output the denoised representation
            # The diffusion loss will be computed separately
            output = self.output_proj(x_denoised)
            
            # Store for loss computation
            self._noise = noise
            self._predicted_x = x_denoised
            self._target_x = x
            self._timesteps = t
            
        else:
            # Inference: Full iterative denoising
            steps = num_inference_steps if num_inference_steps else self.num_steps
            
            # Start from noise
            x_t = torch.randn_like(x)
            
            # Iterative denoising (simplified DDIM-style)
            for i in reversed(range(steps)):
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                
                # Predict denoised x
                x_pred = self._denoise_step(x_t, t, condition)
                
                if i > 0:
                    # DDIM update step (deterministic)
                    alpha_t = self.alphas_cumprod[i]
                    alpha_prev = self.alphas_cumprod[i - 1]
                    
                    # Compute x_{t-1}
                    sigma = 0  # Deterministic (eta=0 in DDIM)
                    pred_x0 = x_pred
                    
                    # Direction pointing to x_t
                    dir_xt = torch.sqrt(1 - alpha_prev - sigma**2) * (x_t - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1 - alpha_t + 1e-8)
                    
                    x_t = torch.sqrt(alpha_prev) * pred_x0 + dir_xt
                else:
                    x_t = x_pred
            
            output = self.output_proj(x_t)
        
        return output
    
    def compute_diffusion_loss(self):
        """
        Compute diffusion training loss (denoising score matching).
        
        This should be called during training after forward() to get the 
        auxiliary diffusion loss.
        
        Returns:
            Scalar loss value
        """
        if not hasattr(self, '_predicted_x') or self._predicted_x is None:
            return torch.tensor(0.0)
        
        # L2 loss between predicted and target clean representation
        loss = F.mse_loss(self._predicted_x, self._target_x)
        
        return 0


class ScoreMatchingMetaLearner(nn.Module):
    """
    Alternative: Score Matching-based Meta-Learner
    
    Instead of full diffusion, this directly learns the score function 
    (gradient of log probability) using denoising score matching.
    
    This is simpler than full diffusion but still captures distribution structure.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=512, 
                 num_noise_levels=5, dropout=0.1):
        super(ScoreMatchingMetaLearner, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_noise_levels = num_noise_levels
        
        # Noise levels (geometric sequence)
        sigmas = torch.exp(torch.linspace(math.log(0.01), math.log(1.0), num_noise_levels))
        self.register_buffer('sigmas', sigmas)
        
        # Score network
        self.score_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for noise level
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, llm_embedding):
        """
        Transform LLM embedding using score-based refinement.
        """
        batch_size = llm_embedding.shape[0]
        device = llm_embedding.device
        
        if self.training:
            # Sample noise level
            noise_level_idx = torch.randint(0, self.num_noise_levels, (batch_size,), device=device)
            sigma = self.sigmas[noise_level_idx].unsqueeze(-1)
            
            # Add noise
            noise = torch.randn(batch_size, self.output_dim, device=device)
            
            # Start from a rough initial estimate
            x = torch.zeros(batch_size, self.output_dim, device=device)
            x_noisy = x + sigma * noise
            
            # Concatenate with noise level and condition
            sigma_input = sigma.expand(-1, 1)  # [batch, 1]
            
            # Use LLM embedding as condition for the score
            h = torch.cat([llm_embedding, sigma_input], dim=-1)
            
            # Predict score (gradient of log prob)
            score = self.score_net(h)
            
            # Store for loss
            self._score = score
            self._noise = noise
            self._sigma = sigma
            
            # Refine using score
            output = x_noisy + sigma * score
        else:
            # Inference: Langevin dynamics
            x = torch.zeros(batch_size, self.output_dim, device=device)
            
            for sigma in reversed(self.sigmas):
                sigma_input = sigma.expand(batch_size, 1)
                h = torch.cat([llm_embedding, sigma_input], dim=-1)
                score = self.score_net(h)
                
                # Langevin update
                step_size = 0.5 * (sigma ** 2)
                noise = torch.randn_like(x)
                x = x + step_size * score + torch.sqrt(2 * step_size) * noise
            
            output = x
            
        return output
    
    def compute_score_loss(self):
        """Denoising score matching loss."""
        if not hasattr(self, '_score'):
            return torch.tensor(0.0)
        
        # Score should predict -noise/sigma
        target = -self._noise / self._sigma
        loss = F.mse_loss(self._score, target)
        return loss


class FlowMatchingMetaLearner(nn.Module):
    """
    Alternative: Flow Matching-based Meta-Learner
    
    Uses Conditional Flow Matching (Lipman et al., 2023) which:
    - Is simpler than diffusion (no noise schedule needed)
    - Learns a velocity field that transports samples
    - Has connections to optimal transport
    
    This is particularly well-suited for distribution matching!
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.1):
        super(FlowMatchingMetaLearner, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Velocity network v(x, t, condition)
        self.velocity_net = nn.Sequential(
            nn.Linear(output_dim + input_dim + 1, hidden_dim),  # x + condition + t
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initial projection for target distribution
        self.target_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, llm_embedding, num_steps=10):
        """
        Transform using flow matching.
        
        The flow transports from a simple distribution (e.g., Gaussian)
        to the target distribution, conditioned on the LLM embedding.
        """
        batch_size = llm_embedding.shape[0]
        device = llm_embedding.device
        
        if self.training:
            # Get target (what we want to transport to)
            x1 = self.target_proj(llm_embedding)
            
            # Sample from base distribution
            x0 = torch.randn_like(x1)
            
            # Sample time uniformly
            t = torch.rand(batch_size, 1, device=device)
            
            # Interpolate (optimal transport path)
            x_t = (1 - t) * x0 + t * x1
            
            # Target velocity (derivative of interpolation)
            target_v = x1 - x0
            
            # Predict velocity
            v_input = torch.cat([x_t, llm_embedding, t], dim=-1)
            predicted_v = self.velocity_net(v_input)
            
            # Store for loss
            self._predicted_v = predicted_v
            self._target_v = target_v
            
            output = x1  # During training, output the target
        else:
            # Inference: ODE integration
            x = torch.randn(batch_size, self.output_dim, device=device)
            
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((batch_size, 1), i * dt, device=device)
                v_input = torch.cat([x, llm_embedding, t], dim=-1)
                v = self.velocity_net(v_input)
                x = x + dt * v
            
            output = x
            
        return output
    
    def compute_flow_loss(self):
        """Flow matching loss (velocity matching)."""
        if not hasattr(self, '_predicted_v'):
            return torch.tensor(0.0)
        
        loss = F.mse_loss(self._predicted_v, self._target_v)
        return loss


class mult_vae_MDDM_Diffusion(BaseModel):
    """
    Mult-VAE with MDDM distribution matching using a Diffusion-based Meta-Learner.
    
    Key differences from standard MDDM:
    1. Meta-learner is a diffusion model instead of simple MLP
    2. Iterative refinement of distribution parameters
    3. Additional diffusion loss for training the meta-learner
    
    Hyperparameters:
    - beta: Mixing coefficient for MDDM (same as original)
    - diffusion_steps: Number of diffusion denoising steps
    - diffusion_hidden: Hidden dimension of diffusion network
    - diffusion_lambda: Weight for auxiliary diffusion loss
    - meta_type: 'diffusion', 'score', or 'flow' for different meta-learner types
    """
    
    def __init__(self, data_handler):
        super(mult_vae_MDDM_Diffusion, self).__init__(data_handler)

        self.beta = self.hyper_config['beta']
        
        # Diffusion meta-learner hyperparameters
        self.diffusion_steps = self.hyper_config.get('diffusion_steps', 10)
        self.diffusion_hidden = self.hyper_config.get('diffusion_hidden', 512)
        self.diffusion_lambda = self.hyper_config.get('diffusion_lambda', 0.1)
        self.meta_type = self.hyper_config.get('meta_type', 'diffusion')  # 'diffusion', 'score', 'flow'

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
        self.usrprf_embeds = torch.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = torch.tensor(configs['itmprf_embeds']).float().cuda()

        # Create the meta-learner based on type
        input_dim = self.itmprf_embeds.shape[1]  # LLM embedding dim (e.g., 1536)
        output_dim = 400  # μ (200) + logvar (200)
        
        if self.meta_type == 'diffusion':
            self.meta_learner = DiffusionMetaLearner(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=self.diffusion_hidden,
                num_diffusion_steps=self.diffusion_steps,
                num_blocks=3,
                dropout=self.hyper_config.get('dropout', 0.5)
            )
        elif self.meta_type == 'score':
            self.meta_learner = ScoreMatchingMetaLearner(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=self.diffusion_hidden,
                num_noise_levels=self.diffusion_steps,
                dropout=self.hyper_config.get('dropout', 0.5)
            )
        elif self.meta_type == 'flow':
            self.meta_learner = FlowMatchingMetaLearner(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=self.diffusion_hidden,
                dropout=self.hyper_config.get('dropout', 0.5)
            )
        else:
            raise ValueError(f"Unknown meta_type: {self.meta_type}")

        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])]
        )

        self.drop = nn.Dropout(self.hyper_config['dropout'])

        self.final_embeds = None
        self.is_training = False

    def encode(self, x, user_emb):
        """
        Encode user interactions and LLM embeddings to distributions.
        
        Returns:
            mu_src, mu_llm, logvar_src, logvar_llm
        """
        h = self.drop(x)
        
        # Combine item embeddings weighted by interactions, plus user embedding
        # This is the "base network g" from the paper
        hidden = torch.matmul(h, self.itmprf_embeds) + user_emb  # [batch, 1536]
        
        # Apply diffusion-based meta-learner instead of simple MLP
        # f_φ: R^{d_s} → R^{2d}
        hidden = self.meta_learner(hidden)  # [batch, 400]

        mu_llm = hidden[:, :200]
        logvar_llm = hidden[:, 200:]

        # Collaborative space encoding (unchanged from original)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]

        return mu, mu_llm, logvar, logvar_llm

    def decode(self, z):
        """Decode latent to item predictions."""
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        if self.is_training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std) + mu
        else:
            z = mu
        return z
    
    def compute_meta_learner_loss(self):
        """
        Compute the auxiliary loss for training the meta-learner.
        
        Returns:
            Scalar loss value depending on meta_type
        """
        if self.meta_type == 'diffusion':
            return self.meta_learner.compute_diffusion_loss()
        elif self.meta_type == 'score':
            return self.meta_learner.compute_score_loss()
        elif self.meta_type == 'flow':
            return self.meta_learner.compute_flow_loss()
        return torch.tensor(0.0)

    def cal_loss(self, user, batch_data):
        """
        Compute total training loss.
        
        Loss = BCE (reconstruction) + KLD (MDDM matching) + λ * meta_loss (diffusion/score/flow)
        """
        self.is_training = True

        user_emb = self.usrprf_embeds[user]

        mu_src, mu_llm, logvar_src, logvar_llm = self.encode(batch_data, user_emb)

        # MDDM: Combine distributions via addition
        mu = mu_src + mu_llm
        logvar = logvar_src + logvar_llm

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        # Reconstruction loss
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * batch_data, -1))

        # MDDM KL divergence (Eq. 15 from paper)
        # β · D_KL(q_φ || p_z) + (1-β) · D_KL(q_φ || p_ψ)
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        KLD_llm = -0.5 * torch.mean(torch.sum(
            1 + torch.log(logvar.exp() / (logvar_llm.exp() + 1e-8) + 1e-8) -
            (mu - mu_llm).pow(2) / (logvar_llm.exp() + 1e-8) - 
            logvar.exp() / (logvar_llm.exp() + 1e-8), dim=1
        ))

        KLD = self.beta * KLD + (1 - self.beta) * KLD_llm
        
        # Auxiliary meta-learner loss
        meta_loss = self.compute_meta_learner_loss()

        loss = BCE + KLD + self.diffusion_lambda * meta_loss
        
        losses = {
            'rec_loss': BCE, 
            'reg_loss': KLD,
            'meta_loss': meta_loss
        }
        return loss, losses

    def full_predict(self, batch_data):
        """
        Full prediction for evaluation.
        """
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        batch_data = self.data_handler.train_data[pck_users.cpu()]
        data = torch.FloatTensor(batch_data.toarray()).to(configs['device'])
        user_emb = self.usrprf_embeds[pck_users]

        mu, mu_llm, logvar, logvar_llm = self.encode(data, user_emb)

        # MDDM combination
        mu = mu + mu_llm
        logvar = logvar + logvar_llm

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        full_preds = self._mask_predict(recon_x, train_mask)
        return full_preds


# Also provide variants for GODM and CPDM with diffusion meta-learner

class mult_vae_GODM_Diffusion(mult_vae_MDDM_Diffusion):
    """
    GODM with Diffusion-based Meta-Learner.
    
    Uses Wasserstein distance for distribution matching instead of KL mixing.
    """
    
    def cal_loss(self, user, batch_data):
        self.is_training = True

        user_emb = self.usrprf_embeds[user]
        mu_src, mu_llm, logvar_src, logvar_llm = self.encode(batch_data, user_emb)

        # Combine distributions
        mu = mu_src + mu_llm
        logvar = logvar_src + logvar_llm

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        # Reconstruction loss
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * batch_data, -1))

        # Standard KL regularization
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        # GODM: 2-Wasserstein distance (Eq. 9 from paper)
        # W_2(p_φ, q_ψ) = ||μ_φ - μ_ψ||_2^2 + Tr(Σ_φ + Σ_ψ - 2(Σ_φ^{1/2} Σ_ψ Σ_φ^{1/2})^{1/2})
        # For diagonal covariances, this simplifies to:
        mean_diff = torch.norm(mu_src - mu_llm, dim=1) ** 2
        std_src = torch.exp(0.5 * logvar_src)
        std_llm = torch.exp(0.5 * logvar_llm)
        var_diff = torch.norm(std_src - std_llm, dim=1) ** 2
        
        WD = torch.mean(torch.sqrt(mean_diff + var_diff + 1e-8))

        KLD = KLD + self.beta * WD
        
        # Meta-learner loss
        meta_loss = self.compute_meta_learner_loss()

        loss = BCE + KLD + self.diffusion_lambda * meta_loss
        
        losses = {'rec_loss': BCE, 'reg_loss': KLD, 'meta_loss': meta_loss}
        return loss, losses


class mult_vae_CPDM_Diffusion(mult_vae_MDDM_Diffusion):
    """
    CPDM with Diffusion-based Meta-Learner.
    
    Uses composite prior for distribution matching.
    """
    
    def cal_loss(self, user, batch_data):
        self.is_training = True

        user_emb = self.usrprf_embeds[user]
        mu_src, mu_llm, logvar_src, logvar_llm = self.encode(batch_data, user_emb)

        # Combine distributions
        mu = mu_src + mu_llm
        logvar = logvar_src + logvar_llm

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        # Reconstruction loss
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * batch_data, -1))

        # Standard KL regularization
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        # CPDM: Composite prior (Eq. 11-12 from paper)
        # p_com = α · N(μ_φ, Σ_φ) + (1-α) · N(μ_ψ, Σ_ψ)
        # For α = 0.5, we get Jensen-Shannon divergence
        mu_mix = (mu + mu_llm) / 2
        logvar_mix = (logvar + logvar_llm) / 2

        KLD_1 = -0.5 * torch.mean(torch.sum(
            1 + torch.log(logvar.exp() / (logvar_mix.exp() + 1e-8) + 1e-8) -
            (mu - mu_mix).pow(2) / (logvar_mix.exp() + 1e-8) - 
            logvar.exp() / (logvar_mix.exp() + 1e-8), dim=1
        ))

        KLD_2 = -0.5 * torch.mean(torch.sum(
            1 + torch.log(logvar_llm.exp() / (logvar_mix.exp() + 1e-8) + 1e-8) -
            (mu_llm - mu_mix).pow(2) / (logvar_mix.exp() + 1e-8) - 
            logvar_llm.exp() / (logvar_mix.exp() + 1e-8), dim=1
        ))

        KLD = KLD + self.beta * (KLD_1 + KLD_2)
        
        # Meta-learner loss
        meta_loss = self.compute_meta_learner_loss()

        loss = BCE + KLD + self.diffusion_lambda * meta_loss
        
        losses = {'rec_loss': BCE, 'reg_loss': KLD, 'meta_loss': meta_loss}
        return loss, losses
