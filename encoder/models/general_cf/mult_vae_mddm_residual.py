"""
Phase 2 Alternative: Mult-VAE MDDM with Residual Denoising

Instead of replacing embeddings, this model learns additive corrections:
- Embeddings = Original + α * Correction
- α starts at 0 and gradually increases (warmup)
- Preserves original information while learning refinements
- Less destructive than full replacement

This approach:
1. Starts identical to baseline (α=0 means no correction)
2. Gradually introduces learned corrections as training progresses
3. Prevents catastrophic information loss from aggressive denoising
"""

import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.base_model import BaseModel
import numpy as np

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class ResidualRefiner(nn.Module):
    """
    Learns additive corrections to LLM embeddings.
    
    Key difference from SemanticDenoiser:
    - Outputs corrections/residuals, not full denoised embeddings
    - Lighter network (fewer parameters)
    - Explicitly preserves original information
    """
    def __init__(self, embedding_dim=1536, hidden_dim=256):
        super(ResidualRefiner, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Correction network - learns what to add/subtract
        # Smaller than denoiser to prevent overfitting
        self.corrector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()  # Bounded corrections in [-1, 1]
        )
        
        # Initialize with small weights for gentle corrections
        self._init_small()
    
    def _init_small(self):
        """Initialize with small weights so initial corrections are minimal."""
        with torch.no_grad():
            for module in self.corrector:
                if isinstance(module, nn.Linear):
                    # Scale down initial weights
                    module.weight.data *= 0.1
                    if module.bias is not None:
                        module.bias.data.zero_()
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch, embedding_dim] Original LLM embeddings
            
        Returns:
            correction: [batch, embedding_dim] Learned correction to add
        """
        correction = self.corrector(embeddings)
        return correction


class mult_vae_MDDM_Residual(BaseModel):
    def __init__(self, data_handler):
        super(mult_vae_MDDM_Residual, self).__init__(data_handler)

        self.beta = self.hyper_config['beta']
        self.correction_scale = self.hyper_config.get('correction_scale', 0.1)  # Max correction strength
        self.warmup_epochs = self.hyper_config.get('warmup_epochs', 10)  # Epochs to reach full correction
        
        self.data_handler = data_handler
        self.current_epoch = 0

        # VAE structure: [item_num, 600, 200, 600, item_num]
        self.p_dims = [200, 600, self.item_num]
        self.q_dims = [self.item_num, 600, 200]

        # Compute mean and variance in parallel
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]

        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])]
        )

        # Load LLM embeddings (keep originals, never modify)
        self.usrprf_embeds_raw = torch.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds_raw = torch.tensor(configs['itmprf_embeds']).float().cuda()
        
        # Residual refiners (lightweight compared to full denoisers)
        self.user_refiner = ResidualRefiner(
            embedding_dim=self.usrprf_embeds_raw.shape[1],
            hidden_dim=256
        )
        
        self.item_refiner = ResidualRefiner(
            embedding_dim=self.itmprf_embeds_raw.shape[1],
            hidden_dim=256
        )
        
        # Refined embeddings (computed on-the-fly)
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

    def set_epoch(self, epoch):
        """Update current epoch for warmup schedule."""
        self.current_epoch = epoch

    def get_correction_alpha(self):
        """
        Compute correction strength based on training progress.
        
        Returns:
            alpha: 0.0 at start, linearly increases to correction_scale by warmup_epochs
        """
        if self.current_epoch < self.warmup_epochs:
            alpha = (self.current_epoch / self.warmup_epochs) * self.correction_scale
        else:
            alpha = self.correction_scale
        return alpha

    def refine_embeddings(self):
        """
        Apply residual corrections to raw embeddings.
        
        Refined = Original + α * Correction
        where α grows from 0 → correction_scale during warmup
        """
        alpha = self.get_correction_alpha()
        
        # Compute corrections
        user_correction = self.user_refiner(self.usrprf_embeds_raw)
        item_correction = self.item_refiner(self.itmprf_embeds_raw)
        
        # Apply with scaling
        self.usrprf_embeds = self.usrprf_embeds_raw + alpha * user_correction
        self.itmprf_embeds = self.itmprf_embeds_raw + alpha * item_correction

    def encode(self, x, user_emb):
        h = self.drop(x)
        
        # Use refined item embeddings
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

    def cal_loss(self, user, batch_data):
        self.is_training = True
        
        # Apply residual corrections (gradients flow through refiners!)
        self.refine_embeddings()

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
        
        loss = BCE + KLD
        
        losses = {
            'rec_loss': BCE, 
            'reg_loss': KLD,
            'correction_alpha': self.get_correction_alpha()  # Track warmup progress
        }
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        # Apply corrections for inference
        self.refine_embeddings()

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
