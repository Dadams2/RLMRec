import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
import numpy as np

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class mult_vae_GODM_Sliced(BaseModel):
    """
    Improved GODM using Max-Sliced Wasserstein Distance for distribution matching.
    
    Key improvements over standard GODM:
    - Computationally efficient: O(n log n) vs O(n^3) for full Wasserstein
    - Smoother gradients for better optimization
    - More stable for high-dimensional Gaussian distributions
    - Sharper alignment through max-sliced variant
    """
    def __init__(self, data_handler):
        super(mult_vae_GODM_Sliced, self).__init__(data_handler)

        self.beta = self.hyper_config['beta']
        
        # Sliced Wasserstein hyperparameters
        self.num_projections = self.hyper_config.get('num_projections', 50)  # Number of random projections
        self.use_max_sliced = self.hyper_config.get('use_max_sliced', True)  # Use max-sliced variant
        
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

    def sample_from_gaussian(self, mu, logvar, num_samples=None):
        """
        Sample points from Gaussian distribution for sliced Wasserstein computation.
        
        Args:
            mu: [batch_size, latent_dim] - mean of Gaussian
            logvar: [batch_size, latent_dim] - log variance of Gaussian
            num_samples: number of samples to draw (default: batch_size)
        
        Returns:
            samples: [num_samples, latent_dim]
        """
        if num_samples is None:
            num_samples = mu.shape[0]
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(num_samples, mu.shape[1], device=mu.device)
        
        # Sample from each Gaussian in the batch
        if num_samples == mu.shape[0]:
            samples = mu + eps * std
        else:
            # For different num_samples, sample uniformly from batch
            idx = torch.randint(0, mu.shape[0], (num_samples,), device=mu.device)
            samples = mu[idx] + eps * std[idx]
        
        return samples

    def sliced_wasserstein_distance(self, mu_src, logvar_src, mu_llm, logvar_llm):
        """
        Compute Sliced Wasserstein Distance between two Gaussian distributions.
        
        Algorithm:
        1. Sample points from both distributions
        2. Generate random projection directions
        3. Project samples onto each direction
        4. Compute 1-D Wasserstein distance (closed-form for 1-D)
        5. Average (or max) over all projections
        
        Args:
            mu_src: [batch_size, latent_dim] - mean of collaborative distribution
            logvar_src: [batch_size, latent_dim] - log variance of collaborative distribution
            mu_llm: [batch_size, latent_dim] - mean of language distribution
            logvar_llm: [batch_size, latent_dim] - log variance of language distribution
        
        Returns:
            swd: scalar - sliced Wasserstein distance
        """
        batch_size, latent_dim = mu_src.shape
        
        # Sample points from both distributions
        num_samples = max(batch_size, 100)  # Use at least 100 samples for stability
        samples_src = self.sample_from_gaussian(mu_src, logvar_src, num_samples)
        samples_llm = self.sample_from_gaussian(mu_llm, logvar_llm, num_samples)
        
        # Generate random projection directions on unit sphere
        # Shape: [num_projections, latent_dim]
        projections = torch.randn(self.num_projections, latent_dim, device=mu_src.device)
        projections = F.normalize(projections, p=2, dim=1)  # Normalize to unit sphere
        
        # Project samples onto each direction
        # Shape: [num_samples, num_projections]
        proj_src = torch.matmul(samples_src, projections.t())
        proj_llm = torch.matmul(samples_llm, projections.t())
        
        # Compute 1-D Wasserstein distance for each projection
        # Sort the projected samples
        proj_src_sorted, _ = torch.sort(proj_src, dim=0)
        proj_llm_sorted, _ = torch.sort(proj_llm, dim=0)
        
        # 1-D Wasserstein-2 distance: mean of squared differences
        wasserstein_1d = torch.mean((proj_src_sorted - proj_llm_sorted) ** 2, dim=0)
        
        if self.use_max_sliced:
            # Max-Sliced: Take maximum over projections for sharper alignment
            swd = torch.max(wasserstein_1d)
        else:
            # Standard Sliced: Average over projections
            swd = torch.mean(wasserstein_1d)
        
        # Take square root to get Wasserstein-2 distance
        swd = torch.sqrt(swd + 1e-8)
        
        return swd

    def analytical_sliced_wasserstein(self, mu_src, logvar_src, mu_llm, logvar_llm):
        """
        Analytical approximation of Sliced Wasserstein for Gaussians.
        
        For Gaussian distributions, we can compute an analytical approximation by:
        1. Projecting the Gaussian parameters (mu, Sigma) onto random directions
        2. Computing 1-D Wasserstein between projected Gaussians analytically
        
        This is faster and has cleaner gradients than sampling-based approach.
        """
        batch_size, latent_dim = mu_src.shape
        
        # Generate random projection directions on unit sphere
        projections = torch.randn(self.num_projections, latent_dim, device=mu_src.device)
        projections = F.normalize(projections, p=2, dim=1)
        
        # Project means: [batch_size, num_projections]
        proj_mu_src = torch.matmul(mu_src, projections.t())
        proj_mu_llm = torch.matmul(mu_llm, projections.t())
        
        # Project variances (diagonal covariance assumed)
        # For diagonal Σ and projection θ: Var[θ^T X] = θ^T Σ θ = Σ_i θ_i^2 σ_i^2
        std_src = torch.exp(0.5 * logvar_src)  # [batch_size, latent_dim]
        std_llm = torch.exp(0.5 * logvar_llm)
        
        # [batch_size, num_projections]
        proj_var_src = torch.matmul(std_src ** 2, projections.t() ** 2)
        proj_var_llm = torch.matmul(std_llm ** 2, projections.t() ** 2)
        
        proj_std_src = torch.sqrt(proj_var_src + 1e-8)
        proj_std_llm = torch.sqrt(proj_var_llm + 1e-8)
        
        # 1-D Wasserstein-2 distance between Gaussians N(μ1, σ1^2) and N(μ2, σ2^2):
        # W_2^2 = (μ1 - μ2)^2 + (σ1 - σ2)^2
        mean_diff = (proj_mu_src - proj_mu_llm) ** 2
        std_diff = (proj_std_src - proj_std_llm) ** 2
        
        # Wasserstein distance for each projection: [batch_size, num_projections]
        wasserstein_1d = mean_diff + std_diff
        
        if self.use_max_sliced:
            # Max over projections, then average over batch
            swd = torch.mean(torch.max(wasserstein_1d, dim=1)[0])
        else:
            # Average over both projections and batch
            swd = torch.mean(wasserstein_1d)
        
        # Take square root to get Wasserstein-2 distance
        swd = torch.sqrt(swd + 1e-8)
        
        return swd

    def cal_loss(self, user, batch_data):
        self.is_training = True

        user_emb = self.usrprf_embeds[user]

        mu_src, mu_llm, logvar_src, logvar_llm = self.encode(batch_data, user_emb)

        # There are two processing strategies:
        # (1) performing downstream tasks after addition, and
        # (2) directly conducting downstream tasks.
        # In practice, we found that these two strategies have their own advantages on different datasets.
        # Therefore, we adopt a unified design that performs reconstruction and matching after addition.
        mu = mu_src + mu_llm
        logvar = logvar_src + logvar_llm

        # Perform downstream tasks directly
        # mu = mu_src
        # logvar = logvar_src

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        # Reconstruction loss
        BCE = - torch.mean(torch.sum(F.log_softmax(recon_x, 1) * batch_data, -1))

        # KL divergence with prior
        KLD = - 0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        # Sliced Wasserstein Distance (Improved GODM)
        # Use analytical version for efficiency and gradient stability
        SWD = self.analytical_sliced_wasserstein(mu_src, logvar_src, mu_llm, logvar_llm)
        
        # Alternative: use sampling-based version (uncomment if preferred)
        # SWD = self.sliced_wasserstein_distance(mu_src, logvar_src, mu_llm, logvar_llm)

        # Total regularization loss (GODM with Sliced Wasserstein)
        KLD = KLD + self.beta * SWD

        loss = BCE + KLD
        losses = {'rec_loss': BCE, 'reg_loss': KLD, 'swd': SWD}
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        batch_data = self.data_handler.train_data[pck_users.cpu()]
        data = torch.FloatTensor(batch_data.toarray()).to(configs['device'])
        user_emb = self.usrprf_embeds[pck_users]

        mu, mu_llm, logvar, logvar_llm = self.encode(data, user_emb)

        # If the downstream tasks are performed directly, this section should be commented out.
        mu = mu + mu_llm
        logvar = logvar + logvar_llm

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        full_preds = self._mask_predict(recon_x, train_mask)
        return full_preds
