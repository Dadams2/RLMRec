import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
import numpy as np

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class mult_vae_MDDM_Alpha(BaseModel):
    """
    MultVAE with Mixing Divergence using α-divergence.
    
    α-divergence generalizes many common divergences:
    - α → 0: Reverse KL (mode-seeking, favors precision)
    - α = 0.5: Hellinger distance (balanced, robust)
    - α = 1.0: Forward KL (mode-covering, favors recall)
    - α → ∞: χ²-divergence
    
    For α ∈ (0,1): D_α(P||Q) = (1/(α(1-α))) * [1 - ∫ p(x)^α * q(x)^(1-α) dx]
    
    This implementation uses a Monte Carlo approximation for Gaussian distributions.
    
    Advantages over KL/MMD:
    - Tunable trade-off between mode-seeking and mode-covering
    - More robust to mismatched supports
    - Better stability when distributions are disjoint
    - Single hyperparameter (α) controls behavior
    """
    def __init__(self, data_handler):
        super(mult_vae_MDDM_Alpha, self).__init__(data_handler)

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

        # α-divergence parameters
        self.alpha = self.hyper_config.get('alpha', 0.5)  # Default: Hellinger distance
        self.num_samples = self.hyper_config.get('num_alpha_samples', 10)  # MC samples
        self.alpha_weight = self.hyper_config.get('alpha_weight', 1.0)  # Scaling factor
        
        # Use analytical form when possible (for special α values)
        self.use_analytical = self.hyper_config.get('use_analytical', True)

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

    def compute_alpha_divergence_gaussian(self, mu_p, logvar_p, mu_q, logvar_q):
        """
        Compute α-divergence between two Gaussian distributions.
        
        For Gaussian distributions, we can use analytical forms for some α values
        or Monte Carlo approximation for general α.
        
        Args:
            mu_p: mean of distribution P (collaborative space)
            logvar_p: log variance of distribution P
            mu_q: mean of distribution Q (language space)  
            logvar_q: log variance of distribution Q
            
        Returns:
            α-divergence value (scalar)
        """
        alpha = self.alpha
        
        # Special case: α = 0.5 (Hellinger distance) - has analytical form
        if self.use_analytical and abs(alpha - 0.5) < 1e-6:
            return self._hellinger_distance_gaussian(mu_p, logvar_p, mu_q, logvar_q)
        
        # Special case: α = 1.0 (Forward KL) - standard KL divergence
        if self.use_analytical and abs(alpha - 1.0) < 1e-6:
            return self._kl_divergence_gaussian(mu_p, logvar_p, mu_q, logvar_q)
        
        # Special case: α → 0 (Reverse KL)
        if self.use_analytical and abs(alpha) < 1e-6:
            return self._kl_divergence_gaussian(mu_q, logvar_q, mu_p, logvar_p)
        
        # General case: Monte Carlo approximation
        return self._alpha_divergence_mc(mu_p, logvar_p, mu_q, logvar_q, alpha)

    def _hellinger_distance_gaussian(self, mu_p, logvar_p, mu_q, logvar_q):
        """
        Analytical Hellinger distance for Gaussian distributions (α=0.5).
        
        For Gaussians: H²(P,Q) = 1 - sqrt(det(Σ_p)^(1/4) * det(Σ_q)^(1/4) / det((Σ_p+Σ_q)/2)^(1/2))
                                     * exp(-1/8 * (μ_p-μ_q)^T * ((Σ_p+Σ_q)/2)^(-1) * (μ_p-μ_q))
        
        Simplified for diagonal covariance:
        H²(P,Q) = 1 - exp(sum(-1/4*log(var_avg) + 1/8*(log(var_p) + log(var_q)) - 1/8*(μ_p-μ_q)²/var_avg))
        """
        var_p = torch.exp(logvar_p)
        var_q = torch.exp(logvar_q)
        
        # Average variance: (Σ_p + Σ_q) / 2
        var_avg = (var_p + var_q) / 2.0 + 1e-8
        
        # Mahalanobis distance term: (μ_p - μ_q)^T * Σ_avg^(-1) * (μ_p - μ_q)
        mu_diff = mu_p - mu_q
        mahala_term = -0.125 * torch.sum(mu_diff ** 2 / var_avg, dim=1)
        
        # Determinant term: sqrt(det(Σ_p) * det(Σ_q) / det(Σ_avg))
        # For diagonal: sum(0.25 * log(var_p) + 0.25 * log(var_q) - 0.5 * log(var_avg))
        det_term = 0.125 * torch.sum(logvar_p + logvar_q - 2.0 * torch.log(var_avg), dim=1)
        
        # Hellinger distance squared: 1 - exp(mahala_term + det_term)
        # Clamp the exponent for numerical stability
        exponent = torch.clamp(mahala_term + det_term, min=-10, max=0)
        hellinger_sq = 1.0 - torch.exp(exponent)
        
        # Clamp result to valid range [0, 1]
        hellinger_sq = torch.clamp(hellinger_sq, min=0.0, max=1.0)
        
        return torch.mean(hellinger_sq) * self.alpha_weight

    def _kl_divergence_gaussian(self, mu_p, logvar_p, mu_q, logvar_q):
        """
        Analytical KL divergence for Gaussian distributions (α=1.0).
        
        KL(P||Q) = 0.5 * [log(det(Σ_q)/det(Σ_p)) - d + Tr(Σ_q^(-1) * Σ_p) + 
                          (μ_q - μ_p)^T * Σ_q^(-1) * (μ_q - μ_p)]
        """
        var_p = torch.exp(logvar_p)
        var_q = torch.exp(logvar_q) + 1e-8  # Add epsilon for stability
        
        # Trace term: Tr(Σ_q^(-1) * Σ_p) = Σ(σ²_p / σ²_q)
        trace_term = torch.sum(var_p / var_q, dim=1)
        
        # Mahalanobis distance term
        mu_diff = mu_p - mu_q
        mahala = torch.sum(mu_diff ** 2 / var_q, dim=1)
        
        # Log determinant term
        logdet = torch.sum(logvar_q - logvar_p, dim=1)
        
        # Dimensionality
        d = mu_p.shape[1]
        
        kl = 0.5 * (logdet - d + trace_term + mahala)
        
        # KL should be non-negative, clamp for numerical stability
        kl = torch.clamp(kl, min=0.0)
        
        return torch.mean(kl) * self.alpha_weight

    def _alpha_divergence_mc(self, mu_p, logvar_p, mu_q, logvar_q, alpha):
        """
        Monte Carlo approximation of α-divergence.
        
        D_α(P||Q) = (1/(α(1-α))) * [1 - ∫ p(x)^α * q(x)^(1-α) dx]
                  ≈ (1/(α(1-α))) * [1 - (1/N) * Σ (p(x_i)/q(x_i))^(1-α)]
        
        where x_i ~ P
        """
        batch_size = mu_p.shape[0]
        
        # Sample from P (collaborative space)
        samples_p = []
        for _ in range(self.num_samples):
            z = self.reparameterize(mu_p, logvar_p)
            samples_p.append(z)
        samples_p = torch.stack(samples_p, dim=1)  # [batch, num_samples, latent_dim]
        
        # Compute log p(x) for each sample
        log_p = self._log_gaussian_prob(samples_p, mu_p.unsqueeze(1), logvar_p.unsqueeze(1))
        
        # Compute log q(x) for each sample  
        log_q = self._log_gaussian_prob(samples_p, mu_q.unsqueeze(1), logvar_q.unsqueeze(1))
        
        # Compute ratio (p/q)^(1-α) in log space
        # log[(p/q)^(1-α)] = (1-α) * (log p - log q)
        log_ratio = (1.0 - alpha) * (log_p - log_q)
        
        # Numerical stability: clamp before exp
        log_ratio = torch.clamp(log_ratio, min=-10, max=10)
        ratio = torch.exp(log_ratio)
        
        # Average over samples
        ratio_mean = torch.mean(ratio, dim=1)
        
        # α-divergence: (1/(α(1-α))) * [1 - E[ratio]]
        if abs(alpha * (1 - alpha)) > 1e-6:
            alpha_div = (1.0 - ratio_mean) / (alpha * (1 - alpha))
        else:
            # Limit case: use KL divergence
            alpha_div = torch.mean(log_p - log_q, dim=1)
        
        return torch.mean(alpha_div) * self.alpha_weight

    def _log_gaussian_prob(self, x, mu, logvar):
        """
        Compute log probability of x under Gaussian N(mu, exp(logvar)).
        
        log p(x) = -0.5 * [d*log(2π) + Σlog(σ²) + Σ((x-μ)/σ)²]
        """
        var = torch.exp(logvar)
        
        # Mahalanobis distance
        diff = x - mu
        mahala = torch.sum(diff ** 2 / var, dim=-1)
        
        # Log determinant
        logdet = torch.sum(logvar, dim=-1)
        
        # Dimensionality
        d = x.shape[-1]
        
        log_prob = -0.5 * (d * np.log(2 * np.pi) + logdet + mahala)
        
        return log_prob

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
        
        return x_pred, mu, logvar, mu_llm, logvar_llm

    def cal_loss(self, user, batch_data):
        # Get user embeddings
        user_emb = self.usrprf_embeds[user]
        
        # Forward pass
        x_pred, mu_collab, logvar_collab, mu_lang, logvar_lang = self.forward(batch_data, user_emb)
        
        # Combined distributions
        mu = mu_collab + mu_lang
        logvar = logvar_collab + logvar_lang
        
        # Reconstruction loss (negative log-likelihood)
        BCE = -torch.mean(torch.sum(F.log_softmax(x_pred, 1) * batch_data, -1))
        
        # Mixing Divergence with α-divergence:
        # L = BCE + β·KL(q||N(0,I)) + (1-β)·D_α(q_collab||q_lang)
        
        # Term 1: Standard KL divergence to Gaussian prior (regularization)
        # KL(N(μ, σ²) || N(0, I)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        KLD_standard = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        
        # Term 2: α-divergence between collaborative and language distributions (matching)
        # This provides more flexible distribution matching than KL or MMD
        Alpha_matching = self.compute_alpha_divergence_gaussian(
            mu_collab, logvar_collab, mu_lang, logvar_lang
        )
        
        # Mixing divergence: balance regularization and matching
        # β controls the trade-off:
        # - High β (e.g., 0.6-0.8): more Gaussian regularization
        # - Low β (e.g., 0.2-0.4): more distribution matching via α-divergence
        mixing_divergence = self.beta * KLD_standard + (1 - self.beta) * Alpha_matching
        
        # Total loss
        loss = BCE + mixing_divergence
        
        # Check for numerical issues and log loss components
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Loss is {'NaN' if torch.isnan(loss) else 'Inf'}!")
            print(f"  BCE: {BCE.item():.4f}")
            print(f"  KLD_standard: {KLD_standard.item():.4f}")
            print(f"  Alpha_matching: {Alpha_matching.item():.4f}")
            print(f"  mu range: [{mu.min().item():.3f}, {mu.max().item():.3f}]")
            print(f"  logvar range: [{logvar.min().item():.3f}, {logvar.max().item():.3f}]")
        
        # Return detailed losses for monitoring
        losses = {
            'rec_loss': BCE,
            'reg_loss': mixing_divergence,
            'kld_standard': KLD_standard,
            'alpha_matching': Alpha_matching
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
