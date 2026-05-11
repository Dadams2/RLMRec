"""
End-to-End Multi-View ProEx Recommendation Model

Combines ProEx's multi-profile environment-based invariance learning with
end-to-end gradient flow through learnable profile embeddings.

Key innovations over DMRec/ProEx:
1. Profile embeddings are trainable (not frozen) — recommendation errors reshape
   the semantic space via gradient backpropagation
2. K virtual profile views per user/item, initialized from LLM embeddings + learnable offsets
3. Interaction-conditioned profile mixer replaces random Dirichlet sampling
4. Multi-environment training with invariance constraint (ProEx)
5. Configurable distribution matching: MDDM, GODM, or MMD

Architecture:
  ProfileBank(K learnable views) → ProfileMixer(interaction-conditioned attention)
  → EnvironmentSampler(Dirichlet-perturbed) → MLP → Distribution
  → DistributionMatcher(MDDM/GODM/MMD) + CollaborativeVAE → Reconstruction
"""

import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.base_model import BaseModel
import numpy as np

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class ProfileBank(nn.Module):
    """
    Manages K learnable profile embedding views per entity.

    Two modes:
    - Virtual (default): view_k = frozen_original + learnable_offset_k
    - Real multi-profile: view_k = real_k_embedding + learnable_offset_k
      (when multi-profile embeddings are provided from LLM generation)
    """
    def __init__(self, raw_embeds, K, init_scale=0.01, multi_embeds=None):
        super().__init__()
        num_entities, embed_dim = raw_embeds.shape
        self.K = K
        self.embed_dim = embed_dim
        self.has_multi = multi_embeds is not None

        if self.has_multi:
            # Real multi-profile embeddings: [N, K, D]
            self.register_buffer('base', multi_embeds)  # [N, K, D]
        else:
            # Virtual mode: single base, K offsets
            self.register_buffer('raw', raw_embeds)  # [N, D]

        # Learnable offsets for K views — small init so view ≈ base
        self.offsets = nn.Parameter(torch.randn(num_entities, K, embed_dim) * init_scale)

    def get_profiles(self, indices):
        """
        Args:
            indices: [batch] entity indices
        Returns:
            [batch, K, D] profile embeddings with gradient flow
        """
        if self.has_multi:
            base = self.base[indices]                          # [B, K, D]
        else:
            base = self.raw[indices].unsqueeze(1)              # [B, 1, D]
        offsets = self.offsets[indices]                         # [B, K, D]
        return base + offsets                                  # [B, K, D]

    def get_all_profiles(self):
        """Returns [N, K, D] for all entities."""
        if self.has_multi:
            base = self.base                                   # [N, K, D]
        else:
            base = self.raw.unsqueeze(1)                       # [N, 1, D]
        return base + self.offsets                             # [N, K, D]

    def drift_loss(self):
        """L2 penalty on offsets to keep profiles near initialization."""
        return torch.mean(self.offsets.pow(2))


class ProfileMixer(nn.Module):
    """
    Interaction-conditioned attention over K profile views.

    Uses the collaborative VAE's latent representation to produce
    data-dependent mixing weights, replacing ProEx's random Dirichlet.
    """
    def __init__(self, input_dim, K, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K),
        )

    def forward(self, collab_hidden):
        """
        Args:
            collab_hidden: [batch, input_dim] — collaborative signal
        Returns:
            [batch, K] softmax attention weights
        """
        return F.softmax(self.net(collab_hidden), dim=-1)


class EnvironmentSampler:
    """
    Generates E sets of mixing weights by perturbing learned attention
    with Dirichlet noise.  Noise anneals from high (random) to low (learned).
    """
    def __init__(self, K, E, dirichlet_alpha=1.0):
        self.K = K
        self.E = E
        self.dirichlet_alpha = dirichlet_alpha

    def sample(self, base_weights, noise_scale):
        """
        Args:
            base_weights: [batch, K] learned attention
            noise_scale: float in [0, 1], 1 = full random, 0 = pure learned
        Returns:
            list of E tensors, each [batch, K]
        """
        envs = []
        batch_size = base_weights.shape[0]
        device = base_weights.device

        for _ in range(self.E):
            if noise_scale > 0:
                alpha = torch.full((batch_size, self.K), self.dirichlet_alpha,
                                   device=device)
                # Dirichlet via Gamma reparameterization (differentiable-ish noise)
                gamma_samples = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).sample()
                dirichlet = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)
                weights = (1 - noise_scale) * base_weights + noise_scale * dirichlet
                # Re-normalize
                weights = weights / weights.sum(dim=-1, keepdim=True)
            else:
                weights = base_weights
            envs.append(weights)
        return envs


class DistributionMatcher(nn.Module):
    """
    Pluggable distribution matching: MDDM, GODM, or MMD.

    Computes the alignment loss between collaborative-space and
    language-space distributions.
    """
    def __init__(self, strategy='mddm', beta=0.5, mmd_bandwidths=None):
        super().__init__()
        self.strategy = strategy
        self.beta = beta

        if strategy == 'mmd':
            self.bandwidths = mmd_bandwidths or [0.1, 0.5, 1.0, 2.0, 5.0]
            self.log_bandwidth_scales = nn.Parameter(torch.zeros(len(self.bandwidths)))

    def forward(self, mu_combined, logvar_combined,
                mu_src, logvar_src, mu_llm, logvar_llm):
        """
        Returns: (total_kl, dict of sub-losses)
        """
        if self.strategy == 'mddm':
            return self._mddm(mu_combined, logvar_combined,
                               mu_src, logvar_src, mu_llm, logvar_llm)
        elif self.strategy == 'godm':
            return self._godm(mu_combined, logvar_combined,
                               mu_src, logvar_src, mu_llm, logvar_llm)
        elif self.strategy == 'mmd':
            return self._mmd(mu_combined, logvar_combined,
                              mu_src, logvar_src, mu_llm, logvar_llm)
        else:
            raise ValueError(f"Unknown matching strategy: {self.strategy}")

    def _mddm(self, mu, logvar, mu_src, logvar_src, mu_llm, logvar_llm):
        """Mixing Divergence Distribution Matching (DMRec Eq. 16-17)."""
        KLD_standard = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        KLD_llm = -0.5 * torch.mean(
            torch.sum(1 + torch.log(logvar.exp() / (logvar_llm.exp() + 1e-7) + 1e-7)
                       - (mu - mu_llm).pow(2) / (logvar_llm.exp() + 1e-7)
                       - logvar.exp() / (logvar_llm.exp() + 1e-7), dim=1))
        kl = self.beta * KLD_standard + (1 - self.beta) * KLD_llm
        return kl, {'kl_std': KLD_standard, 'kl_llm': KLD_llm}

    def _godm(self, mu, logvar, mu_src, logvar_src, mu_llm, logvar_llm):
        """Global Optimality Distribution Matching (Wasserstein)."""
        KLD_standard = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        mean_diff = torch.norm(mu_src - mu_llm, dim=1) ** 2
        var_diff = torch.norm(torch.exp(0.5 * logvar_src) - torch.exp(0.5 * logvar_llm), dim=1) ** 2
        WD = torch.mean(torch.sqrt(mean_diff + var_diff + 1e-7))
        kl = KLD_standard + self.beta * WD
        return kl, {'kl_std': KLD_standard, 'wasserstein': WD}

    def _mmd(self, mu, logvar, mu_src, logvar_src, mu_llm, logvar_llm):
        """Maximum Mean Discrepancy matching."""
        KLD_standard = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        # Sample from both distributions
        z_src = mu_src + torch.exp(0.5 * logvar_src) * torch.randn_like(logvar_src)
        z_llm = mu_llm + torch.exp(0.5 * logvar_llm) * torch.randn_like(logvar_llm)

        mmd_val = self._compute_mmd(z_src, z_llm)
        kl = self.beta * KLD_standard + (1 - self.beta) * mmd_val
        return kl, {'kl_std': KLD_standard, 'mmd': mmd_val}

    def _compute_mmd(self, x, y):
        """Multi-scale RBF MMD."""
        batch_size = x.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=x.device)
        mask = 1 - torch.eye(batch_size, device=x.device)
        n_pairs = mask.sum()
        mmd_total = torch.tensor(0.0, device=x.device)

        for idx, bw in enumerate(self.bandwidths):
            scale = torch.exp(self.log_bandwidth_scales[idx])
            bandwidth = bw * scale

            def rbf(a, b, bw):
                aa = (a ** 2).sum(1, keepdim=True)
                bb = (b ** 2).sum(1, keepdim=True)
                dist = torch.clamp(aa - 2 * a @ b.t() + bb.t(), min=0.0)
                return torch.exp(-dist / (2 * bw ** 2 + 1e-8))

            kxx = (rbf(x, x, bandwidth) * mask).sum() / n_pairs
            kyy = (rbf(y, y, bandwidth) * mask).sum() / n_pairs
            kxy = (rbf(x, y, bandwidth) * mask).sum() / n_pairs
            mmd_total = mmd_total + (kxx + kyy - 2 * kxy)

        return mmd_total


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class mult_vae_ProEx_E2E(BaseModel):
    def __init__(self, data_handler):
        super(mult_vae_ProEx_E2E, self).__init__(data_handler)

        self.data_handler = data_handler

        # --- Hyperparameters ---
        self.beta = self.hyper_config['beta']
        self.K = self.hyper_config.get('num_profiles', 4)
        self.E = self.hyper_config.get('num_environments', 2)
        self.lambda_align = self.hyper_config.get('lambda_align', 1.0)
        self.lambda_contrast = self.hyper_config.get('lambda_contrast', 0.01)
        self.lambda_variance = self.hyper_config.get('lambda_variance', 0.5)
        self.lambda_drift = self.hyper_config.get('lambda_drift', 0.1)
        self.contrast_tau = self.hyper_config.get('contrast_tau', 0.2)
        self.warmup_epochs = self.hyper_config.get('warmup_epochs', 20)
        self.dirichlet_alpha = self.hyper_config.get('dirichlet_alpha', 1.0)
        self.init_scale = self.hyper_config.get('init_scale', 0.01)
        matching_strategy = self.hyper_config.get('matching_strategy', 'mddm')
        mixer_input = self.hyper_config.get('mixer_input', 'latent')  # 'latent' or 'interaction'

        self.current_epoch = 0

        # --- VAE architecture (same as base MDDM) ---
        self.latent_dim = 200
        self.p_dims = [self.latent_dim, 600, self.item_num]
        self.q_dims = [self.item_num, 600, self.latent_dim]
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]

        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in
             zip(temp_q_dims[:-1], temp_q_dims[1:])])

        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in
             zip(self.p_dims[:-1], self.p_dims[1:])])

        self.drop = nn.Dropout(self.hyper_config['dropout'])

        # --- Profile Banks (learnable multi-view embeddings) ---
        raw_usr = torch.tensor(configs['usrprf_embeds']).float().cuda()
        raw_itm = torch.tensor(configs['itmprf_embeds']).float().cuda()
        self.embed_dim = raw_usr.shape[1]  # 1536

        # Use real multi-profile embeddings if available
        multi_usr = None
        multi_itm = None
        if 'usrprf_multi_embeds' in configs:
            multi_usr = torch.tensor(configs['usrprf_multi_embeds']).float().cuda()
            print(f"  [ProEx E2E] Using real multi-profile user embeddings: {multi_usr.shape}")
        if 'itmprf_multi_embeds' in configs:
            multi_itm = torch.tensor(configs['itmprf_multi_embeds']).float().cuda()
            print(f"  [ProEx E2E] Using real multi-profile item embeddings: {multi_itm.shape}")

        self.user_bank = ProfileBank(raw_usr, self.K, init_scale=self.init_scale,
                                     multi_embeds=multi_usr)
        self.item_bank = ProfileBank(raw_itm, self.K, init_scale=self.init_scale,
                                     multi_embeds=multi_itm)

        # --- MLP: semantic embedding → VAE distribution params ---
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 600),
            nn.Tanh(),
            nn.Linear(600, self.latent_dim * 2),  # → mu_llm[200] + logvar_llm[200]
        )

        # --- Profile mixer (interaction-conditioned attention over K views) ---
        if mixer_input == 'latent':
            mixer_in_dim = self.latent_dim * 2   # mu_src + logvar_src
        else:
            mixer_in_dim = self.item_num
        self.mixer_input = mixer_input
        self.user_mixer = ProfileMixer(mixer_in_dim, self.K)
        self.item_mixer = None  # Items use mean-pooling (no per-batch interaction signal)

        # --- Environment sampler ---
        self.env_sampler = EnvironmentSampler(self.K, self.E, self.dirichlet_alpha)

        # --- Distribution matcher ---
        self.matcher = DistributionMatcher(
            strategy=matching_strategy,
            beta=self.beta,
            mmd_bandwidths=[0.1, 0.5, 1.0, 2.0, 5.0] if matching_strategy == 'mmd' else None,
        )

        self.is_training = False

    # ---- epoch / warmup ----
    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _noise_scale(self):
        """Anneal environment noise: 1 (random) → 0 (learned)."""
        if self.warmup_epochs <= 0:
            return 0.0
        return max(0.0, 1.0 - self.current_epoch / self.warmup_epochs)

    # ---- VAE encode / decode (collaborative branch) ----
    def _collab_encode(self, x):
        """Standard collaborative VAE encoder on interaction vector."""
        h = self.drop(x)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.latent_dim]
                logvar = h[:, self.latent_dim:]
        return mu, logvar

    def _decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def _reparameterize(self, mu, logvar):
        if self.is_training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    # ---- language branch ----
    def _language_encode(self, mixed_embed):
        """
        Transform mixed profile embedding → language-space distribution params.
        Args:
            mixed_embed: [batch, embed_dim]
        Returns:
            mu_llm [batch, latent_dim], logvar_llm [batch, latent_dim]
        """
        h = self.mlp(mixed_embed)
        return h[:, :self.latent_dim], h[:, self.latent_dim:]

    def _mix_profiles(self, user_profiles, item_profiles, weights_user):
        """
        Weighted combination of K profile views.

        user_profiles: [batch, K, D]
        item_profiles: [item_num, K, D]
        weights_user:  [batch, K]

        Returns:
            mixed_user_embed: [batch, D]
            mixed_item_embed: [item_num, D]  (mean-pooled; no per-user signal for items)
        """
        # User: attention-weighted
        mixed_user = torch.einsum('bk,bkd->bd', weights_user, user_profiles)

        # Item: mean pool across K views
        mixed_item = item_profiles.mean(dim=1)  # [item_num, D]

        return mixed_user, mixed_item

    # ---- contrastive regularisation (ProEx Eq. 7) ----
    def _contrastive_loss(self, profiles):
        """
        Push apart K profile views per entity.
        profiles: [batch, K, D]
        """
        B, K, D = profiles.shape
        # Normalize
        p = F.normalize(profiles, dim=-1)                     # [B, K, D]
        # Pairwise cosine similarity between K views
        sim = torch.bmm(p, p.transpose(1, 2))                # [B, K, K]

        # Mask out diagonal (self-similarity)
        mask = ~torch.eye(K, dtype=torch.bool, device=profiles.device)  # [K, K]
        mask = mask.unsqueeze(0).expand(B, -1, -1)            # [B, K, K]
        sim_masked = sim.masked_select(mask).view(B, K, K - 1)

        loss = torch.log(1 + torch.exp(
            torch.tensor(1.0 / self.contrast_tau, device=profiles.device))
            * torch.exp(sim_masked / self.contrast_tau).sum(dim=-1)).mean()
        return loss

    # ---- main forward ----
    def cal_loss(self, user, batch_data):
        self.is_training = True

        # 1. Collaborative encoder
        mu_src, logvar_src = self._collab_encode(batch_data)

        # 2. Get K profile views
        user_profiles = self.user_bank.get_profiles(user)       # [B, K, D]
        item_profiles = self.item_bank.get_all_profiles()       # [N, K, D]

        # 3. Compute mixer input
        if self.mixer_input == 'latent':
            mixer_in = torch.cat([mu_src.detach(), logvar_src.detach()], dim=-1)
        else:
            mixer_in = batch_data
        base_weights = self.user_mixer(mixer_in)                # [B, K]

        # 4. Multi-environment forward
        noise_scale = self._noise_scale()
        env_weights = self.env_sampler.sample(base_weights, noise_scale)

        env_losses = []
        total_rec = torch.tensor(0.0, device=batch_data.device)
        total_match = torch.tensor(0.0, device=batch_data.device)

        for weights in env_weights:
            # Mix profiles for this environment
            mixed_user, mixed_item = self._mix_profiles(
                user_profiles, item_profiles, weights)

            # Language-space distribution from mixed semantic embedding
            # Interaction-weighted item semantics + user semantics
            h = self.drop(batch_data)
            semantic_input = torch.matmul(h, mixed_item) + mixed_user  # [B, D]
            mu_llm, logvar_llm = self._language_encode(semantic_input)

            # Combine distributions
            mu = mu_src + mu_llm
            logvar = logvar_src + logvar_llm

            # Reparameterize + decode
            z = self._reparameterize(mu, logvar)
            recon_x = self._decode(z)

            # Reconstruction loss
            bce = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * batch_data, -1))

            # Distribution matching
            kl, _ = self.matcher(mu, logvar, mu_src, logvar_src, mu_llm, logvar_llm)

            env_loss = bce + self.lambda_align * kl
            env_losses.append(env_loss)
            total_rec = total_rec + bce
            total_match = total_match + kl

        # 5. Aggregate environment losses
        env_loss_stack = torch.stack(env_losses)
        loss_mean = env_loss_stack.mean()
        loss_var = env_loss_stack.var() if self.E > 1 else torch.tensor(0.0, device=batch_data.device)

        # 6. Contrastive regularisation on user profiles
        contrast = self._contrastive_loss(user_profiles)

        # 7. Drift regularisation
        drift = self.user_bank.drift_loss() + self.item_bank.drift_loss()

        # Total loss
        loss = (loss_mean
                + self.lambda_contrast * contrast
                + self.lambda_variance * loss_var
                + self.lambda_drift * drift)

        losses = {
            'rec_loss': total_rec.item() / self.E,
            'match_loss': total_match.item() / self.E,
            'env_var': loss_var.item(),
            'contrast': contrast.item(),
            'drift': drift.item(),
            'noise_scale': noise_scale,
        }
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        data = torch.FloatTensor(
            self.data_handler.train_data[pck_users.cpu()].toarray()
        ).to(configs['device'])

        # Collaborative encoder
        mu_src, logvar_src = self._collab_encode(data)

        # Mean-pool K profile views for inference (no environments)
        user_profiles = self.user_bank.get_profiles(pck_users)   # [B, K, D]
        item_profiles = self.item_bank.get_all_profiles()        # [N, K, D]
        mixed_user = user_profiles.mean(dim=1)                   # [B, D]
        mixed_item = item_profiles.mean(dim=1)                   # [N, D]

        # Language branch
        semantic_input = torch.matmul(data, mixed_item) + mixed_user
        mu_llm, logvar_llm = self._language_encode(semantic_input)

        mu = mu_src + mu_llm
        logvar = logvar_src + logvar_llm

        z = self._reparameterize(mu, logvar)
        recon_x = self._decode(z)

        return self._mask_predict(recon_x, train_mask)
