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
        self.device = configs['device']

        # VAE structure: [item_num, 600, 200, 600, item_num]
        self.p_dims = [200, 600, self.item_num]
        self.q_dims = [self.item_num, 600, 200]

        # Compute mean and variance in parallel
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]

        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])]
        )

        # Load LLM embeddings (keep originals, never modify)
        self.usrprf_embeds_raw = torch.tensor(configs['usrprf_embeds']).float().to(self.device)
        self.itmprf_embeds_raw = torch.tensor(configs['itmprf_embeds']).float().to(self.device)
        self.user_texts = configs.get('usrprf_texts', [])
        self.item_texts = configs.get('itmprf_texts', [])
        
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
        self.live_user_override_embeds = torch.zeros_like(self.usrprf_embeds_raw)
        self.live_item_override_embeds = torch.zeros_like(self.itmprf_embeds_raw)
        self.live_user_override_mask = torch.zeros(self.usrprf_embeds_raw.shape[0], dtype=torch.bool, device=self.device)
        self.live_item_override_mask = torch.zeros(self.itmprf_embeds_raw.shape[0], dtype=torch.bool, device=self.device)
        self.live_user_summaries = {}
        self.live_item_summaries = {}

    def _apply_live_overrides(self, embeddings, entity_type):
        if entity_type == 'user' and torch.any(self.live_user_override_mask):
            embeddings = embeddings.clone()
            embeddings[self.live_user_override_mask] = self.live_user_override_embeds[self.live_user_override_mask]
        if entity_type == 'item' and torch.any(self.live_item_override_mask):
            embeddings = embeddings.clone()
            embeddings[self.live_item_override_mask] = self.live_item_override_embeds[self.live_item_override_mask]
        return embeddings

    def update_live_prompt_bank(self, entity_type, indices, embeddings, summaries=None, epoch_idx=None):
        del epoch_idx
        if len(indices) == 0:
            return

        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        embed_tensor = torch.as_tensor(np.asarray(embeddings), dtype=self.usrprf_embeds_raw.dtype, device=self.device)
        if embed_tensor.ndim == 1:
            embed_tensor = embed_tensor.unsqueeze(0)

        if entity_type == 'user':
            if embed_tensor.shape[1] != self.usrprf_embeds_raw.shape[1]:
                raise ValueError('User live prompt embedding width mismatch.')
            self.live_user_override_embeds[index_tensor] = embed_tensor
            self.live_user_override_mask[index_tensor] = True
            if summaries is not None:
                for idx, summary in zip(indices, summaries):
                    self.live_user_summaries[int(idx)] = summary
        else:
            if embed_tensor.shape[1] != self.itmprf_embeds_raw.shape[1]:
                raise ValueError('Item live prompt embedding width mismatch.')
            self.live_item_override_embeds[index_tensor] = embed_tensor
            self.live_item_override_mask[index_tensor] = True
            if summaries is not None:
                for idx, summary in zip(indices, summaries):
                    self.live_item_summaries[int(idx)] = summary

    def get_live_prompt_state(self):
        return {
            'user_embeds': self.live_user_override_embeds.detach().cpu().clone(),
            'user_mask': self.live_user_override_mask.detach().cpu().clone(),
            'item_embeds': self.live_item_override_embeds.detach().cpu().clone(),
            'item_mask': self.live_item_override_mask.detach().cpu().clone(),
            'user_summaries': dict(self.live_user_summaries),
            'item_summaries': dict(self.live_item_summaries),
        }

    def load_live_prompt_state(self, state):
        if state is None:
            return
        self.live_user_override_embeds = state['user_embeds'].to(self.device)
        self.live_user_override_mask = state['user_mask'].to(self.device)
        self.live_item_override_embeds = state['item_embeds'].to(self.device)
        self.live_item_override_mask = state['item_mask'].to(self.device)
        self.live_user_summaries = dict(state.get('user_summaries', {}))
        self.live_item_summaries = dict(state.get('item_summaries', {}))

    def _summarize_anchor_items(self, item_indices, item_texts, top_k_items):
        anchors = []
        for item_idx in item_indices[:top_k_items]:
            text = ''
            if item_texts and int(item_idx) < len(item_texts):
                text = item_texts[int(item_idx)]
            anchors.append({'item_id': int(item_idx), 'text': text})
        return anchors

    def _summarize_support_users(self, user_indices, user_texts, top_k_users):
        supports = []
        for user_idx in user_indices[:top_k_users]:
            text = ''
            if user_texts and int(user_idx) < len(user_texts):
                text = user_texts[int(user_idx)]
            supports.append({'user_id': int(user_idx), 'text': text})
        return supports

    def _compute_encoder_stats(self, user_indices, batch_data):
        self.is_training = False
        self.refine_embeddings()
        user_tensor = torch.as_tensor(user_indices, dtype=torch.long, device=self.device)
        user_emb = self.usrprf_embeds[user_tensor]
        mu_src, mu_llm, logvar_src, logvar_llm = self.encode(batch_data, user_emb)
        return user_tensor, user_emb, mu_src, mu_llm, logvar_src, logvar_llm

    def collect_live_prompt_user_payload(self, user_indices, batch_data, item_texts=None, top_k_items=5):
        with torch.no_grad():
            user_tensor, user_emb, mu_src, mu_llm, logvar_src, logvar_llm = self._compute_encoder_stats(user_indices, batch_data)
            raw_user = self.usrprf_embeds_raw[user_tensor]
            correction = user_emb - raw_user
            payloads = []

            for row_idx, user_idx in enumerate(user_tensor.tolist()):
                interacted = torch.nonzero(batch_data[row_idx] > 0, as_tuple=False).flatten().tolist()
                payloads.append({
                    'entity_id': int(user_idx),
                    'base_profile': self.user_texts[user_idx] if user_idx < len(self.user_texts) else '',
                    'correction_alpha': float(self.get_correction_alpha()),
                    'mu_gap_norm': float(torch.norm(mu_src[row_idx] - mu_llm[row_idx]).item()),
                    'logvar_gap_norm': float(torch.norm(logvar_src[row_idx] - logvar_llm[row_idx]).item()),
                    'correction_norm': float(torch.norm(correction[row_idx]).item()),
                    'anchor_items': self._summarize_anchor_items(interacted, item_texts or self.item_texts, top_k_items),
                })

            return payloads

    def collect_live_prompt_item_payload(self, user_indices, batch_data, user_texts=None, item_texts=None, item_sample_size=32, top_k_support_users=3):
        with torch.no_grad():
            user_tensor, user_emb, mu_src, mu_llm, logvar_src, logvar_llm = self._compute_encoder_stats(user_indices, batch_data)
            item_support = batch_data.sum(dim=0)
            active_items = torch.nonzero(item_support > 0, as_tuple=False).flatten()
            if active_items.numel() == 0:
                return []

            sample_size = min(int(item_sample_size), int(active_items.numel()))
            ranked_items = active_items[torch.topk(item_support[active_items], k=sample_size).indices]
            payloads = []

            for item_idx in ranked_items.tolist():
                support_rows = torch.nonzero(batch_data[:, item_idx] > 0, as_tuple=False).flatten()
                support_user_ids = user_tensor[support_rows].tolist()
                support_user_emb = user_emb[support_rows]
                if support_user_emb.shape[0] == 0:
                    continue

                avg_user_emb = support_user_emb.mean(dim=0, keepdim=True)
                raw_item = self.itmprf_embeds_raw[item_idx].unsqueeze(0)
                refined_item = self.itmprf_embeds[item_idx].unsqueeze(0)
                alignment_before = F.cosine_similarity(raw_item, avg_user_emb, dim=1).item()
                alignment_after = F.cosine_similarity(refined_item, avg_user_emb, dim=1).item()
                support_gap = torch.norm(mu_src[support_rows] - mu_llm[support_rows], dim=1).mean().item()
                logvar_gap = torch.norm(logvar_src[support_rows] - logvar_llm[support_rows], dim=1).mean().item()

                payloads.append({
                    'entity_id': int(item_idx),
                    'base_profile': (item_texts or self.item_texts)[item_idx] if item_idx < len(item_texts or self.item_texts) else '',
                    'correction_alpha': float(self.get_correction_alpha()),
                    'mu_gap_norm': float(support_gap),
                    'logvar_gap_norm': float(logvar_gap),
                    'correction_norm': float(torch.norm(refined_item - raw_item).item()),
                    'support_count': int(support_rows.numel()),
                    'alignment_delta': float(alignment_after - alignment_before),
                    'support_users': self._summarize_support_users(support_user_ids, user_texts or self.user_texts, top_k_support_users),
                })

            return payloads

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
        self.usrprf_embeds = self._apply_live_overrides(self.usrprf_embeds, 'user')
        self.itmprf_embeds = self._apply_live_overrides(self.itmprf_embeds, 'item')

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
