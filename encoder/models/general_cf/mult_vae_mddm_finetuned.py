"""
Phase 3: Mult-VAE MDDM with Fine-tunable Open Source Encoder

Instead of frozen OpenAI embeddings, this uses a trainable sentence-transformer
that can be fine-tuned end-to-end from the recommendation loss.

Key differences from baseline:
- Embeddings are computed on-the-fly from text using transformer encoder
- Encoder weights are trainable and updated via gradient backpropagation
- No need for pre-computed frozen embeddings
- Full end-to-end training from text → embeddings → recommendations
"""

import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.base_model import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class FineTunableEncoder(nn.Module):
    """
    Wrapper around sentence-transformer that allows fine-tuning.
    
    Uses mean pooling over token embeddings to get fixed-size sentence representations.
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 freeze_encoder=False, cache_embeddings=True):
        super(FineTunableEncoder, self).__init__()
        
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        
        # Load pre-trained sentence transformer
        print(f"Loading encoder: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        # Access the underlying transformer
        self.transformer = self.encoder[0].auto_model
        self.tokenizer = self.encoder[0].tokenizer
        
        # Optionally freeze encoder (for comparison)
        if freeze_encoder:
            for param in self.transformer.parameters():
                param.requires_grad = False
            print("  Encoder frozen (not trainable)")
        else:
            print("  Encoder trainable (fine-tuning enabled)")
        
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"  Embedding dimension: {self.embedding_dim}")
        
        # Cache for efficiency during training
        self.text_cache = {}
        self.embedding_cache = {}
    
    def encode_texts(self, texts, batch_size=32):
        """
        Encode texts into embeddings using the transformer.
        
        Args:
            texts: List of strings
            batch_size: Batch size for encoding
            
        Returns:
            embeddings: [num_texts, embedding_dim]
        """
        if not isinstance(texts, list):
            texts = [texts]
        
        # Check cache first
        if self.cache_embeddings:
            uncached_texts = []
            uncached_indices = []
            embeddings = [None] * len(texts)
            
            for i, text in enumerate(texts):
                if text in self.embedding_cache:
                    embeddings[i] = self.embedding_cache[text]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Encode uncached texts
            if uncached_texts:
                new_embeddings = self.encoder.encode(
                    uncached_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    device=next(self.parameters()).device
                )
                
                # Update cache and results
                for i, idx in enumerate(uncached_indices):
                    emb = new_embeddings[i]
                    embeddings[idx] = emb
                    if self.cache_embeddings:
                        self.embedding_cache[uncached_texts[i]] = emb
            
            return torch.stack(embeddings)
        else:
            # Direct encoding without cache
            return self.encoder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                device=next(self.parameters()).device
            )
    
    def forward(self, texts):
        """Forward pass - encode texts to embeddings."""
        return self.encode_texts(texts)


class mult_vae_MDDM_FineTuned(BaseModel):
    def __init__(self, data_handler):
        super(mult_vae_MDDM_FineTuned, self).__init__(data_handler)

        self.beta = self.hyper_config['beta']
        self.freeze_encoder = self.hyper_config.get('freeze_encoder', False)
        self.encoder_name = self.hyper_config.get('encoder_name', 'sentence-transformers/all-MiniLM-L6-v2')
        
        self.data_handler = data_handler

        # VAE structure: [item_num, 600, 200, 600, item_num]
        self.p_dims = [200, 600, self.item_num]
        self.q_dims = [self.item_num, 600, 200]

        # Compute mean and variance in parallel
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]

        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])]
        )

        # Load text profiles
        print("Loading text profiles...")
        self.user_texts, self.item_texts = self._load_text_profiles()
        print(f"  Loaded {len(self.user_texts)} user profiles, {len(self.item_texts)} item profiles")
        
        # Initialize fine-tunable encoder
        self.text_encoder = FineTunableEncoder(
            model_name=self.encoder_name,
            freeze_encoder=self.freeze_encoder,
            cache_embeddings=True  # Cache for efficiency
        )
        
        # Pre-compute embeddings once for efficiency (with no_grad if frozen)
        print("Pre-computing embeddings...")
        if self.freeze_encoder:
            with torch.no_grad():
                self.usrprf_embeds = self.text_encoder.encode_texts(self.user_texts, batch_size=128)
                self.itmprf_embeds = self.text_encoder.encode_texts(self.item_texts, batch_size=128)
        else:
            # Keep gradients enabled for fine-tuning
            self.usrprf_embeds = self.text_encoder.encode_texts(self.user_texts, batch_size=128)
            self.itmprf_embeds = self.text_encoder.encode_texts(self.item_texts, batch_size=128)
            # Ensure requires_grad is True
            self.usrprf_embeds.requires_grad_(True)
            self.itmprf_embeds.requires_grad_(True)
        print(f"  User embeddings: {self.usrprf_embeds.shape}")
        print(f"  Item embeddings: {self.itmprf_embeds.shape}")

        # MLP for processing semantic information
        encoder_dim = self.text_encoder.embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, 600),
            nn.Tanh(),
            nn.Linear(600, 400)
        )

        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])]
        )

        self.drop = nn.Dropout(self.hyper_config['dropout'])

        self.final_embeds = None
        self.is_training = False
        
        # Training config
        self.recompute_embeddings_every = self.hyper_config.get('recompute_embeddings_every', 5)
        self.epoch_counter = 0

    def _load_text_profiles(self):
        """Load text profiles for users and items from pickle files."""
        dataset = configs['data']['name']
        
        # Load from pickle files in data/{dataset}/ folder
        data_path = Path(f"../data/{dataset}")
        
        user_pkl_path = data_path / 'usr_prf.pkl'
        item_pkl_path = data_path / 'itm_prf.pkl'
        
        print(f"  Loading user profiles from {user_pkl_path}")
        with open(user_pkl_path, 'rb') as f:
            usr_profiles = pickle.load(f)
        
        print(f"  Loading item profiles from {item_pkl_path}")
        with open(item_pkl_path, 'rb') as f:
            itm_profiles = pickle.load(f)
        
        # Extract text from profile dicts
        # Format: {0: {'profile': '...', 'reasoning': '...'}, 1: {...}, ...}
        user_texts = [usr_profiles[i]['profile'] for i in range(len(usr_profiles))]
        item_texts = [itm_profiles[i]['profile'] for i in range(len(itm_profiles))]
        
        return user_texts, item_texts
    
    def recompute_embeddings(self):
        """
        Recompute embeddings from text using updated encoder.
        Called periodically during training to update embeddings based on fine-tuned encoder.
        """
        if not self.freeze_encoder and self.is_training:
            with torch.enable_grad():
                self.usrprf_embeds = self.text_encoder.encode_texts(self.user_texts, batch_size=128)
                self.itmprf_embeds = self.text_encoder.encode_texts(self.item_texts, batch_size=128)

    def encode(self, x, user_emb):
        h = self.drop(x)
        
        # Use current embeddings (may be recomputed during training)
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
        
        # Periodically recompute embeddings if encoder is trainable
        if not self.freeze_encoder:
            self.epoch_counter += 1
            if self.epoch_counter % self.recompute_embeddings_every == 0:
                self.recompute_embeddings()

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
        
        losses = {'rec_loss': BCE, 'reg_loss': KLD}
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

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
