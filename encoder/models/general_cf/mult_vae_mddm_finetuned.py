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
from torch.utils.checkpoint import checkpoint
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
    
    Computes embeddings on-the-fly to maintain gradient flow.
    Uses gradient checkpointing for memory efficiency with large models.
    """
    def __init__(self, model_name='Qwen/Qwen3-Embedding-8B', 
                 freeze_encoder=False, use_gradient_checkpointing=True,
                 num_trainable_layers=4):
        super(FineTunableEncoder, self).__init__()
        
        self.model_name = model_name
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.num_trainable_layers = num_trainable_layers
        
        # Load pre-trained sentence transformer
        print(f"Loading encoder: {model_name}")
        self.encoder = SentenceTransformer(model_name, trust_remote_code=True)
        
        # Access the underlying transformer
        self.transformer = self.encoder[0].auto_model
        self.tokenizer = self.encoder[0].tokenizer
        
        # Freeze encoder or use partial freezing
        if freeze_encoder:
            for param in self.transformer.parameters():
                param.requires_grad = False
            print("  Encoder frozen (not trainable)")
        else:
            # Partial freezing: only train last few layers
            self._freeze_early_layers(num_trainable_layers)
            
            # Enable gradient checkpointing for memory efficiency
            if use_gradient_checkpointing and hasattr(self.transformer, 'gradient_checkpointing_enable'):
                self.transformer.gradient_checkpointing_enable()
                print(f"  Encoder partially trainable: last {num_trainable_layers} layers with gradient checkpointing")
            else:
                print(f"  Encoder partially trainable: last {num_trainable_layers} layers")
        
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.freeze_encoder = freeze_encoder
        print(f"  Embedding dimension: {self.embedding_dim}")
    
    def _freeze_early_layers(self, num_trainable_layers):
        """
        Freeze all layers except the last num_trainable_layers.
        This reduces the number of trainable parameters while still allowing
        task-specific adaptation.
        """
        # First, freeze everything
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        # Handle different transformer architectures
        layers_to_train = None
        total_layers = 0
        
        # Try BERT-style (encoder.layer)
        if hasattr(self.transformer, 'encoder') and hasattr(self.transformer.encoder, 'layer'):
            total_layers = len(self.transformer.encoder.layer)
            layers_to_train = self.transformer.encoder.layer[-num_trainable_layers:]
        
        # Try GPT-style or Qwen-style (model.layers or layers)
        elif hasattr(self.transformer, 'layers'):
            total_layers = len(self.transformer.layers)
            layers_to_train = self.transformer.layers[-num_trainable_layers:]
        
        elif hasattr(self.transformer, 'model') and hasattr(self.transformer.model, 'layers'):
            total_layers = len(self.transformer.model.layers)
            layers_to_train = self.transformer.model.layers[-num_trainable_layers:]
        
        # Try T5-style (encoder.block)
        elif hasattr(self.transformer, 'encoder') and hasattr(self.transformer.encoder, 'block'):
            total_layers = len(self.transformer.encoder.block)
            layers_to_train = self.transformer.encoder.block[-num_trainable_layers:]
        
        if layers_to_train is not None:
            # Unfreeze last N layers
            for layer in layers_to_train:
                for param in layer.parameters():
                    param.requires_grad = True
            
            # Also unfreeze final layer norm if it exists
            if hasattr(self.transformer, 'norm'):
                for param in self.transformer.norm.parameters():
                    param.requires_grad = True
            elif hasattr(self.transformer, 'ln_f'):
                for param in self.transformer.ln_f.parameters():
                    param.requires_grad = True
            elif hasattr(self.transformer, 'final_layer_norm'):
                for param in self.transformer.final_layer_norm.parameters():
                    param.requires_grad = True
            
            # Unfreeze pooler if it exists (for final representation)
            if hasattr(self.transformer, 'pooler') and self.transformer.pooler is not None:
                for param in self.transformer.pooler.parameters():
                    param.requires_grad = True
            
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.transformer.parameters())
            print(f"  Frozen {total_layers - num_trainable_layers}/{total_layers} layers")
            print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
        else:
            # Fallback: if structure is different, try to at least freeze embeddings
            print(f"  Warning: Could not detect layer structure")
            
            # Freeze embedding layers
            if hasattr(self.transformer, 'embeddings'):
                for param in self.transformer.embeddings.parameters():
                    param.requires_grad = False
            elif hasattr(self.transformer, 'embed_tokens'):
                for param in self.transformer.embed_tokens.parameters():
                    param.requires_grad = False
            elif hasattr(self.transformer, 'wte'):
                for param in self.transformer.wte.parameters():
                    param.requires_grad = False
            
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.transformer.parameters())
            print(f"  Froze embeddings, trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    def encode_batch(self, texts, batch_size=32):
        """
        Encode a batch of texts into embeddings.
        Maintains gradient flow for fine-tuning by using transformer directly.
        
        Args:
            texts: List of strings
            batch_size: Batch size for encoding
            
        Returns:
            embeddings: [num_texts, embedding_dim]
        """
        if not isinstance(texts, list):
            texts = [texts]
        
        device = next(self.parameters()).device
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            # Forward through transformer (maintains gradients)
            if self.freeze_encoder or not self.training:
                with torch.no_grad():
                    outputs = self.transformer(**encoded)
            else:
                outputs = self.transformer(**encoded)
            
            # Mean pooling
            attention_mask = encoded['attention_mask']
            token_embeddings = outputs[0]  # First element is token embeddings
            
            # Expand attention mask for broadcasting
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # Sum embeddings with mask, then divide by number of tokens
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            all_embeddings.append(embeddings)
        
        # Concatenate all batches
        return torch.cat(all_embeddings, dim=0)
        
        return embeddings
    
    def forward(self, text_indices, all_texts):
        """
        Forward pass - encode selected texts to embeddings.
        
        Args:
            text_indices: Tensor of indices into all_texts
            all_texts: List of all text strings
            
        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        # Get the texts for this batch
        if isinstance(text_indices, torch.Tensor):
            indices = text_indices.cpu().numpy()
        else:
            indices = text_indices
        
        batch_texts = [all_texts[i] for i in indices]
        
        # Encode with gradient flow
        embeddings = self.encode_batch(batch_texts, batch_size=len(batch_texts))
        
        return embeddings


class mult_vae_MDDM_FineTuned(BaseModel):
    def __init__(self, data_handler):
        super(mult_vae_MDDM_FineTuned, self).__init__(data_handler)

        self.beta = self.hyper_config['beta']
        self.freeze_encoder = self.hyper_config.get('freeze_encoder', False)
        self.encoder_name = self.hyper_config.get('encoder_name', 'Qwen/Qwen3-Embedding-8B')
        self.use_gradient_checkpointing = self.hyper_config.get('use_gradient_checkpointing', True)
        self.num_trainable_layers = self.hyper_config.get('num_trainable_layers', 4)
        self.compute_embeddings_every = self.hyper_config.get('compute_embeddings_every', 1)  # Every batch by default
        
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
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            num_trainable_layers=self.num_trainable_layers
        )
        
        # Store encoder dimension
        encoder_dim = self.text_encoder.embedding_dim
        print(f"  Encoder dimension: {encoder_dim}")

        # MLP for processing semantic information
        # Input: encoder_dim (4096 for Qwen), Output: 400 (200 for mu + 200 for logvar)
        # Larger MLP to handle high-dimensional Qwen embeddings
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 400)
        )

        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])]
        )

        self.drop = nn.Dropout(self.hyper_config['dropout'])

        self.final_embeds = None
        self.is_training = False
        self.batch_counter = 0
        
        # For frozen encoder, pre-compute embeddings once
        if self.freeze_encoder:
            print("Pre-computing embeddings (frozen encoder mode)...")
            with torch.no_grad():
                self.usrprf_embeds = self.text_encoder.encode_batch(self.user_texts, batch_size=128)
                self.itmprf_embeds = self.text_encoder.encode_batch(self.item_texts, batch_size=128)
            print(f"  User embeddings: {self.usrprf_embeds.shape}")
            print(f"  Item embeddings: {self.itmprf_embeds.shape}")
        else:
            # For trainable encoder, embeddings computed on-the-fly
            self.usrprf_embeds = None
            self.itmprf_embeds = None
            print("  Embeddings will be computed on-the-fly during training")

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
    
    def get_user_embeddings(self, user_indices):
        """
        Get user embeddings with gradient flow.
        Computes on-the-fly if encoder is trainable, uses cached if frozen.
        """
        if self.freeze_encoder:
            # Use pre-computed embeddings
            return self.usrprf_embeds[user_indices]
        else:
            # Compute on-the-fly with gradient flow
            return self.text_encoder(user_indices, self.user_texts)
    
    def get_item_embeddings(self):
        """
        Get all item embeddings.
        Computes on-the-fly if encoder is trainable, uses cached if frozen.
        """
        if self.freeze_encoder:
            # Use pre-computed embeddings
            return self.itmprf_embeds
        else:
            # Compute on-the-fly with gradient flow
            # For items, we need all embeddings for the full item-item interaction
            # Only recompute periodically for efficiency
            if self.itmprf_embeds is None or self.batch_counter % self.compute_embeddings_every == 0:
                self.itmprf_embeds = self.text_encoder.encode_batch(self.item_texts, batch_size=128)
            return self.itmprf_embeds

    def encode(self, x, user_emb, item_emb):
        h = self.drop(x)
        
        # Compute interaction between user behavior and item semantics
        # x: [batch, item_num] (user's interaction history)
        # item_emb: [item_num, emb_dim] (item semantic embeddings)
        # Result: [batch, emb_dim] (semantic representation of user's interactions)
        hidden = torch.matmul(h, item_emb) + user_emb
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
        self.batch_counter += 1
        
        # Get embeddings with gradient flow
        user_emb = self.get_user_embeddings(user)
        item_emb = self.get_item_embeddings()

        mu_src, mu_llm, logvar_src, logvar_llm = self.encode(batch_data, user_emb, item_emb)

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
        
        # Get embeddings (frozen: cached, trainable: compute)
        user_emb = self.get_user_embeddings(pck_users)
        item_emb = self.get_item_embeddings()

        mu, mu_llm, logvar, logvar_llm = self.encode(data, user_emb, item_emb)

        mu = mu + mu_llm
        logvar = logvar + logvar_llm

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        full_preds = self._mask_predict(recon_x, train_mask)
        return full_preds
