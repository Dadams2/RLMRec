"""
Visualization System for LLM-VAE Embedding Interactions with Aligned PCA
This version uses consistent PCA axes across all embedding stages for better comparability

This script captures and visualizes how LLM embeddings interact with VAE latent spaces
during training, showing the evolution of user/item representations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import procrustes
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# Import project modules
import sys
sys.path.append('..')
from config.configurator import configs
from data_utils.build_data_handler import build_data_handler


class EmbeddingVisualizer:
    """
    Captures and visualizes embedding dynamics during VAE training with aligned PCA
    """
    
    def __init__(self, model_name='mult_vae_godm', dataset='yelp', output_dir='./visualization_outputs'):
        self.model_name = model_name
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for embeddings at different stages
        self.embeddings_history = {
            'llm_raw': [],           # Raw LLM embeddings
            'llm_processed': [],     # After MLP transformation
            'vae_latent': [],        # VAE encoder output (mu)
            'combined': [],          # Combined mu_src + mu_llm
            'reconstructed': [],     # Decoder output
        }
        
        self.metadata_history = []
        self.loss_history = []
        
    def load_data(self):
        """Load the dataset and LLM embeddings"""
        print(f"Loading {self.dataset} dataset...")
        
        # Load user/item profiles and embeddings
        data_path = Path(f'../data/{self.dataset}')
        
        with open(data_path / 'usr_prf.pkl', 'rb') as f:
            self.usr_profiles = pickle.load(f)
        with open(data_path / 'itm_prf.pkl', 'rb') as f:
            self.itm_profiles = pickle.load(f)
        with open(data_path / 'usr_emb_np.pkl', 'rb') as f:
            self.usr_embeddings = pickle.load(f)
        with open(data_path / 'itm_emb_np.pkl', 'rb') as f:
            self.itm_embeddings = pickle.load(f)
            
        print(f"User embeddings shape: {self.usr_embeddings.shape}")
        print(f"Item embeddings shape: {self.itm_embeddings.shape}")
        
        # Extract item categories for clustering (from profiles)
        self.item_categories = self._extract_item_categories()
        self.user_categories = self._extract_user_categories()
        self.user_cluster_descriptions = self.generate_cluster_descriptions(self.user_categories)
        
    def _extract_item_categories(self):
        """
        Extract or infer categories from item profiles for clustering
        Uses simple keyword matching on profile text
        """
        categories = []
        # Common Yelp categories
        keywords = {
            'restaurant': ['restaurant', 'food', 'dining', 'cuisine', 'meal'],
            'bar': ['bar', 'drinks', 'cocktail', 'beer', 'wine'],
            'cafe': ['cafe', 'coffee', 'tea', 'bakery'],
            'shopping': ['shop', 'store', 'retail', 'boutique'],
            'service': ['service', 'salon', 'spa', 'repair'],
            'entertainment': ['entertainment', 'theater', 'cinema', 'show'],
            'hotel': ['hotel', 'accommodation', 'lodging'],
            'other': []
        }
        
        for profile in self.itm_profiles:
            text = self.itm_profiles[profile].get('profile', '').lower()
            category = 'other'
            for cat, words in keywords.items():
                if any(word in text for word in words):
                    category = cat
                    break
            categories.append(category)
            
        return np.array(categories)
    
    def _extract_user_categories(self):
        """
        Cluster users based on their preference patterns
        """
        # Use KMeans clustering on user embeddings
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        user_clusters = kmeans.fit_predict(self.usr_embeddings)
        return user_clusters
    
    def generate_cluster_descriptions(self, user_clusters, top_n_keywords=3):
        """
        Generate human-readable descriptions for user clusters based on their profiles
        
        Args:
            user_clusters: Array of cluster assignments for each user
            top_n_keywords: Number of top keywords to extract per cluster
            
        Returns:
            Dictionary mapping cluster_id to description string
        """
        cluster_descriptions = {}
        n_clusters = len(np.unique(user_clusters))
        
        # Collect profiles for each cluster
        cluster_profiles = {i: [] for i in range(n_clusters)}
        
        for idx, (user_id, profile_data) in enumerate(self.usr_profiles.items()):
            if idx < len(user_clusters):
                cluster_id = user_clusters[idx]
                profile_text = profile_data.get('profile', '')
                if profile_text:
                    cluster_profiles[cluster_id].append(profile_text)
        
        # Build complete corpus for TF-IDF (all clusters treated separately)
        corpus = []
        corpus_cluster_ids = []
        
        for cluster_id in range(n_clusters):
            # Each document is one cluster's combined profiles
            cluster_text = ' '.join(cluster_profiles[cluster_id])
            corpus.append(cluster_text)
            corpus_cluster_ids.append(cluster_id)
        
        if not corpus or all(len(text) == 0 for text in corpus):
            print("Warning: No profile text found")
            return {i: f"Cluster {i}" for i in range(n_clusters)}
        
        # Custom stop words for restaurant reviews
        custom_stop_words = list(set([
            'user', 'likely', 'enjoy', 'based', 'reviews', 'review', 'appreciate', 
            'good', 'great', 'nice', 'place', 'places', 'business', 'businesses',
            'restaurant', 'restaurants', 'offer', 'offers', 'offering', 'food',
            'enjoy', 'enjoyed', 'enjoys', 'like', 'likes', 'prefer', 'prefers'
        ] + list(__import__('sklearn.feature_extraction.text', fromlist=['ENGLISH_STOP_WORDS']).ENGLISH_STOP_WORDS)))
        
        try:
            # TF-IDF with custom parameters to find distinctive terms
            vectorizer = TfidfVectorizer(
                max_features=200,
                stop_words=custom_stop_words,
                ngram_range=(1, 2),
                min_df=1,  # Must appear in at least 1 document
                max_df=0.8,  # Must not appear in more than 80% of documents
                token_pattern=r'\b[a-zA-Z]{3,}\b'  # At least 3 characters
            )
            
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = np.array(vectorizer.get_feature_names_out())
            
            # For each cluster, find most distinctive terms
            for cluster_id in range(n_clusters):
                cluster_vec = tfidf_matrix[cluster_id].toarray()[0]
                
                # Get top scoring terms for this cluster
                top_indices = cluster_vec.argsort()[-20:][::-1]
                top_terms = []
                
                for idx in top_indices:
                    if cluster_vec[idx] > 0:
                        term = feature_names[idx]
                        # Additional filtering for more meaningful terms
                        if len(term) > 2 and term not in ['the', 'and', 'for', 'with']:
                            top_terms.append(term)
                        if len(top_terms) >= top_n_keywords:
                            break
                
                if top_terms:
                    # Capitalize nicely
                    formatted_terms = []
                    for term in top_terms[:3]:
                        if ' ' in term:  # bigram
                            formatted_terms.append(term.title())
                        else:
                            formatted_terms.append(term.capitalize())
                    
                    description = ' / '.join(formatted_terms)
                    cluster_descriptions[cluster_id] = description
                    print(f"Cluster {cluster_id}: {description} (size: {len(cluster_profiles[cluster_id])} users)")
                else:
                    cluster_descriptions[cluster_id] = f"Group {cluster_id + 1}"
                    
        except Exception as e:
            print(f"Warning: Could not generate descriptions: {e}")
            import traceback
            traceback.print_exc()
            return {i: f"Group {i + 1}" for i in range(n_clusters)}
        
        return cluster_descriptions
    
    def capture_embeddings_from_model(self, model, data_handler, n_samples=1000, batch_size=256, fixed_users=None):
        """
        Capture embeddings at different stages of the VAE forward pass
        
        Args:
            model: Trained VAE model
            data_handler: Data handler with train data
            n_samples: Number of samples to capture
            batch_size: Batch size for processing
            fixed_users: Optional array of specific user IDs to use (for consistent sampling across checkpoints)
        """
        model.eval()
        model.is_training = False
        
        embeddings_stage = {
            'llm_raw': [],
            'llm_processed': [],
            'vae_latent': [],
            'combined': [],
            'reconstructed': [],
            'target': [],  # Ground truth interaction data
            'users': [],
            'items_interacted': []
        }
        
        # Sample users (or use provided fixed users)
        if fixed_users is not None:
            sampled_users = fixed_users
        else:
            n_users = data_handler.train_data.shape[0]
            sampled_users = np.random.choice(n_users, size=min(n_samples, n_users), replace=False)
        
        with torch.no_grad():
            for i in range(0, len(sampled_users), batch_size):
                batch_users = sampled_users[i:i+batch_size]
                batch_data = data_handler.train_data[batch_users]
                data = torch.FloatTensor(batch_data.toarray()).cuda()
                
                user_ids = torch.LongTensor(batch_users).cuda()
                
                # Stage 1: Raw LLM embeddings
                user_emb = model.usrprf_embeds[user_ids]  # [batch, 1536]
                embeddings_stage['llm_raw'].append(user_emb.cpu().numpy())
                
                # Stage 2: Item-weighted LLM embeddings
                h = model.drop(data)
                hidden = torch.matmul(h, model.itmprf_embeds) + user_emb  # [batch, 1536]
                
                # Stage 3: After MLP transformation
                hidden_mlp = model.mlp(hidden)  # [batch, 400]
                mu_llm = hidden_mlp[:, :200]
                logvar_llm = hidden_mlp[:, 200:]
                embeddings_stage['llm_processed'].append(mu_llm.cpu().numpy())
                
                # Stage 4: VAE encoder output
                h = model.drop(data)
                for j, layer in enumerate(model.q_layers):
                    h = layer(h)
                    if j != len(model.q_layers) - 1:
                        h = torch.tanh(h)
                    else:
                        mu_src = h[:, :model.q_dims[-1]]
                        logvar_src = h[:, model.q_dims[-1]:]
                
                embeddings_stage['vae_latent'].append(mu_src.cpu().numpy())
                
                # Stage 5: Combined representation
                mu_combined = mu_src + mu_llm
                embeddings_stage['combined'].append(mu_combined.cpu().numpy())
                
                # Stage 6: Decoder output
                z = model.reparameterize(mu_combined, logvar_src + logvar_llm)
                recon_x = model.decode(z)
                embeddings_stage['reconstructed'].append(recon_x.cpu().numpy())
                
                # Stage 7: Target (ground truth)
                embeddings_stage['target'].append(data.cpu().numpy())
                
                embeddings_stage['users'].extend(batch_users.tolist())
                
        # Concatenate all batches
        for key in ['llm_raw', 'llm_processed', 'vae_latent', 'combined', 'reconstructed', 'target']:
            if len(embeddings_stage[key]) > 0:
                embeddings_stage[key] = np.vstack(embeddings_stage[key])
            else:
                print(f"WARNING: No embeddings captured for {key}")
                # Fallback - use appropriate dimension
                if key == 'target':
                    # Target has item dimensionality
                    n_items = data_handler.train_data.shape[1] if hasattr(data_handler, 'train_data') else 1000
                    embeddings_stage[key] = np.zeros((len(sampled_users), n_items))
                else:
                    embeddings_stage[key] = np.zeros((len(sampled_users), 200))
            
        return embeddings_stage
    
    def reduce_dimensions_aligned(self, embeddings_dict, method='pca', n_components=2, 
                                  reference_stage='combined'):
        """
        Reduce dimensionality with aligned axes across all stages
        
        This fits PCA on a reference embedding stage and applies the same transformation
        to all other stages, ensuring consistent interpretation of axes.
        
        Args:
            embeddings_dict: Dictionary of embeddings at different stages
            method: 'pca' or 'tsne' (tsne doesn't support alignment well)
            n_components: Number of dimensions to reduce to
            reference_stage: Which embedding stage to use as reference for PCA fitting
        """
        reduced = {}
        
        if method == 'pca':
            # Fit PCA on the reference stage (combined embeddings)
            reference_emb = embeddings_dict[reference_stage]
            print(f"\nFitting PCA on {reference_stage} embeddings (shape: {reference_emb.shape})")
            
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(reference_emb)
            
            print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
            print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")
            
            # Apply the same transformation to all stages
            for stage_name, emb in embeddings_dict.items():
                if stage_name in ['users', 'items_interacted']:
                    continue
                
                # Special handling for target (interaction vectors)
                if stage_name == 'target' and emb.shape[1] > 1000:
                    # Target has full item space dimensionality - reduce more aggressively
                    print(f"  {stage_name}: High-dimensional interaction vector ({emb.shape[1]} items), reducing to PCA space")
                    # First reduce to manageable size
                    pca_pre = PCA(n_components=min(200, emb.shape[1]), random_state=42)
                    emb_reduced = pca_pre.fit_transform(emb)
                    # Then project to reference space
                    if emb_reduced.shape[1] != reference_emb.shape[1]:
                        emb_padded = np.zeros((emb_reduced.shape[0], reference_emb.shape[1]))
                        emb_padded[:, :emb_reduced.shape[1]] = emb_reduced
                        reduced_emb = pca.transform(emb_padded)
                    else:
                        reduced_emb = pca.transform(emb_reduced)
                    reduced[stage_name] = reduced_emb
                    print(f"  {stage_name}: Transformed to shape {reduced_emb.shape}")
                    continue
                
                # Handle different dimensionalities
                if emb.shape[1] != reference_emb.shape[1]:
                    # For high-dimensional embeddings (like llm_raw with 1536 dims),
                    # first reduce to the same dimensionality as reference
                    if emb.shape[1] > reference_emb.shape[1]:
                        print(f"  {stage_name}: Reducing from {emb.shape[1]} to {reference_emb.shape[1]} dims first")
                        pca_pre = PCA(n_components=reference_emb.shape[1], random_state=42)
                        emb_reduced = pca_pre.fit_transform(emb)
                        reduced_emb = pca.transform(emb_reduced)
                    else:
                        # For lower-dimensional embeddings, pad with zeros
                        print(f"  {stage_name}: Padding from {emb.shape[1]} to {reference_emb.shape[1]} dims")
                        emb_padded = np.zeros((emb.shape[0], reference_emb.shape[1]))
                        emb_padded[:, :emb.shape[1]] = emb
                        reduced_emb = pca.transform(emb_padded)
                else:
                    # Same dimensionality - directly transform
                    reduced_emb = pca.transform(emb)
                
                reduced[stage_name] = reduced_emb
                print(f"  {stage_name}: Transformed to shape {reduced_emb.shape}")
                
        elif method == 'tsne':
            print("Warning: t-SNE doesn't support aligned transformations well.")
            print("Each stage will have its own t-SNE embedding - use PCA for alignment.")
            
            # Fall back to independent t-SNE for each stage
            for stage_name, emb in embeddings_dict.items():
                if stage_name in ['users', 'items_interacted']:
                    continue
                    
                reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
                reduced_emb = reducer.fit_transform(emb)
                reduced[stage_name] = reduced_emb
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return reduced, pca if method == 'pca' else None
    
    def create_static_visualization(self, reduced_embeddings, user_ids, pca=None, 
                                   save_path='embedding_vis_aligned.png'):
        """
        Create a static visualization of embeddings at different stages with aligned axes
        """
        stages = ['llm_raw', 'llm_processed', 'vae_latent', 'combined', 'reconstructed', 'target']
        stage_titles = [
            'Raw LLM Embeddings\n(1536D → 200D → PC1-2)',
            'Processed LLM\n(after MLP, 200D → PC1-2)',
            'VAE Latent Space\n(encoder μ, 200D → PC1-2)',
            'Combined Space\n(VAE + LLM, 200D PCA reference)',
            'Reconstructed\n(decoder output → PC1-2)',
            'Target (Ground Truth)\n(user-item interactions → PC1-2)'
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        # Get user categories
        user_categories = self.user_categories[user_ids]
        
        # Create color palette
        n_categories = len(np.unique(user_categories))
        colors = plt.cm.tab10(np.linspace(0, 1, n_categories))
        
        # Calculate global axis limits for consistency
        all_x = []
        all_y = []
        for stage in stages:
            if stage in reduced_embeddings:
                all_x.extend(reduced_embeddings[stage][:, 0])
                all_y.extend(reduced_embeddings[stage][:, 1])
        
        x_min, x_max = np.percentile(all_x, [1, 99])
        y_min, y_max = np.percentile(all_y, [1, 99])
        
        # Add some padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        
        for idx, (stage, title) in enumerate(zip(stages, stage_titles)):
            ax = axes[idx]
            
            if stage in reduced_embeddings:
                emb = reduced_embeddings[stage]
                
                # Scatter plot with categories
                for cat_id in np.unique(user_categories):
                    mask = user_categories == cat_id
                    label = self.user_cluster_descriptions.get(cat_id, f'Cluster {cat_id}')
                    ax.scatter(emb[mask, 0], emb[mask, 1], 
                             alpha=0.6, s=30, 
                             color=colors[cat_id],
                             label=label,
                             edgecolors='white', linewidth=0.5)
                
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xlabel('PC1 (aligned)', fontsize=10)
                ax.set_ylabel('PC2 (aligned)', fontsize=10)
                
                # Use consistent axis limits
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                
                # Only show legend on first plot
                if idx == 0:
                    ax.legend(markerscale=0.8, fontsize=8, loc='upper right')
                ax.grid(True, alpha=0.3)
        
        # All 6 subplots are now used - no need to remove any
        
        # Add overall title with PCA info
        title_text = f'Embedding Evolution with Aligned PCA: {self.model_name.upper()} on {self.dataset.upper()}'
        if pca is not None:
            var_explained = pca.explained_variance_ratio_
            title_text += f'\nPC1: {var_explained[0]:.1%}, PC2: {var_explained[1]:.1%} of variance'
        
        fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved static visualization to {self.output_dir / save_path}")
        plt.close()
    
    def create_animation_multi_stage(self, checkpoint_dirs, data_handler, n_samples=500, fps=2):
        """
        Create an animation showing embedding evolution across multiple checkpoints
        with aligned PCA axes
        
        Args:
            checkpoint_dirs: List of checkpoint directories (in order)
            data_handler: Data handler
            n_samples: Number of samples to visualize
            fps: Frames per second for animation
        """
        from models.bulid_model import build_model
        
        print("Creating multi-stage animation with aligned PCA...")
        all_reduced_embeddings = []
        
        # Sample users once to keep consistent across checkpoints
        n_users = data_handler.train_data.shape[0]
        sampled_users = np.random.choice(n_users, size=min(n_samples, n_users), replace=False)
        
        # First pass: collect all embeddings to fit global PCA
        print("\nFirst pass: Collecting all embeddings for global PCA fitting...")
        all_combined_embeddings = []
        
        for ckpt_idx, ckpt_dir in enumerate(tqdm(checkpoint_dirs, desc="Collecting embeddings")):
            model = build_model(data_handler).cuda()
            
            checkpoint = torch.load(ckpt_dir, map_location='cuda')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            model.is_training = False
            
            embeddings_stage = self.capture_embeddings_from_model(
                model, data_handler, n_samples=n_samples, 
                fixed_users=sampled_users
            )
            
            all_combined_embeddings.append(embeddings_stage['combined'])
        
        # Fit global PCA on all combined embeddings
        all_combined = np.vstack(all_combined_embeddings)
        print(f"\nFitting global PCA on combined embeddings: {all_combined.shape}")
        global_pca = PCA(n_components=2, random_state=42)
        global_pca.fit(all_combined)
        print(f"Global PCA variance explained: {global_pca.explained_variance_ratio_}")
        
        # Second pass: transform all embeddings with global PCA
        print("\nSecond pass: Transforming embeddings with global PCA...")
        
        for ckpt_idx, ckpt_dir in enumerate(tqdm(checkpoint_dirs, desc="Processing checkpoints")):
            model = build_model(data_handler).cuda()
            
            checkpoint = torch.load(ckpt_dir, map_location='cuda')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            model.is_training = False
            
            embeddings_stage = self.capture_embeddings_from_model(
                model, data_handler, n_samples=n_samples, 
                fixed_users=sampled_users
            )
            
            # Transform with global PCA
            reduced = {}
            for stage_name, emb in embeddings_stage.items():
                if stage_name in ['users', 'items_interacted']:
                    continue
                
                # Handle different dimensionalities
                if emb.shape[1] != all_combined.shape[1]:
                    if emb.shape[1] > all_combined.shape[1]:
                        pca_pre = PCA(n_components=all_combined.shape[1], random_state=42)
                        emb_reduced = pca_pre.fit_transform(emb)
                        reduced_emb = global_pca.transform(emb_reduced)
                    else:
                        emb_padded = np.zeros((emb.shape[0], all_combined.shape[1]))
                        emb_padded[:, :emb.shape[1]] = emb
                        reduced_emb = global_pca.transform(emb_padded)
                else:
                    reduced_emb = global_pca.transform(emb)
                
                reduced[stage_name] = reduced_emb
            
            all_reduced_embeddings.append({
                'reduced': reduced,
                'users': embeddings_stage['users'],
                'checkpoint': ckpt_idx
            })
        
        # Create animation
        self._animate_embeddings(all_reduced_embeddings, fps=fps, global_pca=global_pca)
    
    def _animate_embeddings(self, all_reduced_embeddings, fps=2, global_pca=None):
        """
        Create the actual animation from reduced embeddings with consistent axes
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        stages = ['llm_raw', 'llm_processed', 'vae_latent', 'combined', 'reconstructed', 'target']
        stage_titles = [
            'Raw LLM Embeddings',
            'Processed LLM',
            'VAE Latent Space',
            'Combined Space',
            'Reconstructed',
            'Target (Ground Truth)'
        ]
        
        # Calculate global axis limits
        all_x, all_y = [], []
        for data in all_reduced_embeddings:
            for stage in stages:
                if stage in data['reduced']:
                    emb = data['reduced'][stage]
                    all_x.extend(emb[:, 0])
                    all_y.extend(emb[:, 1])
        
        x_min, x_max = np.percentile(all_x, [1, 99])
        y_min, y_max = np.percentile(all_y, [1, 99])
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        
        # Initialize scatter plots
        scatters = []
        for idx, title in enumerate(stage_titles):
            ax = axes[idx]
            scatter = ax.scatter([], [], alpha=0.6, s=30, c=[], cmap='tab10', 
                               edgecolors='white', linewidth=0.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('PC1 (aligned)')
            ax.set_ylabel('PC2 (aligned)')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.3)
            scatters.append(scatter)
        
        # All 6 subplots are now used
        
        # Checkpoint counter
        checkpoint_text = fig.text(0.72, 0.35, '', fontsize=14, fontweight='bold')
        
        # Add PCA info
        if global_pca is not None:
            var_text = f"PC1: {global_pca.explained_variance_ratio_[0]:.1%}, PC2: {global_pca.explained_variance_ratio_[1]:.1%}"
            fig.text(0.72, 0.32, var_text, fontsize=10)
        
        def update(frame):
            """Update function for animation"""
            data = all_reduced_embeddings[frame]
            reduced = data['reduced']
            users = np.array(data['users'])
            checkpoint = data['checkpoint']
            
            # Get user categories for coloring
            user_categories = self.user_categories[users]
            
            for idx, (stage, scatter) in enumerate(zip(stages, scatters)):
                if stage in reduced:
                    emb = reduced[stage]
                    scatter.set_offsets(emb)
                    scatter.set_array(user_categories)
            
            checkpoint_text.set_text(f'Checkpoint: {checkpoint + 1}/{len(all_reduced_embeddings)}')
            
            return scatters + [checkpoint_text]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(all_reduced_embeddings), 
                           interval=1000//fps, blit=True, repeat=True)
        
        # Save as GIF
        output_path = self.output_dir / f'{self.model_name}_{self.dataset}_animation_aligned.gif'
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=100)
        print(f"Saved animation to {output_path}")
        plt.close()


def main():
    """
    Main execution function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize LLM-VAE embedding dynamics with aligned PCA')
    parser.add_argument('--model', type=str, default='mult_vae_godm', help='Model name')
    parser.add_argument('--dataset', type=str, default='yelp', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory containing multiple checkpoints for animation')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./visualization_outputs', help='Output directory')
    parser.add_argument('--type', type=str, default='static', 
                       choices=['static', 'animation'], 
                       help='Visualization mode')
    
    args = parser.parse_args()
    
    from trainer.logger import Logger
    logger = Logger(configs)
    
    # Initialize visualizer
    visualizer = EmbeddingVisualizer(
        model_name=args.model,
        dataset=args.dataset,
        output_dir=args.output_dir
    )
    
    # Load data
    visualizer.load_data()
    
    # Build data handler
    data_handler = build_data_handler()
    data_handler.load_data()
    
    if args.type == 'static':
        # Load single checkpoint and create static visualization
        from models.bulid_model import build_model
        model = build_model(data_handler).cuda()
        
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {args.checkpoint}")
        
        model.eval()
        
        # Capture embeddings
        print("Capturing embeddings from model...")
        embeddings_stage = visualizer.capture_embeddings_from_model(
            model, data_handler, n_samples=args.n_samples
        )
        
        # Reduce dimensions with aligned PCA
        print("Reducing dimensions with aligned PCA...")
        reduced, pca = visualizer.reduce_dimensions_aligned(
            embeddings_stage, method='pca', reference_stage='combined'
        )
        
        # Create visualization
        print("Creating static visualization...")
        visualizer.create_static_visualization(
            reduced, 
            embeddings_stage['users'],
            pca=pca,
            save_path=f'{args.model}_{args.dataset}_static_aligned.png'
        )
        
    elif args.type == 'animation':
        # Create multi-checkpoint animation with aligned axes
        if not args.checkpoint_dir:
            print("Error: --checkpoint_dir required for animation mode")
            return
        
        # Find all checkpoints in directory
        checkpoint_paths = sorted(Path(args.checkpoint_dir).glob('*.pth'))
        print(f"Found {len(checkpoint_paths)} checkpoints")
        
        visualizer.create_animation_multi_stage(
            checkpoint_paths, 
            data_handler, 
            n_samples=args.n_samples,
            fps=2
        )
    
    print("Visualization complete!")


if __name__ == '__main__':
    main()
