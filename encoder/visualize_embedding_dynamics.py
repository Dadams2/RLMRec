"""
Visualization System for LLM-VAE Embedding Interactions
Inspired by Vision Transformer embedding visualization techniques

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
    Captures and visualizes embedding dynamics during VAE training
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
                
                embeddings_stage['users'].extend(batch_users.tolist())
                
        # Concatenate all batches
        for key in ['llm_raw', 'llm_processed', 'vae_latent', 'combined', 'reconstructed']:
            if len(embeddings_stage[key]) > 0:
                embeddings_stage[key] = np.vstack(embeddings_stage[key])
            else:
                print(f"WARNING: No embeddings captured for {key}")
                embeddings_stage[key] = np.zeros((len(sampled_users), 200))  # Fallback
            
        return embeddings_stage
    
    def reduce_dimensions(self, embeddings_dict, method='pca', n_components=2, reference_embedding=None):
        """
        Reduce dimensionality of embeddings using PCA or t-SNE
        Apply Procrustes alignment if reference is provided
        
        Args:
            embeddings_dict: Dictionary of embeddings at different stages
            method: 'pca' or 'tsne'
            n_components: Number of dimensions to reduce to
            reference_embedding: Reference embedding for Procrustes alignment
        """
        reduced = {}
        
        for stage_name, emb in embeddings_dict.items():
            if stage_name in ['users', 'items_interacted']:
                continue
                
            # Apply dimensionality reduction
            if method == 'pca':
                reducer = PCA(n_components=n_components, random_state=42)
                reduced_emb = reducer.fit_transform(emb)
            elif method == 'tsne':
                reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
                reduced_emb = reducer.fit_transform(emb)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Apply Procrustes alignment for smooth transitions
            if reference_embedding is not None and stage_name in reference_embedding:
                _, reduced_emb, _ = procrustes(reference_embedding[stage_name], reduced_emb)
            
            reduced[stage_name] = reduced_emb
            
        return reduced
    
    def create_static_visualization(self, reduced_embeddings, user_ids, save_path='embedding_vis.png'):
        """
        Create a static visualization of embeddings at different stages
        """
        stages = ['llm_raw', 'llm_processed', 'vae_latent', 'combined', 'reconstructed']
        stage_titles = [
            'Raw LLM Embeddings\n(text-embedding-ada-002)',
            'Processed LLM\n(after MLP)',
            'VAE Latent Space\n(encoder μ)',
            'Combined Space\n(VAE + LLM)',
            'Reconstructed\n(decoder output)'
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Get user categories
        user_categories = self.user_categories[user_ids]
        
        # Create color palette
        n_categories = len(np.unique(user_categories))
        colors = plt.cm.tab10(np.linspace(0, 1, n_categories))
        
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
                
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.legend(markerscale=0.8, fontsize=8, loc='upper right')
                ax.grid(True, alpha=0.3)
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        # Add overall title
        fig.suptitle(f'Embedding Evolution: {self.model_name.upper()} on {self.dataset.upper()}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        print(f"Saved static visualization to {self.output_dir / save_path}")
        plt.close()
    
    def create_animation_multi_stage(self, checkpoint_dirs, data_handler, n_samples=500, fps=2):
        """
        Create an animation showing embedding evolution across multiple checkpoints
        Similar to ViT fine-tuning visualization
        
        Args:
            checkpoint_dirs: List of checkpoint directories (in order)
            data_handler: Data handler
            n_samples: Number of samples to visualize
            fps: Frames per second for animation
        """
        from models.bulid_model import build_model
        
        print("Creating multi-stage animation...")
        all_reduced_embeddings = []
        
        # Sample users once to keep consistent across checkpoints
        n_users = data_handler.train_data.shape[0]
        sampled_users = np.random.choice(n_users, size=min(n_samples, n_users), replace=False)
        
        # Load embeddings from each checkpoint
        for ckpt_idx, ckpt_dir in enumerate(tqdm(checkpoint_dirs, desc="Processing checkpoints")):
            # Load model from checkpoint
            model = build_model(data_handler).cuda()
            
            # Try different checkpoint loading strategies
            checkpoint = torch.load(ckpt_dir, map_location='cuda')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            model.is_training = False
            
            print(f"\nCheckpoint {ckpt_idx + 1}/{len(checkpoint_dirs)}: {ckpt_dir.name}")
            
            # Capture embeddings with fixed user sampling
            embeddings_stage = self.capture_embeddings_from_model(
                model, data_handler, n_samples=n_samples, 
                fixed_users=sampled_users
            )
            
            # Validate embeddings aren't all zeros
            for stage_name, emb in embeddings_stage.items():
                if stage_name not in ['users', 'items_interacted']:
                    emb_mean = np.abs(emb).mean()
                    print(f"  {stage_name}: mean absolute value = {emb_mean:.6f}")
                    if emb_mean < 1e-6:
                        print(f"  WARNING: {stage_name} embeddings are near zero!")
            
            # Reduce dimensions WITHOUT Procrustes (that was causing the collapse)
            reduced = self.reduce_dimensions(embeddings_stage, method='pca', reference_embedding=None)
            
            all_reduced_embeddings.append({
                'reduced': reduced,
                'users': embeddings_stage['users'],
                'checkpoint': ckpt_idx
            })
        
        # Create animation
        self._animate_embeddings(all_reduced_embeddings, fps=fps)
    
    def _animate_embeddings(self, all_reduced_embeddings, fps=2):
        """
        Create the actual animation from reduced embeddings
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        stages = ['llm_raw', 'llm_processed', 'vae_latent', 'combined', 'reconstructed']
        stage_titles = [
            'Raw LLM Embeddings',
            'Processed LLM',
            'VAE Latent Space',
            'Combined Space',
            'Reconstructed'
        ]
        
        # Initialize scatter plots
        scatters = []
        for idx, title in enumerate(stage_titles):
            ax = axes[idx]
            scatter = ax.scatter([], [], alpha=0.6, s=30, c=[], cmap='tab10', edgecolors='white', linewidth=0.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.grid(True, alpha=0.3)
            scatters.append(scatter)
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        # Checkpoint counter
        checkpoint_text = fig.text(0.72, 0.35, '', fontsize=14, fontweight='bold')
        
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
                    
                    # Update axis limits
                    axes[idx].set_xlim(emb[:, 0].min() - 1, emb[:, 0].max() + 1)
                    axes[idx].set_ylim(emb[:, 1].min() - 1, emb[:, 1].max() + 1)
            
            checkpoint_text.set_text(f'Checkpoint: {checkpoint + 1}/{len(all_reduced_embeddings)}')
            
            return scatters + [checkpoint_text]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(all_reduced_embeddings), 
                           interval=1000//fps, blit=True, repeat=True)
        
        # Save as GIF
        output_path = self.output_dir / f'{self.model_name}_{self.dataset}_animation.gif'
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=100)
        print(f"Saved animation to {output_path}")
        plt.close()
    
    def create_distribution_animation(self, model, data_handler, n_frames=50, n_samples=1000):
        """
        Create an animation showing the evolution of distribution parameters (mu, sigma)
        during the encoding process
        """
        print("Creating distribution animation...")
        
        model.eval()
        model.is_training = False
        
        # Sample users
        n_users = data_handler.train_data.shape[0]
        sampled_users = np.random.choice(n_users, size=min(n_samples, n_users), replace=False)
        
        frames_data = []
        
        with torch.no_grad():
            batch_data = data_handler.train_data[sampled_users]
            data = torch.FloatTensor(batch_data.toarray()).cuda()
            user_ids = torch.LongTensor(sampled_users).cuda()
            
            # Get all distributions
            user_emb = model.usrprf_embeds[user_ids]
            h = model.drop(data)
            hidden = torch.matmul(h, model.itmprf_embeds) + user_emb
            hidden_mlp = model.mlp(hidden)
            
            mu_llm = hidden_mlp[:, :200].cpu().numpy()
            logvar_llm = hidden_mlp[:, 200:].cpu().numpy()
            sigma_llm = np.exp(0.5 * logvar_llm)
            
            # VAE encoder
            h = model.drop(data)
            for i, layer in enumerate(model.q_layers):
                h = layer(h)
                if i != len(model.q_layers) - 1:
                    h = torch.tanh(h)
                else:
                    mu_src = h[:, :model.q_dims[-1]].cpu().numpy()
                    logvar_src = h[:, model.q_dims[-1]:].cpu().numpy()
                    sigma_src = np.exp(0.5 * logvar_src)
            
            # Create frames showing interpolation between distributions
            for alpha in np.linspace(0, 1, n_frames):
                mu_interp = (1 - alpha) * mu_src + alpha * mu_llm
                sigma_interp = (1 - alpha) * sigma_src + alpha * sigma_llm
                
                # Reduce to 2D for visualization
                pca = PCA(n_components=2, random_state=42)
                mu_2d = pca.fit_transform(mu_interp)
                
                frames_data.append({
                    'mu': mu_2d,
                    'sigma': sigma_interp[:, :2],  # Just first 2 dimensions for visualization
                    'alpha': alpha,
                    'users': sampled_users
                })
        
        # Create animation
        self._animate_distributions(frames_data)
    
    def _animate_distributions(self, frames_data):
        """
        Animate the distribution parameters
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Get user categories
        users = frames_data[0]['users']
        user_categories = self.user_categories[users]
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            data = frames_data[frame]
            mu = data['mu']
            sigma = data['sigma']
            alpha = data['alpha']
            
            # Plot means
            scatter1 = ax1.scatter(mu[:, 0], mu[:, 1], c=user_categories, 
                                  cmap='tab10', alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
            ax1.set_title(f'Mean (μ) - Interpolation α={alpha:.2f}', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Component 1')
            ax1.set_ylabel('Component 2')
            ax1.grid(True, alpha=0.3)
            
            # Plot ellipses for variance
            for i in range(0, len(mu), 10):  # Plot every 10th for clarity
                from matplotlib.patches import Ellipse
                ellipse = Ellipse(xy=mu[i], width=sigma[i, 0]*2, height=sigma[i, 1]*2,
                                alpha=0.3, facecolor=plt.cm.tab10(user_categories[i] / 10))
                ax2.add_patch(ellipse)
            
            ax2.scatter(mu[:, 0], mu[:, 1], c=user_categories, 
                       cmap='tab10', alpha=0.6, s=10, edgecolors='white', linewidth=0.5)
            ax2.set_title(f'Distributions (μ ± σ) - α={alpha:.2f}', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Component 1')
            ax2.set_ylabel('Component 2')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(mu[:, 0].min() - 5, mu[:, 0].max() + 5)
            ax2.set_ylim(mu[:, 1].min() - 5, mu[:, 1].max() + 5)
            
            return scatter1,
        
        anim = FuncAnimation(fig, update, frames=len(frames_data), 
                           interval=100, blit=False, repeat=True)
        
        output_path = self.output_dir / f'{self.model_name}_{self.dataset}_distribution_anim.gif'
        writer = PillowWriter(fps=10)
        anim.save(output_path, writer=writer, dpi=100)
        print(f"Saved distribution animation to {output_path}")
        plt.close()


def main():
    """
    Main execution function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize LLM-VAE embedding dynamics')
    parser.add_argument('--model', type=str, default='mult_vae_godm', help='Model name')
    parser.add_argument('--dataset', type=str, default='yelp', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory containing multiple checkpoints for animation')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./visualization_outputs', help='Output directory')
    parser.add_argument('--type', type=str, default='animation', 
                       choices=['static', 'animation', 'distribution'], 
                       help='Visualization mode')
    
    args = parser.parse_args()
    
    # print(args.model, args.dataset)
    # Set configurations
    # configs = parse_configure(model=args.model, dataset=args.dataset)


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
        
        # Reduce dimensions
        print("Reducing dimensions with PCA...")
        reduced = visualizer.reduce_dimensions(embeddings_stage, method='pca')
        
        # Create visualization
        print("Creating static visualization...")
        visualizer.create_static_visualization(
            reduced, 
            embeddings_stage['users'],
            save_path=f'{args.model}_{args.dataset}_static.png'
        )
        
    elif args.type == 'distribution':
        # Create distribution animation
        from models.bulid_model import build_model
        model = build_model(data_handler).cuda()
        
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint)
        
        model.eval()
        visualizer.create_distribution_animation(model, data_handler, n_samples=args.n_samples)
        
    elif args.type == 'animation':
        # Create multi-checkpoint animation
        if not args.checkpoint_dir:
            print("Error: --checkpoint_dir required for animation mode")
            return
        
        # Find all checkpoints in directory
        checkpoint_paths = sorted(Path(args.checkpoint_dir).glob('*.pth'))
        print(f"Found {len(checkpoint_paths)} checkpoints")
        
        visualizer.create_animation_multi_stage(
            checkpoint_paths, 
            data_handler, 
            n_samples=args.n_samples
        )
    
    print("Visualization complete!")


if __name__ == '__main__':
    main()
