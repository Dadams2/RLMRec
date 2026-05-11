"""
Phase 1: Semantic Noise Analysis
Analyzes which dimensions of LLM embeddings are important for recommendation
by computing gradient-based importance scores.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json


class SemanticNoiseAnalyzer:
    def __init__(self, model, data_handler, save_dir='./analysis_results'):
        """
        Args:
            model: The trained VAE model (mult_vae_MDDM, CPDM, or GODM)
            data_handler: Data handler with train/validation data
            save_dir: Directory to save analysis results
        """
        self.model = model
        self.data_handler = data_handler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for gradients and statistics
        self.user_grad_importance = []
        self.item_grad_importance = []
        self.user_activation_stats = []
        self.item_activation_stats = []
        
    def compute_gradient_importance(self, num_batches=50, batch_size=256):
        """
        Compute gradient-based importance of semantic embedding dimensions.
        
        Returns:
            dict with user and item importance scores
        """
        print(f"Computing gradient importance over {num_batches} batches...")
        self.model.eval()  # Set to eval but we'll compute gradients
        
        user_grads = []
        item_grads = []
        
        # Sample users
        num_users = self.data_handler.train_data.shape[0]
        sampled_users = np.random.choice(num_users, size=min(num_batches * batch_size, num_users), replace=False)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(sampled_users))
            batch_users = sampled_users[start_idx:end_idx]
            
            if len(batch_users) == 0:
                break
                
            # Get batch data
            batch_users_tensor = torch.LongTensor(batch_users).cuda()
            batch_data = self.data_handler.train_data[batch_users]
            batch_data_tensor = torch.FloatTensor(batch_data.toarray()).cuda()
            
            # Enable gradients for embeddings
            user_emb = self.model.usrprf_embeds[batch_users_tensor].clone().detach().requires_grad_(True)
            item_emb = self.model.itmprf_embeds.clone().detach().requires_grad_(True)
            
            # Forward pass with custom embeddings
            h = self.model.drop(batch_data_tensor)
            hidden = torch.matmul(h, item_emb) + user_emb
            hidden = self.model.mlp(hidden)
            
            mu_llm = hidden[:, :200]
            logvar_llm = hidden[:, 200:]
            
            # Also compute CF path
            h_cf = self.model.drop(batch_data_tensor)
            for i, layer in enumerate(self.model.q_layers):
                h_cf = layer(h_cf)
                if i != len(self.model.q_layers) - 1:
                    h_cf = torch.tanh(h_cf)
                else:
                    mu_cf = h_cf[:, :self.model.q_dims[-1]]
                    logvar_cf = h_cf[:, self.model.q_dims[-1]:]
            
            # Combine
            mu = mu_cf + mu_llm
            logvar = logvar_cf + logvar_llm
            
            # Sample and reconstruct
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std) + mu
            
            recon_x = self.model.decode(z)
            
            # Compute loss (what the model is actually optimizing)
            BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * batch_data_tensor, -1))
            
            # Compute gradients
            if user_emb.grad is not None:
                user_emb.grad.zero_()
            if item_emb.grad is not None:
                item_emb.grad.zero_()
                
            BCE.backward()
            
            # Store gradient magnitudes (importance = how much each dimension affects loss)
            if user_emb.grad is not None:
                user_grads.append(user_emb.grad.abs().mean(dim=0).cpu().detach().numpy())
            if item_emb.grad is not None:
                item_grads.append(item_emb.grad.abs().mean(dim=0).cpu().detach().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed batch {batch_idx + 1}/{num_batches}")
        
        # Aggregate results
        user_importance = np.mean(user_grads, axis=0) if user_grads else None
        item_importance = np.mean(item_grads, axis=0) if item_grads else None
        
        results = {
            'user_importance': user_importance,
            'item_importance': item_importance,
            'user_importance_std': np.std(user_grads, axis=0) if user_grads else None,
            'item_importance_std': np.std(item_grads, axis=0) if item_grads else None,
        }
        
        return results
    
    def compute_activation_statistics(self, num_batches=50, batch_size=256):
        """
        Compute statistics about semantic embedding activations.
        Shows which dimensions have high variance (informative).
        """
        print(f"Computing activation statistics over {num_batches} batches...")
        self.model.eval()
        
        user_activations = []
        item_activations = []
        
        num_users = self.data_handler.train_data.shape[0]
        sampled_users = np.random.choice(num_users, size=min(num_batches * batch_size, num_users), replace=False)
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(sampled_users))
                batch_users = sampled_users[start_idx:end_idx]
                
                if len(batch_users) == 0:
                    break
                
                batch_users_tensor = torch.LongTensor(batch_users).cuda()
                batch_data = self.data_handler.train_data[batch_users]
                batch_data_tensor = torch.FloatTensor(batch_data.toarray()).cuda()
                
                # Get activations
                user_emb = self.model.usrprf_embeds[batch_users_tensor]
                
                # Item activations: weighted by interaction
                h = batch_data_tensor
                item_weighted = torch.matmul(h, self.model.itmprf_embeds)  # [batch, emb_dim]
                
                user_activations.append(user_emb.cpu().numpy())
                item_activations.append(item_weighted.cpu().numpy())
        
        user_activations = np.concatenate(user_activations, axis=0)
        item_activations = np.concatenate(item_activations, axis=0)
        
        results = {
            'user_mean': np.mean(user_activations, axis=0),
            'user_std': np.std(user_activations, axis=0),
            'user_variance': np.var(user_activations, axis=0),
            'item_mean': np.mean(item_activations, axis=0),
            'item_std': np.std(item_activations, axis=0),
            'item_variance': np.var(item_activations, axis=0),
        }
        
        return results
    
    def analyze_noise_correlation(self, gradient_importance, activation_stats):
        """
        Correlate gradient importance with activation statistics to identify noise.
        
        High variance but low gradient = potential noise dimensions
        Low variance but high gradient = important compressed information
        """
        results = {}
        
        # User analysis
        if gradient_importance['user_importance'] is not None:
            user_grad = gradient_importance['user_importance']
            user_var = activation_stats['user_variance']
            
            # Normalize for comparison
            user_grad_norm = (user_grad - user_grad.min()) / (user_grad.max() - user_grad.min() + 1e-8)
            user_var_norm = (user_var - user_var.min()) / (user_var.max() - user_var.min() + 1e-8)
            
            # Compute noise score: high variance, low importance
            user_noise_score = user_var_norm * (1 - user_grad_norm)
            # Signal score: high importance regardless of variance
            user_signal_score = user_grad_norm
            
            results['user'] = {
                'noise_score': user_noise_score,
                'signal_score': user_signal_score,
                'top_noise_dims': np.argsort(user_noise_score)[-50:],  # Top 50 noisy dimensions
                'top_signal_dims': np.argsort(user_signal_score)[-50:],  # Top 50 signal dimensions
            }
        
        # Item analysis
        if gradient_importance['item_importance'] is not None:
            item_grad = gradient_importance['item_importance']
            item_var = activation_stats['item_variance']
            
            item_grad_norm = (item_grad - item_grad.min()) / (item_grad.max() - item_grad.min() + 1e-8)
            item_var_norm = (item_var - item_var.min()) / (item_var.max() - item_var.min() + 1e-8)
            
            item_noise_score = item_var_norm * (1 - item_grad_norm)
            item_signal_score = item_grad_norm
            
            results['item'] = {
                'noise_score': item_noise_score,
                'signal_score': item_signal_score,
                'top_noise_dims': np.argsort(item_noise_score)[-50:],
                'top_signal_dims': np.argsort(item_signal_score)[-50:],
            }
        
        return results
    
    def visualize_results(self, gradient_importance, activation_stats, noise_analysis, model_name='model'):
        """
        Create comprehensive visualizations of the analysis.
        """
        print("Creating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. User gradient importance heatmap
        if gradient_importance['user_importance'] is not None:
            fig, axes = plt.subplots(3, 1, figsize=(16, 12))
            
            # Plot 1: Gradient importance
            user_grad = gradient_importance['user_importance']
            axes[0].plot(user_grad, alpha=0.7, linewidth=0.8)
            axes[0].set_title(f'User Embedding Gradient Importance ({model_name})', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Dimension Index')
            axes[0].set_ylabel('Gradient Magnitude')
            axes[0].grid(True, alpha=0.3)
            
            # Highlight top dimensions
            top_dims = np.argsort(user_grad)[-20:]
            axes[0].scatter(top_dims, user_grad[top_dims], color='red', s=50, zorder=5, label='Top 20 dims')
            axes[0].legend()
            
            # Plot 2: Activation variance
            user_var = activation_stats['user_variance']
            axes[1].plot(user_var, alpha=0.7, linewidth=0.8, color='green')
            axes[1].set_title('User Embedding Activation Variance', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Dimension Index')
            axes[1].set_ylabel('Variance')
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Noise vs Signal scores
            if 'user' in noise_analysis:
                noise_score = noise_analysis['user']['noise_score']
                signal_score = noise_analysis['user']['signal_score']
                
                x = np.arange(len(noise_score))
                axes[2].plot(x, signal_score, alpha=0.7, label='Signal Score', linewidth=0.8)
                axes[2].plot(x, noise_score, alpha=0.7, label='Noise Score', linewidth=0.8, color='orange')
                axes[2].set_title('Signal vs Noise Scores', fontsize=14, fontweight='bold')
                axes[2].set_xlabel('Dimension Index')
                axes[2].set_ylabel('Score (normalized)')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / f'{model_name}_user_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Item gradient importance
        if gradient_importance['item_importance'] is not None:
            fig, axes = plt.subplots(3, 1, figsize=(16, 12))
            
            item_grad = gradient_importance['item_importance']
            axes[0].plot(item_grad, alpha=0.7, linewidth=0.8, color='purple')
            axes[0].set_title(f'Item Embedding Gradient Importance ({model_name})', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Dimension Index')
            axes[0].set_ylabel('Gradient Magnitude')
            axes[0].grid(True, alpha=0.3)
            
            top_dims = np.argsort(item_grad)[-20:]
            axes[0].scatter(top_dims, item_grad[top_dims], color='red', s=50, zorder=5, label='Top 20 dims')
            axes[0].legend()
            
            item_var = activation_stats['item_variance']
            axes[1].plot(item_var, alpha=0.7, linewidth=0.8, color='green')
            axes[1].set_title('Item Embedding Activation Variance', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Dimension Index')
            axes[1].set_ylabel('Variance')
            axes[1].grid(True, alpha=0.3)
            
            if 'item' in noise_analysis:
                noise_score = noise_analysis['item']['noise_score']
                signal_score = noise_analysis['item']['signal_score']
                
                x = np.arange(len(noise_score))
                axes[2].plot(x, signal_score, alpha=0.7, label='Signal Score', linewidth=0.8)
                axes[2].plot(x, noise_score, alpha=0.7, label='Noise Score', linewidth=0.8, color='orange')
                axes[2].set_title('Signal vs Noise Scores', fontsize=14, fontweight='bold')
                axes[2].set_xlabel('Dimension Index')
                axes[2].set_ylabel('Score (normalized)')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / f'{model_name}_item_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Summary statistics plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if gradient_importance['user_importance'] is not None and 'user' in noise_analysis:
            # Distribution of noise vs signal
            noise_score = noise_analysis['user']['noise_score']
            signal_score = noise_analysis['user']['signal_score']
            
            axes[0].hist(noise_score, bins=50, alpha=0.6, label='Noise Score', color='orange', edgecolor='black')
            axes[0].hist(signal_score, bins=50, alpha=0.6, label='Signal Score', color='blue', edgecolor='black')
            axes[0].set_title('User Embedding Score Distribution', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Score Value')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        if gradient_importance['item_importance'] is not None and 'item' in noise_analysis:
            noise_score = noise_analysis['item']['noise_score']
            signal_score = noise_analysis['item']['signal_score']
            
            axes[1].hist(noise_score, bins=50, alpha=0.6, label='Noise Score', color='orange', edgecolor='black')
            axes[1].hist(signal_score, bins=50, alpha=0.6, label='Signal Score', color='blue', edgecolor='black')
            axes[1].set_title('Item Embedding Score Distribution', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Score Value')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{model_name}_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {self.save_dir}")
    
    def save_results(self, gradient_importance, activation_stats, noise_analysis, model_name='model'):
        """
        Save numerical results to JSON for further analysis.
        """
        results = {
            'model_name': model_name,
            'gradient_importance': {
                'user_top_20_dims': np.argsort(gradient_importance['user_importance'])[-20:].tolist() 
                    if gradient_importance['user_importance'] is not None else None,
                'user_top_20_values': sorted(gradient_importance['user_importance'].tolist())[-20:]
                    if gradient_importance['user_importance'] is not None else None,
                'item_top_20_dims': np.argsort(gradient_importance['item_importance'])[-20:].tolist()
                    if gradient_importance['item_importance'] is not None else None,
                'item_top_20_values': sorted(gradient_importance['item_importance'].tolist())[-20:]
                    if gradient_importance['item_importance'] is not None else None,
            },
            'noise_analysis': {
                'user_top_noise_dims': noise_analysis['user']['top_noise_dims'].tolist() 
                    if 'user' in noise_analysis else None,
                'user_top_signal_dims': noise_analysis['user']['top_signal_dims'].tolist()
                    if 'user' in noise_analysis else None,
                'item_top_noise_dims': noise_analysis['item']['top_noise_dims'].tolist()
                    if 'item' in noise_analysis else None,
                'item_top_signal_dims': noise_analysis['item']['top_signal_dims'].tolist()
                    if 'item' in noise_analysis else None,
            },
            'statistics': {
                'user_embedding_dim': len(gradient_importance['user_importance'])
                    if gradient_importance['user_importance'] is not None else None,
                'item_embedding_dim': len(gradient_importance['item_importance'])
                    if gradient_importance['item_importance'] is not None else None,
                'user_avg_gradient': float(np.mean(gradient_importance['user_importance']))
                    if gradient_importance['user_importance'] is not None else None,
                'item_avg_gradient': float(np.mean(gradient_importance['item_importance']))
                    if gradient_importance['item_importance'] is not None else None,
                'user_gradient_sparsity': float(np.sum(gradient_importance['user_importance'] < np.percentile(gradient_importance['user_importance'], 10)) / len(gradient_importance['user_importance']))
                    if gradient_importance['user_importance'] is not None else None,
            }
        }
        
        output_file = self.save_dir / f'{model_name}_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
        return results
    
    def run_full_analysis(self, model_name='model', num_batches=50):
        """
        Run complete analysis pipeline.
        """
        print(f"\n{'='*60}")
        print(f"Starting Semantic Noise Analysis for {model_name}")
        print(f"{'='*60}\n")
        
        # Step 1: Gradient importance
        print("Step 1/4: Computing gradient importance...")
        gradient_importance = self.compute_gradient_importance(num_batches=num_batches)
        
        # Step 2: Activation statistics  
        print("\nStep 2/4: Computing activation statistics...")
        activation_stats = self.compute_activation_statistics(num_batches=num_batches)
        
        # Step 3: Noise analysis
        print("\nStep 3/4: Analyzing noise vs signal...")
        noise_analysis = self.analyze_noise_correlation(gradient_importance, activation_stats)
        
        # Step 4: Visualization
        print("\nStep 4/4: Creating visualizations...")
        self.visualize_results(gradient_importance, activation_stats, noise_analysis, model_name)
        
        # Save results
        results = self.save_results(gradient_importance, activation_stats, noise_analysis, model_name)
        
        # Print summary
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        if gradient_importance['user_importance'] is not None:
            top_user_dims = np.argsort(gradient_importance['user_importance'])[-5:]
            print(f"\nTop 5 User Dimensions (by gradient): {top_user_dims.tolist()}")
            print(f"  Values: {gradient_importance['user_importance'][top_user_dims]}")
            
            if 'user' in noise_analysis:
                print(f"\nTop 5 Noisy User Dimensions: {noise_analysis['user']['top_noise_dims'][-5:].tolist()}")
                print(f"Top 5 Signal User Dimensions: {noise_analysis['user']['top_signal_dims'][-5:].tolist()}")
        
        if gradient_importance['item_importance'] is not None:
            top_item_dims = np.argsort(gradient_importance['item_importance'])[-5:]
            print(f"\nTop 5 Item Dimensions (by gradient): {top_item_dims.tolist()}")
            print(f"  Values: {gradient_importance['item_importance'][top_item_dims]}")
            
            if 'item' in noise_analysis:
                print(f"\nTop 5 Noisy Item Dimensions: {noise_analysis['item']['top_noise_dims'][-5:].tolist()}")
                print(f"Top 5 Signal Item Dimensions: {noise_analysis['item']['top_signal_dims'][-5:].tolist()}")
        
        print(f"\n{'='*60}\n")
        
        return results
