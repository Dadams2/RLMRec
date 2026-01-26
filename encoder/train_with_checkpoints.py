"""
Modified Trainer with Frequent Checkpoint Saving for Animation

This script modifies the trainer to save checkpoints at regular intervals
during training, enabling the creation of training evolution animations.

Usage:
    python train_with_checkpoints.py --model mult_vae_godm --dataset yelp --save_interval 5
"""

import sys
import argparse
from pathlib import Path
import torch

# Add parent directory to path
sys.path.append('..')

from config.configurator import configs
from trainer.logger import Logger
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model
from trainer.trainer import Trainer, VAETrainer


def get_checkpoint_trainer_class(model_name):
    """
    Return the appropriate CheckpointTrainer class based on model type
    """
    # VAE models need VAETrainer
    if 'vae' in model_name.lower():
        base_trainer = VAETrainer
    else:
        base_trainer = Trainer
    
    class CheckpointTrainer(base_trainer):
        """
        Extended trainer that saves checkpoints at regular intervals
        """
        
        def __init__(self, data_handler, logger, save_interval=5, checkpoint_dir=None):
            super().__init__(data_handler, logger)
            self.save_interval = save_interval
            
            if checkpoint_dir is None:
                model_name = configs['model']['name']
                dataset_name = configs['data']['name']
                checkpoint_dir = f'./checkpoint/{model_name}/training_evolution_{dataset_name}'
            
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
            print(f"Save interval: every {save_interval} epochs")
        
        def _save_checkpoint(self, epoch, model, is_initial=False):
            """Save a checkpoint"""
            checkpoint_path = self.checkpoint_dir / f'epoch_{epoch:03d}.pth'
            
            model_state_dict = model.state_dict()
            torch.save(model_state_dict, checkpoint_path)
            
            if is_initial:
                print(f"✓ Saved initial checkpoint: {checkpoint_path}")
            else:
                print(f"✓ Saved checkpoint at epoch {epoch}: {checkpoint_path}")
        
        def train(self, model):
            """
            Override train method to add checkpoint saving
            """
            print(f"Training with checkpoint saving (interval: {self.save_interval} epochs)")
            
            # Save initial random state
            self._save_checkpoint(0, model, is_initial=True)
            
            # Access training configuration
            num_epochs = configs['train']['epoch']
            
            # Store original train_epoch method
            original_train_epoch = self.train_epoch
            
            # Wrap train_epoch to save checkpoints
            epoch_counter = [0]  # Use list to allow modification in nested function
            
            def wrapped_train_epoch(model, epoch_idx):
                result = original_train_epoch(model, epoch_idx)
                epoch_counter[0] = epoch_idx + 1
                
                # Save checkpoint at intervals
                if (epoch_idx + 1) % self.save_interval == 0:
                    self._save_checkpoint(epoch_counter[0], model)
                
                return result
            
            # Replace method temporarily
            self.train_epoch = wrapped_train_epoch
            
            try:
                # Call parent train method
                super(CheckpointTrainer, self).train(model)
                
                # Save final checkpoint
                self._save_checkpoint(epoch_counter[0], model, is_initial=False)
                
            finally:
                # Restore original method
                self.train_epoch = original_train_epoch
            
            print(f"\n{'='*80}")
            print(f"Training complete! Checkpoints saved to: {self.checkpoint_dir}")
            print(f"Total checkpoints: {len(list(self.checkpoint_dir.glob('*.pth')))}")
            print(f"{'='*80}")
    
    return CheckpointTrainer


def main():
    parser = argparse.ArgumentParser(description='Train model with regular checkpoint saving')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--save_interval', type=int, default=5, 
                       help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Directory to save checkpoints (default: auto-generated)')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA device')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    
    args = parser.parse_args()
    
    # Set configurations
    configs['model']['name'] = args.model
    configs['data']['name'] = args.dataset
    configs['train']['seed'] = args.seed
    
    # Set CUDA device
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    
    # Initialize logger
    logger = Logger(configs)
    
    print(f"\n{'='*80}")
    print(f"Training Configuration:")
    print(f"{'='*80}")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Save Interval: {args.save_interval} epochs")
    print(f"  CUDA Device: {args.cuda}")
    print(f"  Random Seed: {args.seed}")
    print(f"{'='*80}\n")
    
    # Build data handler and model
    print("Building data handler...")
    data_handler = build_data_handler()
    data_handler.load_data()
    
    print("Building model...")
    model = build_model(data_handler).cuda()
    
    # Get appropriate trainer class for this model type
    CheckpointTrainerClass = get_checkpoint_trainer_class(args.model)
    
    # Create checkpoint trainer
    trainer = CheckpointTrainerClass(
        data_handler,
        logger,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train model
    print("\nStarting training...\n")
    trainer.train(model)
    
    print("\n✓ Training completed successfully!")
    print(f"\nTo create an animation, run:")
    print(f"python visualize_embedding_dynamics.py \\")
    print(f"    --model {args.model} \\")
    print(f"    --dataset {args.dataset} \\")
    print(f"    --mode animation \\")
    print(f"    --checkpoint_dir {trainer.checkpoint_dir} \\")
    print(f"    --n_samples 500")


if __name__ == '__main__':
    main()
