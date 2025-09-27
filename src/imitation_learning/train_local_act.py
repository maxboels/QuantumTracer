#!/usr/bin/env python3
"""
Local Training Script for Tracer RC Car ACT Policy
Trains ACT policy using locally recorded demonstrations
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from datetime import datetime
import json

# Setup LeRobot imports
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

try:
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType
except ImportError as e:
    print(f"LeRobot import error: {e}")
    print("Please run setup_lerobot.py first")
    sys.exit(1)

from local_dataset_loader import TracerLocalDataset, create_data_loaders, compute_dataset_stats

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TracerACTTrainer:
    """Trainer for ACT policy on Tracer RC car data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize dataset and dataloaders
        self.setup_dataset()
        
        # Initialize model
        self.setup_model()
        
        # Initialize optimizer and scheduler
        self.setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def setup_dataset(self):
        """Setup dataset and data loaders"""
        logger.info("Setting up dataset...")
        
        # Image transforms
        image_transforms = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create data loaders
        self.train_loader, self.val_loader = create_data_loaders(
            data_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            train_split=self.config['train_split'],
            val_split=self.config['val_split'],
            num_workers=self.config['num_workers'],
            transforms=image_transforms
        )
        
        # Compute dataset statistics
        full_dataset = TracerLocalDataset(self.config['data_dir'], transforms=image_transforms)
        self.dataset_stats = compute_dataset_stats(full_dataset)
        
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples: {len(self.val_loader.dataset)}")
    
    def setup_model(self):
        """Setup ACT model with appropriate configuration"""
        logger.info("Setting up ACT model...")
        
        # Define input and output features
        input_features = {
            'observation.image_front': PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640)
            ),
            'observation.state': PolicyFeature(
                type=FeatureType.STATE,
                shape=(2,)  # [steering, throttle] 
            )
        }
        
        output_features = {
            'action': PolicyFeature(
                type=FeatureType.ACTION,
                shape=(2,)  # [steering_command, throttle_command]
            )
        }
        
        # Create ACT configuration
        self.act_config = ACTConfig(
            n_obs_steps=self.config['n_obs_steps'],
            chunk_size=self.config['chunk_size'],
            n_action_steps=self.config['n_action_steps'],
            dim_model=self.config['hidden_size'],
            n_encoder_layers=self.config['n_encoder_layers'],
            n_decoder_layers=self.config['n_decoder_layers'],
            n_heads=self.config['n_heads'],
            dim_feedforward=self.config['feedforward_dim'],
            dropout=self.config['dropout'],
            kl_weight=self.config['kl_weight'],
            temporal_ensemble_coeff=self.config['temporal_ensemble_coeff'],
            vision_backbone=self.config['vision_encoder'],
            pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1' if self.config['pretrained_backbone'] else None,
            replace_final_stride_with_dilation=self.config.get('replace_final_stride_with_dilation', False),
            device=str(self.device),
            use_amp=self.config.get('use_amp', False),
            input_features=input_features,
            output_features=output_features
        )
        
        # Initialize model
        self.model = ACTPolicy(self.act_config).to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['max_steps'],
            eta_min=self.config['learning_rate'] * 0.01
        )
    
    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare batch for training"""
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device, non_blocking=True)
            elif isinstance(batch[key], dict):
                for subkey in batch[key]:
                    if isinstance(batch[key][subkey], torch.Tensor):
                        batch[key][subkey] = batch[key][subkey].to(self.device, non_blocking=True)
        
        return batch
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Prepare batch
        batch = self.prepare_batch(batch)
        
        try:
            # Forward pass
            output = self.model.forward(batch)
            
            # Extract loss (ACT returns tuple with loss and additional info)
            if isinstance(output, tuple):
                loss = output[0]
            else:
                loss = output
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip_norm']
                )
            
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            return float('inf')
    
    def val_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single validation step"""
        self.model.eval()
        
        with torch.no_grad():
            # Prepare batch
            batch = self.prepare_batch(batch)
            
            try:
                # Forward pass
                output = self.model.forward(batch)
                
                # Extract loss
                if isinstance(output, tuple):
                    loss = output[0]
                else:
                    loss = output
                
                return loss.item()
                
            except Exception as e:
                logger.error(f"Error in validation step: {e}")
                return float('inf')
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        epoch_losses = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.train_step(batch)
            epoch_losses.append(loss)
            
            self.global_step += 1
            
            # Log progress
            if batch_idx % self.config['log_freq'] == 0:
                logger.info(f"Step {self.global_step}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss:.4f}")
            
            # Update scheduler
            self.scheduler.step()
            
            if self.global_step >= self.config['max_steps']:
                break
        
        return np.mean(epoch_losses)
    
    def validate_epoch(self) -> float:
        """Validate for one epoch"""
        epoch_losses = []
        
        for batch in self.val_loader:
            loss = self.val_step(batch)
            epoch_losses.append(loss)
        
        return np.mean(epoch_losses)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'act_config': self.act_config.__dict__,
            'dataset_stats': self.dataset_stats
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.config['max_epochs']):
            if self.global_step >= self.config['max_steps']:
                break
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{self.config['max_epochs']}")
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Best Val Loss: {self.best_val_loss:.4f}")
            logger.info(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch, is_best)
        
        # Save final model
        self.save_checkpoint(epoch, is_best=True)
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

def get_default_config() -> Dict[str, Any]:
    """Get default training configuration"""
    return {
        # Data
        'data_dir': '/home/maxboels/projects/QuantumTracer/src/imitation_learning/data',
        'batch_size': 8,
        'num_workers': 4,
        'train_split': 0.8,
        'val_split': 0.2,
        
        # Model
        'n_obs_steps': 1,
        'chunk_size': 32,
        'n_action_steps': 1,  # Must be 1 when using temporal ensembling
        'hidden_size': 512,
        'n_encoder_layers': 4,
        'n_decoder_layers': 7,
        'n_heads': 8,
        'feedforward_dim': 3200,
        'dropout': 0.1,
        'kl_weight': 10.0,
        'temporal_ensemble_coeff': 0.01,
        'vision_encoder': 'resnet18',
        'pretrained_backbone': True,
        
        # Training
        'max_epochs': 100,
        'max_steps': 50000,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'grad_clip_norm': 1.0,
        'log_freq': 10,
        'save_freq': 10,
        
        # Hardware
        'device': 'cuda',
        'use_amp': False,  # Mixed precision training
        
        # Output
        'output_dir': f'./outputs/tracer_act_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }

def main():
    parser = argparse.ArgumentParser(description='Train ACT policy for Tracer RC car')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    parser.add_argument('--data_dir', type=str, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Get default config
    config = get_default_config()
    
    # Override with command line arguments
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.max_epochs:
        config['max_epochs'] = args.max_epochs
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.device:
        config['device'] = args.device
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    # Create trainer and start training
    trainer = TracerACTTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()