#!/usr/bin/env python3
"""
Official HuggingFace ACT Training Script for Tracer RC Car
Uses the official LeRobot ACT implementation with proper configuration
"""

import os
import sys
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import time
import csv
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
import argparse
from PIL import Image

# Setup LeRobot imports
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

try:
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.configs.types import NormalizationMode, PolicyFeature, FeatureType
    print("✅ Successfully imported official ACT implementation")
except ImportError as e:
    print(f"❌ LeRobot import error: {e}")
    print("Please run setup_lerobot.py first")
    sys.exit(1)

# Import our local dataset loader
from local_dataset_loader import TracerLocalDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVLogger:
    """Simple CSV logger for tracking training progress"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV files
        self.batch_log_file = self.log_dir / "batch_metrics.csv"
        self.epoch_log_file = self.log_dir / "epoch_metrics.csv"
        
        # Write headers
        with open(self.batch_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'batch_loss', 'learning_rate', 'timestamp'])
            
        with open(self.epoch_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'best_val_loss', 'learning_rate', 'timestamp'])
    
    def log_batch(self, step: int, batch_loss: float, learning_rate: float):
        with open(self.batch_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, batch_loss, learning_rate, time.time()])
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, best_val_loss: float, learning_rate: float):
        with open(self.epoch_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, best_val_loss, learning_rate, time.time()])

class TracerACTDataset(Dataset):
    """Dataset wrapper that formats data for official ACT implementation"""
    
    def __init__(self, data_dir: str, transforms=None):
        self.base_dataset = TracerLocalDataset(data_dir, transforms=transforms)
        print(f"📊 Loaded {len(self.base_dataset)} samples for official ACT training")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get data from base dataset
        item = self.base_dataset[idx]
        
        # Reformat for official ACT implementation
        # The official implementation expects specific key naming
        formatted_item = {
            # Observations (what the policy sees)
            'observation.images.front': item['observation']['image_front'],  # [3, H, W] tensor
            'observation.state': item['observation']['state'],              # [2] tensor with current steering/throttle
            
            # Actions (what the policy should predict) 
            'action': item['action'],                                       # [2] tensor with target steering/throttle
        }
        
        return formatted_item

class OfficialACTTrainer:
    """Official HuggingFace ACT trainer for Tracer RC car"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        print(f"🚀 Initializing Official ACT Trainer")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Training resolution: {config['image_size']}")
        print(f"📦 Batch size: {config['batch_size']}")
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_dir']) / f"official_act_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.setup_dataset()
        self.setup_model()
        self.setup_training()
        
        # Save configuration
        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Output directory: {self.output_dir}")
    
    def setup_dataset(self):
        """Setup dataset and data loaders"""
        print("📚 Setting up official ACT dataset...")
        
        # Image preprocessing for ACT
        image_transforms = transforms.Compose([
            transforms.Resize(self.config['image_size']),
            transforms.ToTensor(),
            # Note: ACT uses its own normalization, so we don't normalize here
        ])
        
        # Create dataset
        full_dataset = TracerACTDataset(
            self.config['data_dir'], 
            transforms=image_transforms
        )
        
        # Split into train/val
        train_size = int(self.config['train_split'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        print(f"📊 Dataset split: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
    
    def setup_model(self):
        """Setup the official ACT model with proper configuration"""
        print("🤖 Setting up official ACT model...")
        
        # Create ACT configuration
        act_config = ACTConfig(
            # Input/Output configuration
            n_obs_steps=1,
            chunk_size=self.config['chunk_size'],
            n_action_steps=self.config['chunk_size'],  # Use full chunk for training
            
            # Input/Output features - this is critical for the official implementation
            input_features={
                'observation.images.front': PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, self.config['image_size'][0], self.config['image_size'][1])  # [C, H, W]
                ),
                'observation.state': PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(2,)  # [steering, throttle]
                ),
            },
            output_features={
                'action': PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=(2,)  # [steering_cmd, throttle_cmd]
                ),
            },
            
            # Vision backbone
            vision_backbone=self.config['vision_backbone'],
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            replace_final_stride_with_dilation=False,
            
            # Transformer architecture
            dim_model=self.config['hidden_dim'],
            n_heads=self.config['num_heads'],
            dim_feedforward=self.config.get('feedforward_dim', 3200),
            n_encoder_layers=self.config['num_encoder_layers'],
            n_decoder_layers=self.config['num_decoder_layers'],
            
            # VAE settings
            use_vae=self.config.get('use_vae', True),
            latent_dim=self.config.get('latent_dim', 32),
            n_vae_encoder_layers=self.config.get('vae_encoder_layers', 4),
            
            # Training parameters
            dropout=self.config.get('dropout', 0.1),
            kl_weight=self.config.get('kl_weight', 10.0),
            
            # Optimizer settings (will be used by the policy)
            optimizer_lr=self.config['learning_rate'],
            optimizer_weight_decay=self.config.get('weight_decay', 1e-4),
            optimizer_lr_backbone=self.config.get('backbone_lr', 1e-5),
        )
        
        # Create the policy
        self.policy = ACTPolicy(act_config)
        self.policy.to(self.device)
        
        # Calculate dataset statistics for normalization
        print("📈 Computing dataset statistics for normalization...")
        self.compute_dataset_stats()
        
        # Setup pre/post processors
        self.preprocessor, self.postprocessor = make_act_pre_post_processors(
            act_config, self.dataset_stats
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.policy.parameters())
        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        
        print(f"🧠 Model parameters: {trainable_params:,} trainable / {total_params:,} total")
        print(f"🎯 Chunk size: {act_config.chunk_size}")
        print(f"🔬 VAE enabled: {act_config.use_vae}")
    
    def compute_dataset_stats(self):
        """Compute dataset statistics for normalization"""
        print("📊 Computing dataset statistics...")
        
        # Collect all data for statistics
        all_images = []
        all_states = []
        all_actions = []
        
        # Use a subset for efficiency if dataset is large
        sample_size = min(len(self.train_dataset), 1000)
        indices = torch.randperm(len(self.train_dataset))[:sample_size]
        
        for idx in indices:
            item = self.train_dataset[idx.item()]
            all_images.append(item['observation.images.front'])
            all_states.append(item['observation.state'])
            all_actions.append(item['action'])
        
        # Stack tensors
        all_images = torch.stack(all_images)  # [N, 3, H, W]
        all_states = torch.stack(all_states)  # [N, 2]  
        all_actions = torch.stack(all_actions)  # [N, 2]
        
        # Compute statistics
        self.dataset_stats = {
            'observation.images.front': {
                'mean': torch.tensor([0.485, 0.456, 0.406]),  # ImageNet stats for pretrained backbone
                'std': torch.tensor([0.229, 0.224, 0.225]),
            },
            'observation.state': {
                'mean': all_states.mean(dim=0),
                'std': all_states.std(dim=0),
            },
            'action': {
                'mean': all_actions.mean(dim=0), 
                'std': all_actions.std(dim=0),
            }
        }
        
        print(f"📈 Dataset statistics computed from {sample_size} samples")
        print(f"   State mean: {self.dataset_stats['observation.state']['mean']}")
        print(f"   State std: {self.dataset_stats['observation.state']['std']}")
        print(f"   Action mean: {self.dataset_stats['action']['mean']}")
        print(f"   Action std: {self.dataset_stats['action']['std']}")
    
    def setup_training(self):
        """Setup optimizer, scheduler, and logging"""
        print("⚙️ Setting up training components...")
        
        # Use the policy's optimized parameter grouping
        param_groups = self.policy.get_optim_params()
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        # CSV logger
        self.logger = CSVLogger(self.output_dir / "logs")
        
        # Training state
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        print(f"📚 Optimizer: AdamW with LR={self.config['learning_rate']}")
        print(f"📈 Scheduler: CosineAnnealing")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.policy.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass through policy
            self.optimizer.zero_grad()
            
            # The ACT policy handles normalization internally via preprocessors
            output = self.policy(batch)
            
            # The policy returns a dict with 'loss' key
            loss = output['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.log_batch(self.global_step, loss.item(), current_lr)
            self.global_step += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"    Batch {batch_idx}/{len(self.train_loader)}: Loss={loss.item():.6f}")
        
        avg_train_loss = total_loss / num_batches
        return avg_train_loss
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.policy.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                output = self.policy(batch)
                loss = output['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = total_loss / num_batches
        return avg_val_loss
    
    def train(self):
        """Main training loop"""
        print(f"🎯 Starting official ACT training for {self.config['num_epochs']} epochs...")
        
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            epoch_start = time.time()
            
            print(f"\n📚 Epoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate  
            val_loss = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Log epoch metrics
            self.logger.log_epoch(epoch, train_loss, val_loss, self.best_val_loss, current_lr)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"✅ Epoch {epoch+1} complete ({epoch_time:.1f}s)")
            print(f"   Train Loss: {train_loss:.6f}")
            print(f"   Val Loss: {val_loss:.6f}")
            print(f"   Best Val Loss: {self.best_val_loss:.6f}")
            print(f"   Learning Rate: {current_lr:.2e}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training complete! Total time: {total_time/3600:.1f} hours")
        print(f"🏆 Best validation loss: {self.best_val_loss:.6f}")
        print(f"💾 Models saved to: {self.output_dir}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'dataset_stats': self.dataset_stats,
        }
        
        # Save latest checkpoint
        latest_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model to {best_path}")

def get_default_config() -> Dict[str, Any]:
    """Get default configuration optimized for Tracer RC car"""
    return {
        # Data
        'data_dir': '/home/maxboels/projects/QuantumTracer/src/imitation_learning/data',
        'train_split': 0.8,
        
        # Model architecture  
        'image_size': (480, 640),  # Full resolution
        'chunk_size': 32,          # ACT chunk size
        'hidden_dim': 512,         # Transformer hidden dimension
        'num_heads': 8,            # Multi-head attention
        'num_encoder_layers': 4,   # Transformer encoder layers
        'num_decoder_layers': 1,   # Transformer decoder layers (ACT default due to bug)
        'vision_backbone': 'resnet18',  # Vision encoder
        
        # VAE settings
        'use_vae': True,
        'latent_dim': 32,
        'vae_encoder_layers': 4,
        'kl_weight': 10.0,
        
        # Training
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-5,     # ACT paper default
        'backbone_lr': 1e-5,       # Backbone learning rate
        'weight_decay': 1e-4,
        'dropout': 0.1,
        
        # Hardware
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        
        # Output
        'output_dir': './outputs'
    }

def main():
    parser = argparse.ArgumentParser(description='Train official HuggingFace ACT policy for Tracer RC car')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/maxboels/projects/QuantumTracer/src/imitation_learning/data',
                       help='Path to training data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    config['data_dir'] = args.data_dir
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['output_dir'] = args.output_dir
    
    if args.device == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config['device'] = args.device
    
    # Validate data directory
    if not Path(config['data_dir']).exists():
        print(f"❌ Data directory not found: {config['data_dir']}")
        return
    
    print("🎬 Official HuggingFace ACT Training for Tracer RC Car")
    print("=" * 60)
    
    # Create trainer and start training
    trainer = OfficialACTTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()