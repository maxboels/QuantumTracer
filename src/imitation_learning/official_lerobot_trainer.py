#!/usr/bin/env python3
"""
Official LeRobot ACT Training Script for Tracer RC Car
Uses the official LeRobot ACT implementation with our custom dataset loader
Designed for training on all available episodes with comprehensive logging
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
import csv
from PIL import Image
import time

# Setup LeRobot imports
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

try:
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.configs.types import NormalizationMode, PolicyFeature, FeatureType
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
    print("✅ Successfully imported official LeRobot ACT implementation")
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
    """CSV logger for tracking training progress"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV files
        self.batch_log_file = self.log_dir / "batch_metrics.csv"
        self.epoch_log_file = self.log_dir / "epoch_metrics.csv"
        
        # Write headers
        with open(self.batch_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'epoch', 'batch_loss', 'learning_rate', 'timestamp'])
            
        with open(self.epoch_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'best_val_loss', 'learning_rate', 
                           'epoch_time', 'total_samples', 'timestamp'])
    
    def log_batch(self, step: int, epoch: int, batch_loss: float, learning_rate: float):
        with open(self.batch_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, epoch, batch_loss, learning_rate, datetime.now().isoformat()])
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, best_val_loss: float, 
                  learning_rate: float, epoch_time: float, total_samples: int):
        with open(self.epoch_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, best_val_loss, learning_rate, 
                           epoch_time, total_samples, datetime.now().isoformat()])

class TracerOfficialACTDataset(Dataset):
    """Dataset wrapper for official LeRobot ACT training"""
    
    def __init__(self, data_dir: str, transforms=None):
        self.base_dataset = TracerLocalDataset(
            data_dir=data_dir,
            transforms=transforms,
            sync_tolerance=0.05,
            episode_length=None  # Use all available data
        )
        
        # Log dataset statistics
        logger.info(f"📊 Dataset loaded:")
        logger.info(f"   Total samples: {len(self.base_dataset)}")
        
        # Simple logging without accessing internal samples
        logger.info(f"   Ready for training with LeRobot ACT")
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        # Convert to LeRobot format
        observation = {
            "image_front": sample['image'],  # Shape: [C, H, W]
            "state": sample['action']        # Shape: [2] (current state as previous action)
        }
        
        action = sample['action']  # Shape: [2]
        
        return {
            "observation": observation,
            "action": action
        }

class OfficialLeRobotACTTrainer:
    """Trainer using official LeRobot ACT implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_dir']) / f"lerobot_act_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Setup logging
        self.csv_logger = CSVLogger(self.output_dir / "logs")
        
        logger.info(f"🚀 Starting official LeRobot ACT training")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🎯 Device: {self.device}")
        
        self.setup_dataset()
        self.setup_model()
        self.setup_training()
        
    def setup_dataset(self):
        """Setup dataset and data loaders"""
        logger.info("📊 Setting up dataset...")
        
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((self.config['image_height'], self.config['image_width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset
        dataset = TracerOfficialACTDataset(
            data_dir=self.config['data_dir'],
            transforms=transform
        )
        
        # Split dataset
        train_size = int(self.config['train_split'] * len(dataset))
        val_size = len(dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config['seed'])
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=False
        )
        
        logger.info(f"   Training samples: {len(self.train_dataset)}")
        logger.info(f"   Validation samples: {len(self.val_dataset)}")
        
    def setup_model(self):
        """Setup official LeRobot ACT model"""
        logger.info("🤖 Setting up official LeRobot ACT model...")
        
        # Create ACT configuration
        act_config = ACTConfig(
            # Observation configuration
            n_obs_steps=self.config['n_obs_steps'],
            
            # Input/Output structure
            chunk_size=self.config['chunk_size'],
            n_action_steps=self.config['n_action_steps'],
            
            # Architecture parameters
            dim_model=self.config['hidden_size'],
            n_encoder_layers=self.config['n_encoder_layers'],
            n_decoder_layers=self.config['n_decoder_layers'],
            n_heads=self.config['n_heads'],
            dim_feedforward=self.config['feedforward_dim'],
            dropout=self.config['dropout'],
            kl_weight=self.config['kl_weight'],
            
            # Vision encoder
            vision_backbone=self.config['vision_encoder'],
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1" if self.config['pretrained_backbone'] else None,
            
            # Device
            device=self.device,
        )
        
        # Create policy
        self.policy = ACTPolicy(act_config)
        self.policy.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.policy.parameters())
        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
        
    def setup_training(self):
        """Setup optimizer and scheduler"""
        logger.info("⚙️ Setting up training components...")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=self.config['betas']
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.step = 0
        
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.policy.train()
        
        epoch_losses = []
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            for key in batch:
                if isinstance(batch[key], dict):
                    for sub_key in batch[key]:
                        batch[key][sub_key] = batch[key][sub_key].to(self.device)
                else:
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.policy(batch)
            loss = output['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Logging
            epoch_losses.append(loss.item())
            
            if batch_idx % self.config['log_freq'] == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.csv_logger.log_batch(self.step, epoch, loss.item(), current_lr)
                
                logger.info(f"Epoch {epoch:3d} [{batch_idx:3d}/{len(self.train_loader):3d}] "
                          f"Loss: {loss.item():.6f} LR: {current_lr:.2e}")
            
            self.step += 1
        
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = np.mean(epoch_losses)
        
        return avg_train_loss, epoch_time
    
    def validate(self):
        """Validation step"""
        self.policy.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], dict):
                        for sub_key in batch[key]:
                            batch[key][sub_key] = batch[key][sub_key].to(self.device)
                    else:
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                output = self.policy(batch)
                loss = output['loss']
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"💾 Saved best model (val_loss: {self.best_val_loss:.6f})")
    
    def train(self):
        """Main training loop"""
        logger.info(f"🚀 Starting training for {self.config['num_epochs']} epochs")
        logger.info(f"📊 Training samples: {len(self.train_dataset)}")
        logger.info(f"📊 Validation samples: {len(self.val_dataset)}")
        
        for epoch in range(self.config['num_epochs']):
            # Training
            train_loss, epoch_time = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate()
            
            # Scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Logging
            self.csv_logger.log_epoch(
                epoch, train_loss, val_loss, self.best_val_loss, 
                current_lr, epoch_time, len(self.train_dataset)
            )
            
            logger.info(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                       f"Best: {self.best_val_loss:.6f} | LR: {current_lr:.2e} | "
                       f"Time: {epoch_time:.1f}s")
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_freq'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        logger.info("🎉 Training completed!")
        logger.info(f"📊 Best validation loss: {self.best_val_loss:.6f}")
        logger.info(f"📁 Models saved in: {self.output_dir}")

def get_lerobot_config() -> Dict[str, Any]:
    """Get configuration for LeRobot ACT training"""
    return {
        # Data settings
        'data_dir': '/home/maxboels/projects/QuantumTracer/src/imitation_learning/data',
        'output_dir': './outputs',
        
        # Dataset splits
        'train_split': 0.8,
        'batch_size': 8,
        'num_workers': 4,
        
        # Image settings (matching your data)
        'image_width': 640,
        'image_height': 480,
        
        # ACT model configuration
        'n_obs_steps': 1,
        'chunk_size': 32,
        'n_action_steps': 32,
        'hidden_size': 512,
        'n_encoder_layers': 4,
        'n_decoder_layers': 7,
        'n_heads': 8,
        'feedforward_dim': 3200,
        'dropout': 0.1,
        'kl_weight': 10.0,
        
        # Vision encoder
        'vision_encoder': 'resnet18',
        'pretrained_backbone': True,
        
        # Training settings
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'betas': [0.9, 0.999],
        'grad_clip': 1.0,
        
        # Logging
        'log_freq': 10,
        'save_freq': 5,
        
        # System
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
    }

def main():
    parser = argparse.ArgumentParser(description='Official LeRobot ACT Training')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory') 
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Override device')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_lerobot_config()
    
    # Override with command line arguments
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.device:
        config['device'] = args.device
    
    # Validate data directory
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    episodes = list(data_dir.glob('episode_*'))
    logger.info(f"Found {len(episodes)} episodes in {data_dir}")
    
    if len(episodes) == 0:
        logger.error("No episodes found!")
        return
    
    # Start training
    trainer = OfficialLeRobotACTTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()