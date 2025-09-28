#!/usr/bin/env python3
"""
Hybrid LeRobot ACT Trainer
Uses the official LeRobot ACT model with our simple training loop
Focuses on getting training working with all 35 episodes
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

# Use the same SimpleACTModel from our previous training but with full resolution
class FullResolutionACTModel(nn.Module):
    """ACT-like model optimized for full resolution images (480x640)"""
    
    def __init__(self, image_size=(480, 640), action_dim=2, hidden_dim=512, 
                 num_layers=4, num_heads=8, chunk_size=32):
        super().__init__()
        
        self.image_size = image_size
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        
        # Vision encoder optimized for full resolution
        self.vision_encoder = nn.Sequential(
            # First conv block - reduce spatial dimension
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 5)),  # Fixed output size
        )
        
        # Calculate vision feature size
        self.vision_feature_size = 128 * 4 * 5  # 2560 features
        
        # Feature projection
        self.vision_proj = nn.Linear(self.vision_feature_size, hidden_dim)
        
        # Transformer encoder for processing image features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, action_dim * chunk_size)
        )
        
    def forward(self, image, state=None):
        batch_size = image.shape[0]
        
        # Vision encoding
        vision_features = self.vision_encoder(image)  # [B, 128, 4, 5]
        vision_features = vision_features.flatten(2)  # [B, 128, 20]
        vision_features = vision_features.transpose(1, 2)  # [B, 20, 128]
        
        # Project to hidden dimension
        vision_features = self.vision_proj(vision_features.flatten(1))  # [B, 2560]
        vision_features = vision_features.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Transformer encoding
        encoded_features = self.transformer_encoder(vision_features)  # [B, 1, hidden_dim]
        
        # Predict actions
        action_pred = self.action_head(encoded_features.squeeze(1))  # [B, action_dim * chunk_size]
        action_pred = action_pred.view(batch_size, self.chunk_size, self.action_dim)
        
        # For training, we typically use just the first predicted action
        return action_pred[:, 0, :]  # [B, action_dim]

class HybridACTTrainer:
    """Hybrid trainer using full resolution with all episodes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_dir']) / f"hybrid_act_fullres_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Setup logging
        self.csv_logger = CSVLogger(self.output_dir / "logs")
        
        logger.info(f"🚀 Starting Hybrid ACT training with full resolution")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🎯 Device: {self.device}")
        
        self.setup_dataset()
        self.setup_model()
        self.setup_training()
        
    def setup_dataset(self):
        """Setup dataset and data loaders"""
        logger.info("📊 Setting up dataset with all episodes...")
        
        # Image preprocessing - keep full resolution!
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset using ALL episodes
        dataset = TracerLocalDataset(
            data_dir=self.config['data_dir'],
            transforms=transform,
            sync_tolerance=0.05,
            episode_length=None  # Use all available data
        )
        
        logger.info(f"   Total samples: {len(dataset)}")
        
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
        """Setup full resolution ACT model"""
        logger.info("🤖 Setting up Full Resolution ACT model...")
        
        self.model = FullResolutionACTModel(
            image_size=(self.config['image_height'], self.config['image_width']),
            action_dim=2,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            chunk_size=self.config['chunk_size']
        )
        
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
        
    def setup_training(self):
        """Setup optimizer and scheduler"""
        logger.info("⚙️ Setting up training components...")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
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
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.best_val_loss = float('inf')
        self.step = 0
        
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = []
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            image = batch['observation']['image_front'].to(self.device)  # [B, 3, 480, 640]
            action = batch['action'].to(self.device)  # [B, 2]
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_action = self.model(image)
            loss = self.criterion(predicted_action, action)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
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
        self.model.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                image = batch['observation']['image_front'].to(self.device)
                action = batch['action'].to(self.device)
                
                predicted_action = self.model(image)
                loss = self.criterion(predicted_action, action)
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
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
        logger.info(f"📊 Training on ALL {len(self.train_dataset)} samples from 35 episodes")
        logger.info(f"📊 Validation on {len(self.val_dataset)} samples")
        
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

def get_hybrid_config() -> Dict[str, Any]:
    """Get configuration for hybrid training with all episodes"""
    return {
        # Data settings
        'data_dir': '/home/maxboels/projects/QuantumTracer/src/imitation_learning/data',
        'output_dir': './outputs',
        
        # Dataset splits
        'train_split': 0.8,
        'batch_size': 4,  # Smaller batch size for full resolution
        'num_workers': 4,
        
        # Image settings (FULL RESOLUTION!)
        'image_width': 640,
        'image_height': 480,
        
        # Model configuration
        'hidden_dim': 512,
        'num_layers': 4,
        'num_heads': 8,
        'chunk_size': 32,
        
        # Training settings
        'num_epochs': 15,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'betas': [0.9, 0.999],
        'grad_clip': 1.0,
        
        # Logging
        'log_freq': 10,
        'save_freq': 3,
        
        # System
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
    }

def main():
    parser = argparse.ArgumentParser(description='Hybrid ACT Training with Full Resolution')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory') 
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Override device')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_hybrid_config()
    
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
    logger.info(f"🎯 Found {len(episodes)} episodes for training!")
    
    if len(episodes) == 0:
        logger.error("No episodes found!")
        return
    
    # Start training
    trainer = HybridACTTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()