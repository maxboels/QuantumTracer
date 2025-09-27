#!/usr/bin/env python3
"""
Enhanced ACT Training Script with CSV Logging (TensorBoard alternative)
Full resolution training with automatic episode detection and progress tracking
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from datetime import datetime
import json
from PIL import Image
import csv

# Import from the existing script
from simple_act_trainer import SimplifiedTracerDataset, SimpleACTModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVLogger:
    """Simple CSV logger as TensorBoard alternative"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.batch_log_path = self.log_dir / 'batch_metrics.csv'
        self.epoch_log_path = self.log_dir / 'epoch_metrics.csv'
        
        # Write headers
        with open(self.batch_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'batch_loss', 'learning_rate', 'timestamp'])
        
        with open(self.epoch_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'best_val_loss', 'learning_rate', 'timestamp'])
    
    def log_batch(self, step: int, batch_loss: float, learning_rate: float):
        """Log batch metrics"""
        with open(self.batch_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, batch_loss, learning_rate, datetime.now().isoformat()])
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, best_val_loss: float, learning_rate: float):
        """Log epoch metrics"""
        with open(self.epoch_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, best_val_loss, learning_rate, datetime.now().isoformat()])

class EnhancedACTTrainer:
    """Enhanced trainer with CSV logging and full episode support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup CSV logging
        self.csv_logger = CSVLogger(self.output_dir / 'logs')
        logger.info(f"CSV logs will be saved to: {self.output_dir / 'logs'}")
        
        # Save config
        with open(self.output_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Setup dataset and data loaders
        self.setup_dataset()
        
        # Setup model
        self.setup_model()
        
        # Setup training components
        self.setup_training()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def setup_dataset(self):
        """Setup dataset with automatic episode detection"""
        logger.info("Setting up dataset with automatic episode detection...")
        
        # Find all episodes in data directory
        data_path = Path(self.config['data_dir'])
        episode_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and (d / "episode_data.json").exists()])
        
        if not episode_dirs:
            raise RuntimeError(f"No valid episodes found in {data_path}")
        
        logger.info(f"Found {len(episode_dirs)} episodes:")
        for episode_dir in episode_dirs:
            logger.info(f"  - {episode_dir.name}")
        
        # Image transforms for full resolution
        transform = transforms.Compose([
            transforms.Resize((480, 640)),  # Full resolution
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset
        full_dataset = SimplifiedTracerDataset(
            self.config['data_dir'],
            transforms=transform,
            episode_length=self.config.get('episode_length')  # Use all samples if None
        )
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
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
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
        
        logger.info(f"Dataset split: {len(self.train_dataset)} train, {len(self.val_dataset)} validation")
        logger.info(f"Using full resolution: 480x640 pixels")
    
    def setup_model(self):
        """Setup the model for full resolution"""
        self.model = SimpleACTModel(
            image_size=(480, 640),  # Full resolution
            action_dim=2,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            chunk_size=self.config['chunk_size']
        ).to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        logger.info(f"Model architecture optimized for full resolution training")
    
    def setup_training(self):
        """Setup training components"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['max_epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        self.criterion = nn.MSELoss()
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, (images, actions) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predicted_actions = self.model(images)
            loss = self.criterion(predicted_actions, actions)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Log metrics
            epoch_losses.append(loss.item())
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to CSV
            self.csv_logger.log_batch(self.global_step, loss.item(), current_lr)
            
            if batch_idx % self.config['log_freq'] == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}, LR: {current_lr:.2e}")
            
            self.global_step += 1
        
        return np.mean(epoch_losses)
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = []
        
        with torch.no_grad():
            for images, actions in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                actions = actions.to(self.device, non_blocking=True)
                
                predicted_actions = self.model(images)
                loss = self.criterion(predicted_actions, actions)
                
                epoch_losses.append(loss.item())
        
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
            'config': self.config
        }
        
        # Save regular checkpoint
        if (epoch + 1) % self.config['save_freq'] == 0:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path, _use_new_zipfile_serialization=False)
            logger.info(f"✅ New best model saved! Validation loss: {self.best_val_loss:.6f}")
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting enhanced training with full resolution...")
        logger.info(f"Training for {self.config['max_epochs']} epochs")
        logger.info(f"Batch size: {self.config['batch_size']}, Learning rate: {self.config['learning_rate']}")
        
        for epoch in range(self.config['max_epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step()
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Log epoch metrics
            current_lr = self.scheduler.get_last_lr()[0]
            self.csv_logger.log_epoch(epoch, train_loss, val_loss, self.best_val_loss, current_lr)
            
            # Log to console
            logger.info(f"{'='*60}")
            logger.info(f"Epoch {epoch+1}/{self.config['max_epochs']} Summary:")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss:   {val_loss:.6f}")
            logger.info(f"  Best Val:   {self.best_val_loss:.6f} {'(NEW!)' if is_best else ''}")
            logger.info(f"  Learn Rate: {current_lr:.2e}")
            logger.info(f"{'='*60}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
        
        logger.info("🎉 Training completed!")
        logger.info(f"📊 Final Results:")
        logger.info(f"  Best validation loss: {self.best_val_loss:.6f}")
        logger.info(f"  Training logs: {self.output_dir / 'logs'}")
        logger.info(f"  Model checkpoint: {self.output_dir / 'best_model.pth'}")

def get_enhanced_config():
    """Configuration optimized for full resolution training"""
    return {
        # Data
        'data_dir': '/home/maxboels/projects/QuantumTracer/src/imitation_learning/data',
        'batch_size': 6,      # Balanced for full resolution + GPU memory
        'num_workers': 4,     # Conservative for stability
        'episode_length': None,  # Use ALL samples from each episode
        
        # Model (full resolution optimized)
        'hidden_dim': 512,    # Good capacity for full resolution
        'num_layers': 4,      # Deep enough for complex visual patterns
        'num_heads': 8,       # Standard attention heads
        'chunk_size': 32,     # Standard chunk size
        
        # Training
        'max_epochs': 50,
        'learning_rate': 1e-4,  # Conservative for stability
        'weight_decay': 1e-4,
        'log_freq': 5,       # More frequent logging
        'save_freq': 5,      # Save every 5 epochs
        
        # Hardware
        'device': 'cuda',
        
        # Output
        'output_dir': f'./outputs/enhanced_full_res_act_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }

def main():
    parser = argparse.ArgumentParser(description='Enhanced ACT training with full resolution and CSV logging')
    parser.add_argument('--data_dir', type=str, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--max_epochs', type=int, help='Maximum epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Get default config
    config = get_enhanced_config()
    
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
    
    # Create and run trainer
    trainer = EnhancedACTTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()