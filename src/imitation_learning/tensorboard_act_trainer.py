#!/usr/bin/env python3
"""
TensorBoard ACT Trainer for Tracer RC Car
Enhanced training script with comprehensive TensorBoard logging and automatic episode detection
Optimized for training on all available episodes with real-time loss monitoring
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

# TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
    print("WARNING: TensorBoard not available. Please install tensorboard: pip install tensorboard")

# Import from existing scripts
from simple_act_trainer import SimplifiedTracerDataset, SimpleACTModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TensorBoardACTTrainer:
    """Enhanced ACT trainer with comprehensive TensorBoard logging"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_dir']) / f"tensorboard_act_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(str(self.output_dir / 'tensorboard_logs'))
            logger.info(f"TensorBoard logging enabled. Run: tensorboard --logdir {self.output_dir / 'tensorboard_logs'}")
        else:
            self.writer = None
            logger.warning("TensorBoard not available. Training will proceed without TensorBoard logging.")
        
        # Initialize components
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def setup_dataset(self):
        """Setup dataset with automatic episode detection"""
        logger.info("Setting up dataset with automatic episode detection...")
        
        data_dir = self.config['data_dir']
        data_path = Path(data_dir)
        episodes = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('episode_')])
        
        logger.info(f"Found {len(episodes)} episodes for training:")
        for i, episode in enumerate(episodes):
            logger.info(f"  {i+1:2d}. {episode.name}")
        
        # Create dataset
        full_dataset = SimplifiedTracerDataset(
            data_dir=data_dir,
            transforms=transforms.Compose([
                transforms.Resize((self.config['image_height'], self.config['image_width'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            episode_length=self.config.get('episode_length', None)
        )
        
        logger.info(f"Total synchronized samples: {len(full_dataset)}")
        
        # Split dataset
        train_size = int(self.config['train_split'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Log dataset info to TensorBoard
        if self.writer:
            self.writer.add_scalar('Dataset/Total_Episodes', len(episodes), 0)
            self.writer.add_scalar('Dataset/Total_Samples', len(full_dataset), 0)
            self.writer.add_scalar('Dataset/Train_Samples', len(train_dataset), 0)
            self.writer.add_scalar('Dataset/Val_Samples', len(val_dataset), 0)
            
    def setup_model(self):
        """Setup model, optimizer, and loss function"""
        logger.info("Setting up model architecture...")
        
        # Create model
        self.model = SimpleACTModel(
            image_size=(self.config['image_height'], self.config['image_width']),
            action_dim=2,  # steering and throttle
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            chunk_size=self.config['chunk_size']
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.95)
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs'],
            eta_min=self.config['learning_rate'] * 0.1
        )
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Log model info to TensorBoard
        if self.writer:
            self.writer.add_scalar('Model/Total_Parameters', total_params, 0)
            self.writer.add_scalar('Model/Trainable_Parameters', trainable_params, 0)
            self.writer.add_scalar('Hyperparameters/Learning_Rate', self.config['learning_rate'], 0)
            self.writer.add_scalar('Hyperparameters/Batch_Size', self.config['batch_size'], 0)
            self.writer.add_scalar('Hyperparameters/Hidden_Dim', self.config['hidden_dim'], 0)
            
    def train_epoch(self, epoch: int):
        """Train for one epoch with detailed logging"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, actions) in enumerate(self.train_loader):
            images = images.to(self.device)
            actions = actions.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions, actions)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update tracking
            batch_loss = loss.item()
            epoch_loss += batch_loss
            self.global_step += 1
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('Train/Batch_Loss', batch_loss, self.global_step)
                self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                # Log gradients periodically
                if self.global_step % 100 == 0:
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    self.writer.add_scalar('Train/Gradient_Norm', total_norm, self.global_step)
            
            # Progress logging
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch:2d}/{self.config["epochs"]:2d} '
                          f'[{batch_idx:3d}/{num_batches:3d}] '
                          f'Loss: {batch_loss:.6f} '
                          f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}')
        
        avg_epoch_loss = epoch_loss / num_batches
        self.train_losses.append(avg_epoch_loss)
        
        return avg_epoch_loss
    
    def validate_epoch(self, epoch: int):
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, actions in self.val_loader:
                images = images.to(self.device)
                actions = actions.to(self.device)
                
                predictions = self.model(images)
                loss = self.criterion(predictions, actions)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / num_batches
        self.val_losses.append(avg_val_loss)
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'global_step': self.global_step
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 New best model saved: {best_path} (val_loss: {val_loss:.6f})")
    
    def train(self):
        """Main training loop with comprehensive logging"""
        logger.info("Starting training with TensorBoard logging...")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.config['epochs']}")
        
        # Save training configuration
        config_path = self.output_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        start_time = datetime.now()
        
        try:
            for epoch in range(1, self.config['epochs'] + 1):
                epoch_start = datetime.now()
                
                # Train epoch
                train_loss = self.train_epoch(epoch)
                
                # Validate epoch
                val_loss = self.validate_epoch(epoch)
                
                # Update scheduler
                self.scheduler.step()
                
                # Check if best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                # Log epoch metrics to TensorBoard
                if self.writer:
                    self.writer.add_scalar('Train/Epoch_Loss', train_loss, epoch)
                    self.writer.add_scalar('Validation/Epoch_Loss', val_loss, epoch)
                    self.writer.add_scalar('Validation/Best_Loss', self.best_val_loss, epoch)
                    
                    # Add learning rate
                    self.writer.add_scalar('Train/Epoch_LR', self.optimizer.param_groups[0]['lr'], epoch)
                    
                    # Add timing info
                    epoch_time = (datetime.now() - epoch_start).total_seconds()
                    self.writer.add_scalar('Training/Epoch_Duration', epoch_time, epoch)
                
                # Save checkpoint
                if epoch % 5 == 0 or is_best:
                    self.save_checkpoint(epoch, val_loss, is_best)
                
                # Progress report
                epoch_time = datetime.now() - epoch_start
                logger.info(f'Epoch {epoch:2d}/{self.config["epochs"]:2d} '
                          f'Train: {train_loss:.6f} Val: {val_loss:.6f} '
                          f'Best: {self.best_val_loss:.6f} '
                          f'Time: {epoch_time.total_seconds():.1f}s')
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Final save
            self.save_checkpoint(epoch, val_loss)
            
            # Training summary
            total_time = datetime.now() - start_time
            logger.info(f"\n🎯 Training completed in {total_time}")
            logger.info(f"📊 Best validation loss: {self.best_val_loss:.6f}")
            logger.info(f"💾 Models saved to: {self.output_dir}")
            
            if self.writer:
                # Log final summary
                self.writer.add_text('Training/Summary', 
                    f"Total time: {total_time}\n"
                    f"Best validation loss: {self.best_val_loss:.6f}\n"
                    f"Total episodes: {len(list(Path(self.config['data_dir']).glob('episode_*')))}\n"
                    f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
                
                self.writer.close()
                logger.info(f"📈 TensorBoard logs saved to: {self.output_dir / 'tensorboard_logs'}")
                logger.info("🔍 View training progress with: tensorboard --logdir " + 
                          str(self.output_dir / 'tensorboard_logs'))

def get_tensorboard_config() -> Dict[str, Any]:
    """Get optimized configuration for TensorBoard training"""
    return {
        # Data settings
        'data_dir': '/home/maxboels/projects/QuantumTracer/src/imitation_learning/data',
        'output_dir': './outputs',
        
        # Model architecture
        'image_width': 640,
        'image_height': 480,
        'hidden_dim': 512,
        'num_layers': 4,
        'num_heads': 8,
        'chunk_size': 16,
        
        # Training settings
        'batch_size': 4,  # Adjusted for larger dataset
        'epochs': 20,     # More epochs for larger dataset
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'train_split': 0.8,
        
        # System settings
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Optional settings
        'episode_length': None,  # Use all available data
    }

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ACT model with TensorBoard logging')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/maxboels/projects/QuantumTracer/src/imitation_learning/data',
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Get base configuration
    config = get_tensorboard_config()
    
    # Override with command line arguments
    config.update({
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
    })
    
    if args.device != 'auto':
        config['device'] = args.device
    
    # Create trainer and run
    trainer = TensorBoardACTTrainer(config)
    trainer.setup_dataset()
    trainer.setup_model()
    trainer.train()

if __name__ == "__main__":
    main()