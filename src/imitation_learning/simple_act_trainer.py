#!/usr/bin/env python3
"""
Simplified ACT Training Script for Tracer RC Car
Uses only PyTorch components to avoid LeRobot configuration complexity
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

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
    logging.warning("TensorBoard not available. Training will proceed without TensorBoard logging.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedTracerDataset(Dataset):
    """Simplified dataset for Tracer RC car that works with basic PyTorch"""
    
    def __init__(self, data_dir: str, transforms=None, episode_length: int = None):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.episode_length = episode_length
        
        # Load all synchronized samples
        self.samples = self._load_all_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def _load_all_samples(self):
        """Load all synchronized frame-control pairs from ALL episodes"""
        samples = []
        
        # Get all episode directories and sort them for consistent ordering
        episode_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and (d / "episode_data.json").exists()])
        
        logger.info(f"Found {len(episode_dirs)} episodes to load")
        
        for episode_dir in episode_dirs:
            episode_data_path = episode_dir / "episode_data.json"
            
            try:
                with open(episode_data_path, 'r') as f:
                    episode_data = json.load(f)
                
                # Extract frame samples
                frame_samples = {
                    fs['frame_id']: {
                        'timestamp': fs['timestamp'],
                        'image_path': episode_dir / fs['image_path']
                    }
                    for fs in episode_data.get('frame_samples', [])
                }
                
                # Extract control samples
                control_samples = episode_data.get('control_samples', [])
                
                # Match frames with controls
                episode_samples = []
                for control in control_samples:
                    # Find the closest frame
                    best_frame_id = None
                    min_time_diff = float('inf')
                    
                    for frame_id, frame_data in frame_samples.items():
                        time_diff = abs(frame_data['timestamp'] - control['system_timestamp'])
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            best_frame_id = frame_id
                    
                    # Only include if sync is reasonable (within 50ms)
                    if best_frame_id is not None and min_time_diff < 0.05:
                        sample = {
                            'image_path': frame_samples[best_frame_id]['image_path'],
                            'steering': control['steering_normalized'],
                            'throttle': control['throttle_normalized'],
                            'timestamp': control['system_timestamp']
                        }
                        episode_samples.append(sample)
                
                # Limit episode length if specified
                if self.episode_length:
                    episode_samples = episode_samples[:self.episode_length]
                
                samples.extend(episode_samples)
                logger.info(f"Episode {episode_data['episode_id']}: {len(episode_samples)} samples")
                
            except Exception as e:
                logger.error(f"Error loading episode {episode_dir}: {e}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            if self.transforms:
                image = self.transforms(image)
        except Exception as e:
            logger.error(f"Error loading image {sample['image_path']}: {e}")
            # Create dummy image
            image = torch.zeros(3, 480, 640, dtype=torch.float32)
        
        # Action vector [steering, throttle]
        action = torch.tensor([sample['steering'], sample['throttle']], dtype=torch.float32)
        
        return image, action

class SimpleACTModel(nn.Module):
    """Simplified ACT-like model using only PyTorch"""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (480, 640),  # Full resolution for better quality
        action_dim: int = 2,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        chunk_size: int = 32
    ):
        super().__init__()
        
        # Vision encoder (Full resolution CNN)
        self.vision_encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, 7, stride=2, padding=3),  # 480x640 -> 240x320
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),      # 240x320 -> 120x160
            
            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # 120x160 -> 60x80
            
            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # 60x80 -> 30x40
            
            # Fourth conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 10)),             # -> 8x10 for full res
            
            nn.Flatten(),
            nn.Linear(256 * 8 * 10, hidden_dim),       # Adjusted for full res
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Transformer encoder for processing visual features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'  # More efficient activation
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Action decoder with residual connection
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, action_dim * chunk_size)
        )
        
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        
    def forward(self, images):
        batch_size = images.shape[0]
        
        # Encode images
        visual_features = self.vision_encoder(images)  # [batch_size, hidden_dim]
        
        # Add sequence dimension for transformer (we use single timestep)
        visual_features = visual_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Process through transformer
        encoded_features = self.transformer_encoder(visual_features)  # [batch_size, 1, hidden_dim]
        
        # Decode actions
        action_logits = self.action_head(encoded_features[:, 0])  # [batch_size, action_dim * chunk_size]
        
        # Reshape to action chunks
        actions = action_logits.view(batch_size, self.chunk_size, self.action_dim)
        
        # For training, we typically just use the first action
        return actions[:, 0]  # [batch_size, action_dim]

class SimpleACTTrainer:
    """Simplified trainer for ACT model"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard logging (if available)
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
            logger.info(f"TensorBoard logs will be saved to: {self.output_dir / 'tensorboard'}")
        else:
            self.writer = None
            logger.info("TensorBoard not available - skipping tensorboard logging")
        
        # Setup dataset and data loaders
        self.setup_dataset()
        
        # Setup model
        self.setup_model()
        
        # Setup training components
        self.setup_training()
    
    def setup_dataset(self):
        """Setup dataset and data loaders"""
        logger.info("Setting up dataset...")
        
        # Image transforms with light augmentation (full resolution)
        transform = transforms.Compose([
            transforms.Resize((480, 640)),  # Full resolution for better quality
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create full dataset
        full_dataset = SimplifiedTracerDataset(
            self.config['data_dir'],
            transforms=transform,
            episode_length=self.config.get('episode_length')
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
            pin_memory=True if self.config['num_workers'] > 0 else False,
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
        logger.info(f"Using full resolution: 480x640 for high-quality training")
    
    def setup_model(self):
        """Setup the model"""
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
        logger.info(f"Model optimized for edge deployment on Raspberry Pi 5 with AI HAT")
    
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
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.global_step = 0
    
    def train_epoch(self):
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
            
            epoch_losses.append(loss.item())
            
            # Log to TensorBoard (if available)
            if self.writer:
                self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            if batch_idx % self.config['log_freq'] == 0:
                logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
            
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
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
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
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log to TensorBoard (if available)
            if self.writer:
                self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
                self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
                self.writer.add_scalar('Train/LearningRate', self.scheduler.get_last_lr()[0], epoch)
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                if self.writer:
                    self.writer.add_scalar('Val/BestLoss', self.best_val_loss, epoch)
            
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
        
        # Close TensorBoard writer (if available)
        if self.writer:
            self.writer.close()
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        if TENSORBOARD_AVAILABLE:
            logger.info(f"TensorBoard logs saved to: {self.output_dir / 'tensorboard'}")
            logger.info("View training progress with: tensorboard --logdir <output_dir>/tensorboard")

def get_default_config():
    """Get default configuration optimized for full resolution GPU training"""
    return {
        # Data
        'data_dir': '/home/maxboels/projects/QuantumTracer/src/imitation_learning/data',
        'batch_size': 8,   # Reduced for full resolution (more memory usage)
        'num_workers': 6,  # Good balance for GPU
        'episode_length': None,  # Use full episodes (no length limit)
        
        # Model (optimized for full resolution)
        'hidden_dim': 512,    # Increased for full resolution
        'num_layers': 4,      # Good depth for learning
        'num_heads': 8,
        'chunk_size': 32,     # Standard chunk size
        
        # Training
        'max_epochs': 100,
        'learning_rate': 1e-4,  # Conservative for stability
        'weight_decay': 1e-4,
        'log_freq': 10,
        'save_freq': 10,
        
        # Hardware
        'device': 'cuda',
        
        # Output
        'output_dir': f'./outputs/full_res_act_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }

def main():
    parser = argparse.ArgumentParser(description='Train simplified ACT policy for Tracer RC car')
    parser.add_argument('--data_dir', type=str, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum epochs')
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
    
    # Create trainer and start training
    trainer = SimpleACTTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()