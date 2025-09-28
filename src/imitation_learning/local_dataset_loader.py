#!/usr/bin/env python3
"""
Local Dataset Loader for Tracer RC Car Imitation Learning
Loads locally recorded episodes and converts them to LeRobot compatible format
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TracerLocalDataset(Dataset):
    """Dataset class for locally recorded Tracer RC car episodes"""
    
    def __init__(
        self,
        data_dir: str,
        transforms=None,
        sync_tolerance: float = 0.05,  # 50ms tolerance for timestamp matching
        episode_length: int = None,  # Optional: limit episode length
    ):
        """
        Initialize the Tracer local dataset
        
        Args:
            data_dir: Path to directory containing episode folders
            transforms: Image transforms to apply
            sync_tolerance: Tolerance in seconds for timestamp synchronization
            episode_length: Optional limit on episode length for training
        """
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.sync_tolerance = sync_tolerance
        self.episode_length = episode_length
        
        # Load all episodes
        self.episodes = self._load_episodes()
        self.synchronized_data = self._synchronize_all_episodes()
        
        logger.info(f"Loaded {len(self.episodes)} episodes with {len(self.synchronized_data)} synchronized samples")
    
    def _load_episodes(self) -> List[Dict]:
        """Load all episode data from the data directory"""
        episodes = []
        
        for episode_dir in self.data_dir.iterdir():
            if not episode_dir.is_dir():
                continue
                
            episode_data_path = episode_dir / "episode_data.json"
            if not episode_data_path.exists():
                logger.warning(f"No episode_data.json found in {episode_dir}")
                continue
            
            try:
                with open(episode_data_path, 'r') as f:
                    episode_data = json.load(f)
                
                # Add episode directory path
                episode_data['episode_dir'] = episode_dir
                episodes.append(episode_data)
                
            except Exception as e:
                logger.error(f"Error loading episode {episode_dir}: {e}")
                continue
        
        return episodes
    
    def _synchronize_episode(self, episode: Dict) -> List[Dict]:
        """Synchronize frames and controls for a single episode"""
        synchronized_samples = []
        
        try:
            # Extract frame samples with proper timestamps
            frame_samples = []
            for frame_sample in episode.get('frame_samples', []):
                frame_samples.append({
                    'timestamp': frame_sample['timestamp'],
                    'frame_id': frame_sample['frame_id'],
                    'image_path': episode['episode_dir'] / frame_sample['image_path']
                })
            
            # Extract control samples
            control_samples = episode.get('control_samples', [])
            
            if not frame_samples or not control_samples:
                logger.warning(f"Episode {episode['episode_id']} has no frames or controls")
                return []
            
            # Sort by timestamp
            frame_samples.sort(key=lambda x: x['timestamp'])
            control_samples.sort(key=lambda x: x['system_timestamp'])
            
            # For each frame, find the closest control sample
            for frame_sample in frame_samples:
                frame_time = frame_sample['timestamp']
                
                # Find closest control sample
                best_control = None
                min_time_diff = float('inf')
                
                for control_sample in control_samples:
                    time_diff = abs(control_sample['system_timestamp'] - frame_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_control = control_sample
                
                # Only include if within tolerance
                if best_control and min_time_diff <= self.sync_tolerance:
                    synchronized_sample = {
                        'episode_id': episode['episode_id'],
                        'frame_id': frame_sample['frame_id'],
                        'image_path': frame_sample['image_path'],
                        'timestamp': frame_time,
                        'steering': best_control['steering_normalized'],
                        'throttle': best_control['throttle_normalized'],
                        'time_sync_error': min_time_diff
                    }
                    synchronized_samples.append(synchronized_sample)
            
            # Optionally limit episode length
            if self.episode_length and len(synchronized_samples) > self.episode_length:
                synchronized_samples = synchronized_samples[:self.episode_length]
            
        except Exception as e:
            logger.error(f"Error synchronizing episode {episode.get('episode_id', 'unknown')}: {e}")
        
        return synchronized_samples
    
    def _synchronize_all_episodes(self) -> List[Dict]:
        """Synchronize frames and controls for all episodes"""
        all_synchronized = []
        
        for episode in self.episodes:
            episode_sync = self._synchronize_episode(episode)
            all_synchronized.extend(episode_sync)
            logger.info(f"Episode {episode['episode_id']}: {len(episode_sync)} synchronized samples")
        
        return all_synchronized
    
    def __len__(self) -> int:
        return len(self.synchronized_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset"""
        sample = self.synchronized_data[idx]
        
        try:
            # Load and process image
            image = Image.open(sample['image_path']).convert('RGB')
            
            if self.transforms:
                image = self.transforms(image)
            else:
                # Default transforms: resize and normalize
                image = image.resize((640, 480))
                image = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
                image = image.permute(2, 0, 1)  # HWC to CHW
            
            # Prepare control actions
            action = torch.tensor([
                sample['steering'],
                sample['throttle']
            ], dtype=torch.float32)
            
            # Create observation dict matching LeRobot format
            observation = {
                'image_front': image,
                'state': action,  # Current state (for state-action pairs)
            }
            
            return {
                'observation': observation,
                'action': action,
                'episode_id': sample['episode_id'],
                'frame_id': sample['frame_id'],
                'timestamp': sample['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return a dummy sample to avoid training interruption
            dummy_image = torch.zeros(3, 480, 640, dtype=torch.float32)
            dummy_action = torch.zeros(2, dtype=torch.float32)
            return {
                'observation': {
                    'image_front': dummy_image,
                    'state': dummy_action,
                },
                'action': dummy_action,
                'episode_id': 'dummy',
                'frame_id': 0,
                'timestamp': 0.0
            }

def create_data_loaders(
    data_dir: str,
    batch_size: int = 8,
    train_split: float = 0.8,
    val_split: float = 0.2,
    num_workers: int = 4,
    transforms=None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation data loaders"""
    
    # Create dataset
    full_dataset = TracerLocalDataset(data_dir, transforms=transforms)
    
    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def compute_dataset_stats(dataset: TracerLocalDataset) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compute normalization statistics for the dataset"""
    logger.info("Computing dataset statistics...")
    
    # Collect all actions for statistics
    all_actions = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        all_actions.append(sample['action'].numpy())
    
    all_actions = np.array(all_actions)
    
    # Compute mean and std for actions
    action_mean = torch.tensor(np.mean(all_actions, axis=0), dtype=torch.float32)
    action_std = torch.tensor(np.std(all_actions, axis=0), dtype=torch.float32)
    
    # For images, we'll use standard ImageNet stats
    image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    
    stats = {
        'observation.image_front': {
            'mean': image_mean,
            'std': image_std
        },
        'observation.state': {
            'mean': action_mean,
            'std': action_std
        },
        'action': {
            'mean': action_mean,
            'std': action_std
        }
    }
    
    logger.info(f"Action statistics - Mean: {action_mean}, Std: {action_std}")
    
    return stats

if __name__ == "__main__":
    # Test the dataset loader
    data_dir = "/home/maxboels/projects/QuantumTracer/src/imitation_learning/data"
    
    # Create dataset
    dataset = TracerLocalDataset(data_dir)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['observation']['image_front'].shape}")
    print(f"Action shape: {sample['action'].shape}")
    print(f"Action values: {sample['action']}")
    
    # Compute stats
    stats = compute_dataset_stats(dataset)
    print(f"Dataset statistics computed: {list(stats.keys())}")