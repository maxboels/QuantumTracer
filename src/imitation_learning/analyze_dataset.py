#!/usr/bin/env python3
"""
Data Analysis and Visualization Tool for Tracer RC Car Dataset
Analyzes recorded episodes and visualizes training data
"""

import os
import sys
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import seaborn as sns
from typing import List, Dict, Any

from local_dataset_loader import TracerLocalDataset

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    """Analyzer for Tracer RC car dataset"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.dataset = TracerLocalDataset(data_dir)
        
    def analyze_episodes(self) -> Dict[str, Any]:
        """Analyze episode-level statistics"""
        logger.info("Analyzing episodes...")
        
        episodes = self.dataset.episodes
        
        episode_stats = {
            'total_episodes': len(episodes),
            'episode_durations': [],
            'total_frames': [],
            'total_controls': [],
            'avg_fps': [],
            'sync_errors': []
        }
        
        for episode in episodes:
            duration = episode.get('duration', 0)
            metadata = episode.get('metadata', {})
            
            episode_stats['episode_durations'].append(duration)
            episode_stats['total_frames'].append(metadata.get('total_frames', 0))
            episode_stats['total_controls'].append(metadata.get('total_control_samples', 0))
            episode_stats['avg_fps'].append(metadata.get('avg_frame_rate', 0))
        
        # Compute sync errors
        for sample in self.dataset.synchronized_data:
            episode_stats['sync_errors'].append(sample['time_sync_error'])
        
        # Convert to numpy for statistics
        for key in ['episode_durations', 'total_frames', 'total_controls', 'avg_fps', 'sync_errors']:
            if episode_stats[key]:
                arr = np.array(episode_stats[key])
                episode_stats[f'{key}_mean'] = float(np.mean(arr))
                episode_stats[f'{key}_std'] = float(np.std(arr))
                episode_stats[f'{key}_min'] = float(np.min(arr))
                episode_stats[f'{key}_max'] = float(np.max(arr))
        
        return episode_stats
    
    def analyze_actions(self) -> Dict[str, Any]:
        """Analyze action distribution and statistics"""
        logger.info("Analyzing actions...")
        
        steering_values = []
        throttle_values = []
        
        for sample in self.dataset.synchronized_data:
            steering_values.append(sample['steering'])
            throttle_values.append(sample['throttle'])
        
        steering_arr = np.array(steering_values)
        throttle_arr = np.array(throttle_values)
        
        action_stats = {
            'total_samples': len(steering_values),
            'steering': {
                'mean': float(np.mean(steering_arr)),
                'std': float(np.std(steering_arr)),
                'min': float(np.min(steering_arr)),
                'max': float(np.max(steering_arr)),
                'median': float(np.median(steering_arr)),
                'range': float(np.ptp(steering_arr))
            },
            'throttle': {
                'mean': float(np.mean(throttle_arr)),
                'std': float(np.std(throttle_arr)),
                'min': float(np.min(throttle_arr)),
                'max': float(np.max(throttle_arr)),
                'median': float(np.median(throttle_arr)),
                'range': float(np.ptp(throttle_arr))
            }
        }
        
        return action_stats, steering_arr, throttle_arr
    
    def create_visualizations(self, output_dir: str = "./analysis_output"):
        """Create comprehensive visualizations of the dataset"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Creating visualizations in {output_dir}")
        
        # Set style
        plt.style.use('seaborn-v0_8' if hasattr(plt, 'style') else 'default')
        sns.set_palette("husl")
        
        # Analyze data
        episode_stats = self.analyze_episodes()
        action_stats, steering_arr, throttle_arr = self.analyze_actions()
        
        # 1. Episode Overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dataset Episode Overview', fontsize=16)
        
        # Duration distribution
        axes[0, 0].hist(episode_stats['episode_durations'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Episode Duration Distribution')
        axes[0, 0].set_xlabel('Duration (seconds)')
        axes[0, 0].set_ylabel('Count')
        
        # Frame count distribution
        axes[0, 1].hist(episode_stats['total_frames'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Frames per Episode')
        axes[0, 1].set_xlabel('Frame Count')
        axes[0, 1].set_ylabel('Count')
        
        # FPS distribution
        axes[1, 0].hist(episode_stats['avg_fps'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Average FPS per Episode')
        axes[1, 0].set_xlabel('FPS')
        axes[1, 0].set_ylabel('Count')
        
        # Sync error distribution
        axes[1, 1].hist(episode_stats['sync_errors'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Timestamp Sync Errors')
        axes[1, 1].set_xlabel('Sync Error (seconds)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].axvline(x=0.05, color='r', linestyle='--', label='Tolerance')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'episode_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Action Analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Action Analysis', fontsize=16)
        
        # Steering distribution
        axes[0, 0].hist(steering_arr, bins=50, alpha=0.7, edgecolor='black', color='blue')
        axes[0, 0].set_title('Steering Distribution')
        axes[0, 0].set_xlabel('Steering Value')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].axvline(x=np.mean(steering_arr), color='r', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        # Throttle distribution  
        axes[0, 1].hist(throttle_arr, bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].set_title('Throttle Distribution')
        axes[0, 1].set_xlabel('Throttle Value')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].axvline(x=np.mean(throttle_arr), color='r', linestyle='--', label='Mean')
        axes[0, 1].legend()
        
        # Steering vs Throttle scatter
        axes[0, 2].scatter(steering_arr, throttle_arr, alpha=0.5, s=1)
        axes[0, 2].set_title('Steering vs Throttle')
        axes[0, 2].set_xlabel('Steering')
        axes[0, 2].set_ylabel('Throttle')
        
        # Time series for first episode (if available)
        if len(self.dataset.synchronized_data) > 100:
            first_episode_data = [s for s in self.dataset.synchronized_data 
                                if s['episode_id'] == self.dataset.synchronized_data[0]['episode_id']]
            
            if first_episode_data:
                times = [(s['timestamp'] - first_episode_data[0]['timestamp']) 
                        for s in first_episode_data]
                steering_ts = [s['steering'] for s in first_episode_data]
                throttle_ts = [s['throttle'] for s in first_episode_data]
                
                axes[1, 0].plot(times, steering_ts, label='Steering', alpha=0.8)
                axes[1, 0].set_title('Steering Time Series (First Episode)')
                axes[1, 0].set_xlabel('Time (seconds)')
                axes[1, 0].set_ylabel('Steering Value')
                
                axes[1, 1].plot(times, throttle_ts, label='Throttle', color='green', alpha=0.8)
                axes[1, 1].set_title('Throttle Time Series (First Episode)')
                axes[1, 1].set_xlabel('Time (seconds)')
                axes[1, 1].set_ylabel('Throttle Value')
                
                axes[1, 2].plot(times, steering_ts, label='Steering', alpha=0.8)
                axes[1, 2].plot(times, throttle_ts, label='Throttle', alpha=0.8)
                axes[1, 2].set_title('Combined Time Series (First Episode)')
                axes[1, 2].set_xlabel('Time (seconds)')
                axes[1, 2].set_ylabel('Value')
                axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'action_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Sample Images
        self.create_sample_visualization(output_dir)
        
        # 4. Save statistics
        stats_summary = {
            'episode_stats': episode_stats,
            'action_stats': action_stats
        }
        
        with open(output_dir / 'dataset_statistics.json', 'w') as f:
            json.dump(stats_summary, f, indent=2)
        
        logger.info(f"Visualizations saved to {output_dir}")
        
        return stats_summary
    
    def create_sample_visualization(self, output_dir: Path):
        """Create visualization showing sample images with their corresponding actions"""
        logger.info("Creating sample image visualization...")
        
        # Select diverse samples
        indices = np.linspace(0, len(self.dataset) - 1, 12, dtype=int)
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('Sample Images with Actions', fontsize=16)
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(indices):
                sample_idx = indices[idx]
                try:
                    sample = self.dataset[sample_idx]
                    
                    # Get image
                    image = sample['observation']['image_front']
                    if isinstance(image, torch.Tensor):
                        # Convert from tensor to numpy
                        image = image.permute(1, 2, 0).cpu().numpy()
                        # Denormalize if needed
                        if image.min() < 0:  # Assume normalized
                            mean = np.array([0.485, 0.456, 0.406])
                            std = np.array([0.229, 0.224, 0.225])
                            image = image * std + mean
                        image = np.clip(image, 0, 1)
                    
                    ax.imshow(image)
                    
                    # Add action info
                    action = sample['action']
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    
                    steering, throttle = action[0], action[1]
                    ax.set_title(f'S: {steering:.3f}, T: {throttle:.3f}', fontsize=10)
                    ax.axis('off')
                    
                except Exception as e:
                    logger.error(f"Error visualizing sample {sample_idx}: {e}")
                    ax.text(0.5, 0.5, 'Error loading\nimage', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sample_images.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """Print dataset summary"""
        episode_stats = self.analyze_episodes()
        action_stats, _, _ = self.analyze_actions()
        
        print("\n" + "="*60)
        print("TRACER RC CAR DATASET ANALYSIS")
        print("="*60)
        
        print(f"\n📊 EPISODE STATISTICS:")
        print(f"  Total Episodes: {episode_stats['total_episodes']}")
        print(f"  Total Synchronized Samples: {action_stats['total_samples']}")
        print(f"  Average Episode Duration: {episode_stats.get('episode_durations_mean', 0):.2f}s")
        print(f"  Total Recording Time: {sum(episode_stats['episode_durations']):.2f}s")
        print(f"  Average FPS: {episode_stats.get('avg_fps_mean', 0):.1f}")
        
        print(f"\n🎮 ACTION STATISTICS:")
        print(f"  Steering Range: [{action_stats['steering']['min']:.3f}, {action_stats['steering']['max']:.3f}]")
        print(f"  Steering Mean±Std: {action_stats['steering']['mean']:.3f}±{action_stats['steering']['std']:.3f}")
        print(f"  Throttle Range: [{action_stats['throttle']['min']:.3f}, {action_stats['throttle']['max']:.3f}]")
        print(f"  Throttle Mean±Std: {action_stats['throttle']['mean']:.3f}±{action_stats['throttle']['std']:.3f}")
        
        print(f"\n⏱️  SYNCHRONIZATION:")
        print(f"  Average Sync Error: {episode_stats.get('sync_errors_mean', 0)*1000:.2f}ms")
        print(f"  Max Sync Error: {episode_stats.get('sync_errors_max', 0)*1000:.2f}ms")
        
        print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Analyze Tracer RC car dataset')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/maxboels/projects/QuantumTracer/src/imitation_learning/data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./analysis_output',
                       help='Output directory for visualizations')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create visualizations')
    parser.add_argument('--summary_only', action='store_true',
                       help='Only print summary without creating visualizations')
    
    args = parser.parse_args()
    
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory {args.data_dir} not found")
        return 1
    
    try:
        # Create analyzer
        analyzer = DatasetAnalyzer(args.data_dir)
        
        # Print summary
        analyzer.print_summary()
        
        # Create visualizations unless summary only
        if not args.summary_only:
            if args.visualize or True:  # Default to creating visualizations
                analyzer.create_visualizations(args.output_dir)
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())