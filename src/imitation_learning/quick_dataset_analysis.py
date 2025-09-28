#!/usr/bin/env python3
"""
Quick Dataset Analysis for All Episodes
Analyzes the expanded dataset before training
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add local imports
sys.path.append(str(Path(__file__).parent))
from local_dataset_loader import TracerLocalDataset

def analyze_all_episodes(data_dir: str):
    """Analyze all episodes in the dataset"""
    
    data_path = Path(data_dir)
    episodes = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('episode_')])
    
    print(f"🔍 Analyzing {len(episodes)} episodes...")
    print("=" * 60)
    
    episode_stats = []
    total_samples = 0
    all_steering = []
    all_throttle = []
    
    for episode_dir in episodes:
        episode_data_path = episode_dir / "episode_data.json"
        
        if not episode_data_path.exists():
            print(f"⚠️  No episode_data.json in {episode_dir.name}")
            continue
            
        with open(episode_data_path, 'r') as f:
            episode_data = json.load(f)
        
        # Extract metadata
        metadata = episode_data.get('metadata', {})
        duration = episode_data.get('duration', 0)
        
        frame_count = len(episode_data.get('frame_samples', []))
        control_count = len(episode_data.get('control_samples', []))
        
        # Extract control values for analysis
        control_samples = episode_data.get('control_samples', [])
        steering_values = [s['steering_normalized'] for s in control_samples]
        throttle_values = [s['throttle_normalized'] for s in control_samples]
        
        all_steering.extend(steering_values)
        all_throttle.extend(throttle_values)
        
        episode_stats.append({
            'name': episode_dir.name,
            'duration': duration,
            'frames': frame_count,
            'controls': control_count,
            'avg_frame_rate': frame_count / duration if duration > 0 else 0,
            'avg_control_rate': control_count / duration if duration > 0 else 0,
            'steering_mean': np.mean(steering_values) if steering_values else 0,
            'steering_std': np.std(steering_values) if steering_values else 0,
            'throttle_mean': np.mean(throttle_values) if throttle_values else 0,
            'throttle_std': np.std(throttle_values) if throttle_values else 0,
        })
        
        total_samples += min(frame_count, control_count)  # Synchronized samples
        
        print(f"{episode_dir.name}: {duration:.1f}s, {frame_count} frames, {control_count} controls")
    
    print("=" * 60)
    print(f"📊 DATASET SUMMARY:")
    print(f"   Total episodes: {len(episode_stats)}")
    print(f"   Total duration: {sum(s['duration'] for s in episode_stats):.1f}s")
    print(f"   Total synchronized samples: ~{total_samples}")
    print(f"   Average episode duration: {np.mean([s['duration'] for s in episode_stats]):.1f}s")
    print(f"   Average frames per episode: {np.mean([s['frames'] for s in episode_stats]):.1f}")
    
    print(f"\n🎮 CONTROL ANALYSIS:")
    print(f"   Steering range: [{np.min(all_steering):.3f}, {np.max(all_steering):.3f}]")
    print(f"   Steering mean ± std: {np.mean(all_steering):.3f} ± {np.std(all_steering):.3f}")
    print(f"   Throttle range: [{np.min(all_throttle):.3f}, {np.max(all_throttle):.3f}]") 
    print(f"   Throttle mean ± std: {np.mean(all_throttle):.3f} ± {np.std(all_throttle):.3f}")
    
    # Check data variety
    steering_unique = len(set(np.round(all_steering, 3)))
    throttle_unique = len(set(np.round(all_throttle, 3))) 
    
    print(f"\n📈 DATA VARIETY:")
    print(f"   Unique steering values: {steering_unique}")
    print(f"   Unique throttle values: {throttle_unique}")
    print(f"   Steering variety score: {steering_unique/len(all_steering)*100:.1f}%")
    print(f"   Throttle variety score: {throttle_unique/len(all_throttle)*100:.1f}%")
    
    return episode_stats, all_steering, all_throttle

def test_dataset_loader(data_dir: str):
    """Test the dataset loader with all episodes"""
    print(f"\n🧪 TESTING DATASET LOADER:")
    print("=" * 40)
    
    try:
        dataset = TracerLocalDataset(
            data_dir=data_dir,
            transforms=None,
            sync_tolerance=0.05,
            episode_length=None
        )
        
        print(f"   ✅ Dataset loaded successfully")
        print(f"   📊 Total samples: {len(dataset)}")
        
        # Test a few samples
        for i in [0, len(dataset)//2, len(dataset)-1]:
            sample = dataset[i]
            print(f"   Sample {i}: image {sample['image'].shape}, action {sample['action'].shape}")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Dataset loader error: {e}")
        return False

def main():
    data_dir = "/home/maxboels/projects/QuantumTracer/src/imitation_learning/data"
    
    # Analyze episodes
    episode_stats, all_steering, all_throttle = analyze_all_episodes(data_dir)
    
    # Test dataset loader
    dataset_ok = test_dataset_loader(data_dir)
    
    print(f"\n🎯 TRAINING READINESS:")
    print("=" * 30)
    if len(episode_stats) >= 10:
        print("   ✅ Sufficient episodes for training")
    else:
        print("   ⚠️  Limited episodes - may need more data")
    
    if len(set(np.round(all_steering, 2))) > 10:
        print("   ✅ Good steering variety")
    else:
        print("   ⚠️  Limited steering variety")
        
    if dataset_ok:
        print("   ✅ Dataset loader working")
    else:
        print("   ❌ Dataset loader issues")
    
    print(f"\n🚀 Ready to train with {len(episode_stats)} episodes!")

if __name__ == "__main__":
    main()