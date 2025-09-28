#!/usr/bin/env python3
"""
Simple Dataset Analysis for All Episodes
Basic analysis without external dependencies
"""

import os
import json
from pathlib import Path

def analyze_episodes(data_dir: str):
    """Analyze all episodes in the dataset"""
    
    data_path = Path(data_dir)
    episodes = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('episode_')])
    
    print(f"🔍 Analyzing {len(episodes)} episodes...")
    print("=" * 80)
    
    total_duration = 0
    total_frames = 0
    total_controls = 0
    all_steering = []
    all_throttle = []
    
    for i, episode_dir in enumerate(episodes, 1):
        episode_data_path = episode_dir / "episode_data.json"
        
        if not episode_data_path.exists():
            print(f"⚠️  Episode {i:2d}: {episode_dir.name} - No episode_data.json")
            continue
            
        with open(episode_data_path, 'r') as f:
            episode_data = json.load(f)
        
        # Extract basic info
        duration = episode_data.get('duration', 0)
        frame_count = len(episode_data.get('frame_samples', []))
        control_count = len(episode_data.get('control_samples', []))
        
        # Extract control values
        control_samples = episode_data.get('control_samples', [])
        steering_values = [s['steering_normalized'] for s in control_samples]
        throttle_values = [s['throttle_normalized'] for s in control_samples]
        
        all_steering.extend(steering_values)
        all_throttle.extend(throttle_values)
        
        total_duration += duration
        total_frames += frame_count
        total_controls += control_count
        
        # Calculate rates
        frame_rate = frame_count / duration if duration > 0 else 0
        control_rate = control_count / duration if duration > 0 else 0
        
        # Calculate steering/throttle stats
        if steering_values:
            steering_min = min(steering_values)
            steering_max = max(steering_values)
            steering_avg = sum(steering_values) / len(steering_values)
        else:
            steering_min = steering_max = steering_avg = 0
            
        if throttle_values:
            throttle_min = min(throttle_values)
            throttle_max = max(throttle_values)
            throttle_avg = sum(throttle_values) / len(throttle_values)
        else:
            throttle_min = throttle_max = throttle_avg = 0
        
        print(f"Episode {i:2d}: {episode_dir.name}")
        print(f"   Duration: {duration:6.1f}s | Frames: {frame_count:4d} ({frame_rate:5.1f} fps) | Controls: {control_count:4d} ({control_rate:5.1f} hz)")
        print(f"   Steering: [{steering_min:6.3f}, {steering_max:6.3f}] avg={steering_avg:6.3f}")
        print(f"   Throttle: [{throttle_min:6.3f}, {throttle_max:6.3f}] avg={throttle_avg:6.3f}")
        print()
    
    print("=" * 80)
    print(f"📊 DATASET SUMMARY:")
    print(f"   Episodes: {len(episodes)}")
    print(f"   Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"   Total frames: {total_frames}")
    print(f"   Total controls: {total_controls}")
    print(f"   Average episode: {total_duration/len(episodes):.1f}s, {total_frames//len(episodes)} frames")
    
    if all_steering:
        steering_min = min(all_steering)
        steering_max = max(all_steering)
        steering_avg = sum(all_steering) / len(all_steering)
        steering_range = steering_max - steering_min
        
        print(f"\n🎮 STEERING ANALYSIS:")
        print(f"   Range: [{steering_min:.3f}, {steering_max:.3f}] (span: {steering_range:.3f})")
        print(f"   Average: {steering_avg:.3f}")
        print(f"   Unique values: {len(set([round(s, 3) for s in all_steering]))}")
    
    if all_throttle:
        throttle_min = min(all_throttle)
        throttle_max = max(all_throttle)
        throttle_avg = sum(all_throttle) / len(all_throttle)
        throttle_range = throttle_max - throttle_min
        
        print(f"\n⚡ THROTTLE ANALYSIS:")
        print(f"   Range: [{throttle_min:.3f}, {throttle_max:.3f}] (span: {throttle_range:.3f})")
        print(f"   Average: {throttle_avg:.3f}")
        print(f"   Unique values: {len(set([round(t, 3) for t in all_throttle]))}")
    
    # Estimate training samples
    estimated_samples = min(total_frames, total_controls)
    train_samples = int(estimated_samples * 0.8)
    val_samples = estimated_samples - train_samples
    
    print(f"\n🚀 TRAINING ESTIMATES:")
    print(f"   Estimated synchronized samples: ~{estimated_samples}")
    print(f"   Training samples (~80%): ~{train_samples}")
    print(f"   Validation samples (~20%): ~{val_samples}")
    
    # Quality assessment
    print(f"\n✅ DATASET QUALITY:")
    if len(episodes) >= 20:
        print(f"   Episodes: Excellent ({len(episodes)} episodes)")
    elif len(episodes) >= 10:
        print(f"   Episodes: Good ({len(episodes)} episodes)")
    else:
        print(f"   Episodes: Fair ({len(episodes)} episodes - could use more)")
    
    if total_duration >= 300:  # 5 minutes
        print(f"   Duration: Excellent ({total_duration/60:.1f} minutes)")
    elif total_duration >= 120:  # 2 minutes
        print(f"   Duration: Good ({total_duration/60:.1f} minutes)")
    else:
        print(f"   Duration: Fair ({total_duration/60:.1f} minutes - could use more)")
    
    if all_steering and len(set([round(s, 2) for s in all_steering])) > 20:
        print("   Steering variety: Good")
    else:
        print("   Steering variety: Limited - check for diverse driving scenarios")
    
    return len(episodes), estimated_samples

def main():
    data_dir = "/home/maxboels/projects/QuantumTracer/src/imitation_learning/data"
    
    if not Path(data_dir).exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    num_episodes, num_samples = analyze_episodes(data_dir)
    
    print(f"\n🎯 READY FOR TRAINING!")
    print(f"   Run: python official_lerobot_trainer.py")
    print(f"   Expected training time: ~{num_episodes * 2} minutes")

if __name__ == "__main__":
    main()