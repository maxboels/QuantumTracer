#!/usr/bin/env python3
"""
Training Data Visualization Script
Plot steering and throttle signals for each episode to analyze signal variation
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime

def load_episode_data(episode_dir: Path):
    """Load episode data and extract control signals"""
    episode_data_path = episode_dir / "episode_data.json"
    
    if not episode_data_path.exists():
        return None, None, None
    
    with open(episode_data_path, 'r') as f:
        episode_data = json.load(f)
    
    # Extract control samples
    control_samples = episode_data.get('control_samples', [])
    
    if not control_samples:
        return None, None, None
    
    # Extract time, steering, and throttle
    times = []
    steering_values = []
    throttle_values = []
    
    start_time = control_samples[0]['system_timestamp']
    
    for sample in control_samples:
        times.append(sample['system_timestamp'] - start_time)  # Relative time in seconds
        steering_values.append(sample['steering_normalized'])
        throttle_values.append(sample['throttle_normalized'])
    
    return np.array(times), np.array(steering_values), np.array(throttle_values)

def plot_episode_signals(episode_dir: Path, output_dir: Path):
    """Plot steering and throttle signals for a single episode"""
    times, steering, throttle = load_episode_data(episode_dir)
    
    if times is None:
        print(f"❌ No data found for episode {episode_dir.name}")
        return None
    
    # Calculate statistics
    steering_mean = np.mean(steering)
    steering_std = np.std(steering)
    steering_range = np.max(steering) - np.min(steering)
    
    throttle_mean = np.mean(throttle)
    throttle_std = np.std(throttle)
    throttle_range = np.max(throttle) - np.min(throttle)
    
    print(f"📊 {episode_dir.name}:")
    print(f"   Steering: μ={steering_mean:.4f}, σ={steering_std:.4f}, range={steering_range:.4f}")
    print(f"   Throttle: μ={throttle_mean:.4f}, σ={throttle_std:.4f}, range={throttle_range:.4f}")
    print(f"   Duration: {times[-1]:.1f}s, {len(times)} samples")
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot steering
    ax1.plot(times, steering, 'b-', linewidth=1, label='Steering Signal')
    ax1.axhline(y=steering_mean, color='b', linestyle='--', alpha=0.7, label=f'Mean: {steering_mean:.3f}')
    ax1.fill_between(times, steering_mean - steering_std, steering_mean + steering_std, alpha=0.2, color='b', label=f'±1σ: {steering_std:.3f}')
    ax1.set_ylabel('Steering (Normalized PWM)')
    ax1.set_title(f'Episode: {episode_dir.name} - Training Signal Analysis')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(np.min(steering) - 0.05, np.max(steering) + 0.05)
    
    # Plot throttle
    ax2.plot(times, throttle, 'r-', linewidth=1, label='Throttle Signal')
    ax2.axhline(y=throttle_mean, color='r', linestyle='--', alpha=0.7, label=f'Mean: {throttle_mean:.3f}')
    ax2.fill_between(times, throttle_mean - throttle_std, throttle_mean + throttle_std, alpha=0.2, color='r', label=f'±1σ: {throttle_std:.3f}')
    ax2.set_ylabel('Throttle (Normalized PWM)')
    ax2.set_xlabel('Time (seconds)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(np.min(throttle) - 0.05, np.max(throttle) + 0.05)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f"{episode_dir.name}_signals.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📈 Plot saved: {output_file}")
    
    return {
        'episode': episode_dir.name,
        'duration': times[-1],
        'samples': len(times),
        'steering_stats': {'mean': steering_mean, 'std': steering_std, 'range': steering_range},
        'throttle_stats': {'mean': throttle_mean, 'std': throttle_std, 'range': throttle_range}
    }

def plot_combined_analysis(data_dir: Path, output_dir: Path, stats_list):
    """Create combined analysis plots"""
    
    # 1. Signal variation comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = [stat['episode'] for stat in stats_list]
    steering_stds = [stat['steering_stats']['std'] for stat in stats_list]
    throttle_stds = [stat['throttle_stats']['std'] for stat in stats_list]
    steering_ranges = [stat['steering_stats']['range'] for stat in stats_list]
    throttle_ranges = [stat['throttle_stats']['range'] for stat in stats_list]
    
    # Standard deviation comparison
    x_pos = np.arange(len(episodes))
    width = 0.35
    
    ax1.bar(x_pos - width/2, steering_stds, width, label='Steering', color='blue', alpha=0.7)
    ax1.bar(x_pos + width/2, throttle_stds, width, label='Throttle', color='red', alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Standard Deviation')
    ax1.set_title('Signal Variation (Standard Deviation)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(episodes, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Range comparison
    ax2.bar(x_pos - width/2, steering_ranges, width, label='Steering', color='blue', alpha=0.7)
    ax2.bar(x_pos + width/2, throttle_ranges, width, label='Throttle', color='red', alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Range (Max - Min)')
    ax2.set_title('Signal Range')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(episodes, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Load and plot all episodes together for comparison
    all_steering = []
    all_throttle = []
    episode_boundaries = [0]
    
    for episode_dir in sorted(data_dir.iterdir()):
        if episode_dir.is_dir() and (episode_dir / "episode_data.json").exists():
            times, steering, throttle = load_episode_data(episode_dir)
            if times is not None:
                all_steering.extend(steering)
                all_throttle.extend(throttle)
                episode_boundaries.append(len(all_steering))
    
    # Combined steering plot
    ax3.plot(all_steering, 'b-', linewidth=0.5, alpha=0.8)
    ax3.set_ylabel('Steering (Normalized)')
    ax3.set_title('All Episodes - Steering Signal')
    ax3.grid(True, alpha=0.3)
    
    # Add episode boundaries
    for boundary in episode_boundaries[1:-1]:
        ax3.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    # Combined throttle plot
    ax4.plot(all_throttle, 'r-', linewidth=0.5, alpha=0.8)
    ax4.set_ylabel('Throttle (Normalized)')
    ax4.set_xlabel('Sample Index')
    ax4.set_title('All Episodes - Throttle Signal')
    ax4.grid(True, alpha=0.3)
    
    # Add episode boundaries
    for boundary in episode_boundaries[1:-1]:
        ax4.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save combined analysis
    combined_file = output_dir / "combined_signal_analysis.png"
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Combined analysis saved: {combined_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize training signal variation for RC car episodes')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/maxboels/projects/QuantumTracer/src/imitation_learning/data',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str,
                        default='./signal_analysis',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    print("🔍 Analyzing training signal variation...")
    print(f"📁 Data directory: {data_dir}")
    print(f"📊 Output directory: {output_dir}")
    print("=" * 60)
    
    # Find all episodes
    episode_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / "episode_data.json").exists()])
    
    if not episode_dirs:
        print("❌ No valid episodes found!")
        return
    
    print(f"Found {len(episode_dirs)} episodes to analyze:")
    
    # Process each episode
    stats_list = []
    for episode_dir in episode_dirs:
        stats = plot_episode_signals(episode_dir, output_dir)
        if stats:
            stats_list.append(stats)
    
    if stats_list:
        print("\n" + "=" * 60)
        print("📈 Creating combined analysis...")
        plot_combined_analysis(data_dir, output_dir, stats_list)
        
        print("\n" + "=" * 60)
        print("✅ Analysis complete!")
        print(f"📊 Individual plots: {output_dir}/episode_*_signals.png")
        print(f"📈 Combined analysis: {output_dir}/combined_signal_analysis.png")
        
        # Summary statistics
        total_samples = sum(stat['samples'] for stat in stats_list)
        total_duration = sum(stat['duration'] for stat in stats_list)
        avg_steering_std = np.mean([stat['steering_stats']['std'] for stat in stats_list])
        avg_throttle_std = np.mean([stat['throttle_stats']['std'] for stat in stats_list])
        
        print(f"\n📋 Dataset Summary:")
        print(f"   Episodes: {len(stats_list)}")
        print(f"   Total samples: {total_samples}")
        print(f"   Total duration: {total_duration:.1f}s")
        print(f"   Avg steering variation (σ): {avg_steering_std:.4f}")
        print(f"   Avg throttle variation (σ): {avg_throttle_std:.4f}")
        
        # Training signal quality assessment
        print(f"\n🎯 Training Signal Quality Assessment:")
        if avg_steering_std > 0.01:
            print(f"   ✅ Steering: Good variation (σ={avg_steering_std:.4f})")
        else:
            print(f"   ⚠️  Steering: Low variation (σ={avg_steering_std:.4f}) - consider more diverse maneuvers")
            
        if avg_throttle_std > 0.01:
            print(f"   ✅ Throttle: Good variation (σ={avg_throttle_std:.4f})")
        else:
            print(f"   ⚠️  Throttle: Low variation (σ={avg_throttle_std:.4f}) - consider more speed changes")

if __name__ == "__main__":
    main()