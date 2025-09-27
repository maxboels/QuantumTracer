#!/usr/bin/env python3
"""
Episode Video Animation Creator
Creates animated videos showing camera frames with live steering and throttle plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from pathlib import Path
import argparse
from datetime import datetime
import cv2

def load_episode_data(episode_dir: Path):
    """Load episode data and synchronize frames with control signals"""
    episode_data_path = episode_dir / "episode_data.json"
    
    if not episode_data_path.exists():
        return None
    
    with open(episode_data_path, 'r') as f:
        episode_data = json.load(f)
    
    # Extract frame and control samples
    frame_samples = episode_data.get('frame_samples', [])
    control_samples = episode_data.get('control_samples', [])
    
    if not frame_samples or not control_samples:
        return None
    
    # Synchronize frames with controls by timestamp
    synchronized_data = []
    
    for frame_sample in frame_samples:
        frame_time = frame_sample['timestamp']
        frame_path = episode_dir / frame_sample['image_path']
        
        # Find closest control sample
        best_control = None
        min_time_diff = float('inf')
        
        for control_sample in control_samples:
            time_diff = abs(control_sample['system_timestamp'] - frame_time)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                best_control = control_sample
        
        # Only include if within reasonable sync tolerance (50ms)
        if best_control and min_time_diff < 0.05 and frame_path.exists():
            synchronized_data.append({
                'timestamp': frame_time,
                'frame_path': frame_path,
                'steering': best_control['steering_normalized'],
                'throttle': best_control['throttle_normalized'],
                'frame_id': frame_sample['frame_id']
            })
    
    # Sort by timestamp
    synchronized_data.sort(key=lambda x: x['timestamp'])
    
    print(f"📊 {episode_dir.name}: {len(synchronized_data)} synchronized frame-control pairs")
    
    return synchronized_data

def create_episode_animation(episode_dir: Path, output_dir: Path, fps: int = 10):
    """Create animated video for a single episode"""
    
    print(f"🎬 Creating animation for {episode_dir.name}...")
    
    # Load synchronized data
    sync_data = load_episode_data(episode_dir)
    if not sync_data:
        print(f"❌ No synchronized data for {episode_dir.name}")
        return None
    
    # Prepare data arrays
    n_frames = len(sync_data)
    times = np.array([data['timestamp'] - sync_data[0]['timestamp'] for data in sync_data])
    steering_values = np.array([data['steering'] for data in sync_data])
    throttle_values = np.array([data['throttle'] for data in sync_data])
    
    # Load first image to get dimensions
    first_img = Image.open(sync_data[0]['frame_path'])
    img_width, img_height = first_img.size
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Episode: {episode_dir.name}', fontsize=16, fontweight='bold')
    
    # Layout: 2x2 grid
    # Top left: Camera image
    # Top right: Episode info
    # Bottom left: Steering plot
    # Bottom right: Throttle plot
    
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.2, 1])
    
    # Camera image subplot
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.set_title('Camera View', fontsize=14, fontweight='bold')
    ax_img.axis('off')
    
    # Info subplot
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    
    # Steering plot
    ax_steering = fig.add_subplot(gs[1, 0])
    ax_steering.set_title('Steering Signal', fontsize=12, fontweight='bold')
    ax_steering.set_ylabel('Steering (Normalized PWM)')
    ax_steering.set_xlabel('Time (seconds)')
    ax_steering.grid(True, alpha=0.3)
    ax_steering.set_xlim(0, times[-1])
    ax_steering.set_ylim(np.min(steering_values) - 0.05, np.max(steering_values) + 0.05)
    
    # Throttle plot
    ax_throttle = fig.add_subplot(gs[1, 1])
    ax_throttle.set_title('Throttle Signal', fontsize=12, fontweight='bold')
    ax_throttle.set_ylabel('Throttle (Normalized PWM)')
    ax_throttle.set_xlabel('Time (seconds)')
    ax_throttle.grid(True, alpha=0.3)
    ax_throttle.set_xlim(0, times[-1])
    ax_throttle.set_ylim(np.min(throttle_values) - 0.05, np.max(throttle_values) + 0.05)
    
    # Initialize plot elements
    img_display = ax_img.imshow(np.zeros((img_height, img_width, 3)))
    
    # Steering plot elements
    steering_line, = ax_steering.plot([], [], 'b-', linewidth=2, label='Steering')
    steering_point, = ax_steering.plot([], [], 'bo', markersize=8)
    ax_steering.legend()
    
    # Throttle plot elements
    throttle_line, = ax_throttle.plot([], [], 'r-', linewidth=2, label='Throttle')
    throttle_point, = ax_throttle.plot([], [], 'ro', markersize=8)
    ax_throttle.legend()
    
    # Info text elements
    info_texts = {
        'frame': ax_info.text(0.05, 0.9, '', fontsize=12, fontweight='bold'),
        'time': ax_info.text(0.05, 0.8, '', fontsize=12),
        'steering_val': ax_info.text(0.05, 0.65, '', fontsize=12, color='blue'),
        'throttle_val': ax_info.text(0.05, 0.55, '', fontsize=12, color='red'),
        'stats': ax_info.text(0.05, 0.35, '', fontsize=10),
    }
    
    # Add episode statistics
    episode_stats = (
        f"Total Frames: {n_frames}\n"
        f"Duration: {times[-1]:.1f}s\n"
        f"Frame Rate: ~{n_frames/times[-1]:.1f} FPS\n\n"
        f"Steering Range: [{np.min(steering_values):.3f}, {np.max(steering_values):.3f}]\n"
        f"Throttle Range: [{np.min(throttle_values):.3f}, {np.max(throttle_values):.3f}]"
    )
    ax_info.text(0.05, 0.15, episode_stats, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    
    def animate(frame_idx):
        """Animation function"""
        if frame_idx >= n_frames:
            return []
        
        current_data = sync_data[frame_idx]
        current_time = times[frame_idx]
        
        # Load and display current frame
        try:
            img = Image.open(current_data['frame_path'])
            img_array = np.array(img)
            img_display.set_array(img_array)
        except Exception as e:
            print(f"Warning: Could not load frame {current_data['frame_path']}: {e}")
        
        # Update steering plot
        steering_line.set_data(times[:frame_idx+1], steering_values[:frame_idx+1])
        steering_point.set_data([current_time], [current_data['steering']])
        
        # Update throttle plot
        throttle_line.set_data(times[:frame_idx+1], throttle_values[:frame_idx+1])
        throttle_point.set_data([current_time], [current_data['throttle']])
        
        # Update info texts
        info_texts['frame'].set_text(f"Frame: {frame_idx+1}/{n_frames}")
        info_texts['time'].set_text(f"Time: {current_time:.2f}s")
        info_texts['steering_val'].set_text(f"Steering: {current_data['steering']:.4f}")
        info_texts['throttle_val'].set_text(f"Throttle: {current_data['throttle']:.4f}")
        
        return [img_display, steering_line, steering_point, throttle_line, throttle_point] + list(info_texts.values())
    
    # Create animation
    print(f"🎥 Rendering {n_frames} frames at {fps} FPS...")
    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames, 
        interval=1000//fps, blit=True, repeat=True
    )
    
    # Save animation
    output_file = output_dir / f"{episode_dir.name}_animation.mp4"
    
    try:
        # Use ffmpeg writer for better quality
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='QuantumTracer'), bitrate=1800)
        
        anim.save(str(output_file), writer=writer)
        plt.close(fig)
        
        print(f"✅ Animation saved: {output_file}")
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Error saving animation: {e}")
        print("   Trying alternative format...")
        
        try:
            # Fallback to pillow writer
            output_file_gif = output_dir / f"{episode_dir.name}_animation.gif"
            anim.save(str(output_file_gif), writer='pillow', fps=fps)
            plt.close(fig)
            
            print(f"✅ GIF animation saved: {output_file_gif}")
            return str(output_file_gif)
            
        except Exception as e2:
            print(f"❌ Error saving GIF: {e2}")
            plt.close(fig)
            return None

def create_combined_animation(data_dir: Path, output_dir: Path, fps: int = 8):
    """Create a combined animation showing multiple episodes"""
    
    print("🎬 Creating combined episode animation...")
    
    # Find all episodes
    episode_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / "episode_data.json").exists()])
    
    if len(episode_dirs) < 2:
        print("⚠️  Need at least 2 episodes for combined animation")
        return None
    
    # Load all episode data
    all_episodes_data = []
    for episode_dir in episode_dirs:
        sync_data = load_episode_data(episode_dir)
        if sync_data:
            all_episodes_data.append({
                'name': episode_dir.name,
                'data': sync_data
            })
    
    if not all_episodes_data:
        print("❌ No synchronized data found for combined animation")
        return None
    
    # Create figure for combined view
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Multi-Episode Training Data Visualization', fontsize=18, fontweight='bold')
    
    # Create grid: episodes side by side, plots below
    n_episodes = len(all_episodes_data)
    gs = fig.add_gridspec(3, n_episodes, height_ratios=[1.5, 1, 1])
    
    # Initialize subplots for each episode
    episode_axes = []
    plot_elements = []
    
    for i, episode_data in enumerate(all_episodes_data):
        episode_name = episode_data['name']
        sync_data = episode_data['data']
        
        # Episode image subplot
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.set_title(f'{episode_name}\nCamera View', fontsize=12, fontweight='bold')
        ax_img.axis('off')
        
        # Load first image for dimensions
        first_img = Image.open(sync_data[0]['frame_path'])
        img_display = ax_img.imshow(np.zeros_like(np.array(first_img)))
        
        episode_axes.append({
            'img': ax_img,
            'img_display': img_display,
            'data': sync_data
        })
    
    # Combined steering plot
    ax_steering_combined = fig.add_subplot(gs[1, :])
    ax_steering_combined.set_title('Combined Steering Signals', fontsize=14, fontweight='bold')
    ax_steering_combined.set_ylabel('Steering (Normalized PWM)')
    ax_steering_combined.grid(True, alpha=0.3)
    
    # Combined throttle plot
    ax_throttle_combined = fig.add_subplot(gs[2, :])
    ax_throttle_combined.set_title('Combined Throttle Signals', fontsize=14, fontweight='bold')
    ax_throttle_combined.set_ylabel('Throttle (Normalized PWM)')
    ax_throttle_combined.set_xlabel('Time (seconds)')
    ax_throttle_combined.grid(True, alpha=0.3)
    
    # Prepare combined data
    combined_steering_lines = []
    combined_throttle_lines = []
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    max_time = 0
    for i, episode_data in enumerate(all_episodes_data):
        sync_data = episode_data['data']
        times = np.array([data['timestamp'] - sync_data[0]['timestamp'] for data in sync_data])
        max_time = max(max_time, times[-1])
        
        color = colors[i % len(colors)]
        
        # Initialize empty lines
        steering_line, = ax_steering_combined.plot([], [], color=color, linewidth=2, 
                                                 label=f'{episode_data["name"]} Steering')
        throttle_line, = ax_throttle_combined.plot([], [], color=color, linewidth=2, 
                                                 label=f'{episode_data["name"]} Throttle')
        
        combined_steering_lines.append(steering_line)
        combined_throttle_lines.append(throttle_line)
    
    ax_steering_combined.set_xlim(0, max_time)
    ax_throttle_combined.set_xlim(0, max_time)
    ax_steering_combined.legend()
    ax_throttle_combined.legend()
    
    # Set y-limits based on all data
    all_steering = [data['steering'] for episode_data in all_episodes_data for data in episode_data['data']]
    all_throttle = [data['throttle'] for episode_data in all_episodes_data for data in episode_data['data']]
    
    ax_steering_combined.set_ylim(min(all_steering) - 0.05, max(all_steering) + 0.05)
    ax_throttle_combined.set_ylim(min(all_throttle) - 0.05, max(all_throttle) + 0.05)
    
    plt.tight_layout()
    
    # Find max frames for animation
    max_frames = max(len(episode_data['data']) for episode_data in all_episodes_data)
    
    def animate_combined(frame_idx):
        """Animation function for combined view"""
        elements = []
        
        for i, (episode_data, steering_line, throttle_line) in enumerate(zip(all_episodes_data, combined_steering_lines, combined_throttle_lines)):
            sync_data = episode_data['data']
            
            if frame_idx < len(sync_data):
                # Update image
                try:
                    current_data = sync_data[frame_idx]
                    img = Image.open(current_data['frame_path'])
                    episode_axes[i]['img_display'].set_array(np.array(img))
                except:
                    pass
                
                # Update plot lines
                times = np.array([data['timestamp'] - sync_data[0]['timestamp'] for data in sync_data[:frame_idx+1]])
                steering_values = np.array([data['steering'] for data in sync_data[:frame_idx+1]])
                throttle_values = np.array([data['throttle'] for data in sync_data[:frame_idx+1]])
                
                steering_line.set_data(times, steering_values)
                throttle_line.set_data(times, throttle_values)
            
            elements.extend([episode_axes[i]['img_display'], steering_line, throttle_line])
        
        return elements
    
    # Create animation
    print(f"🎥 Rendering combined animation with {max_frames} frames at {fps} FPS...")
    anim = animation.FuncAnimation(
        fig, animate_combined, frames=max_frames,
        interval=1000//fps, blit=True, repeat=True
    )
    
    # Save combined animation
    output_file = output_dir / f"combined_episodes_animation.mp4"
    
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='QuantumTracer'), bitrate=2400)
        
        anim.save(str(output_file), writer=writer)
        plt.close(fig)
        
        print(f"✅ Combined animation saved: {output_file}")
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Error saving combined animation: {e}")
        plt.close(fig)
        return None

def main():
    parser = argparse.ArgumentParser(description='Create episode animations with live steering/throttle plots')
    parser.add_argument('--data_dir', type=str,
                        default='/home/maxboels/projects/QuantumTracer/src/imitation_learning/data',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str,
                        default='./episode_animations',
                        help='Output directory for animations')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for animation')
    parser.add_argument('--episodes', type=str, nargs='+',
                        help='Specific episodes to animate (default: all)')
    parser.add_argument('--combined', action='store_true',
                        help='Also create combined multi-episode animation')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    print("🎬 Creating episode animations with live plots...")
    print(f"📁 Data directory: {data_dir}")
    print(f"🎥 Output directory: {output_dir}")
    print(f"⏱️  Frame rate: {args.fps} FPS")
    print("=" * 60)
    
    # Find episodes to process
    if args.episodes:
        episode_dirs = [data_dir / ep for ep in args.episodes if (data_dir / ep).exists()]
    else:
        episode_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / "episode_data.json").exists()])
    
    if not episode_dirs:
        print("❌ No valid episodes found!")
        return
    
    print(f"Found {len(episode_dirs)} episodes to animate:")
    for ep_dir in episode_dirs:
        print(f"  📁 {ep_dir.name}")
    
    # Create individual episode animations
    successful_animations = []
    
    for episode_dir in episode_dirs:
        result = create_episode_animation(episode_dir, output_dir, args.fps)
        if result:
            successful_animations.append(result)
    
    # Create combined animation if requested
    if args.combined and len(episode_dirs) > 1:
        print("\n" + "=" * 60)
        combined_result = create_combined_animation(data_dir, output_dir, max(6, args.fps//2))
        if combined_result:
            successful_animations.append(combined_result)
    
    print("\n" + "=" * 60)
    print("✅ Animation creation complete!")
    print(f"🎥 Created {len(successful_animations)} animations:")
    for anim_file in successful_animations:
        print(f"   📹 {Path(anim_file).name}")
    
    print(f"\n🎬 View animations in: {output_dir}")

if __name__ == "__main__":
    main()