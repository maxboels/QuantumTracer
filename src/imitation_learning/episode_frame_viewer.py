#!/usr/bin/env python3
"""
Quick Episode Frame Viewer
Browse through episode frames with keyboard controls
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import argparse

class EpisodeViewer:
    def __init__(self, episode_dir):
        self.episode_dir = Path(episode_dir)
        self.current_idx = 0
        self.sync_data = self.load_episode_data()
        
        if not self.sync_data:
            raise ValueError(f"No synchronized data found in {episode_dir}")
        
        self.setup_plot()
        
    def load_episode_data(self):
        """Load and synchronize episode data"""
        episode_data_path = self.episode_dir / "episode_data.json"
        
        if not episode_data_path.exists():
            return None
        
        with open(episode_data_path, 'r') as f:
            episode_data = json.load(f)
        
        frame_samples = episode_data.get('frame_samples', [])
        control_samples = episode_data.get('control_samples', [])
        
        synchronized_data = []
        
        for frame_sample in frame_samples:
            frame_time = frame_sample['timestamp']
            frame_path = self.episode_dir / frame_sample['image_path']
            
            best_control = None
            min_time_diff = float('inf')
            
            for control_sample in control_samples:
                time_diff = abs(control_sample['system_timestamp'] - frame_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_control = control_sample
            
            if best_control and min_time_diff < 0.05 and frame_path.exists():
                synchronized_data.append({
                    'timestamp': frame_time,
                    'frame_path': frame_path,
                    'steering': best_control['steering_normalized'],
                    'throttle': best_control['throttle_normalized'],
                    'frame_id': frame_sample['frame_id']
                })
        
        synchronized_data.sort(key=lambda x: x['timestamp'])
        print(f"Loaded {len(synchronized_data)} synchronized frames")
        return synchronized_data
        
    def setup_plot(self):
        """Setup the matplotlib figure"""
        self.fig, ((self.ax_img, self.ax_info), (self.ax_steering, self.ax_throttle)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle(f'Episode Viewer: {self.episode_dir.name}', fontsize=16)
        
        # Image subplot
        self.ax_img.set_title('Camera Frame')
        self.ax_img.axis('off')
        
        # Info subplot
        self.ax_info.axis('off')
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        
        # Control plots
        self.ax_steering.set_title('Steering Timeline')
        self.ax_steering.set_ylabel('Steering (Normalized PWM)')
        self.ax_steering.grid(True)
        
        self.ax_throttle.set_title('Throttle Timeline')
        self.ax_throttle.set_ylabel('Throttle (Normalized PWM)')
        self.ax_throttle.set_xlabel('Frame Index')
        self.ax_throttle.grid(True)
        
        # Plot all data
        indices = list(range(len(self.sync_data)))
        steering_vals = [data['steering'] for data in self.sync_data]
        throttle_vals = [data['throttle'] for data in self.sync_data]
        
        self.ax_steering.plot(indices, steering_vals, 'b-', alpha=0.7)
        self.ax_throttle.plot(indices, throttle_vals, 'r-', alpha=0.7)
        
        # Current position markers
        self.steering_marker, = self.ax_steering.plot([], [], 'bo', markersize=10)
        self.throttle_marker, = self.ax_throttle.plot([], [], 'ro', markersize=10)
        
        self.update_display()
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
        
    def update_display(self):
        """Update the display for current frame"""
        if not self.sync_data:
            return
            
        current_data = self.sync_data[self.current_idx]
        
        # Update image
        try:
            img = Image.open(current_data['frame_path'])
            self.ax_img.imshow(img)
            self.ax_img.set_title(f'Frame {self.current_idx + 1}/{len(self.sync_data)}')
        except Exception as e:
            self.ax_img.set_title(f'Frame {self.current_idx + 1}/{len(self.sync_data)} - Error loading')
        
        # Update info
        self.ax_info.clear()
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
        
        info_text = f"""Frame: {self.current_idx + 1}/{len(self.sync_data)}
Frame ID: {current_data['frame_id']}

Steering: {current_data['steering']:.4f}
Throttle: {current_data['throttle']:.4f}

Controls:
← Previous frame
→ Next frame
Space: Play/Pause
R: Reset to start
Q: Quit"""
        
        self.ax_info.text(0.05, 0.95, info_text, fontsize=11, verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Update position markers
        self.steering_marker.set_data([self.current_idx], [current_data['steering']])
        self.throttle_marker.set_data([self.current_idx], [current_data['throttle']])
        
        self.fig.canvas.draw()
        
    def on_key_press(self, event):
        """Handle keyboard input"""
        if event.key == 'right' or event.key == ' ':
            self.next_frame()
        elif event.key == 'left':
            self.prev_frame()
        elif event.key == 'r':
            self.current_idx = 0
            self.update_display()
        elif event.key == 'q':
            plt.close(self.fig)
            
    def next_frame(self):
        """Go to next frame"""
        if self.current_idx < len(self.sync_data) - 1:
            self.current_idx += 1
            self.update_display()
            
    def prev_frame(self):
        """Go to previous frame"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
            
    def show(self):
        """Show the viewer"""
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Browse episode frames interactively')
    parser.add_argument('episode', type=str, help='Episode directory to view')
    args = parser.parse_args()
    
    episode_path = Path(args.episode)
    if not episode_path.exists():
        # Try relative to data directory
        data_dir = Path('/home/maxboels/projects/QuantumTracer/src/imitation_learning/data')
        episode_path = data_dir / args.episode
    
    if not episode_path.exists():
        print(f"❌ Episode directory not found: {args.episode}")
        return
        
    try:
        viewer = EpisodeViewer(episode_path)
        print("🎮 Use arrow keys to navigate, space to advance, 'r' to reset, 'q' to quit")
        viewer.show()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()