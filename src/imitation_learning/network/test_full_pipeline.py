#!/usr/bin/env python3
"""
Full Pipeline Test Script
========================

This script tests the complete autonomous driving pipeline:
1. Loads recorded episodes as camera input
2. Connects to remote inference server
3. Gets AI predictions
4. Displays results (without actually controlling RC car)

This is perfect for:
- Testing the network infrastructure
- Validating the trained model
- Debugging the pipeline without hardware risk
"""

import cv2
import json
import os
import time
import logging
import argparse
import numpy as np
import socket
import struct
from pathlib import Path
from typing import Optional, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineTestClient:
    """Test client that simulates RC car using recorded episodes"""
    
    def __init__(self, 
                 server_ip: str,
                 server_port: int = 8888,
                 data_dir: str = "data",
                 playback_speed: float = 1.0):
        
        self.server_ip = server_ip
        self.server_port = server_port
        self.data_dir = Path(data_dir)
        self.playback_speed = playback_speed
        
        self.socket = None
        self.stats = {
            'frames_processed': 0,
            'inference_times': [],
            'network_errors': 0,
            'start_time': 0
        }
    
    def connect_to_server(self) -> bool:
        """Connect to inference server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            
            logger.info(f"🔗 Connecting to {self.server_ip}:{self.server_port}...")
            self.socket.connect((self.server_ip, self.server_port))
            
            logger.info("✅ Connected to inference server")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def recv_exact(self, size: int) -> Optional[bytes]:
        """Receive exactly 'size' bytes"""
        try:
            data = b''
            while len(data) < size:
                chunk = self.socket.recv(size - len(data))
                if not chunk:
                    return None
                data += chunk
            return data
        except:
            return None
    
    def send_frame_get_prediction(self, frame: np.ndarray) -> Optional[dict]:
        """Send frame and get AI prediction"""
        try:
            start_time = time.time()
            
            # Encode frame
            success, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                return None
            
            frame_data = encoded_frame.tobytes()
            
            # Prepare message
            message = {
                'frame_size': len(frame_data),
                'sensor_data': {'timestamp': time.time()},
                'timestamp': time.time()
            }
            
            message_json = json.dumps(message).encode()
            
            # Send message
            self.socket.send(struct.pack('I', len(message_json)))
            self.socket.send(message_json)
            self.socket.send(frame_data)
            
            # Receive response
            response_size_data = self.recv_exact(4)
            if not response_size_data:
                return None
                
            response_size = struct.unpack('I', response_size_data)[0]
            response_data = self.recv_exact(response_size)
            
            if not response_data:
                return None
            
            response = json.loads(response_data.decode())
            
            # Track timing
            inference_time = time.time() - start_time
            self.stats['inference_times'].append(inference_time)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Inference error: {e}")
            self.stats['network_errors'] += 1
            return None
    
    def load_episode_data(self, episode_path: Path) -> Tuple[List[np.ndarray], List[dict]]:
        """Load frames and actions from an episode"""
        frames = []
        actions = []
        
        # Load frames
        for frame_file in sorted((episode_path / "frames").glob("*.jpg")):
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frames.append(frame)
        
        # Load actions
        actions_file = episode_path / "actions.json"
        if actions_file.exists():
            with open(actions_file, 'r') as f:
                actions_data = json.load(f)
                actions = actions_data.get('actions', [])
        
        logger.info(f"📂 Loaded episode: {len(frames)} frames, {len(actions)} actions")
        return frames, actions
    
    def test_single_episode(self, episode_name: str) -> bool:
        """Test pipeline on a single episode"""
        episode_path = self.data_dir / episode_name
        
        if not episode_path.exists():
            logger.error(f"❌ Episode not found: {episode_path}")
            return False
        
        logger.info(f"🎬 Testing episode: {episode_name}")
        
        # Load episode data
        frames, true_actions = self.load_episode_data(episode_path)
        
        if not frames:
            logger.error("❌ No frames found in episode")
            return False
        
        # Create display window
        cv2.namedWindow('Pipeline Test', cv2.WINDOW_RESIZED)
        cv2.resizeWindow('Pipeline Test', 800, 600)
        
        # Process frames
        errors = []
        predictions = []
        
        for i, frame in enumerate(frames):
            frame_start = time.time()
            
            # Get AI prediction
            response = self.send_frame_get_prediction(frame)
            
            if response and 'control_command' in response:
                cmd = response['control_command']
                steering_pred = cmd.get('steering_target', 0.0)
                throttle_pred = cmd.get('throttle_target', 0.0)
                confidence = cmd.get('confidence', 0.0)
                
                predictions.append({
                    'steering': steering_pred,
                    'throttle': throttle_pred,
                    'confidence': confidence
                })
                
                # Get true action for comparison
                true_steering = 0.0
                true_throttle = 0.0
                if i < len(true_actions):
                    action = true_actions[i]
                    if isinstance(action, dict):
                        true_steering = action.get('steering', 0.0)
                        true_throttle = action.get('throttle', 0.0)
                    elif isinstance(action, list) and len(action) >= 2:
                        true_steering = action[0]
                        true_throttle = action[1]
                
                # Calculate errors
                steering_error = abs(steering_pred - true_steering)
                throttle_error = abs(throttle_pred - true_throttle)
                errors.append({
                    'steering_error': steering_error,
                    'throttle_error': throttle_error
                })
                
                # Create visualization
                display_frame = self.create_visualization(
                    frame, 
                    steering_pred, throttle_pred, confidence,
                    true_steering, true_throttle,
                    steering_error, throttle_error,
                    i, len(frames)
                )
                
                cv2.imshow('Pipeline Test', display_frame)
                
                # Status logging
                if i % 10 == 0:
                    avg_inference = np.mean(self.stats['inference_times'][-10:]) * 1000
                    logger.info(
                        f"Frame {i+1}/{len(frames)}: "
                        f"steering={steering_pred:.2f}±{steering_error:.2f}, "
                        f"throttle={throttle_pred:.2f}±{throttle_error:.2f}, "
                        f"conf={confidence:.2f}, "
                        f"inference={avg_inference:.1f}ms"
                    )
                
                self.stats['frames_processed'] += 1
                
            else:
                logger.warning(f"⚠️  No prediction for frame {i}")
            
            # Control playback speed
            frame_duration = time.time() - frame_start
            target_duration = 1.0 / (30.0 * self.playback_speed)
            
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # ESC
                logger.info("⏹️  Stopped by user")
                break
        
        cv2.destroyAllWindows()
        
        # Calculate final statistics
        if errors:
            avg_steering_error = np.mean([e['steering_error'] for e in errors])
            avg_throttle_error = np.mean([e['throttle_error'] for e in errors])
            avg_confidence = np.mean([p['confidence'] for p in predictions])
            
            logger.info("📊 Episode Statistics:")
            logger.info(f"   Avg steering error: {avg_steering_error:.3f}")
            logger.info(f"   Avg throttle error: {avg_throttle_error:.3f}")
            logger.info(f"   Avg confidence: {avg_confidence:.3f}")
        
        return len(errors) > 0
    
    def create_visualization(self, frame: np.ndarray, 
                           pred_steering: float, pred_throttle: float, confidence: float,
                           true_steering: float, true_throttle: float,
                           steering_error: float, throttle_error: float,
                           frame_idx: int, total_frames: int) -> np.ndarray:
        """Create visualization frame with predictions and ground truth"""
        
        # Create larger canvas
        h, w = frame.shape[:2]
        canvas = np.zeros((h + 200, w, 3), dtype=np.uint8)
        canvas[:h, :] = frame
        
        # Info panel
        info_y = h + 20
        
        # Title
        cv2.putText(canvas, f"Frame {frame_idx+1}/{total_frames}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Predictions vs Ground Truth
        info_y += 30
        cv2.putText(canvas, "AI PREDICTION:", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        info_y += 25
        cv2.putText(canvas, f"Steering: {pred_steering:+.2f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(canvas, f"Throttle: {pred_throttle:+.2f}", 
                   (200, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        info_y += 25
        cv2.putText(canvas, "GROUND TRUTH:", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        info_y += 25
        cv2.putText(canvas, f"Steering: {true_steering:+.2f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(canvas, f"Throttle: {true_throttle:+.2f}", 
                   (200, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Errors and confidence
        info_y += 30
        error_color = (0, 255, 255) if steering_error < 0.1 else (0, 165, 255)
        cv2.putText(canvas, f"Steering Error: {steering_error:.3f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, error_color, 1)
        
        error_color = (0, 255, 255) if throttle_error < 0.1 else (0, 165, 255)
        cv2.putText(canvas, f"Throttle Error: {throttle_error:.3f}", 
                   (250, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, error_color, 1)
        
        # Confidence
        conf_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
        cv2.putText(canvas, f"Confidence: {confidence:.3f}", 
                   (450, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
        
        # Control visualization (steering wheel and throttle bar)
        self.draw_controls_visualization(canvas, pred_steering, pred_throttle, w - 150, info_y - 80)
        
        return canvas
    
    def draw_controls_visualization(self, canvas: np.ndarray, steering: float, throttle: float, x: int, y: int):
        """Draw steering wheel and throttle visualization"""
        # Steering wheel
        center = (x, y)
        radius = 30
        cv2.circle(canvas, center, radius, (100, 100, 100), 2)
        
        # Steering indicator
        angle = steering * np.pi / 2  # Convert to radians
        end_x = int(center[0] + radius * 0.8 * np.sin(angle))
        end_y = int(center[1] - radius * 0.8 * np.cos(angle))
        cv2.line(canvas, center, (end_x, end_y), (0, 255, 0), 3)
        
        # Throttle bar
        bar_x = x + 50
        bar_height = 60
        bar_width = 15
        
        cv2.rectangle(canvas, (bar_x, y - bar_height//2), 
                     (bar_x + bar_width, y + bar_height//2), (100, 100, 100), 2)
        
        # Throttle fill
        fill_height = int(abs(throttle) * bar_height)
        fill_color = (0, 255, 0) if throttle > 0 else (0, 0, 255)
        
        if throttle > 0:
            fill_y = y - fill_height//2
        else:
            fill_y = y
        
        cv2.rectangle(canvas, (bar_x + 1, fill_y), 
                     (bar_x + bar_width - 1, fill_y + fill_height), fill_color, -1)
    
    def test_all_episodes(self) -> None:
        """Test pipeline on all available episodes"""
        episodes = [d.name for d in self.data_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('episode_')]
        
        if not episodes:
            logger.error(f"❌ No episodes found in {self.data_dir}")
            return
        
        episodes.sort()
        logger.info(f"🎬 Found {len(episodes)} episodes")
        
        for episode in episodes:
            logger.info(f"\n{'='*50}")
            success = self.test_single_episode(episode)
            if not success:
                logger.warning(f"⚠️  Issues with episode {episode}")
        
        # Final statistics
        if self.stats['frames_processed'] > 0:
            total_time = time.time() - self.stats['start_time']
            avg_fps = self.stats['frames_processed'] / total_time
            avg_inference = np.mean(self.stats['inference_times']) * 1000
            
            logger.info(f"\n📊 Overall Statistics:")
            logger.info(f"   Total frames: {self.stats['frames_processed']}")
            logger.info(f"   Average FPS: {avg_fps:.1f}")
            logger.info(f"   Average inference: {avg_inference:.1f}ms")
            logger.info(f"   Network errors: {self.stats['network_errors']}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.socket:
            self.socket.close()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Test full autonomous driving pipeline')
    parser.add_argument('--server-ip', required=True, help='Inference server IP address')
    parser.add_argument('--server-port', type=int, default=8888, help='Inference server port')
    parser.add_argument('--data-dir', default='data', help='Directory with episode data')
    parser.add_argument('--episode', help='Test specific episode (optional)')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier')
    
    args = parser.parse_args()
    
    print("🧪 QuantumTracer Pipeline Test")
    print("=" * 40)
    print(f"Server: {args.server_ip}:{args.server_port}")
    print(f"Data: {args.data_dir}")
    print(f"Speed: {args.speed}x")
    print("\n🎮 Controls: 'q' or ESC to stop")
    print()
    
    # Create test client
    client = PipelineTestClient(
        server_ip=args.server_ip,
        server_port=args.server_port,
        data_dir=args.data_dir,
        playback_speed=args.speed
    )
    
    try:
        if not client.connect_to_server():
            return 1
        
        client.stats['start_time'] = time.time()
        
        if args.episode:
            client.test_single_episode(args.episode)
        else:
            client.test_all_episodes()
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user")
    except Exception as e:
        logger.error(f"❌ Test error: {e}")
        return 1
    finally:
        client.cleanup()
    
    return 0

if __name__ == "__main__":
    exit(main())