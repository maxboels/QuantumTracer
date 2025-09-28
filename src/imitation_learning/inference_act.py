#!/usr/bin/env python3
"""
Inference Script for Tracer RC Car ACT Policy
Loads trained model and runs inference for autonomous navigation
Optimized for edge deployment (Raspberry Pi 5 with AI HAT)
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import json
import threading
import queue

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

# Setup LeRobot imports
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

try:
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType
except ImportError as e:
    print(f"LeRobot import error: {e}")
    print("Please run setup_lerobot.py first")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TracerACTInference:
    """Inference engine for Tracer RC car using trained ACT policy"""
    
    def __init__(self, model_path: str, device: str = 'cpu', optimize_for_edge: bool = True):
        """
        Initialize the inference engine
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cpu', 'cuda')
            optimize_for_edge: Whether to optimize for edge deployment
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.optimize_for_edge = optimize_for_edge
        
        # Performance monitoring
        self.inference_times = []
        self.frame_count = 0
        self.start_time = None
        
        # Load model
        self.load_model()
        
        # Setup image preprocessing
        self.setup_preprocessing()
        
        # Action history for temporal consistency
        self.action_history = []
        self.max_history_length = 10
        
        logger.info(f"Inference engine initialized on {self.device}")
        if optimize_for_edge:
            logger.info("Edge optimization enabled")
    
    def load_model(self):
        """Load the trained ACT model"""
        logger.info(f"Loading model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract configuration
        act_config_dict = checkpoint['act_config']
        
        # Recreate ACT configuration
        self.act_config = ACTConfig(**act_config_dict)
        self.act_config.device = str(self.device)
        
        # Create and load model
        self.model = ACTPolicy(self.act_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Dataset statistics for normalization
        self.dataset_stats = checkpoint.get('dataset_stats', {})
        
        # Edge optimizations
        if self.optimize_for_edge:
            self.optimize_model_for_edge()
        
        logger.info("Model loaded successfully")
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
    
    def optimize_model_for_edge(self):
        """Apply optimizations for edge deployment"""
        logger.info("Applying edge optimizations...")
        
        # Set to inference mode
        self.model.eval()
        
        # Disable gradient computation permanently
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        # Try to use torch.jit.script for optimization
        try:
            logger.info("Attempting TorchScript compilation...")
            # Note: ACT model might not be fully scriptable, so we'll wrap inference
            # self.model = torch.jit.script(self.model)
            # logger.info("TorchScript compilation successful")
        except Exception as e:
            logger.warning(f"TorchScript compilation failed: {e}")
        
        # Set memory efficient settings
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def setup_preprocessing(self):
        """Setup image preprocessing pipeline"""
        self.image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess camera image for model input"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        image_tensor = self.image_transforms(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image: np.ndarray, current_state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Run inference on a single image
        
        Args:
            image: Camera image as numpy array
            current_state: Optional current state [steering, throttle]
        
        Returns:
            Dictionary with predicted actions and metadata
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Prepare current state (use zeros if not provided)
            if current_state is None:
                current_state = np.array([0.0, 0.0], dtype=np.float32)
            
            state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Prepare batch for model
            batch = {
                'observation': {
                    'image_front': image_tensor,
                    'state': state_tensor
                }
            }
            
            # Run inference
            with torch.no_grad():
                if hasattr(self.model, 'select_action'):
                    # Use the select_action method for inference
                    action = self.model.select_action(batch)
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy().flatten()
                else:
                    # Fallback to forward pass
                    output = self.model.forward(batch)
                    if isinstance(output, tuple):
                        action = output[0]
                    else:
                        action = output
                    
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy().flatten()
            
            # Post-process actions
            steering, throttle = action[:2] if len(action) >= 2 else (action[0], 0.0)
            
            # Apply safety constraints
            steering = np.clip(steering, -1.0, 1.0)
            throttle = np.clip(throttle, -1.0, 1.0)
            
            # Smooth actions using history
            steering, throttle = self.smooth_actions(steering, throttle)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.frame_count += 1
            
            result = {
                'steering': float(steering),
                'throttle': float(throttle),
                'inference_time_ms': inference_time * 1000,
                'frame_count': self.frame_count
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                'steering': 0.0,
                'throttle': 0.0,
                'inference_time_ms': 0.0,
                'frame_count': self.frame_count,
                'error': str(e)
            }
    
    def smooth_actions(self, steering: float, throttle: float, alpha: float = 0.7) -> tuple:
        """Apply temporal smoothing to actions"""
        # Add to history
        self.action_history.append([steering, throttle])
        
        # Limit history length
        if len(self.action_history) > self.max_history_length:
            self.action_history.pop(0)
        
        # Apply exponential smoothing
        if len(self.action_history) > 1:
            prev_steering, prev_throttle = self.action_history[-2]
            steering = alpha * steering + (1 - alpha) * prev_steering
            throttle = alpha * throttle + (1 - alpha) * prev_throttle
        
        return steering, throttle
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        inference_times = np.array(self.inference_times)
        
        return {
            'avg_inference_time_ms': float(np.mean(inference_times) * 1000),
            'max_inference_time_ms': float(np.max(inference_times) * 1000),
            'min_inference_time_ms': float(np.min(inference_times) * 1000),
            'avg_fps': float(1.0 / np.mean(inference_times)),
            'total_frames': self.frame_count
        }
    
    def reset(self):
        """Reset inference state"""
        self.action_history.clear()
        if hasattr(self.model, 'reset'):
            self.model.reset()

class CameraCapture:
    """Camera capture class for real-time inference"""
    
    def __init__(self, camera_id: int = 0, fps: int = 30, resolution: tuple = (640, 480)):
        self.camera_id = camera_id
        self.fps = fps
        self.resolution = resolution
        
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.capture_thread = None
        self.is_running = False
    
    def start(self):
        """Start camera capture"""
        logger.info(f"Starting camera capture (ID: {self.camera_id})")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def _capture_loop(self):
        """Camera capture loop"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Add frame to queue (drop old frames if queue is full)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    try:
                        self.frame_queue.get_nowait()  # Remove old frame
                        self.frame_queue.put(frame)    # Add new frame
                    except queue.Empty:
                        pass
            time.sleep(1.0 / self.fps)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop camera capture"""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join()
        if self.cap:
            self.cap.release()

def main():
    parser = argparse.ArgumentParser(description='Run ACT inference for Tracer RC car')
    parser.add_argument('model_path', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                       help='Device to run inference on')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera ID')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS')
    parser.add_argument('--resolution', type=str, default='640x480', 
                       help='Camera resolution (WIDTHxHEIGHT)')
    parser.add_argument('--save_video', type=str, help='Save output video to path')
    parser.add_argument('--duration', type=int, default=60, help='Run duration in seconds')
    parser.add_argument('--edge_optimization', action='store_true', 
                       help='Enable edge optimizations')
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    try:
        # Initialize inference engine
        inference_engine = TracerACTInference(
            model_path=args.model_path,
            device=args.device,
            optimize_for_edge=args.edge_optimization
        )
        
        # Initialize camera
        camera = CameraCapture(
            camera_id=args.camera_id,
            fps=args.fps,
            resolution=(width, height)
        )
        camera.start()
        
        # Setup video writer if requested
        video_writer = None
        if args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(args.save_video, fourcc, args.fps, (width, height))
        
        logger.info("Starting inference loop...")
        start_time = time.time()
        
        try:
            while (time.time() - start_time) < args.duration:
                # Get frame
                frame = camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Run inference
                result = inference_engine.predict(frame)
                
                # Log results
                if inference_engine.frame_count % 30 == 0:  # Log every 30 frames
                    stats = inference_engine.get_performance_stats()
                    logger.info(f"Frame {result['frame_count']}: "
                              f"Steering={result['steering']:.3f}, "
                              f"Throttle={result['throttle']:.3f}, "
                              f"FPS={stats.get('avg_fps', 0):.1f}")
                
                # Save video frame if requested
                if video_writer is not None:
                    # Add action overlay to frame
                    overlay_frame = frame.copy()
                    cv2.putText(overlay_frame, f"S: {result['steering']:.2f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(overlay_frame, f"T: {result['throttle']:.2f}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    video_writer.write(overlay_frame)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            # Cleanup
            camera.stop()
            if video_writer:
                video_writer.release()
            
            # Final statistics
            stats = inference_engine.get_performance_stats()
            logger.info("Final Performance Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value:.3f}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())