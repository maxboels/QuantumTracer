#!/usr/bin/env python3
"""
Remote Inference Server (Laptop/GPU Side)
=========================================

Receives camera frames and sensor data from Raspberry Pi, runs ML inference,
and sends control commands back to the RC car.

This runs on your laptop with GPU for fast model inference.

Architecture:
    Raspberry Pi → WiFi → This Server (GPU) → WiFi → Raspberry Pi

Usage:
    python3 remote_inference_server.py --model-path ./trained_model.pth --port 8888
"""

import argparse
import socket
import json
import threading
import time
import struct
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

# Add parent directory to path to import our models
sys.path.insert(0, str(Path(__file__).parent.parent))

# ML/AI imports (you'll need to adjust these based on your model framework)
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available - using dummy model")
    PYTORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DummyModel:
    """Placeholder model for testing without trained weights"""
    
    def __init__(self):
        self.device = 'cpu'
        logger.info("Using dummy model (random predictions)")
    
    def predict(self, frame: np.ndarray, sensor_data: Dict) -> Tuple[float, float, float]:
        """Generate dummy predictions"""
        # Use current steering as baseline with small random variations
        current_steering = sensor_data.get('steering_current', 0.0)
        current_throttle = sensor_data.get('throttle_current', 0.0)
        
        # Small random adjustments (for testing)
        steering = np.clip(current_steering + np.random.normal(0, 0.1), -1.0, 1.0)
        throttle = np.clip(current_throttle + np.random.normal(0, 0.05), 0.0, 1.0)
        confidence = np.random.uniform(0.7, 0.95)
        
        return float(steering), float(throttle), float(confidence)

class PyTorchModel:
    """PyTorch-based imitation learning model"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._setup_transforms()
        
        logger.info(f"PyTorch model loaded on {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                logger.info("Using CPU (no GPU available)")
        return device
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        try:
            # Load checkpoint dictionary
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                # Handle checkpoint dictionary format
                if 'model_state_dict' in checkpoint:
                    # This is our training checkpoint format
                    from simple_act_trainer import SimpleACTModel
                    
                    # Get model config from checkpoint
                    config = checkpoint.get('config', {})
                    
                    # Create model instance
                    model = SimpleACTModel(
                        image_size=(config.get('image_height', 480), config.get('image_width', 640)),
                        action_dim=2,
                        hidden_dim=config.get('hidden_dim', 512),
                        num_layers=config.get('num_layers', 4),
                        num_heads=config.get('num_heads', 8),
                        chunk_size=config.get('chunk_size', 16)
                    )
                    
                    # Load trained weights
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(self.device)
                    model.eval()
                    
                    logger.info(f"Loaded SimpleACTModel from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
                    return model
                else:
                    logger.error("Checkpoint dictionary doesn't contain 'model_state_dict'")
                    return None
            else:
                # Direct model format
                checkpoint.eval()
                return checkpoint
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.info("Falling back to dummy model")
            return None
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((480, 640)),  # Match training resolution
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, frame: np.ndarray, sensor_data: Dict) -> Tuple[float, float, float]:
        """Run model inference"""
        if self.model is None:
            # Fallback to dummy predictions
            return DummyModel().predict(frame, sensor_data)
        
        try:
            # Preprocess frame
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Transform and add batch dimension
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            
            # Add sensor data if your model uses it
            # sensor_tensor = torch.tensor([
            #     sensor_data.get('steering_current', 0.0),
            #     sensor_data.get('throttle_current', 0.0)
            # ]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Forward pass (adjust based on your model architecture)
                outputs = self.model(input_tensor)
                
                # Extract steering and throttle predictions
                steering = float(outputs[0][0].cpu())  # Adjust indexing
                throttle = float(outputs[0][1].cpu())  # Adjust indexing
                
                # Calculate confidence (you might have a separate confidence output)
                confidence = 0.85  # Placeholder
                
                return steering, throttle, confidence
                
        except Exception as e:
            logger.error(f"Inference error: {e}")
            # Fallback to safe values
            return 0.0, 0.0, 0.0

class ClientHandler:
    """Handle individual client connections"""
    
    def __init__(self, client_socket: socket.socket, client_address: Tuple, model):
        self.socket = client_socket
        self.address = client_address
        self.model = model
        self.running = True
        
        # Statistics
        self.stats = {
            'frames_received': 0,
            'predictions_sent': 0,
            'connection_start': time.time(),
            'last_frame_time': 0,
            'avg_inference_time': 0.0
        }
        
        logger.info(f"New client connected: {client_address}")
    
    def recv_exact(self, size: int) -> Optional[bytes]:
        """Receive exactly 'size' bytes with robust error handling"""
        try:
            data = b''
            bytes_received = 0
            
            while bytes_received < size:
                chunk = self.socket.recv(min(4096, size - bytes_received))
                if not chunk:
                    return None
                data += chunk
                bytes_received += len(chunk)
                
                # Progress for large transfers
                if size > 10000 and bytes_received % 50000 == 0:
                    progress = (bytes_received / size) * 100
                    logger.debug(f"Receiving: {bytes_received}/{size} bytes ({progress:.1f}%)")
            
            return data
        except Exception as e:
            logger.error(f"Error receiving data: {e}")
            return None
    
    def handle_client(self):
        """Main client handling loop"""
        inference_times = []
        
        try:
            # Set socket options for reliable transfer
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)  # 1MB
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)  # 1MB
            self.socket.settimeout(30.0)  # 30 second timeout
            
            while self.running:
                # Receive message header
                header_size_data = self.recv_exact(4)
                if not header_size_data:
                    break
                    
                header_size = struct.unpack('I', header_size_data)[0]
                header_data = self.recv_exact(header_size)
                
                if not header_data:
                    break
                
                # Parse header
                message = json.loads(header_data.decode())
                frame_size = message['frame_size']
                sensor_data = message['sensor_data']
                
                # Receive frame data
                frame_data = self.recv_exact(frame_size)
                if not frame_data:
                    break
                
                # Decode frame
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Run inference
                    start_time = time.time()
                    steering, throttle, confidence = self.model.predict(frame, sensor_data)
                    inference_time = time.time() - start_time
                    
                    inference_times.append(inference_time)
                    if len(inference_times) > 100:
                        inference_times = inference_times[-50:]  # Keep recent times
                    
                    # Send control command
                    response = {
                        'control_command': {
                            'timestamp': time.time(),
                            'steering_target': steering,
                            'throttle_target': throttle,
                            'confidence': confidence
                        },
                        'server_stats': {
                            'inference_time': inference_time,
                            'avg_inference_time': np.mean(inference_times)
                        }
                    }
                    
                    response_data = json.dumps(response).encode()
                    self.socket.send(struct.pack('I', len(response_data)))
                    self.socket.send(response_data)
                    
                    # Update stats
                    self.stats['frames_received'] += 1
                    self.stats['predictions_sent'] += 1
                    self.stats['last_frame_time'] = time.time()
                    self.stats['avg_inference_time'] = np.mean(inference_times)
                    
                    # Log periodic stats
                    if self.stats['frames_received'] % 30 == 0:  # Every 30 frames
                        self._log_stats()
                
        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            self.socket.close()
            logger.info(f"Client disconnected: {self.address}")
    
    def _log_stats(self):
        """Log current statistics"""
        uptime = time.time() - self.stats['connection_start']
        fps = self.stats['frames_received'] / uptime if uptime > 0 else 0
        
        logger.info(f"📊 Client {self.address}: "
                   f"Frames={self.stats['frames_received']}, "
                   f"FPS={fps:.1f}, "
                   f"AvgInference={self.stats['avg_inference_time']*1000:.1f}ms")

class InferenceServer:
    """Main inference server"""
    
    def __init__(self, port: int = 8888, model_path: Optional[str] = None):
        self.port = port
        self.model_path = model_path
        self.server_socket = None
        self.running = False
        
        # Initialize model
        if PYTORCH_AVAILABLE and model_path:
            self.model = PyTorchModel(model_path)
        else:
            self.model = DummyModel()
        
        # Client management
        self.clients = []
    
    def start_server(self):
        """Start the inference server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Increase buffer sizes
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)
            
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(5)
            
            self.running = True
            logger.info(f"🚀 Inference server started on port {self.port}")
            logger.info(f"Model: {type(self.model).__name__}")
            
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    
                    # Create client handler
                    client_handler = ClientHandler(client_socket, client_address, self.model)
                    
                    # Start client handling in separate thread
                    client_thread = threading.Thread(
                        target=client_handler.handle_client,
                        daemon=True
                    )
                    client_thread.start()
                    
                    self.clients.append(client_handler)
                    
                except socket.error as e:
                    if self.running:
                        logger.error(f"Socket error: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.stop_server()
    
    def stop_server(self):
        """Stop the server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        
        # Stop all client handlers
        for client in self.clients:
            client.running = False
        
        logger.info("🛑 Server stopped")

def main():
    parser = argparse.ArgumentParser(description='Remote Inference Server (Laptop/GPU)')
    parser.add_argument('--port', type=int, default=8888, help='Server port')
    parser.add_argument('--model-path', type=str, help='Path to trained model file')
    parser.add_argument('--dummy-model', action='store_true', help='Use dummy model for testing')
    
    args = parser.parse_args()
    
    print("🧠 RC Car Remote Inference Server")
    print("=" * 40)
    print(f"Port: {args.port}")
    print(f"Model: {args.model_path if args.model_path else 'Dummy Model'}")
    
    if args.dummy_model:
        args.model_path = None
    
    server = InferenceServer(port=args.port, model_path=args.model_path)
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")

if __name__ == '__main__':
    main()