#!/usr/bin/env python3
"""
RC Car Remote Inference Client
==============================

This script runs on the Raspberry Pi and:
1. Captures camera frames
2. Sends frames to remote inference server (laptop)  
3. Receives AI-generated control commands
4. Applies commands to RC car servos

Architecture:
    Pi Camera → This Script → WiFi → Inference Server (GPU) → WiFi → This Script → RC Car
"""

import cv2
import socket
import json
import struct
import time
import logging
import argparse
import threading
import queue
from typing import Optional, Tuple
import numpy as np

# Import our RC controller
from rc_car_controller import UnifiedRCController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RemoteInferenceClient:
    """Client that connects to remote inference server for autonomous driving"""
    
    def __init__(self, 
                 server_ip: str,
                 server_port: int = 8888,
                 camera_device: int = 0,
                 controller_method: str = "gpio",
                 frame_width: int = 640,
                 frame_height: int = 480,
                 jpeg_quality: int = 80):
        
        self.server_ip = server_ip
        self.server_port = server_port
        self.camera_device = camera_device
        self.controller_method = controller_method
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.jpeg_quality = jpeg_quality
        
        # Components
        self.camera = None
        self.controller = None
        self.socket = None
        
        # Threading and control
        self.running = False
        self.inference_thread = None
        self.stats_thread = None
        
        # Statistics
        self.stats = {
            'frames_sent': 0,
            'commands_received': 0,
            'connection_start': 0,
            'last_inference_time': 0,
            'avg_inference_time': 0,
            'network_errors': 0
        }
        self.inference_times = []
        
        # Command queue for thread safety
        self.command_queue = queue.Queue(maxsize=5)
        
    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("🚀 Initializing RC Car Remote Inference Client...")
        
        # Initialize camera
        if not self._init_camera():
            return False
            
        # Initialize controller
        if not self._init_controller():
            return False
            
        # Connect to inference server  
        if not self._connect_to_server():
            return False
            
        logger.info("✅ All components initialized successfully")
        return True
    
    def _init_camera(self) -> bool:
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(self.camera_device)
            
            if not self.camera.isOpened():
                logger.error(f"❌ Failed to open camera {self.camera_device}")
                return False
                
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Test capture
            ret, frame = self.camera.read()
            if not ret:
                logger.error("❌ Failed to capture test frame")
                return False
                
            logger.info(f"✅ Camera initialized: {frame.shape}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Camera initialization failed: {e}")
            return False
    
    def _init_controller(self) -> bool:
        """Initialize RC car controller"""
        try:
            self.controller = UnifiedRCController(method=self.controller_method)
            logger.info(f"✅ RC controller initialized ({self.controller_method})")
            return True
        except Exception as e:
            logger.error(f"❌ RC controller initialization failed: {e}")
            return False
    
    def _connect_to_server(self) -> bool:
        """Connect to remote inference server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)
            
            logger.info(f"🔗 Connecting to inference server at {self.server_ip}:{self.server_port}...")
            self.socket.connect((self.server_ip, self.server_port))
            
            # Increase buffer sizes for better performance
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)
            
            logger.info("✅ Connected to inference server")
            return True
            
        except Exception as e:
            logger.error(f"❌ Server connection failed: {e}")
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
    
    def start_autonomous_driving(self):
        """Start autonomous driving mode"""
        if not self.initialize():
            logger.error("❌ Initialization failed")
            return
            
        self.running = True
        self.stats['connection_start'] = time.time()
        
        # Start command processing thread
        command_thread = threading.Thread(target=self._command_processor, daemon=True)
        command_thread.start()
        
        # Start statistics thread
        stats_thread = threading.Thread(target=self._stats_reporter, daemon=True)
        stats_thread.start()
        
        logger.info("🏎️  Starting autonomous driving...")
        logger.info("⚠️  Press Ctrl+C for emergency stop")
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("🛑 Emergency stop requested by user")
        except Exception as e:
            logger.error(f"❌ Fatal error in main loop: {e}")
        finally:
            self._cleanup()
    
    def _main_loop(self):
        """Main inference loop"""
        frame_count = 0
        last_fps_time = time.time()
        
        while self.running:
            start_time = time.time()
            
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                logger.warning("⚠️  Failed to capture frame")
                continue
            
            # Get current sensor data (you can expand this)
            sensor_data = {
                'steering_current': getattr(self, 'last_steering', 0.0),
                'throttle_current': getattr(self, 'last_throttle', 0.0),
                'timestamp': time.time()
            }
            
            # Send frame for inference
            try:
                control_command = self._send_frame_get_command(frame, sensor_data)
                
                if control_command:
                    # Queue command for processing
                    try:
                        self.command_queue.put(control_command, block=False)
                    except queue.Full:
                        logger.warning("⚠️  Command queue full, dropping command")
                    
                    # Update statistics
                    inference_time = time.time() - start_time
                    self.inference_times.append(inference_time)
                    if len(self.inference_times) > 50:
                        self.inference_times = self.inference_times[-25:]
                    
                    self.stats['frames_sent'] += 1
                    self.stats['commands_received'] += 1
                    self.stats['last_inference_time'] = inference_time
                    self.stats['avg_inference_time'] = np.mean(self.inference_times)
                
            except Exception as e:
                logger.error(f"❌ Inference error: {e}")
                self.stats['network_errors'] += 1
                
                # Safety: stop the car on network errors
                self.controller.emergency_stop()
                time.sleep(0.1)
            
            # Frame rate control
            frame_count += 1
            if frame_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - last_fps_time)
                logger.debug(f"📊 Running at {fps:.1f} FPS")
                last_fps_time = current_time
    
    def _send_frame_get_command(self, frame: np.ndarray, sensor_data: dict) -> Optional[dict]:
        """Send frame to server and get control command"""
        try:
            # Encode frame as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            success, encoded_frame = cv2.imencode('.jpg', frame, encode_params)
            
            if not success:
                logger.error("❌ Failed to encode frame")
                return None
            
            frame_data = encoded_frame.tobytes()
            
            # Prepare message
            message = {
                'frame_size': len(frame_data),
                'sensor_data': sensor_data,
                'timestamp': time.time()
            }
            
            message_json = json.dumps(message).encode()
            
            # Send message header
            self.socket.send(struct.pack('I', len(message_json)))
            self.socket.send(message_json)
            
            # Send frame data
            self.socket.send(frame_data)
            
            # Receive response
            response_size_data = self.recv_exact(4)
            if not response_size_data:
                return None
                
            response_size = struct.unpack('I', response_size_data)[0]
            response_data = self.recv_exact(response_size)
            
            if not response_data:
                return None
            
            # Parse response
            response = json.loads(response_data.decode())
            return response.get('control_command')
            
        except Exception as e:
            logger.error(f"❌ Network communication error: {e}")
            raise
    
    def _command_processor(self):
        """Process incoming control commands in separate thread"""
        while self.running:
            try:
                # Get command from queue with timeout
                command = self.command_queue.get(timeout=0.1)
                
                if command and 'steering_target' in command and 'throttle_target' in command:
                    steering = command['steering_target']
                    throttle = command['throttle_target']
                    confidence = command.get('confidence', 0.0)
                    
                    # Apply confidence threshold
                    if confidence < 0.3:  # Low confidence - reduce speed
                        throttle *= 0.5
                        logger.warning(f"⚠️  Low confidence ({confidence:.2f}), reducing throttle")
                    
                    # Apply commands to RC car
                    self.controller.set_controls(steering, throttle)
                    
                    # Store last values for sensor data
                    self.last_steering = steering
                    self.last_throttle = throttle
                    
                    logger.debug(f"🎮 Applied: steering={steering:.2f}, throttle={throttle:.2f}, conf={confidence:.2f}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Command processing error: {e}")
    
    def _stats_reporter(self):
        """Report statistics periodically"""
        while self.running:
            time.sleep(10)  # Report every 10 seconds
            
            if self.stats['frames_sent'] > 0:
                uptime = time.time() - self.stats['connection_start']
                fps = self.stats['frames_sent'] / uptime
                
                logger.info(
                    f"📊 Stats: {self.stats['frames_sent']} frames, "
                    f"{fps:.1f} FPS, "
                    f"avg inference: {self.stats['avg_inference_time']*1000:.1f}ms, "
                    f"errors: {self.stats['network_errors']}"
                )
    
    def _cleanup(self):
        """Cleanup all resources"""
        logger.info("🧹 Cleaning up...")
        
        self.running = False
        
        # Stop RC car
        if self.controller:
            self.controller.emergency_stop()
            time.sleep(0.2)
            self.controller.cleanup()
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Close network connection
        if self.socket:
            self.socket.close()
        
        logger.info("✅ Cleanup completed")

def main():
    parser = argparse.ArgumentParser(description='RC Car Remote Inference Client')
    parser.add_argument('--server-ip', required=True, help='IP address of inference server')
    parser.add_argument('--server-port', type=int, default=8888, help='Port of inference server')
    parser.add_argument('--camera-device', type=int, default=0, help='Camera device ID')
    parser.add_argument('--controller', choices=['gpio', 'arduino'], default='gpio',
                       help='RC car controller method')
    parser.add_argument('--arduino-port', default='/dev/ttyACM0',
                       help='Arduino serial port (if using arduino controller)')
    parser.add_argument('--frame-width', type=int, default=640, help='Camera frame width')
    parser.add_argument('--frame-height', type=int, default=480, help='Camera frame height')
    parser.add_argument('--jpeg-quality', type=int, default=80, help='JPEG compression quality')
    
    args = parser.parse_args()
    
    print("🏎️  QuantumTracer Remote Inference Client")
    print("=" * 50)
    print(f"Server: {args.server_ip}:{args.server_port}")
    print(f"Camera: /dev/video{args.camera_device}")
    print(f"Controller: {args.controller}")
    print(f"Resolution: {args.frame_width}x{args.frame_height}")
    print()
    
    # Create and start client
    client = RemoteInferenceClient(
        server_ip=args.server_ip,
        server_port=args.server_port,
        camera_device=args.camera_device,
        controller_method=args.controller,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        jpeg_quality=args.jpeg_quality
    )
    
    try:
        client.start_autonomous_driving()
    except Exception as e:
        logger.error(f"❌ Client error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())