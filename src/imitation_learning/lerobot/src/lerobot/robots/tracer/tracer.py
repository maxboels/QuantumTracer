#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np
import torch

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_tracer import TracerConfig

# Try to import GPIO library (graceful fallback for development)
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    logging.warning("RPi.GPIO not available. Running in simulation mode.")

logger = logging.getLogger(__name__)


class TracerServoController:
    """Handles PWM servo control for steering and throttle"""
    
    def __init__(self, steering_pin: int, throttle_pin: int, 
                 steering_range: tuple[int, int], throttle_range: tuple[int, int]):
        self.steering_pin = steering_pin
        self.throttle_pin = throttle_pin
        self.steering_range = steering_range
        self.throttle_range = throttle_range
        self.steering_pwm = None
        self.throttle_pwm = None
        
        if HAS_GPIO:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.steering_pin, GPIO.OUT)
            GPIO.setup(self.throttle_pin, GPIO.OUT)
            
            # Create PWM instances (50Hz is standard for servos)
            self.steering_pwm = GPIO.PWM(self.steering_pin, 50)
            self.throttle_pwm = GPIO.PWM(self.throttle_pin, 50)
            
    def start(self):
        """Start PWM signals"""
        if HAS_GPIO and self.steering_pwm and self.throttle_pwm:
            self.steering_pwm.start(7.5)  # Neutral position (1.5ms pulse width)
            self.throttle_pwm.start(7.5)
            
    def stop(self):
        """Stop PWM signals and cleanup"""
        if HAS_GPIO:
            if self.steering_pwm:
                self.steering_pwm.stop()
            if self.throttle_pwm:
                self.throttle_pwm.stop()
            GPIO.cleanup()

    def set_steering(self, normalized_value: float):
        """Set steering position from normalized value [-1, 1]"""
        # Clamp value
        normalized_value = np.clip(normalized_value, -1.0, 1.0)
        
        # Convert to PWM duty cycle (1ms = 5%, 2ms = 10% for 50Hz)
        pulse_width_us = np.interp(normalized_value, [-1, 1], self.steering_range)
        duty_cycle = (pulse_width_us / 20000) * 100  # 20ms period for 50Hz
        
        if HAS_GPIO and self.steering_pwm:
            self.steering_pwm.ChangeDutyCycle(duty_cycle)
        else:
            logger.info(f"Steering: {normalized_value:.3f} -> {pulse_width_us:.0f}μs -> {duty_cycle:.1f}%")

    def set_throttle(self, normalized_value: float):
        """Set throttle position from normalized value [-1, 1]"""
        # Clamp value
        normalized_value = np.clip(normalized_value, -1.0, 1.0)
        
        # Convert to PWM duty cycle
        pulse_width_us = np.interp(normalized_value, [-1, 1], self.throttle_range)
        duty_cycle = (pulse_width_us / 20000) * 100
        
        if HAS_GPIO and self.throttle_pwm:
            self.throttle_pwm.ChangeDutyCycle(duty_cycle)
        else:
            logger.info(f"Throttle: {normalized_value:.3f} -> {pulse_width_us:.0f}μs -> {duty_cycle:.1f}%")


class Tracer(Robot):
    """
    Tracer RC Car Robot for imitation learning.
    
    This robot implements a simple RC car with camera input for navigation tasks.
    It supports steering and throttle control via PWM servos and camera observations.
    """

    config_class = TracerConfig
    name = "tracer"

    def __init__(self, config: TracerConfig):
        super().__init__(config)
        self.config = config
                
        # Initialize servo controller
        self.servo_controller = TracerServoController(
            steering_pin=config.steering_pin,
            throttle_pin=config.throttle_pin,
            steering_range=config.steering_range,
            throttle_range=config.throttle_range
        )
        
        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # Current state
        self.current_steering = 0.0
        self.current_throttle = 0.0
        self.is_emergency_stopped = False
        
        # Emergency stop setup
        if config.enable_emergency_stop and HAS_GPIO:
            GPIO.setup(config.emergency_stop_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(
                config.emergency_stop_pin, 
                GPIO.FALLING, 
                callback=self._emergency_stop_callback,
                bouncetime=300
            )

    def _emergency_stop_callback(self, channel):
        """Emergency stop callback"""
        logger.warning("Emergency stop activated!")
        self.is_emergency_stopped = True
        self.servo_controller.set_steering(0.0)
        self.servo_controller.set_throttle(0.0)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation features for the robot"""
        features = {}
        
        # Add camera features
        for name, camera_config in self.config.cameras.items():
            features[f"observation.image_{name}"] = (
                camera_config.height,
                camera_config.width, 
                3  # RGB channels
            )
        
        # Add state features (current servo positions)
        features["observation.state"] = (2,)  # [steering, throttle]
        
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action features for the robot"""
        return {
            "action": (2,)  # [steering_command, throttle_command]
        }

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected"""
        # Check if cameras are connected
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        return cameras_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the robot hardware"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected")
        
        logger.info(f"Connecting {self}...")
        
        # Start servo controller
        self.servo_controller.start()
        
        # Connect cameras
        for name, camera in self.cameras.items():
            logger.info(f"Connecting camera: {name}")
            camera.connect()
        
        # Reset to neutral position
        self.servo_controller.set_steering(0.0)
        self.servo_controller.set_throttle(0.0)
        
        logger.info(f"{self} connected successfully!")

    @property 
    def is_calibrated(self) -> bool:
        """Tracer doesn't require calibration"""
        return True

    def calibrate(self) -> None:
        """Tracer doesn't require calibration"""
        logger.info(f"{self} doesn't require calibration")

    def configure(self) -> None:
        """Configure robot settings"""
        logger.info(f"Configuring {self}...")
        # Any additional configuration can go here

    def get_observation(self) -> dict[str, Any]:
        """Get current observation from cameras and state"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        observation = {}
        
        # Capture camera frames
        for name, camera in self.cameras.items():
            frame = camera.read()
            # Convert to torch tensor and normalize to [0, 1]
            frame_tensor = torch.from_numpy(frame).float() / 255.0
            # Ensure correct shape (H, W, C)
            if frame_tensor.shape[-1] != 3:
                frame_tensor = frame_tensor.permute(2, 0, 1).permute(1, 2, 0)
            observation[f"observation.image_{name}"] = frame_tensor
        
        # Add current state (servo positions)
        state = torch.tensor([self.current_steering, self.current_throttle], dtype=torch.float32)
        observation["observation.state"] = state
        
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action commands to the robot"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        if self.is_emergency_stopped:
            logger.warning("Robot is emergency stopped, ignoring action")
            return {"success": False, "message": "Emergency stopped"}

        # Extract action values
        if isinstance(action["action"], torch.Tensor):
            steering_cmd, throttle_cmd = action["action"].cpu().numpy()
        else:
            steering_cmd, throttle_cmd = action["action"]

        # Apply safety limits
        steering_cmd = np.clip(steering_cmd, -1.0, 1.0)
        throttle_cmd = np.clip(throttle_cmd, -self.config.max_throttle, self.config.max_throttle)
        
        # Send commands to servos
        self.servo_controller.set_steering(steering_cmd)
        self.servo_controller.set_throttle(throttle_cmd)
        
        # Update internal state
        self.current_steering = steering_cmd
        self.current_throttle = throttle_cmd
        
        return {
            "success": True, 
            "steering": steering_cmd, 
            "throttle": throttle_cmd
        }

    def disconnect(self) -> None:
        """Disconnect from robot hardware"""
        logger.info(f"Disconnecting {self}...")
        
        # Stop servos in neutral position
        self.servo_controller.set_steering(0.0)
        self.servo_controller.set_throttle(0.0)
        self.servo_controller.stop()
        
        # Disconnect cameras
        for camera in self.cameras.values():
            camera.disconnect()
            
        logger.info(f"{self} disconnected")

    def __del__(self):
        """Cleanup on destruction"""
        try:
            if self.is_connected:
                self.disconnect()
        except:
            pass