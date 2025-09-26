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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("tracer")
@dataclass
class TracerConfig(RobotConfig):
    """Configuration for Tracer RC car robot"""
    
    # Servo control pins (GPIO pins on Raspberry Pi)
    steering_pin: int = 18
    throttle_pin: int = 19
    
    # PWM settings for servo control
    steering_range: tuple[int, int] = (1000, 2000)  # PWM microseconds
    throttle_range: tuple[int, int] = (1000, 2000)  # PWM microseconds
    
    # Safety limits
    max_steering_angle: float = 45.0  # degrees
    max_throttle: float = 0.5  # normalized [-1, 1]
    
    # Enable emergency stop
    enable_emergency_stop: bool = True
    emergency_stop_pin: int = 21
    
    # Camera configuration - support single or dual cameras
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {
        "front": CameraConfig(
            device_id="/dev/video0",
            fps=30,
            width=640,
            height=480
        )
    })
    
    # Control frequency
    control_frequency: float = 30.0  # Hz
    
    # Recording settings
    episode_time_s: float = 60.0  # Maximum episode duration in seconds