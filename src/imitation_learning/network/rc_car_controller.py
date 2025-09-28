#!/usr/bin/env python3
"""
RC Car Arduino Control Interface
================================

This script handles communication between Raspberry Pi and Arduino microcontroller
for precise servo control of the RC car.

Two approaches supported:
1. Direct GPIO PWM control (current approach)
2. Serial communication with Arduino (alternative approach)
"""

import serial
import struct
import time
import json
from typing import Tuple, Optional
import RPi.GPIO as GPIO
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArduinoSerialController:
    """
    Serial communication with Arduino for RC car control
    
    Arduino should be programmed to receive JSON commands like:
    {"steering": 0.5, "throttle": 0.3}
    """
    
    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to Arduino"""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0,
                write_timeout=1.0
            )
            
            # Wait for Arduino to initialize
            time.sleep(2.0)
            
            # Send test command
            if self._send_test_command():
                self.connected = True
                logger.info(f"✅ Connected to Arduino on {self.port}")
                return True
            else:
                logger.error("❌ Arduino test command failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Arduino connection failed: {e}")
            return False
    
    def _send_test_command(self) -> bool:
        """Send test command to verify Arduino communication"""
        try:
            test_command = {"steering": 0.0, "throttle": 0.0, "test": True}
            return self.send_command(test_command)
        except:
            return False
    
    def send_command(self, command: dict) -> bool:
        """Send control command to Arduino"""
        if not self.connected or not self.serial_connection:
            logger.error("Arduino not connected")
            return False
            
        try:
            # Convert to JSON and send
            json_str = json.dumps(command) + '\n'
            self.serial_connection.write(json_str.encode('utf-8'))
            
            # Wait for acknowledgment (optional)
            if self.serial_connection.in_waiting > 0:
                response = self.serial_connection.readline().decode('utf-8').strip()
                logger.debug(f"Arduino response: {response}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send command to Arduino: {e}")
            return False
    
    def set_controls(self, steering: float, throttle: float) -> bool:
        """
        Set steering and throttle
        
        Args:
            steering: -1.0 (full left) to +1.0 (full right)
            throttle: 0.0 (stop) to +1.0 (full speed)
        """
        # Clamp values
        steering = max(-1.0, min(1.0, steering))
        throttle = max(0.0, min(1.0, throttle))
        
        command = {
            "steering": steering,
            "throttle": throttle,
            "timestamp": time.time()
        }
        
        return self.send_command(command)
    
    def emergency_stop(self) -> bool:
        """Send emergency stop command"""
        return self.send_command({"emergency_stop": True, "steering": 0.0, "throttle": 0.0})
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.serial_connection:
            self.emergency_stop()  # Safety first
            self.serial_connection.close()
            self.connected = False
            logger.info("🔌 Disconnected from Arduino")

class GPIOPWMController:
    """
    Direct GPIO PWM control (your current approach)
    Controls servos directly from Raspberry Pi GPIO pins
    """
    
    def __init__(self, 
                 throttle_pin: int = 13, 
                 steering_pin: int = 18,
                 throttle_freq: int = 900,
                 steering_freq: int = 50):
        self.throttle_pin = throttle_pin
        self.steering_pin = steering_pin
        self.throttle_freq = throttle_freq  
        self.steering_freq = steering_freq
        
        # Speed damping factor from environment
        self.speed_damping = float(os.getenv("SPEED_DAMPING_FACTOR", 0.4))
        assert 0 < self.speed_damping <= 1, "SPEED_DAMPING_FACTOR must be between 0 and 1"
        
        self.initialized = False
        self.throttle_pwm = None
        self.steering_pwm = None
        
    def initialize(self):
        """Initialize GPIO and PWM"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.steering_pin, GPIO.OUT)
            GPIO.setup(self.throttle_pin, GPIO.OUT)
            
            # Create PWM objects
            self.throttle_pwm = GPIO.PWM(self.throttle_pin, self.throttle_freq)
            self.steering_pwm = GPIO.PWM(self.steering_pin, self.steering_freq)
            
            # Start PWM at neutral positions
            self.steering_pwm.start(7.7)  # Neutral steering (7.7% duty cycle)
            self.throttle_pwm.start(0)    # Stopped (0% duty cycle)
            
            self.initialized = True
            logger.info("✅ GPIO PWM controller initialized")
            
        except Exception as e:
            logger.error(f"❌ GPIO PWM initialization failed: {e}")
            self.initialized = False
    
    def set_throttle(self, throttle: float):
        """
        Set throttle/speed
        
        Args:
            throttle: 0.0 (stop) to 1.0 (full speed)
        """
        if not self.initialized:
            logger.error("GPIO PWM not initialized")
            return
            
        # Clamp and validate
        throttle = max(0.0, min(1.0, throttle))
        
        # Apply speed damping and convert to duty cycle
        duty_cycle = int(throttle * 70 * self.speed_damping)
        
        try:
            self.throttle_pwm.ChangeDutyCycle(duty_cycle)
            logger.debug(f"Throttle set to {throttle:.2f} (duty: {duty_cycle}%)")
        except Exception as e:
            logger.error(f"❌ Failed to set throttle: {e}")
    
    def set_steering(self, steering: float):
        """
        Set steering angle
        
        Args:
            steering: -1.0 (full left) to +1.0 (full right)
        """
        if not self.initialized:
            logger.error("GPIO PWM not initialized")
            return
            
        # Clamp and validate
        steering = max(-1.0, min(1.0, steering))
        
        # Convert to servo duty cycle: -1 to 1 -> 5.7% to 9.7% (neutral at 7.7%)
        duty_cycle = 7.7 + steering * 2.0
        
        try:
            self.steering_pwm.ChangeDutyCycle(duty_cycle)
            logger.debug(f"Steering set to {steering:.2f} (duty: {duty_cycle:.1f}%)")
        except Exception as e:
            logger.error(f"❌ Failed to set steering: {e}")
    
    def set_controls(self, steering: float, throttle: float):
        """Set both steering and throttle"""
        self.set_steering(steering)
        self.set_throttle(throttle)
    
    def emergency_stop(self):
        """Emergency stop - set to safe neutral positions"""
        if self.initialized:
            self.set_controls(steering=0.0, throttle=0.0)
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if self.initialized:
            try:
                self.emergency_stop()
                time.sleep(0.1)  # Brief delay for safety
                
                if self.throttle_pwm:
                    self.throttle_pwm.stop()
                if self.steering_pwm:
                    self.steering_pwm.stop()
                    
                GPIO.cleanup()
                self.initialized = False
                logger.info("🧹 GPIO PWM cleaned up")
                
            except Exception as e:
                logger.error(f"❌ GPIO cleanup error: {e}")

class UnifiedRCController:
    """
    Unified controller that can use either Arduino serial or GPIO PWM
    """
    
    def __init__(self, method: str = "gpio", **kwargs):
        """
        Args:
            method: "gpio" for direct PWM, "arduino" for serial communication
            **kwargs: Additional parameters for the chosen method
        """
        self.method = method
        self.controller = None
        
        if method == "gpio":
            self.controller = GPIOPWMController(**kwargs)
            self.controller.initialize()
        elif method == "arduino":
            self.controller = ArduinoSerialController(**kwargs)
            if not self.controller.connect():
                logger.error("Failed to connect to Arduino, falling back to GPIO")
                self.method = "gpio"
                self.controller = GPIOPWMController()
                self.controller.initialize()
        else:
            raise ValueError(f"Unknown control method: {method}")
    
    def set_controls(self, steering: float, throttle: float):
        """Set steering and throttle using the chosen method"""
        return self.controller.set_controls(steering, throttle)
    
    def emergency_stop(self):
        """Emergency stop using the chosen method"""
        return self.controller.emergency_stop()
    
    def cleanup(self):
        """Clean up resources"""
        if self.method == "gpio":
            self.controller.cleanup()
        elif self.method == "arduino":
            self.controller.disconnect()

# Convenience functions for backward compatibility
def create_rc_controller(method: str = "gpio", **kwargs) -> UnifiedRCController:
    """Factory function to create RC controller"""
    return UnifiedRCController(method=method, **kwargs)

def test_controller(controller: UnifiedRCController):
    """Test the RC controller with a simple sequence"""
    logger.info("🧪 Testing RC controller...")
    
    try:
        # Test sequence
        movements = [
            (0.0, 0.0),   # Neutral
            (0.5, 0.2),   # Right turn, slow speed
            (-0.5, 0.2),  # Left turn, slow speed  
            (0.0, 0.0),   # Back to neutral
        ]
        
        for i, (steering, throttle) in enumerate(movements):
            logger.info(f"Test {i+1}: steering={steering:.1f}, throttle={throttle:.1f}")
            controller.set_controls(steering, throttle)
            time.sleep(1.0)
        
        logger.info("✅ Controller test completed")
        
    except Exception as e:
        logger.error(f"❌ Controller test failed: {e}")
    finally:
        controller.emergency_stop()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RC Car Controller Test')
    parser.add_argument('--method', choices=['gpio', 'arduino'], default='gpio',
                       help='Control method to use')
    parser.add_argument('--port', default='/dev/ttyACM0',
                       help='Arduino serial port (for arduino method)')
    parser.add_argument('--test', action='store_true',
                       help='Run test sequence')
    
    args = parser.parse_args()
    
    # Create controller
    if args.method == 'arduino':
        controller = create_rc_controller('arduino', port=args.port)
    else:
        controller = create_rc_controller('gpio')
    
    try:
        if args.test:
            test_controller(controller)
        else:
            logger.info("Controller ready. Use Ctrl+C to exit.")
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        logger.info("👋 Shutting down...")
    finally:
        controller.cleanup()