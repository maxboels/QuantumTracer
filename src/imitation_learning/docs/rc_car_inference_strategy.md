# RC Car Inference Implementation Strategy

## Option 1: Extend LeRobot's Replay Script (Recommended)

### Modify `lerobot/scripts/control_robot.py replay`

**Advantages**:
- Leverages LeRobot's model loading, preprocessing, and safety features
- Maintains compatibility with their ecosystem
- Easier to debug and maintain
- Automatic logging and monitoring

**Implementation**:
```python
# rc_car/scripts/rc_car_inference.py
from lerobot.scripts.control_robot import replay
from rc_car.robots.rc_car_robot import RCCarRobot

def rc_car_replay():
    """RC car specific replay using LeRobot infrastructure"""
    
    # Use LeRobot's replay with RC car robot
    replay(
        robot_path="rc_car.robots.RCCarRobot",
        robot_config="rc_car",
        policy_path="./checkpoints/act_rc_car_best.ckpt",
        env_config="rc_car_real"
    )
```

### Key Differences to Handle:

#### 1. Action Space Adaptation
```python
# In RCCarRobot.apply_action()
def apply_action(self, action):
    """Convert model output to servo commands"""
    
    # Model outputs: [steering_norm, throttle_norm] in [-1, 1]
    steering_norm, throttle_norm = action
    
    # Convert to PWM microseconds
    steering_pwm = self.norm_to_pwm(steering_norm, self.steering_range)
    throttle_pwm = self.norm_to_pwm(throttle_norm, self.throttle_range)
    
    # Send to servo controller
    self.set_servo_pwm(self.steering_pin, steering_pwm)
    self.set_servo_pwm(self.throttle_pin, throttle_pwm)
```

#### 2. Dual Camera Handling
```python
# In RCCarRobot.get_observation()
def get_observation(self):
    """Get dual camera observation"""
    
    front_frame, back_frame = self.camera_system.get_frames()
    
    return {
        "observation.image_front": self.preprocess_image(front_frame),
        "observation.image_back": self.preprocess_image(back_frame),
        "observation.state": torch.tensor(self.get_current_servo_positions())
    }
```

#### 3. Model Output Processing
```python
# In RCCarRobot - handle ACT's action chunking
def process_model_output(self, policy_output):
    """Process ACT model output (action chunks)"""
    
    # ACT outputs action sequences, take first action
    action_sequence = policy_output["action"]  # Shape: [chunk_size, 2]
    current_action = action_sequence[0]  # Take first action: [steering, throttle]
    
    return current_action
```

## Option 2: Custom Inference Script

### When to Use Custom:
- Need very specific optimizations
- Edge deployment with strict latency requirements
- Custom safety logic beyond LeRobot's

**Implementation**:
```python
# rc_car/scripts/custom_inference.py
from rc_car.inference.rc_car_engine import RCCarInferenceEngine

def run_custom_inference():
    """Custom inference optimized for RC car"""
    
    engine = RCCarInferenceEngine(
        model_path="./checkpoints/act_rc_car.onnx",  # ONNX for edge
        camera_config={"front": 0, "back": 1},
        servo_config={"steering": 18, "throttle": 19}
    )
    
    engine.run(duration=60)  # Run for 60 seconds
```

## Recommended Implementation Plan

### Phase 1: Extend LeRobot (MVP)
```bash
# Start with LeRobot's infrastructure
cd QuantumTracer/src/imitation_learning

# Create RC car robot that works with their replay script
python lerobot/scripts/control_robot.py replay \
  --robot-path rc_car.robots.RCCarRobot \
  --policy-path ./checkpoints/act_rc_car.ckpt
```

### Phase 2: Optimize for Edge (Production)
```bash
# Convert to optimized format and deploy custom inference
python rc_car/scripts/convert_model.py --checkpoint act_rc_car.ckpt --output act_rc_car.onnx
python rc_car/scripts/edge_inference.py --model act_rc_car.onnx
```

## Key Differences Implementation

### 1. Joint Space Mapping
```python
class RCCarRobot(ManipulatorRobot):
    def __init__(self, config):
        # RC car has 2 DOF vs manipulator's 6+ DOF
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),    # [steering, throttle]
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Servo configuration
        self.steering_range = config.steering_servo.range  # [1000, 2000] μs
        self.throttle_range = config.throttle_servo.range  # [1000, 2000] μs
```

### 2. Video Input Processing
```python
class DualCameraProcessor:
    def process_observation(self, observation):
        """Process dual camera input for model"""
        
        # Handle dual camera inputs
        front_img = observation["observation.image_front"]
        back_img = observation["observation.image_back"]
        
        # Apply same preprocessing as training
        processed_obs = {
            "image_front": self.transform(front_img),
            "image_back": self.transform(back_img),
            "state": observation["observation.state"]
        }
        
        return processed_obs
```

### 3. Servo Command Interface
```python
class ServoController:
    def __init__(self, steering_pin=18, throttle_pin=19):
        # Initialize PWM (pigpio, RPi.GPIO, etc.)
        import pigpio
        self.pi = pigpio.pi()
        
    def set_servo_commands(self, steering_norm, throttle_norm):
        """Convert normalized commands to servo PWM"""
        
        # Apply safety limits
        steering_norm = np.clip(steering_norm, -0.8, 0.8)  # Limit steering
        throttle_norm = np.clip(throttle_norm, -0.5, 0.5)  # Limit speed
        
        # Convert to PWM
        steering_pwm = 1500 + int(steering_norm * 400)  # 1100-1900 μs
        throttle_pwm = 1500 + int(throttle_norm * 300)   # 1200-1800 μs
        
        # Send to servos
        self.pi.set_servo_pulsewidth(self.steering_pin, steering_pwm)
        self.pi.set_servo_pulsewidth(self.throttle_pin, throttle_pwm)
```

## Command Usage Comparison

### LeRoBot Standard (Manipulator):
```bash
python lerobot/scripts/control_robot.py replay \
  --robot-path lerobot.robots.so100.So100Robot \
  --policy-path policy.ckpt
```

### RC Car Extension:
```bash
python lerobot/scripts/control_robot.py replay \
  --robot-path rc_car.robots.RCCarRobot \
  --robot-config rc_car \
  --policy-path rc_car_policy.ckpt \
  --env-config rc_car_real
```

### Custom RC Car (Edge Optimized):
```bash
python rc_car/scripts/edge_inference.py \
  --model rc_car_act.onnx \
  --config rc_car_edge.yaml \
  --duration 120
```

## Recommendation

**Start with Option 1** (extending LeRobot's replay script):
1. Faster development and debugging
2. Leverages their robust infrastructure
3. Easy to maintain and update
4. Good for initial testing and validation

**Move to Option 2** (custom inference) for production:
1. Optimized for edge deployment
2. ONNX/TensorRT integration
3. Custom safety and performance optimizations
4. Minimal dependencies for deployment

This gives you the best development experience while maintaining the path to optimized deployment.