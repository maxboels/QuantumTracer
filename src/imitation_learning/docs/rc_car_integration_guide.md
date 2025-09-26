# RC Car Integration with LeRobot Codebase

## Project Structure Overview

```
QuantumTracer/src/imitation_learning/  # Our working root
├── lerobot/                          # Cloned LeRobot repository
├── setup_lerobot.py                  # Path configuration
├── rc_car/                           # Our RC car extensions (to create)
│   ├── robots/                       # RC car robot implementation
│   ├── cameras/                      # Dual camera setup
│   ├── teleoperators/               # RC controller teleop
│   ├── configs/                     # RC car specific configs
│   └── scripts/                     # RC car workflows
└── datasets/                        # Local dataset storage
```

## Integration Strategy

### Phase 1: Extend LeRobot's Architecture
Rather than replacing LeRobot's functionality, we'll **extend** their existing patterns to support RC cars:

1. **Robot Implementation**: Create RC car robot class following their `ManipulatorRobot` pattern
2. **Camera Integration**: Extend their camera system for dual USB cameras
3. **Teleoperator**: Create RC controller teleoperator following their patterns
4. **Configuration**: Add RC car configs to their config system
5. **Recording**: Use their recording infrastructure with RC car adaptations

### Phase 2: Leverage Existing Commands
Use LeRobot's existing CLI commands with our extensions:
- `python lerobot/scripts/control_robot.py record` (with RC car robot)
- `python lerobot/scripts/train.py` (with RC car dataset)
- `python lerobot/scripts/control_robot.py replay` (for inference)

## Required Extensions

### 1. RC Car Robot Class (`rc_car/robots/rc_car_robot.py`)

**Purpose**: Implement LeRobot's robot interface for RC car hardware
**Extends**: `lerobot.robots.ManipulatorRobot`

```python
class RCCarRobot(ManipulatorRobot):
    """RC Car robot implementation following LeRobot patterns"""
    
    def __init__(self, config):
        # Initialize servo controllers (steering, throttle)
        # Setup dual camera system
        # Configure action/observation spaces
    
    def get_observation(self):
        # Return: {"image_front": tensor, "image_back": tensor, "state": servo_positions}
    
    def apply_action(self, action):
        # Apply [steering, throttle] commands to servos
    
    def get_action(self):
        # Read current servo positions for leader-follower setup
```

**Key Features**:
- Dual camera capture (front/back)
- Servo control (steering/throttle)
- Leader-follower recording capability
- Safety limits and emergency stop

### 2. Dual Camera System (`rc_car/cameras/dual_usb_camera.py`)

**Purpose**: Extend LeRobot's camera system for dual USB cameras
**Extends**: `lerobot.cameras.Camera`

```python
class DualUSBCamera(Camera):
    """Dual USB camera system for RC car"""
    
    def __init__(self, front_camera_id=0, back_camera_id=1):
        # Initialize both USB cameras
        # Setup threading for concurrent capture
    
    def async_read(self):
        # Return both camera frames synchronously
```

### 3. RC Controller Teleoperator (`rc_car/teleoperators/rc_teleoperator.py`)

**Purpose**: Interface with RC transmitter for human demonstrations
**Extends**: `lerobot.teleoperators.TeleOperator`

```python
class RCTeleOperator(TeleOperator):
    """RC transmitter teleoperator for recording demonstrations"""
    
    def __init__(self, config):
        # Connect to RC receiver/Arduino for reading human inputs
    
    def get_action(self):
        # Read steering/throttle from RC transmitter
        # Return normalized [-1, 1] values
```

### 4. Configuration Files (`rc_car/configs/`)

**Purpose**: RC car specific configurations following LeRobot's config patterns

**`robot/rc_car.yaml`**:
```yaml
_target_: rc_car.robots.RCCarRobot
robot_type: rc_car
calibration_dir: ~/.cache/calibration/rc_car

# Servo configuration
steering_servo:
  pin: 18
  range: [1000, 2000]  # PWM microseconds
  
throttle_servo:
  pin: 19
  range: [1000, 2000]

# Camera configuration  
cameras:
  front:
    device_id: 0
    resolution: [320, 240]
  back:
    device_id: 1
    resolution: [320, 240]
```

**`env/rc_car_real.yaml`**:
```yaml
_target_: rc_car.envs.RCCarRealEnv
robot_config: rc_car
max_episode_steps: 1000
```

### 5. Recording Script (`rc_car/scripts/record_rc_car.py`)

**Purpose**: RC car specific recording workflow using LeRobot's infrastructure

```python
# Extends lerobot/scripts/control_robot.py for RC car recording
def record_rc_car_episodes():
    # Use LeRobot's recording infrastructure
    # Handle dual camera synchronization
    # Record leader-follower data
    # Save in LeRobot dataset format
```

## Leveraging LeRobot's Existing Infrastructure

### 1. Recording System
**Use**: `lerobot.common.robot_devices.control_robot.record()`
**Extend**: Add RC car robot and teleoperator to the recording loop

### 2. Dataset System  
**Use**: `lerobot.common.datasets.lerobot_dataset.LeRobotDataset`
**Benefit**: Automatic HuggingFace dataset compatibility, caching, transforms

### 3. Training System
**Use**: `lerobot.scripts.train.py` with ACTPolicy
**Extend**: Add RC car specific data preprocessing

### 4. Policy System
**Use**: `lerobot.policies.act.ACTPolicy` directly
**Benefit**: Pre-implemented, tested, and optimized

## Implementation Plan

### Step 1: Create Robot Interface
```bash
# Create RC car robot following LeRobot patterns
mkdir -p rc_car/robots
# Implement RCCarRobot class extending ManipulatorRobot
```

### Step 2: Extend Camera System  
```bash
# Create dual camera system
mkdir -p rc_car/cameras
# Implement DualUSBCamera extending Camera base class
```

### Step 3: Create Teleoperator
```bash
# RC controller interface
mkdir -p rc_car/teleoperators  
# Implement RCTeleOperator for human demonstrations
```

### Step 4: Configuration Integration
```bash
# Add RC car configs to LeRobot config system
mkdir -p rc_car/configs
# Create robot, environment, and policy configs
```

### Step 5: Recording Integration
```bash
# Use LeRobot's recording with RC car extensions
python lerobot/scripts/control_robot.py record \
  --robot-path rc_car.robots.RCCarRobot \
  --robot-config rc_car \
  --env-config rc_car_real
```

### Step 6: Training Integration  
```bash
# Use LeRobot's training system directly
python lerobot/scripts/train.py \
  --config-name act \
  --config-path rc_car/configs \
  dataset_repo_id=local_rc_car_dataset
```

## Command Usage Examples

### Recording Demonstrations
```bash
cd QuantumTracer/src/imitation_learning

# Record RC car demonstrations using LeRobot infrastructure
python lerobot/scripts/control_robot.py record \
  --robot-path rc_car.robots.RCCarRobot \
  --robot-config rc_car \
  --teleop-config rc_controller \
  --env-config rc_car_real \
  --episode-buffer-size 100 \
  --num-episodes 50
```

### Training Model
```bash
# Train ACT policy on RC car data using LeRobot
python lerobot/scripts/train.py \
  --config-name act_rc_car \
  hydra.run.dir=./outputs/train \
  dataset_repo_id=local://./datasets/rc_car_demos
```

### Running Inference
```bash
# Deploy trained model using LeRobot infrastructure  
python lerobot/scripts/control_robot.py replay \
  --robot-path rc_car.robots.RCCarRobot \
  --robot-config rc_car \
  --policy-path ./outputs/train/checkpoints/last.ckpt \
  --env-config rc_car_real
```

## Edge Deployment Strategy

### Model Optimization
**Current Approach**: PyTorch models for training
**Edge Optimization**: Convert to optimized formats for deployment

1. **ONNX Conversion**: For cross-platform compatibility
```python
# Convert trained ACT model to ONNX
torch.onnx.export(policy.model, example_input, "rc_car_act.onnx")
```

2. **TensorRT Optimization**: For Jetson Nano acceleration
```python  
# Convert ONNX to TensorRT engine for Jetson
import tensorrt as trt
# Build optimized engine for Jetson hardware
```

3. **Quantization**: Reduce model size/memory
```python
# INT8 quantization for edge deployment
torch.quantization.quantize_dynamic(policy.model, {torch.nn.Linear}, dtype=torch.qint8)
```

### Deployment Pipeline
```bash
# 1. Train on laptop with full precision
python lerobot/scripts/train.py --config-name act_rc_car

# 2. Convert to ONNX
python rc_car/scripts/convert_to_onnx.py --checkpoint ./outputs/train/last.ckpt

# 3. Optimize for Jetson (on Jetson device)
python rc_car/scripts/optimize_for_jetson.py --onnx-path rc_car_act.onnx

# 4. Deploy on RC car
python rc_car/scripts/deploy_edge.py --engine-path rc_car_act.trt
```

## Benefits of This Approach

1. **Leverage Existing Code**: Use LeRobot's tested recording, training, and dataset infrastructure
2. **Maintainability**: Follow their established patterns and conventions  
3. **Future Compatibility**: Easy to integrate new LeRobot features
4. **Documentation**: Benefit from their extensive documentation and examples
5. **Community**: Access to LeRobot community support and improvements

## Next Steps for VS Code Agent

1. **Create Robot Interface**: Implement `RCCarRobot` class following `ManipulatorRobot` pattern
2. **Dual Camera System**: Extend their camera system for dual USB cameras
3. **RC Teleoperator**: Create RC controller interface following their teleoperator pattern  
4. **Configuration Files**: Add RC car configs to their config system
5. **Integration Testing**: Test recording/training pipeline with RC car extensions

This approach maximizes reuse of LeRobot's robust infrastructure while adding the minimal extensions needed for RC car support.