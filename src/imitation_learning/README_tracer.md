# Tracer RC Car Integration with LeRobot

This directory contains the integration of the Tracer RC car with the LeRobot codebase for imitation learning using the ACT (Action Chunking Transformer) policy.

## 🏗️ Architecture

The integration follows LeRobot's architecture patterns:

```
lerobot/src/lerobot/robots/tracer/
├── __init__.py                 # Module exports
├── config_tracer.py           # Robot configuration
└── tracer.py                  # Main robot implementation

config/
├── env/tracer_real.yaml       # Environment config
├── policy/act_tracer.yaml     # ACT policy config
└── train/tracer_act.yaml      # Training config

Scripts:
├── tracer_pipeline.py         # Complete pipeline script
├── test_tracer_integration.py # Integration tests
└── setup_lerobot.py          # Environment setup
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate the lerobot conda environment
conda activate lerobot

# Navigate to the integration directory
cd QuantumTracer/src/imitation_learning

# Test the integration
python test_tracer_integration.py

# Setup LeRobot paths
python setup_lerobot.py
```

### 2. Install Additional Dependencies

```bash
# For Raspberry Pi (with GPIO support)
pip install -r requirements_tracer.txt

# For development (simulation mode)
# GPIO dependency will be gracefully skipped
```

### 3. Run Complete Pipeline

```bash
# Run everything: record -> train -> inference
python tracer_pipeline.py --step all --num-episodes 20

# Or run individual steps:

# Step 1: Record demonstrations
python tracer_pipeline.py --step record --num-episodes 10

# Step 2: Train ACT policy
python tracer_pipeline.py --step train --dataset-path ./outputs/datasets/tracer_demos

# Step 3: Run inference
python tracer_pipeline.py --step inference --model-path ./outputs/models/tracer_act/checkpoint_best.safetensors
```

## 🎮 Manual Control & Recording

### Using LeRobot's Built-in Commands

```bash
# Record demonstrations using LeRobot's control_robot script
python -m lerobot.scripts.control_robot record \
  --robot-path lerobot.robots.tracer.Tracer \
  --robot-config ./config/env/tracer_real.yaml \
  --fps 30 \
  --root ./data \
  --repo-id tracer_navigation_demos \
  --num-episodes 10

# Train using LeRobot's train script  
python -m lerobot.scripts.train \
  --config-path ./config/train/tracer_act.yaml \
  --dataset.local_dir ./data/tracer_navigation_demos

# Deploy trained model
python -m lerobot.scripts.control_robot replay \
  --robot-path lerobot.robots.tracer.Tracer \
  --robot-config ./config/env/tracer_real.yaml \
  --policy-path ./outputs/models/tracer_act/checkpoint_best.safetensors
```

## 🔧 Hardware Setup

### Raspberry Pi Connections

```
Tracer RC Car Hardware:
├── Steering Servo    → GPIO Pin 18 (PWM)
├── Throttle ESC      → GPIO Pin 19 (PWM) 
├── Emergency Stop    → GPIO Pin 21 (Input, Pull-up)
├── Front Camera      → USB /dev/video0
└── Optional 2nd Cam  → USB /dev/video1
```

### Camera Configuration

The robot supports single or dual cameras:

```yaml
# Single camera setup (default)
cameras:
  front:
    device_id: "/dev/video0"
    fps: 30
    width: 640
    height: 480

# Dual camera setup
cameras:
  front:
    device_id: "/dev/video0"
    fps: 30
    width: 640  
    height: 480
  back:
    device_id: "/dev/video1"
    fps: 30
    width: 640
    height: 480
```

## 🧠 Model & Training Details

### ACT Policy Configuration

The Tracer uses the ACT (Action Chunking Transformer) policy with:

- **Input**: Camera images (640x480 RGB) + robot state (2D: steering, throttle)
- **Output**: Action commands (2D: steering_cmd, throttle_cmd)
- **Action Chunking**: 32 timesteps ahead prediction
- **Vision Encoder**: ResNet18 backbone
- **Transformer**: 4 encoder + 7 decoder layers

### Training Parameters

- **Episodes**: 20+ demonstration episodes recommended
- **Batch Size**: 8
- **Training Steps**: 50,000
- **Learning Rate**: Adaptive with LeRobot's preset
- **Augmentation**: Standard vision augmentations applied

## 📊 Data Format

The robot outputs LeRobot-compatible datasets:

```python
{
  "observation.image_front": [H, W, 3],  # RGB camera feed
  "observation.state": [2],              # [steering, throttle] 
  "action": [2],                         # [steering_cmd, throttle_cmd]
  "next.reward": float,                  # Task reward (optional)
  "episode_index": int,
  "frame_index": int,
  "timestamp": float
}
```

## 🛡️ Safety Features

- **Emergency Stop**: Hardware button support (GPIO pin 21)
- **Safety Limits**: Configurable max steering angle and throttle
- **Graceful Degradation**: Simulation mode when GPIO unavailable
- **Timeout Protection**: Episodes automatically terminate after max duration

## 🎯 Navigation Task

The robot is designed for red balloon navigation:

1. **Observation**: Front-facing camera captures environment
2. **State**: Current steering and throttle positions  
3. **Action**: Predicted steering and throttle commands
4. **Goal**: Navigate towards red balloon target

### Task-Specific Notes

- Train on diverse lighting conditions
- Include failure recovery demonstrations  
- Consider obstacle avoidance scenarios
- Record smooth human demonstrations at 30 FPS

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   python setup_lerobot.py  # Re-run setup
   ```

2. **Camera Not Found**:
   ```bash
   lsusb                           # Check USB cameras
   ls /dev/video*                  # Check video devices
   ```

3. **GPIO Permission Errors**:
   ```bash
   sudo usermod -a -G gpio $USER   # Add user to GPIO group
   # Logout and login again
   ```

4. **Training Memory Issues**:
   - Reduce batch size in config/train/tracer_act.yaml
   - Use CPU training if GPU memory insufficient

### Development Mode

For development without hardware:

```python
# The robot automatically detects simulation mode
# when RPi.GPIO is not available
robot = Tracer(config)  # Works without GPIO hardware
```

## 🚀 Performance Optimization

### For Edge Deployment

1. **Model Optimization**:
   ```bash
   # Convert to ONNX for faster inference
   python -c "
   import torch
   from lerobot.policies.act import ACTPolicy
   
   policy = ACTPolicy.from_pretrained('path/to/checkpoint')
   torch.onnx.export(policy, ...)
   "
   ```

2. **Camera Optimization**:
   - Lower resolution for faster processing
   - Hardware-accelerated video encoding
   - Multi-threaded camera capture

### Monitoring

- Logs are automatically saved during training
- Use Weights & Biases for experiment tracking
- Monitor control frequency during deployment

## 📈 Next Steps

1. **Collect More Data**: 50+ episodes for robust performance
2. **Task Variations**: Different environments and lighting
3. **Multi-Camera**: Add rear camera for better spatial awareness  
4. **Reward Engineering**: Task-specific reward functions
5. **Sim-to-Real**: Train in simulation, deploy on hardware

## 🤝 Contributing

This integration follows LeRobot's contribution patterns. To extend:

1. Add new robot variants in `robots/tracer/`
2. Create task-specific environments in `config/env/`
3. Implement custom policies following LeRobot patterns
4. Add comprehensive tests and documentation