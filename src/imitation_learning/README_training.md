# Tracer RC Car ACT Training Pipeline

This directory contains the complete training pipeline for the Tracer RC car using Action Chunking Transformer (ACT) policy with locally recorded demonstrations.

## 📋 Overview

Your dataset contains:
- **2 episodes** with synchronized camera frames and control data
- **447 total frames** at 640x480 resolution 
- **~8-10 seconds** of demonstration per episode
- **30 FPS** camera data with corresponding steering/throttle controls

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Make sure you're in the lerobot environment
conda activate lerobot

# Install additional dependencies
pip install -r requirements_additional.txt

# Verify LeRobot setup
python setup_lerobot.py
```

### 2. Analyze Your Dataset

```bash
# Get comprehensive dataset analysis
python analyze_dataset.py --data_dir ./data --visualize

# This will create:
# - analysis_output/episode_overview.png
# - analysis_output/action_analysis.png  
# - analysis_output/sample_images.png
# - analysis_output/dataset_statistics.json
```

### 3. Train the ACT Model

```bash
# Basic training with default settings
python train_local_act.py --data_dir ./data --output_dir ./outputs/training_run_1

# Advanced training with custom parameters
python train_local_act.py \
    --data_dir ./data \
    --output_dir ./outputs/training_run_2 \
    --batch_size 4 \
    --max_epochs 200 \
    --learning_rate 5e-5 \
    --device cuda
```

### 4. Run Inference

```bash
# Test inference with trained model
python inference_act.py ./outputs/training_run_1/best_model.pth \
    --device cpu \
    --camera_id 0 \
    --duration 30 \
    --edge_optimization \
    --save_video output_demo.avi
```

## 📁 File Structure

```
├── local_dataset_loader.py      # Custom dataset loader for local data
├── train_local_act.py           # Main training script
├── inference_act.py             # Inference engine for deployment
├── analyze_dataset.py           # Dataset analysis and visualization
├── requirements_additional.txt  # Additional dependencies
└── data/                        # Your recorded episodes
    ├── episode_20250927_193855/
    └── episode_20250927_193924/
```

## 🔧 Key Features

### Local Dataset Loading
- **Automatic synchronization** between camera frames and control data
- **Timestamp-based matching** with configurable tolerance (50ms default)
- **Data validation** and error handling
- **Compatible with LeRobot** format expectations

### Training Optimizations
- **Vision transformer** with ResNet18 backbone
- **Action chunking** for temporal consistency (32 timesteps)
- **Mixed precision training** support
- **Gradient clipping** and learning rate scheduling
- **Validation monitoring** with early stopping

### Edge Deployment Ready
- **CPU/GPU inference** support
- **Model optimization** for Raspberry Pi deployment
- **Real-time camera integration**
- **Action smoothing** for stable control
- **Performance monitoring**

## 📊 Expected Training Results

With your current dataset:
- **Training samples**: ~357 synchronized frame-action pairs
- **Training time**: 15-30 minutes on GPU, 1-2 hours on CPU
- **Memory usage**: ~2-4GB GPU memory
- **Model size**: ~50-100MB

### Recommendations for Better Results

1. **More Data**: Collect 20-50 episodes for robust training
2. **Diverse Scenarios**: Include different lighting, surfaces, obstacles
3. **Longer Episodes**: 30-60 seconds per episode for better coverage
4. **Balanced Actions**: Ensure good distribution of steering/throttle values

## ⚙️ Configuration Options

### Training Parameters

```python
config = {
    # Data settings
    'batch_size': 8,          # Reduce if GPU memory limited
    'num_workers': 4,         # CPU threads for data loading
    'train_split': 0.8,       # 80% train, 20% validation
    
    # Model architecture
    'chunk_size': 32,         # Action sequence length
    'hidden_size': 512,       # Transformer hidden dimension
    'n_encoder_layers': 4,    # Encoder depth
    'n_decoder_layers': 7,    # Decoder depth
    'vision_encoder': 'resnet18',  # Backbone network
    
    # Training
    'max_epochs': 100,        # Maximum training epochs
    'learning_rate': 1e-4,    # Initial learning rate
    'weight_decay': 1e-4,     # L2 regularization
}
```

### Edge Deployment Settings

```python
# For Raspberry Pi 5 + AI HAT
inference_config = {
    'device': 'cpu',                    # Use CPU for Raspberry Pi
    'optimize_for_edge': True,          # Enable optimizations
    'batch_size': 1,                    # Single frame inference
    'temporal_smoothing': True,         # Smooth actions over time
}
```

## 🎯 Next Steps for Production

### 1. Post-Training Quantization

```bash
# Convert to ONNX for edge deployment
python convert_to_onnx.py ./outputs/training_run_1/best_model.pth

# Apply INT8 quantization
python quantize_model.py model.onnx model_int8.onnx
```

### 2. Hardware Integration

```bash
# Test on Raspberry Pi 5
python inference_act.py model_int8.onnx \
    --device cpu \
    --camera_id 0 \
    --optimize_for_edge \
    --target_fps 15
```

### 3. Data Collection Pipeline

- Set up automated data collection during manual driving
- Implement data quality checks and filtering
- Create continuous learning pipeline

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   python train_local_act.py --batch_size 4 --device cpu
   ```

2. **Camera Not Found**
   ```bash
   # List available cameras
   v4l2-ctl --list-devices
   python inference_act.py model.pth --camera_id 1
   ```

3. **Low Training Performance**
   - Increase dataset size (collect more episodes)
   - Reduce model complexity (smaller `hidden_size`)
   - Increase learning rate for faster convergence

### Performance Expectations

- **Training**: 50-100 iterations per second on GPU
- **Inference**: 15-30 FPS on Raspberry Pi 5
- **Memory**: <2GB RAM for inference

## 📈 Monitoring and Evaluation

The training script provides:
- **Real-time loss monitoring**
- **Validation metrics**
- **Model checkpointing**
- **Performance statistics**

Monitor training progress:
```bash
# View training logs
tail -f outputs/training_run_1/training.log

# Analyze training curves
python plot_training_curves.py outputs/training_run_1/
```

## 🤝 Contributing

To improve the training pipeline:
1. Collect more diverse demonstration data
2. Experiment with different model architectures
3. Add domain randomization for robustness
4. Implement online learning capabilities