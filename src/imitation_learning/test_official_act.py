#!/usr/bin/env python3
"""
Official ACT Inference Script for Tracer RC Car
Tests inference using the official HuggingFace ACT implementation
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
import argparse
import json
import time

# Setup LeRobot imports
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

try:
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.configs.types import NormalizationMode, PolicyFeature, FeatureType
    print("✅ Official ACT implementation imported successfully")
except ImportError as e:
    print(f"❌ LeRobot import error: {e}")
    print("Please run setup_lerobot.py first")
    sys.exit(1)

def load_official_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained official ACT model from checkpoint"""
    
    print(f"📂 Loading official ACT model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    dataset_stats = checkpoint['dataset_stats']
    
    # Recreate ACT configuration
    act_config = ACTConfig(
        # Input/Output
        n_obs_steps=1,
        chunk_size=config['chunk_size'],
        n_action_steps=config['chunk_size'],
        
        # Input/Output features
        input_features={
            'observation.images.front': PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, config['image_size'][0], config['image_size'][1])
            ),
            'observation.state': PolicyFeature(
                type=FeatureType.STATE,
                shape=(2,)
            ),
        },
        output_features={
            'action': PolicyFeature(
                type=FeatureType.ACTION,
                shape=(2,)
            ),
        },
        
        # Architecture
        vision_backbone=config['vision_backbone'],
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        dim_model=config['hidden_dim'],
        n_heads=config['num_heads'],
        dim_feedforward=config.get('feedforward_dim', 3200),
        n_encoder_layers=config['num_encoder_layers'],
        n_decoder_layers=config['num_decoder_layers'],
        
        # VAE
        use_vae=config.get('use_vae', True),
        latent_dim=config.get('latent_dim', 32),
        n_vae_encoder_layers=config.get('vae_encoder_layers', 4),
        
        # Training
        dropout=config.get('dropout', 0.1),
        kl_weight=config.get('kl_weight', 10.0),
        
        # Optimizer (not used in inference but needed for config)
        optimizer_lr=config['learning_rate'],
        optimizer_weight_decay=config.get('weight_decay', 1e-4),
        optimizer_lr_backbone=config.get('backbone_lr', 1e-5),
    )
    
    # Create policy
    policy = ACTPolicy(act_config)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.to(device)
    policy.eval()
    
    # Setup pre/post processors
    preprocessor, postprocessor = make_act_pre_post_processors(
        act_config, dataset_stats
    )
    
    print(f"🤖 Model loaded successfully")
    print(f"🏆 Best validation loss: {checkpoint['best_val_loss']:.6f}")
    print(f"📊 Trained for {checkpoint['epoch']+1} epochs")
    
    return policy, preprocessor, postprocessor, config

def preprocess_image(image_path: str, target_size: tuple = (480, 640)):
    """Preprocess single image for inference"""
    
    # Load and resize image
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        # Note: Official ACT handles normalization internally
    ])
    
    img_tensor = transform(img)  # [3, H, W]
    return img_tensor

def predict_action(policy, preprocessor, postprocessor, image_tensor, current_state=None, device='cuda'):
    """Predict action using official ACT policy"""
    
    # Default state if none provided (neutral steering/throttle)
    if current_state is None:
        current_state = torch.tensor([0.0, 0.0], dtype=torch.float32)  # [steering, throttle]
    
    # Prepare batch for ACT
    batch = {
        'observation.images.front': image_tensor.unsqueeze(0).to(device),  # [1, 3, H, W]
        'observation.state': current_state.unsqueeze(0).to(device),        # [1, 2]
    }
    
    # Run preprocessing
    processed_batch = preprocessor(batch)
    
    # Get prediction from policy
    with torch.no_grad():
        start_time = time.time()
        action_output = policy.select_action(processed_batch)
        inference_time = time.time() - start_time
    
    # Run postprocessing to get final actions
    postprocessed = postprocessor({'action': action_output})
    predicted_actions = postprocessed['action']  # Should be [chunk_size, 2]
    
    # Take first action from the chunk
    next_action = predicted_actions[0].cpu().numpy()  # [2]
    
    return {
        'steering': float(next_action[0]),
        'throttle': float(next_action[1]),
        'full_chunk': predicted_actions.cpu().numpy(),
        'inference_time_ms': inference_time * 1000
    }

def test_on_sample_data(model_info, data_dir: str, device: str = 'cuda', num_samples: int = 5):
    """Test model on sample data from training set"""
    
    policy, preprocessor, postprocessor, config = model_info
    data_path = Path(data_dir)
    
    print(f"\n🧪 Testing on {num_samples} samples from {data_dir}")
    print("=" * 80)
    
    # Find episode directories
    episode_dirs = [d for d in data_path.iterdir() if d.is_dir() and (d / "episode_data.json").exists()]
    
    if not episode_dirs:
        print("❌ No episode directories found!")
        return
    
    sample_count = 0
    total_inference_time = 0
    
    for episode_dir in episode_dirs:
        if sample_count >= num_samples:
            break
            
        # Load episode data
        with open(episode_dir / "episode_data.json", 'r') as f:
            episode_data = json.load(f)
        
        frame_samples = episode_data.get('frame_samples', [])
        control_samples = episode_data.get('control_samples', [])
        
        # Test a few frames from this episode
        for i, frame_sample in enumerate(frame_samples[::len(frame_samples)//3]):  # Sample every third
            if sample_count >= num_samples:
                break
                
            frame_path = episode_dir / frame_sample['image_path']
            if not frame_path.exists():
                continue
            
            # Find corresponding control data
            frame_time = frame_sample['timestamp']
            closest_control = min(control_samples, 
                                key=lambda x: abs(x['system_timestamp'] - frame_time))
            
            # Prepare input
            image_tensor = preprocess_image(str(frame_path), config['image_size'])
            current_state = torch.tensor([
                closest_control['steering_normalized'],
                closest_control['throttle_normalized']
            ])
            
            # Predict
            result = predict_action(policy, preprocessor, postprocessor, 
                                  image_tensor, current_state, device)
            
            total_inference_time += result['inference_time_ms']
            sample_count += 1
            
            print(f"📸 Sample {sample_count}: {frame_sample['image_path']}")
            print(f"   🎯 Ground Truth - Steering: {closest_control['steering_normalized']:.4f}, "
                  f"Throttle: {closest_control['throttle_normalized']:.4f}")
            print(f"   🤖 Predicted   - Steering: {result['steering']:.4f}, "
                  f"Throttle: {result['throttle']:.4f}")
            print(f"   ⚡ Inference Time: {result['inference_time_ms']:.2f}ms")
            print(f"   📊 Chunk Shape: {result['full_chunk'].shape}")
            print()
    
    if sample_count > 0:
        avg_inference_time = total_inference_time / sample_count
        fps = 1000 / avg_inference_time
        print(f"📊 Performance Summary:")
        print(f"   Average inference time: {avg_inference_time:.2f}ms")
        print(f"   Theoretical FPS: {fps:.1f}")

def benchmark_inference(model_info, device: str = 'cuda', num_iterations: int = 100):
    """Benchmark inference speed"""
    
    policy, preprocessor, postprocessor, config = model_info
    
    print(f"\n⚡ Benchmarking inference speed ({num_iterations} iterations)")
    print("=" * 60)
    
    # Create dummy input
    dummy_image = torch.randn(1, 3, *config['image_size']).to(device)
    dummy_state = torch.randn(1, 2).to(device)
    
    dummy_batch = {
        'observation.images.front': dummy_image,
        'observation.state': dummy_state,
    }
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            processed = preprocessor(dummy_batch)
            _ = policy.select_action(processed)
    
    # Benchmark
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            processed = preprocessor(dummy_batch)
            _ = policy.select_action(processed)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    total_time = time.time() - start_time
    
    avg_time = (total_time * 1000) / num_iterations  # ms
    fps = 1000 / avg_time
    
    print(f"🏁 Benchmark Results:")
    print(f"   Average inference time: {avg_time:.2f}ms")
    print(f"   Theoretical FPS: {fps:.1f}")
    print(f"   Total time for {num_iterations} iterations: {total_time:.2f}s")

def compare_with_ground_truth(model_info, data_dir: str, device: str = 'cuda'):
    """Compare predictions with ground truth across dataset"""
    
    policy, preprocessor, postprocessor, config = model_info
    data_path = Path(data_dir)
    
    print(f"\n📈 Comparing predictions with ground truth")
    print("=" * 60)
    
    all_predictions = []
    all_ground_truth = []
    
    # Process all episodes
    episode_dirs = [d for d in data_path.iterdir() if d.is_dir() and (d / "episode_data.json").exists()]
    
    for episode_dir in episode_dirs[:1]:  # Limit to first episode for speed
        print(f"Processing {episode_dir.name}...")
        
        # Load episode data  
        with open(episode_dir / "episode_data.json", 'r') as f:
            episode_data = json.load(f)
        
        frame_samples = episode_data.get('frame_samples', [])
        control_samples = episode_data.get('control_samples', [])
        
        for frame_sample in frame_samples[::5]:  # Sample every 5th frame
            frame_path = episode_dir / frame_sample['image_path']
            if not frame_path.exists():
                continue
                
            # Find corresponding control
            frame_time = frame_sample['timestamp']
            closest_control = min(control_samples,
                                key=lambda x: abs(x['system_timestamp'] - frame_time))
            
            # Get prediction
            image_tensor = preprocess_image(str(frame_path), config['image_size'])
            current_state = torch.tensor([
                closest_control['steering_normalized'],
                closest_control['throttle_normalized']
            ])
            
            result = predict_action(policy, preprocessor, postprocessor,
                                  image_tensor, current_state, device)
            
            all_predictions.append([result['steering'], result['throttle']])
            all_ground_truth.append([closest_control['steering_normalized'], 
                                   closest_control['throttle_normalized']])
    
    # Calculate metrics
    predictions = np.array(all_predictions)
    ground_truth = np.array(all_ground_truth)
    
    mse = np.mean((predictions - ground_truth) ** 2, axis=0)
    mae = np.mean(np.abs(predictions - ground_truth), axis=0)
    std_pred = np.std(predictions, axis=0)
    std_gt = np.std(ground_truth, axis=0)
    
    print(f"📊 Evaluation Results ({len(all_predictions)} samples):")
    print(f"   Steering - MSE: {mse[0]:.6f}, MAE: {mae[0]:.6f}")
    print(f"   Throttle - MSE: {mse[1]:.6f}, MAE: {mae[1]:.6f}")
    print(f"   Prediction Std - Steering: {std_pred[0]:.4f}, Throttle: {std_pred[1]:.4f}")
    print(f"   Ground Truth Std - Steering: {std_gt[0]:.4f}, Throttle: {std_gt[1]:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Test official ACT model inference')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/maxboels/projects/QuantumTracer/src/imitation_learning/data',
                       help='Path to test data')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of test samples')
    parser.add_argument('--benchmark', action='store_true', help='Run inference benchmark')
    parser.add_argument('--compare', action='store_true', help='Compare with ground truth')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print("🤖 Official HuggingFace ACT Inference Test")
    print("=" * 60)
    print(f"📂 Model: {args.checkpoint}")
    print(f"📊 Data: {args.data_dir}")
    print(f"🔧 Device: {args.device}")
    
    # Load model
    model_info = load_official_model(args.checkpoint, args.device)
    
    # Run tests
    test_on_sample_data(model_info, args.data_dir, args.device, args.num_samples)
    
    if args.benchmark:
        benchmark_inference(model_info, args.device)
    
    if args.compare:
        compare_with_ground_truth(model_info, args.data_dir, args.device)
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    main()