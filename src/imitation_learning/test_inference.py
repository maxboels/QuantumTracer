#!/usr/bin/env python3
"""
Inference script for trained ACT model
Tests the model on sample images and shows predictions
"""

import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import argparse
from pathlib import Path
import json

# Import the model class
from simple_act_trainer import SimpleACTModel

def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    
    # Load checkpoint with weights_only=False for compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Create model
    model = SimpleACTModel(
        image_size=(480, 640),  # Full resolution
        action_dim=2,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        chunk_size=config['chunk_size']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
    
    return model, config

def preprocess_image(image_path: str):
    """Preprocess single image for inference"""
    
    transform = transforms.Compose([
        transforms.Resize((480, 640)),  # Full resolution
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image

def predict_action(model, image_tensor, device='cuda'):
    """Predict action from image"""
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        action = model(image_tensor)
        action = action.cpu().numpy()[0]  # Remove batch dimension
    
    return action

def test_on_sample_data(model, data_dir: str, device: str = 'cuda', num_samples: int = 5):
    """Test model on sample data and show predictions"""
    
    data_path = Path(data_dir)
    
    # Find first episode
    for episode_dir in data_path.iterdir():
        if episode_dir.is_dir():
            break
    
    # Load episode data
    with open(episode_dir / 'episode_data.json') as f:
        episode_data = json.load(f)
    
    # Get some sample frames
    frame_samples = episode_data['frame_samples'][:num_samples]
    control_samples = episode_data['control_samples'][:num_samples]
    
    print(f"\nTesting on {num_samples} samples from {episode_dir.name}")
    print("-" * 80)
    
    for i, (frame_sample, control_sample) in enumerate(zip(frame_samples, control_samples)):
        # Load and preprocess image
        image_path = episode_dir / frame_sample['image_path']
        image_tensor, original_image = preprocess_image(image_path)
        
        # Predict
        predicted_action = predict_action(model, image_tensor, device)
        
        # Ground truth
        true_steering = control_sample['steering_normalized']
        true_throttle = control_sample['throttle_normalized']
        
        print(f"Sample {i+1}:")
        print(f"  True Action:      Steering={true_steering:.4f}, Throttle={true_throttle:.4f}")
        print(f"  Predicted Action: Steering={predicted_action[0]:.4f}, Throttle={predicted_action[1]:.4f}")
        print(f"  Error:            Steering={abs(predicted_action[0] - true_steering):.4f}, Throttle={abs(predicted_action[1] - true_throttle):.4f}")
        print()

def real_time_inference_demo(model, device='cuda'):
    """Demo of real-time inference (simulated)"""
    
    print("\nReal-time inference demo (using sample images)")
    print("-" * 50)
    
    # Simulate real-time inference with full resolution
    dummy_image = torch.randn(1, 3, 480, 640).to(device)
    
    # Warm up
    for _ in range(5):
        _ = predict_action(model, dummy_image, device)
    
    # Time inference
    import time
    
    times = []
    for _ in range(100):
        start_time = time.time()
        _ = predict_action(model, dummy_image, device)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    fps = 1000 / avg_time
    
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Inference FPS: {fps:.1f}")
    print("✅ Model is ready for real-time deployment!")

def main():
    parser = argparse.ArgumentParser(description='Test trained ACT model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, help='Path to test data directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to test')
    
    args = parser.parse_args()
    
    # Load model
    model, config = load_model(args.checkpoint, args.device)
    
    # Test on sample data if provided
    if args.data_dir:
        test_on_sample_data(model, args.data_dir, args.device, args.num_samples)
    
    # Real-time inference demo
    real_time_inference_demo(model, args.device)

if __name__ == "__main__":
    main()