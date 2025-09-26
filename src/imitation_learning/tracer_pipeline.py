#!/usr/bin/env python3
"""
Tracer RC Car Pipeline Script
Handles recording, training, and inference for the Tracer RC car using LeRobot infrastructure.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the LeRobot environment"""
    try:
        # Import and run setup
        sys.path.insert(0, str(Path(__file__).parent))
        import setup_lerobot
        success = setup_lerobot.main()
        if not success:
            logger.error("Failed to setup LeRobot environment")
            return False
        logger.info("✅ LeRobot environment setup complete")
        return True
    except Exception as e:
        logger.error(f"Failed to setup environment: {e}")
        return False

def record_demonstrations(output_dir: str, num_episodes: int = 10):
    """Record demonstration episodes using LeRobot's recording infrastructure"""
    logger.info(f"Recording {num_episodes} demonstration episodes...")
    
    cmd = [
        "python", "-m", "lerobot.scripts.control_robot",
        "record",
        "--robot-path", "lerobot.robots.tracer.Tracer", 
        "--robot-config", "./config/env/tracer_real.yaml",
        "--fps", "30",
        "--root", output_dir,
        "--repo-id", "tracer_navigation_demos",
        "--num-episodes", str(num_episodes),
        "--warmup-time-s", "2",
        "--episode-time-s", "60",
        "--reset-time-s", "5",
    ]
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        logger.info("✅ Recording completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Recording failed: {e}")
        return False

def train_policy(dataset_path: str, output_dir: str):
    """Train ACT policy on recorded demonstrations"""
    logger.info("Training ACT policy on demonstrations...")
    
    # First, check if dataset exists
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset not found at: {dataset_path}")
        return False
        
    cmd = [
        "python", "-m", "lerobot.scripts.train",
        "--config-path", "./config/train/tracer_act.yaml",
        "--dataset.local_dir", str(dataset_path),
        "--output_dir", output_dir,
        "--policy.device", "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
        "--wandb.enable", "true",
        "--eval.n_episodes", "3",  # Reduce for faster training iterations
    ]
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        logger.info("✅ Training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        return False

def run_inference(model_path: str, duration: int = 60):
    """Run inference with trained model"""
    logger.info(f"Running inference for {duration} seconds...")
    
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"Model not found at: {model_path}")
        return False
        
    cmd = [
        "python", "-m", "lerobot.scripts.control_robot", 
        "replay",
        "--robot-path", "lerobot.robots.tracer.Tracer",
        "--robot-config", "./config/env/tracer_real.yaml", 
        "--policy-path", str(model_path),
        "--num-episodes", "1",
        "--max-episode-steps", str(duration * 30),  # 30 FPS
    ]
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        logger.info("✅ Inference completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Inference failed: {e}")
        return False

def run_all(output_dir: str = "./outputs", num_episodes: int = 10):
    """Run complete pipeline: record -> train -> deploy"""
    logger.info("Running complete Tracer RC car pipeline...")
    
    # Step 1: Setup
    if not setup_environment():
        return False
        
    # Step 2: Record demonstrations
    dataset_dir = Path(output_dir) / "datasets" / "tracer_demos"
    if not record_demonstrations(str(dataset_dir), num_episodes):
        return False
        
    # Step 3: Train policy  
    model_dir = Path(output_dir) / "models" / "tracer_act"
    if not train_policy(str(dataset_dir), str(model_dir)):
        return False
        
    # Step 4: Find best checkpoint
    checkpoint_dir = model_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.safetensors"))
        if checkpoints:
            best_checkpoint = sorted(checkpoints)[-1]  # Use latest checkpoint
            logger.info(f"Using checkpoint: {best_checkpoint}")
            
            # Step 5: Run inference
            return run_inference(str(best_checkpoint), duration=30)
    
    logger.error("No trained model found")
    return False

def main():
    parser = argparse.ArgumentParser(description="Tracer RC Car Pipeline")
    parser.add_argument("--step", choices=["setup", "record", "train", "inference", "all"], 
                      default="all", help="Pipeline step to run")
    parser.add_argument("--output-dir", default="./outputs", 
                      help="Output directory for datasets and models")
    parser.add_argument("--dataset-path", help="Path to recorded dataset (for training)")
    parser.add_argument("--model-path", help="Path to trained model (for inference)")
    parser.add_argument("--num-episodes", type=int, default=10, 
                      help="Number of episodes to record")
    parser.add_argument("--duration", type=int, default=60,
                      help="Inference duration in seconds")
    
    args = parser.parse_args()
    
    success = False
    
    if args.step == "setup":
        success = setup_environment()
    elif args.step == "record":
        success = record_demonstrations(
            args.dataset_path or f"{args.output_dir}/datasets/tracer_demos",
            args.num_episodes
        )
    elif args.step == "train":
        if not args.dataset_path:
            logger.error("--dataset-path required for training")
            sys.exit(1)
        success = train_policy(
            args.dataset_path, 
            f"{args.output_dir}/models/tracer_act"
        )
    elif args.step == "inference":
        if not args.model_path:
            logger.error("--model-path required for inference")
            sys.exit(1)
        success = run_inference(args.model_path, args.duration)
    elif args.step == "all":
        success = run_all(args.output_dir, args.num_episodes)
    
    if success:
        logger.info("🎉 Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()