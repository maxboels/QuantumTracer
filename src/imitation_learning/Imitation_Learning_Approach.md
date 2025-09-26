

# Activate environment
conda activate lerobot
cd QuantumTracer/src/imitation_learning

# Test setup
python setup_lerobot.py

# Run complete pipeline
python rc_car_pipeline.py --step all

# Or run individual steps:
python rc_car_pipeline.py --step dataset
python rc_car_pipeline.py --step train --dataset-path ./datasets/rc_car_demos
python rc_car_pipeline.py --step deploy --model-path ./models/act_rc_car_best.pth

# Deploy on edge device
python act_inference.py --model ./models/act_rc_car_best.pth --mode autopilot