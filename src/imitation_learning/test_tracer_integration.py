#!/usr/bin/env python3
"""
Test script for Tracer RC car integration
"""

import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

def test_tracer_import():
    """Test that Tracer robot can be imported"""
    try:
        from lerobot.robots.tracer import Tracer, TracerConfig
        print("✅ Tracer robot import successful")
        return True
    except ImportError as e:
        print(f"❌ Tracer robot import failed: {e}")
        return False

def test_tracer_config():
    """Test Tracer configuration"""
    try:
        from lerobot.robots.tracer import TracerConfig
        
        config = TracerConfig(
            id="test_tracer",
            steering_pin=18,
            throttle_pin=19
        )
        print(f"✅ Tracer config created: {config.id}")
        return True
    except Exception as e:
        print(f"❌ Tracer config failed: {e}")
        return False

def test_tracer_robot():
    """Test Tracer robot instantiation"""
    try:
        from lerobot.robots.tracer import Tracer, TracerConfig
        from lerobot.cameras.opencv import OpenCVCameraConfig
        
        # Create config with simulation-safe settings
        config = TracerConfig(
            id="test_tracer",
            steering_pin=18,
            throttle_pin=19,
            cameras={
                "front": OpenCVCameraConfig(
                    device_id=0,  # Try default camera
                    fps=30,
                    width=640,
                    height=480
                )
            }
        )
        
        # Create robot (should work even without GPIO hardware)
        robot = Tracer(config)
        
        print(f"✅ Tracer robot created: {robot}")
        print(f"   Observation features: {robot.observation_features}")
        print(f"   Action features: {robot.action_features}")
        print(f"   Is calibrated: {robot.is_calibrated}")
        
        return True
    except Exception as e:
        print(f"❌ Tracer robot creation failed: {e}")
        return False

def test_act_config():
    """Test ACT policy configuration"""
    try:
        from lerobot.policies.act import ACTConfig
        
        config = ACTConfig(
            n_obs_steps=1,
            input_features={
                "observation.image_front": {"dtype": "float32", "shape": [480, 640, 3]},
                "observation.state": {"dtype": "float32", "shape": [2]}
            },
            output_features={
                "action": {"dtype": "float32", "shape": [2]}
            }
        )
        print("✅ ACT config created successfully")
        return True
    except Exception as e:
        print(f"❌ ACT config creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Tracer RC car integration...")
    print("=" * 50)
    
    tests = [
        test_tracer_import,
        test_tracer_config, 
        test_tracer_robot,
        test_act_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Tracer integration is working.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)