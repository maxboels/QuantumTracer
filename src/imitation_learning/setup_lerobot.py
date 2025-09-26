#!/usr/bin/env python3
"""
Setup script to configure LeRobot imports from cloned repository
Run this before importing lerobot in your scripts
"""

import sys
import os
import subprocess
from pathlib import Path

def setup_lerobot_environment():
    """Setup LeRobot paths and virtual environment"""
    
    # Get current script location (imitation_learning directory)
    current_dir = Path(__file__).parent.absolute()
    print(f"Working from: {current_dir}")
    
    # Path to cloned lerobot repo (in same directory)
    lerobot_src_path = current_dir / "lerobot" / "src"
    
    print(f"Looking for lerobot at: {lerobot_src_path}")
    
    # Verify paths exist
    if not lerobot_src_path.exists():
        raise FileNotFoundError(f"LeRobot source not found at: {lerobot_src_path}")
    
    if not (lerobot_src_path / "lerobot").exists():
        raise FileNotFoundError(f"LeRobot package not found at: {lerobot_src_path / 'lerobot'}")
    
    # Add to Python path
    lerobot_src_str = str(lerobot_src_path)
    if lerobot_src_str not in sys.path:
        sys.path.insert(0, lerobot_src_str)
        print(f"✅ Added to Python path: {lerobot_src_str}")
    
    return lerobot_src_path

def activate_lerobot_venv():
    """Check if we're in the lerobot virtual environment"""
    venv_name = os.environ.get('CONDA_DEFAULT_ENV') or os.environ.get('VIRTUAL_ENV')
    
    if venv_name and 'lerobot' in str(venv_name).lower():
        print(f"✅ Using virtual environment: {venv_name}")
        return True
    else:
        print("⚠️  Warning: Not in 'lerobot' virtual environment")
        print("   Activate with: conda activate lerobot")
        return False

def test_lerobot_import():
    """Test that lerobot imports correctly"""
    try:
        import lerobot
        print(f"✅ LeRobot imported from: {lerobot.__file__}")
        print(f"✅ LeRobot version: {lerobot.__version__}")
        
        # Test specific imports
        from lerobot import policies
        print("✅ Policies module imported")
        
        try:
            from lerobot.cameras import OpenCVCamera
            print("✅ Camera module imported")
        except ImportError:
            print("⚠️  Camera module not available (may need additional dependencies)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("LeRobot Environment Setup")
    print("=" * 60)
    
    # Check virtual environment
    activate_lerobot_venv()
    
    # Setup paths
    try:
        lerobot_path = setup_lerobot_environment()
        print(f"✅ LeRobot path configured: {lerobot_path}")
    except FileNotFoundError as e:
        print(f"❌ Setup failed: {e}")
        return False
    
    # Test imports
    if test_lerobot_import():
        print("✅ Setup completed successfully!")
        return True
    else:
        print("❌ Setup failed - imports not working")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)