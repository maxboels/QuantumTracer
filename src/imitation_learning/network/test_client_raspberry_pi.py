#!/usr/bin/env python3
"""
Simple Network Test Client (Raspberry Pi)
=========================================
Tests basic network communication before running the full inference system.
"""

import socket
import json
import time
import cv2
import numpy as np
import struct

def test_connection(server_ip, server_port=8889):
    """Test basic socket connection"""
    print(f"🔗 Testing connection to {server_ip}:{server_port}...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((server_ip, server_port))
        
        # Send a simple test message
        test_msg = {"test": "hello from raspberry pi", "timestamp": time.time()}
        msg_data = json.dumps(test_msg).encode()
        
        # Send length as big-endian 4-byte integer (to match server)
        sock.send(len(msg_data).to_bytes(4, byteorder='big'))
        sock.send(msg_data)
        
        # Wait for response
        try:
            response_length = int.from_bytes(sock.recv(4), byteorder='big')
            response_data = sock.recv(response_length)
            response = json.loads(response_data.decode())
            print(f"✅ Server response: {response.get('message', 'No message')}")
            sock.close()
            return True
        except socket.timeout:
            print("⏰ Timeout waiting for response")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_camera():
    """Test camera capture"""
    print("📷 Testing camera...")
    
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print(f"✅ Camera working: {frame.shape}")
            return True
        else:
            print("❌ Camera failed to capture")
            return False
            
    except Exception as e:
        print(f"❌ Camera error: {e}")
        return False

def test_frame_sending(server_ip, server_port=8889):
    """Test sending a camera frame"""
    print(f"📸 Testing frame sending to {server_ip}:{server_port}...")
    
    # Capture a test frame
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not capture frame")
        return False
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect((server_ip, server_port))
        
        # Encode frame as JPEG
        _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_data = encoded_frame.tobytes()
        
        print(f"   Frame size: {len(frame_data)} bytes")
        
        # Send frame data
        sock.send(len(frame_data).to_bytes(4, byteorder='big'))
        sock.send(frame_data)
        
        # Wait for response
        try:
            response_length = int.from_bytes(sock.recv(4), byteorder='big')
            response_data = sock.recv(response_length)
            response_text = response_data.decode()
            print(f"✅ Server response: {response_text}")
            sock.close()
            return True
        except socket.timeout:
            print("⏰ Timeout waiting for frame response")
            return False
        
    except Exception as e:
        print(f"❌ Frame sending failed: {e}")
        return False

def main():
    print("🧪 Network Communication Test - Raspberry Pi Client")
    print("=" * 50)
    
    # Get server IP from user
    server_ip = input("Enter your laptop's IP address (192.168.1.33): ").strip()
    if not server_ip:
        server_ip = "192.168.1.33"  # Default to your laptop's IP
    
    print(f"🎯 Target server: {server_ip}:8889")
    print("📍 Pi IP address:", socket.gethostbyname(socket.gethostname()))
    print()
    
    # Test 1: Basic connection
    print("TEST 1: Basic Connection")
    if not test_connection(server_ip):
        print("❌ Basic connection failed - check server is running")
        return
    print()
    
    # Test 2: Camera
    print("TEST 2: Camera")
    if not test_camera():
        print("❌ Camera test failed")
        return
    print()
    
    # Test 3: Frame sending
    print("TEST 3: Frame Sending")
    if test_frame_sending(server_ip):
        print("🎉 All tests passed! Network communication is working.")
    else:
        print("❌ Frame sending failed")

if __name__ == '__main__':
    main()