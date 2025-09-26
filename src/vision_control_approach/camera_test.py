#!/usr/bin/env python3
"""
Test script for the camera functionality in BasicDetector.
This script captures images from your laptop camera and detects red objects.
"""

import numpy as np
import cv2
import time
from vision import BasicDetector

def main():
    print("Initializing BasicDetector with camera...")
    
    # Create detector with default settings
    detector = BasicDetector(
        red_min=(200, 0, 0),    # Minimum red RGB values
        red_max=(255, 80, 80),  # Maximum red RGB values
        erosion_kernel_size = 10,
        camera_id=0,            # Use default camera (usually laptop webcam)
        target_size=(1024, 1024)  # Resize images to 256x256
    )
    
    print("Starting camera test. Press 'q' to quit, 's' to save current frame.")
    
    try:
        frame_count = 0
        while True:
            # Get image from camera
            img = detector.get_image()
            
            if img is None:
                print("Failed to get image from camera")
                break
            
            frame_count += 1
            
            # Analyze the image for red objects
            bbox, img_size, mask = detector.analyse_img(img)
            
            # Convert back to BGR for OpenCV display
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Draw detection results on the image
            if bbox is not None:
                x, y, width, height = bbox
                
                # Calculate center for drawing circle
                center_x = x + width // 2
                center_y = y + height // 2
                cv2.circle(img_bgr, (center_x, center_y), 5, (0, 255, 0), -1)
                
                # Draw bounding box using (x, y, width, height) format
                cv2.rectangle(img_bgr, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Add text with object info
                text = f"Red object: {width}x{height} at ({center_x},{center_y})"
                cv2.putText(img_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add bbox info
                bbox_text = f"BBox: ({x},{y}) {width}x{height}"
                cv2.putText(img_bgr, bbox_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                cv2.putText(img_bgr, "No red object detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add frame counter
            cv2.putText(img_bgr, f"Frame: {frame_count}", (10, img_bgr.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the image
            cv2.imshow('Red Object Detection', img_bgr)
            
            # Also show the mask in a separate window
            if mask is not None:
                cv2.imshow('Detection Mask', mask.astype(np.uint8) * 255)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f'captured_frame_{int(time.time())}.jpg'
                cv2.imwrite(filename, img_bgr)
                print(f"Saved frame as {filename}")
            
            # Add small delay to reduce CPU usage
            time.sleep(0.03)  # ~30 FPS
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean up
        detector.release_camera()
        cv2.destroyAllWindows()
        print("Camera test finished.")

if __name__ == "__main__":
    main()