import numpy as np
import cv2
from typing import Tuple
from scipy.ndimage import binary_erosion

class Input:

    def __init__(self):

        # horizon length of past snapshots
        self.H = 10

        # RGB, width height
        self.img = np.zeros([256,256,3])
        # snapshoots, dim (acceleration, steering)
        self.output_history = np.zeros([self.H,2])
    
    def set_img(self, img):
        # accept either (H,W,3) RGB uint8
        if img is None:
            return
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1,2,0))
        if img.shape[0] != 256 or img.shape[1] != 256:
            img = cv2.resize(img, (256,256))
        self.img = img

    def set_input(self, output): 
        # add the new recording:
        self.output_history = np.vstack([self.output_history, output.reshape(1, 2)])
        # drop the oldest recording:
        self.ouput_history = self.ouput_history[1:]


class BasicDetector:

    def __init__(self,
                 red_min: Tuple[int] = (200, 0, 0),
                 red_max: Tuple[int] = (255, 80, 80),
                 erosion_kernel_size: int = 3,
                 enable_noise_reduction: bool = True,
                 camera_id: int = 0,
                 target_size: Tuple[int, int] = (256, 256)):
        # RGB intervals for saturated red
        self.red_min = np.array(red_min)
        self.red_max = np.array(red_max)
        
        # Morphological operations parameters
        self.erosion_kernel_size = erosion_kernel_size
        self.enable_noise_reduction = enable_noise_reduction
        # Create the structuring element (kernel) for erosion
        self.erosion_structure = np.ones((erosion_kernel_size, erosion_kernel_size))
        
        # Camera parameters
        self.camera_id = camera_id
        self.target_size = target_size
        self.cap = None
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize the camera capture object."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera with ID {self.camera_id}")
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera {self.camera_id} initialized successfully")
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            self.cap = None
    
    def get_image(self):
        """
        Capture an image from the laptop camera.
        
        Returns:
            np.ndarray: RGB image of shape (256, 256, 3), or None if capture failed
        """
        if self.cap is None:
            print("Camera not initialized. Attempting to reinitialize...")
            self._initialize_camera()
            if self.cap is None:
                return None
        
        try:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to capture frame from camera")
                return None
            
            # Convert BGR (OpenCV default) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to target size (256, 256)
            resized_frame = cv2.resize(frame_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            return resized_frame
            
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None
    
    def release_camera(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera released")
    
    def __del__(self):
        """Destructor to ensure camera is released."""
        self.release_camera()


    def analyse_img(self, img):
        """
        Input: img HxWx3 RGB uint8
        Returns: 
            - bbox (x_min, y_min, width, height) or None: bounding box in (x,y,w,h) format
            - img_size (width, height): image dimensions  
            - mask (H,W): binary mask of detected red pixels
        """
        if img is None:
            return None, None, None
        
        # Convert RGB to HSV for better color detection under varying lighting
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Create mask using HSV ranges (more robust to shadows/lighting)
        # For red objects: Hue ~0-15 or ~165-180, Saturation >30, Value >50
        mask1 = cv2.inRange(hsv, (0, 200, 50), (30, 255, 255))      # Red hue range 1
        mask2 = cv2.inRange(hsv, (150, 200, 50), (179, 255, 255))   # Red hue range 2  
        mask = mask1 | mask2  # Combine both red ranges

        # mask1 = cv2.inRange(hsv, (0, 50, 50), (15, 255, 255))        # Pure red range
        # mask2 = cv2.inRange(hsv, (170, 50, 50), (179, 255, 255))     # Dark red range  
        mask3 = cv2.inRange(hsv, (15, 200, 50), (35, 255, 255))       # Orange-red range (plastic shift)
        mask = mask1 | mask2 | mask3  # Combine all ranges
        
        # Debug info (remove this later)
        # print(f"Mask pixels - Pure red: {mask1.sum()}, Dark red: {mask2.sum()}, Orange: {mask3.sum()}, Total: {mask.sum()}")
        
        
        # RGB approach is simpler but less robust to lighting changes
        mask_rgb1 = np.all((img >= self.red_min) & (img <= self.red_max), axis=-1)  # Original RGB range
        # 
        # Additional RGB ranges for plastic balloon variations
        # Bright red areas
        red_min_bright = np.array([180, 0, 0])
        red_max_bright = np.array([255, 100, 100])
        mask_rgb2 = np.all((img >= red_min_bright) & (img <= red_max_bright), axis=-1)
        # 
        # Darker red areas (shadows)
        red_min_dark = np.array([100, 0, 0])
        red_max_dark = np.array([200, 20, 20])
        mask_rgb3 = np.all((img >= red_min_dark) & (img <= red_max_dark), axis=-1)
        # 
        # Orange-ish red (plastic color shift)
        red_min_orange = np.array([180, 20, 0])
        red_max_orange = np.array([255, 50, 30])
        mask_rgb4 = np.all((img >= red_min_orange) & (img <= red_max_orange), axis=-1)
        # 
        # Combine all RGB masks
        mask = mask_rgb1 | mask_rgb2 | mask_rgb3 | mask_rgb4
        print(f"RGB Mask pixels - Original: {mask_rgb1.sum()}, Bright: {mask_rgb2.sum()}, Dark: {mask_rgb3.sum()}, Orange: {mask_rgb4.sum()}, Total: {mask.sum()}")
        
        # To switch between HSV and RGB approaches, comment/uncomment the relevant sections above
        
        
        # Find indices of True values
        true_indices = np.argwhere(mask)
        if true_indices.size == 0:
            return None, None, mask  # No cluster found
        # Cluster analysis: bounding box

        # Remove isolated pixels (noise) by checking neighbors
        # mask is shape (256, 256), True for red pixels
        if self.enable_noise_reduction:
            # Erode mask to remove pixels without connected neighbors
            cleaned_mask = binary_erosion(mask, structure=self.erosion_structure, border_value=0)
        else:
            # Skip noise reduction if disabled
            cleaned_mask = mask

        # Only keep pixels that are part of a cluster (not isolated)
        true_indices = np.argwhere(cleaned_mask)
        
        # Check if any red pixels remain after cleaning
        if true_indices.size == 0:
            return None, None, mask  # No cluster found after cleaning
            
        y_min, x_min = true_indices.min(axis=0)
        y_max, x_max = true_indices.max(axis=0)
        
        # Convert to (x, y, width, height) format - consistent (x,y) coordinates
        bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)  # (x, y, width, height)
        img_size = (img.shape[1], img.shape[0])  # (width, height) of image

        return bbox, img_size, cleaned_mask  # Return the mask actually used for bbox calculation