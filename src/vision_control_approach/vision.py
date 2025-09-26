import numpy as np
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
    
    def set_img(self, img, ):
        # Transpose to (256, 256, 3) for easier pixel-wise comparison
        # img = np.transpose(img, (1, 2, 0)) if needed transpose array into correct form
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
                 enable_noise_reduction: bool = True):
        # RGB intervals for saturated red
        self.red_min = np.array(red_min)
        self.red_max = np.array(red_max)
        
        # Morphological operations parameters
        self.erosion_kernel_size = erosion_kernel_size
        self.enable_noise_reduction = enable_noise_reduction
        # Create the structuring element (kernel) for erosion
        self.erosion_structure = np.ones((erosion_kernel_size, erosion_kernel_size))

    def analyse_img(self, img):
        # img shape: (256, 256, 3)
        # Create mask for saturated red pixels
        mask = np.all((img >= self.red_min) & (img <= self.red_max), axis=-1)
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
        y_min, x_min = true_indices.min(axis=0)
        y_max, x_max = true_indices.max(axis=0)
        center = ((y_min + y_max) // 2, (x_min + x_max) // 2)
        dimension = (y_max - y_min + 1, x_max - x_min + 1)
        return center, dimension, mask