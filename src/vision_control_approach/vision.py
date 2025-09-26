import numpy as np
from typing import Tuple

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
        img = np.transpose(img, (1, 2, 0))
        self.img = img

    def set_input(self, output): 
        # add the new recording:
        self.output_history = np.vstack([self.output_history, output.reshape(1, 2)])
        # drop the oldest recording:
        self.ouput_history = self.ouput_history[1:]


class BasicDetector:
    
    def __init__(self,
                 red_min: Tuple[int] = (200, 0, 0),
                 red_max: Tuple[int] = (255, 80, 80)):
        # RGB intervals for saturated red
        self.red_min = np.array(red_min)
        self.red_max = np.array(red_max)

    def analyse_img(self, img):
        # img shape: (3, 256, 256)
        # Create mask for saturated red pixels
        mask = np.all((img >= self.red_min) & (img <= self.red_max), axis=-1)
        # Find indices of True values
        true_indices = np.argwhere(mask)
        if true_indices.size == 0:
            return None, None, mask  # No cluster found
        # Cluster analysis: bounding box
        y_min, x_min = true_indices.min(axis=0)
        y_max, x_max = true_indices.max(axis=0)
        center = ((y_min + y_max) // 2, (x_min + x_max) // 2)
        dimension = (y_max - y_min + 1, x_max - x_min + 1)
        return center, dimension, mask