import numpy as np

class Input:

    def __init__(self):

        # horizon length of past snapshots
        self.H = 10

        # RGB, width height
        self.image = np.zeros([3,256,256])
        # snapshoots, dim (acceleration, steering)
        self.output_history = np.zeros([self.H,2])
    
    def set_image(self, image, ):
        self.image = image

    def set_input(self, output): 
        # add the new recording:
        self.output_history = np.vstack([self.output_history, output.reshape(1, 2)])
        # drop the oldest recording:
        self.ouput_history = self.ouput_history[1:]
        