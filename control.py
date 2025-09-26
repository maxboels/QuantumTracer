import numpy as np

class Output():

    def __init__(self):
        self.steering = 0.5
        self.acceleration = 0.5
        self.output = np.array([self.acceleration, self.steering])

    def pass_to_controller():
        # access the interface of the car

        return True