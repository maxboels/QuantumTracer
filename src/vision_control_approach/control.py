import numpy as np
from typing import List


class Output():

    def __init__(self):
        self.steering = 0.5
        self.acceleration = 0.5
        self.output = np.array([self.acceleration, self.steering])

    def set_control(self, control_values: List[float]):
        # self.acceleration, self.steering = control_values

        self.output = np.array(control_values)


    def pass_to_controller():
        # access the interface of the car to pass the 

        return True