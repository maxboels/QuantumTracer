# src/vision_control_approach/control.py
import numpy as np

class BasicController:

    def __init__(self, Kp_dist):
        self.Kp_dist = Kp_dist

    def default_movement(DEFAULT_ANGLE):

        angle = DEFAULT_ANGLE
        throttle = 0.3

        return throttle, angle

    def get_command(self, distance, angle):
        """
        The angle is expected to be in range [-1,1] where 0 is straight ahead,

        This is a bit iffy (the estimator should probably return angle in degrees or radians
        and the controller should handle the conversion to steering command).

        However, this will do for now.
        """

        # throttle control
        raw = self.Kp_dist * (distance)
        print(f"Kp_dist: {self.Kp_dist}, distance: {distance}m => raw throttle {raw}")
        throttle = float(np.clip(raw, 0.0, 1.0))

        return throttle, angle
