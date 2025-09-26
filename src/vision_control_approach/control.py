import numpy as np
from typing import List


class Output():

    def __init__(self):
        self.steering = 0.5
        self.acceleration = 0.5
        self.command = np.array([self.acceleration, self.steering])

    def set_control(self, control_values: List[float]):
        # self.acceleration, self.steering = control_values
        self.command = np.array(control_values)
    

class BasicController:
    
    def __init__(self, detector, img_width=1024, img_height=1024):
        self.detector = detector
        self.fixed_angle = 0.25
        self.no_angle = 0.5
        self.img_width = img_width
        self.img_height = img_height

    def steer(self, bbox, img_size):
        """
        Control logic based on bounding box detection.
        
        Args:
            bbox: (x, y, width, height) - bounding box of detected object
            img_size: (width, height) - image dimensions
        """
        ##### decision logic ######

        # Extract center and object size from bbox
        x, y, width, height = bbox
        img_w, img_h = img_size

        center = x + width / 2  # horizontal center of bbox
        obj_size = [width, height]

        # dump angle rule:
        angle = self.no_angle
        if center >= img_w / 2 + 10:
            angle += self.fixed_angle
        elif center <= img_w / 2 - 10:
            angle -= self.fixed_angle

        # dump acceleration rule: 
        acc = 6 / sum(obj_size)  # 6 since kernel=3 and multiplied by two dimensions x,y which are summed

        ##### pass data to the controller ######
        
        output = Output()
        output.set_control([acc, angle])
        self.pass_to_controller(output)

        return True
    
    def run(self, img):

        center, object_size, img_size, mask = self.detector.analyse_img(img)

        self.steer(center, object_size, img_size)

    def pass_to_controller(self, output):
        # access the interface of the car to pass the
        # 
        print(output) 

        return True




        