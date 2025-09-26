import numpy as np
import cv2

class BasicDetector:
    def __init__(self,
                 hsv_lower1=(0,120,70),
                 hsv_upper1=(10,255,255),
                 hsv_lower2=(170,120,70),
                 hsv_upper2=(180,255,255),
                 morph_kernel_size=5,
                 min_area_px=20):
        self.hsv_lower1 = np.array(hsv_lower1, dtype=np.uint8)
        self.hsv_upper1 = np.array(hsv_upper1, dtype=np.uint8)
        self.hsv_lower2 = np.array(hsv_lower2, dtype=np.uint8)
        self.hsv_upper2 = np.array(hsv_upper2, dtype=np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        self.min_area_px = min_area_px

    def analyse_img(self, img):
        """
        Input: img HxWx3 RGB uint8
        Returns: center (y,x) or None, dimension (diameter_px,diameter_px) or None, mask (H,W)
        """
        if img is None:
            return None, None, None
        # ensure uint8 RGB
        img_u8 = img.astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        mask2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        # clean
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        # contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, None, mask
        # pick largest contour
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area_px:
            return None, None, mask
        (x,y), radius = cv2.minEnclosingCircle(c)
        diameter = 2.0 * float(radius)
        # convert to integer center in (row,col) format to match control code (y,x)
        center = (int(round(y)), int(round(x)))
        diam = int(round(diameter))
        return center, diam, mask
