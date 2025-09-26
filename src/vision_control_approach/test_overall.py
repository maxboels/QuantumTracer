import vision
import control

img_size = 256

detector = vision.BasicDetector()
controller = control.BasicController(detector, img_size)

print(controller.command)
print()