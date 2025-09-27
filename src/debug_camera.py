import numpy as np, cv2, os
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720), "format": "BGR888"}))

print("Starting camera...")
picam2.start()

# diagnostic: capture one frame, print channel means, save variations
req = picam2.capture_request()
frame = req.make_array("main")   # raw array from camera
req.release()

print("shape:", frame.shape, "dtype:", frame.dtype)
print("channel means (c0,c1,c2):", float(frame[:,:,0].mean()), float(frame[:,:,1].mean()), float(frame[:,:,2].mean()))

# Save as-is and swapped to compare (two files)
cv2.imwrite("frame_as_is.png", frame)                 # write raw array
cv2.imwrite("frame_swapped_rgb.png", frame[:,:,::-1]) # channel-reversed copy
print("Saved frame_as_is.png and frame_swapped_rgb.png in cwd. Inspect both.")