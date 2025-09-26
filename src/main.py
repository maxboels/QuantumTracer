from picamera2 import Picamera2
import os
from datetime import datetime
from vision import BasicDetector
from control import BasicController
from actuator_controls import ActuatorControls
from time import sleep

output_dir = "saved_frames"  # Directory to save frames
os.makedirs(output_dir, exist_ok=True)

FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))
TIMEOUT = int(os.getenv("TIMEOUT", "20"))  # seconds

detector = BasicDetector()
actuator = ActuatorControls()

start_timestamp = None

def process_frame(request):
    frame = request.make_array("main")

    detector = BasicDetector()
    ctrl = BasicController(detector, params={}, img_size=FRAME_WIDTH)

    # Detect object location and size
    coords, diameter, _ = detector.analyse_img(frame)
    if coords is None or diameter is None:
        print("No object detected")
        return

    print(f"Detected object at {coords} with diameter {diameter}px")
    
    # Determine steering and throttle from estimated object position
    throttle_angle = ctrl.get_command(coords, diameter)

    if throttle_angle is None:
        print("Steering control could not be determined. Keeping course unchanged.")
        return
    
    throttle, angle = throttle_angle
    print(f"Throttle: {throttle:.2f}, Angle: {angle:.2f}")

    # Convert the throttle and angle to actuator commands
    actuator.set_fwd_speed(throttle)
    actuator.set_steering_angle(angle)

    # Check for timeout
    if (datetime.now().timestamp() - start_timestamp) > TIMEOUT:
        actuator.stop()
        return
    
    sleep(0.2)


def main():
    global start_timestamp
    # UTC timestamp of current time
    start_timestamp = datetime.now().timestamp()
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}))
    picam2.post_callback = process_frame
    picam2.start()

    # Keep the script alive
    running = True

    try:
        while running:
            pass
    except KeyboardInterrupt:
        picam2.stop()

if __name__ == "__main__":
    main()
