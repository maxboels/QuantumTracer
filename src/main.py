from picamera2 import Picamera2
import os
from datetime import datetime
from vision import BasicDetector, MJPEGStreamer
from control import BasicController
from actuator_controls import ActuatorControls
from time import sleep
from datetime import datetime


output_dir = "saved_frames"  # Directory to save frames
os.makedirs(output_dir, exist_ok=True)

FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))
TIMEOUT = int(os.getenv("TIMEOUT", "20"))  # seconds
STREAMING_ENABLED = os.getenv("STREAMING_ENABLED", "false") == "true"
STREAM_PORT = int(os.getenv("STREAM_PORT", "8081"))  # where you'll view on your laptop

detector = BasicDetector()
ctrl = BasicController(detector, params={"lookup_csv": "src/px_to_m.csv"}, img_size=FRAME_WIDTH)
actuator = ActuatorControls()

streamer = None
if STREAMING_ENABLED:
    streamer = MJPEGStreamer(port=STREAM_PORT)  # visit http://<pi-ip>:8081/

start_timestamp = None

def process_frame(request):
    frame_rgb = request.make_array("main")  # HxWx3 RGB uint8

    # Detect object location and size
    coords, diameter, mask = detector.analyse_img(frame_rgb)


    if STREAMING_ENABLED:
        # Compose a single debug frame: original | mask | overlay
        debug_bgr = detector.make_debug_view(frame_rgb, mask, coords, diameter)
        streamer.update(debug_bgr)  # publish to MJPEG stream

    if coords is None or diameter is None:
        print("No object detected")
        # still stream the debug view; don't actuate
        sleep(0.02)
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
    
    sleep(0.01)


def main():
    global start_timestamp
    # UTC timestamp of current time
    start_timestamp = datetime.now().timestamp()

    # Start streamer
    streamer.start()
    print(f"[MJPEG] Streaming debug view at http://0.0.0.0:{STREAM_PORT}/ (open from your laptop via the Pi's IP)")

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}))
    picam2.post_callback = process_frame
    picam2.start()

    # Keep the script alive
    try:
        while True:
            sleep(0.1)
    except KeyboardInterrupt:
        picam2.stop()
    finally:
        actuator.stop()
        streamer.stop()

if __name__ == "__main__":
    main()
