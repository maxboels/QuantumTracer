from picamera2 import Picamera2
import os
from datetime import datetime
from src.position_estimation import PositionEstimator
from vision import BasicDetector, MJPEGStreamer
from control import BasicController
from actuator_controls import ActuatorControls
import time
import math
import queue

# --- Globals for smoothing and frame timing ---
_smooth_alpha = 0.6
_outlier_fraction = 0.5
_last_proc_time = 0.0
_min_proc_dt = 0.0

def compute_distance_uncertainty(S_m, f_px, p_px,
                                 sigma_S_frac=0.02, sigma_f_frac=0.03, sigma_p_px=0.7):
    rel_S = sigma_S_frac
    rel_f = sigma_f_frac
    rel_p = sigma_p_px / max(1.0, p_px)
    return math.sqrt(rel_S*rel_S + rel_f*rel_f + rel_p*rel_p)

_prev_distance = None
_prev_angle = None

def smooth_and_reject(d, a, alpha=_smooth_alpha, outlier_frac=_outlier_fraction):
    global _prev_distance, _prev_angle
    if _prev_distance is None:
        _prev_distance, _prev_angle = d, a
        return d, a, False
    if abs(d - _prev_distance) > outlier_frac * max(1e-6, _prev_distance):
        return _prev_distance, _prev_angle, True
    d_s = alpha * d + (1.0 - alpha) * _prev_distance
    a_s = alpha * a + (1.0 - alpha) * _prev_angle
    _prev_distance, _prev_angle = d_s, a_s
    return d_s, a_s, False


output_dir = "saved_frames"
os.makedirs(output_dir, exist_ok=True)

FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))
TIMEOUT = int(os.getenv("TIMEOUT", "20"))  # seconds
STREAMING_ENABLED = False
STREAM_PORT = int(os.getenv("STREAM_PORT", "8081"))

detector = BasicDetector()
estimator = PositionEstimator(params={"lookup_csv": "px_to_m.csv", "img_size": FRAME_WIDTH})
ctrl = BasicController()
actuator = ActuatorControls()

streamer = MJPEGStreamer(port=STREAM_PORT) if STREAMING_ENABLED else None
start_timestamp = None

def process_frame(request):
    global _last_proc_time, start_timestamp

    # optional throttle (non-blocking)
    if _min_proc_dt > 0.0:
        now = time.perf_counter()
        if (now - _last_proc_time) < _min_proc_dt:
            return
        _last_proc_time = now

    frame_rgb = request.make_array("main")  # HxWx3 RGB uint8

    # Detect object location and size (center (x_px,y_px) floats, diameter float px)
    coords, diameter, mask = detector.analyse_img(frame_rgb)

    if STREAMING_ENABLED and streamer is not None:
        debug_bgr = detector.make_debug_view(frame_rgb, mask, coords, diameter)
        if debug_bgr is not None:
            streamer.update(debug_bgr)

    if coords is None or diameter is None:
        return

    # Estimate distance and angle to object
    distance_angle = estimator.estimate(coords, diameter)
    if distance_angle is None:
        return
    distance, angle = distance_angle

    # Uncertainty (use estimator.f_px if available)
    S_m = 0.125  # balloon diameter in meters (fallback)
    f_px = getattr(estimator, "f_px", None)
    if f_px is None:
        f_px = 4.74 * FRAME_WIDTH / 6.45
    rel_unc = compute_distance_uncertainty(S_m, f_px, diameter)
    abs_unc = rel_unc * distance
    print(f"Detected center={coords}, diam={diameter:.2f}px -> dist={distance:.2f}m ±{abs_unc:.2f}m ({rel_unc*100:.1f}%) angle={angle:.2f}°")

    # Smooth and reject outliers. Use smoothed values for control.
    dist_s, ang_s, rejected = smooth_and_reject(distance, angle)
    if rejected:
        print("Outlier detected. Skipping actuation.")
        return

    throttle_angle = ctrl.get_command(dist_s, ang_s)
    if throttle_angle is None:
        print("Steering control could not be determined. Keeping course unchanged.")
        return

    throttle, steer_angle = throttle_angle
    print(f"Throttle: {throttle:.2f}, Steer: {steer_angle:.2f}")

    actuator.set_fwd_speed(throttle)
    actuator.set_steering_angle(steer_angle)

    # Timeout check (non-blocking)
    if start_timestamp and ((datetime.now().timestamp() - start_timestamp) > TIMEOUT):
        actuator.stop()
        return


def main():
    global start_timestamp, streamer

    start_timestamp = datetime.now().timestamp()

    if STREAMING_ENABLED and streamer is not None:
        streamer.start()
        print(f"[MJPEG] Streaming debug view at http://0.0.0.0:{STREAM_PORT}/")

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}))
    picam2.post_callback = process_frame

    # Lock camera controls for stable geometry (daytime indoors/outdoors)
    try:
        controls = {
            "ExposureTime": 8000,    # µs, ~1/125s
            "AnalogueGain": 1.2,
            "AwbEnable": False,
            "ColourGains": (1.2, 1.0)
        }
        picam2.set_controls(controls)
    except Exception as e:
        print("Warning: could not set Picamera2 controls:", e)

    try:
        picam2.set_controls({"AfMode": 0})
    except Exception:
        pass

    picam2.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        actuator.stop()
        if streamer is not None:
            streamer.stop()


if __name__ == "__main__":
    main()
