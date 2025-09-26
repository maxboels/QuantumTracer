# test_runner.py
import cv2, time, argparse, os, csv, json
import numpy as np
from src.vision_control_approach.vision import Input, BasicDetector
from src.vision_control_approach.control import BasicController

# load params.json if present
params = {}
if os.path.exists("params.json"):
    try:
        with open("params.json","r") as fh:
            params = json.load(fh)
            print("Loaded params.json:", params)
    except Exception as e:
        print("Failed to read params.json:", e)

# allow specifying lookup_csv path (if created by calibrate_target)
if "lookup_csv" not in params and os.path.exists("lookup_px_to_m.csv"):
    params["lookup_csv"] = "lookup_px_to_m.csv"

# runtime params defaults
RUNTIME = {
    "focal_length_px": params.get("focal_length_px", 600.0),
    "target_width_m": params.get("target_width_m", 0.12),
    "desired_dist_m": params.get("desired_dist_m", 0.6),
    "collision_threshold_m": params.get("collision_threshold_m", 0.45),
    "lunge_power_pct": params.get("lunge_power_pct", 0.85),
    "lunge_time_ms": params.get("lunge_time_ms", 300),
    "steer_scale": params.get("steer_scale", 0.45),
    "deadzone_px": params.get("deadzone_px", 8),
    "Kp_dist": params.get("Kp_dist", 0.7),
    "smoothing_alpha": params.get("smoothing_alpha", 0.4),
    "min_bbox_w_px": params.get("min_bbox_w_px", 6),
    "min_diameter_px": params.get("min_diameter_px", 6),
    "k": params.get("k", None),
    "lookup_csv": params.get("lookup_csv", None)
}
PARAMS_RUNTIME = RUNTIME

CSV_HEADER = ["timestamp_ms","frame_id","image_path","bbox_x1","bbox_y1","bbox_x2","bbox_y2","bbox_conf","track_id","rel_x_m","rel_y_m","rel_dist_m","rel_angle_deg","target_vx_mps","vehicle_speed_mps","steering_cmd_deg","throttle_cmd_pct","brake_cmd_pct","mode","battery_v","hit_bool","hit_type","balloon_popped","collision_time_ms","run_id"]
DIAG_HEADER = ["timestamp_ms","frame_id","cx_raw","diam_raw","cx_s","d_s","rel_dist_m","in_lunge"]

def make_synthetic_image():
    import numpy as np, cv2
    img = np.zeros((256,256,3), dtype=np.uint8)
    cv2.circle(img, (128,120), 25, (0,0,255), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def main(mode):
    out_dir = "logs/test_run"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "log.csv")
    diag_path = os.path.join(out_dir, "diag.csv")
    f = open(csv_path, "w", newline="")
    fd = open(diag_path, "w", newline="")
    writer = csv.writer(f); diag_writer = csv.writer(fd)
    writer.writerow(CSV_HEADER); diag_writer.writerow(DIAG_HEADER)

    detector = BasicDetector()
    inp = Input()
    ctrl = BasicController(detector, PARAMS_RUNTIME)
    cap = None
    frame_id = 0
    if mode == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: cannot open webcam"); return
        # try to set resolution (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    try:
        while True:
            if mode == "synth":
                img = make_synthetic_image(); time.sleep(0.05)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("no frame"); break
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256,256))
            inp.set_img(img)
            out = ctrl.steer(inp.img, acc_request=0.4)
            c, dim, mask = detector.analyse_img(inp.img)
            bbox = (0,0,0,0); bbox_conf = 0.0; rel_dist = 0.0; rel_angle = 0.0
            if c is not None and dim is not None:
                cy, cx = c; bh = dim[0]; bw = dim[0]
                x1 = cx - bw//2; y1 = cy - bh//2; x2 = cx + bw//2; y2 = cy + bh//2
                bbox = (x1,y1,x2,y2); bbox_conf = 1.0
                # compute raw rel_dist using params k if present (use raw diam)
                if PARAMS_RUNTIME.get("k"):
                    rel_dist = PARAMS_RUNTIME["k"] / float(bw) if bw>0 else 0.0
                else:
                    rel_dist = (PARAMS_RUNTIME["target_width_m"] * PARAMS_RUNTIME["focal_length_px"]) / float(bw) if bw>0 else 0.0
                rel_angle = np.degrees(np.arctan2((cx - 128) * rel_dist / PARAMS_RUNTIME["focal_length_px"], rel_dist)) if rel_dist else 0.0

            vis = cv2.cvtColor(inp.img.copy(), cv2.COLOR_RGB2BGR)
            if c is not None:
                x1,y1,x2,y2 = bbox
                cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                cv2.circle(vis, (int((x1+x2)//2), int((y1+y2)//2)), 3, (255,0,0), -1)
            debug = getattr(out, "_debug", {}) or {}
            rd = debug.get("rel_dist_m")
            if rd is None: rd = 0.0
            lunge = bool(debug.get("in_lunge", False))
            cv2.putText(vis, f"acc:{out.acc:.2f} angle:{out.angle:.2f} dist:{rd:.2f}m LUNGE:{int(lunge)}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            img_path = os.path.join(out_dir, f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(img_path, vis)
            timestamp_ms = int(time.time()*1000)
            row = [
                timestamp_ms, frame_id, img_path,
                bbox[0], bbox[1], bbox[2], bbox[3], bbox_conf, 0,
                None, None,
                rel_dist, rel_angle,
                0.0, 0.0, out.angle, out.acc, 0.0, "auto", 0.0, False, "", False, 0, "test_run"
            ]
            writer.writerow(row)
            diag_row = [
                timestamp_ms, frame_id,
                debug.get("cx_raw"), debug.get("diam_raw"),
                debug.get("cx_s"), debug.get("d_s"),
                debug.get("rel_dist_m"), int(debug.get("in_lunge", False))
            ]
            diag_writer.writerow(diag_row)
            f.flush(); fd.flush()
            frame_id += 1

            cv2.imshow("vis", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or frame_id>2000:
                break
    finally:
        if cap: cap.release()
        f.close(); fd.close()
        cv2.destroyAllWindows()
        print("log saved to", csv_path)
        print("diag saved to", diag_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synth","webcam"], default="synth")
    args = parser.parse_args()
    main(args.mode)
