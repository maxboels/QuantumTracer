# calibrate_target.py
# Captures the actual red target (tag/balloon) and computes calibration constant k = pw_px * D
import cv2, json, time, numpy as np, os
from src.vision_control_approach.vision import BasicDetector

det = BasicDetector()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: cannot open webcam"); raise SystemExit(1)

print("Calibration for your actual red target.")
print(" - Stand or place the red object at the known distance D (meters).")
print(" - Measure D with a tape measure. Typical small tag: D between 0.10 and 2.0 m.")
print(" - Press 'c' to capture a sample when mask and enclosing circle look correct.")
print(" - Press 'q' to finish and compute calibration.")

samples = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256))
    center, dim, mask = det.analyse_img(img)
    vis = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    if center is not None:
        cy, cx = center
        d = dim[0]
        x1 = int(cx - d//2); y1 = int(cy - d//2); x2 = int(cx + d//2); y2 = int(cy + d//2)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(vis, f"d_px:{d:.1f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
    cv2.imshow("calibrate (c capture, q quit)", vis)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if center is None:
            print("No detection. Move target into view and try again.")
        else:
            samples.append(float(dim[0])); print(f"Captured diameter px = {dim[0]:.1f}")
    elif key == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()

if len(samples) == 0:
    print("No samples captured. Aborting."); raise SystemExit(1)

pw_px = float(np.median(np.array(samples)))
print(f"Median pixel diameter: {pw_px:.2f} px")
# ask user for physical diameter W and distance D
def ask_float(prompt, default):
    try:
        v = input(f"{prompt} [default {default}]: ").strip()
        return float(v) if v != "" else float(default)
    except:
        return float(default)
W = ask_float("Enter real target width / diameter W in meters (e.g. balloon ~0.12, tag maybe 0.05)", 0.12)
D = ask_float("Enter distance D (meters) at which you captured the samples", 0.5)

# compute calibration constant k = pw_px * D
k = pw_px * D
# compute focal length as f = k / W
f = k / W if W>0 else 0.0

params = {"k": float(k), "focal_length_px": float(f), "target_width_m": float(W)}
with open("params.json", "w") as fh:
    json.dump(params, fh, indent=2)
print("Saved params.json:", params)

# generate simple lookup CSV using k: px -> dist = k / px for px in reasonable range
csv_path = os.path.join("lookup_px_to_m.csv")
with open(csv_path, "w") as fh:
    fh.write("px,dist_m\n")
    for px in range(4,501,4):
        d = k/px
        fh.write(f"{px},{d:.6f}\n")
print("Wrote lookup_px_to_m.csv (editable). You can set lookup_csv in params.json if you want to use it.")
