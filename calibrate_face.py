# calibrate_face.py
import cv2
import numpy as np
import time
import json
import os

CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: cannot open webcam")
    raise SystemExit(1)

face_cascade = cv2.CascadeClassifier(CASCADE)
print("Instructions:")
print(" - Position yourself facing the camera at a known distance (D).")
print(" - Measure distance D in meters (e.g. 1.00) BEFORE capturing.")
print(" - Press 'c' to capture a sample when face box looks correct.")
print(" - Press 'q' to finish and compute calibration.")

samples = []
frame_id = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, (640,480))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        # draw faces
        for (x,y,w,h) in faces:
            cv2.rectangle(frame_small, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame_small, f"samples: {len(samples)}  frame:{frame_id}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),1)
        cv2.imshow("calib (press c to capture, q to quit)", frame_small)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # capture largest face width in px (w)
            if len(faces) == 0:
                print("No face detected in this frame. Try again.")
            else:
                # pick largest face
                faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
                x,y,w,h = faces_sorted[0]
                samples.append(int(w))
                print(f"Captured sample #{len(samples)}: face_width_px = {w}")
        elif key == ord('q'):
            break
        frame_id += 1
finally:
    cap.release()
    cv2.destroyAllWindows()

if len(samples) == 0:
    print("No samples captured. Run again and press 'c' when face detected.")
    raise SystemExit(1)

# compute median pixel width
pw_px = int(np.median(np.array(samples)))
print(f"\nMedian face width in pixels from samples: {pw_px} px")

# ask user for real-world width and distance
def ask_float(prompt, default):
    try:
        v = input(f"{prompt} [default {default}]: ").strip()
        if v == "":
            return float(default)
        return float(v)
    except:
        return float(default)

print("\nNow provide calibration measurements.")
W = ask_float("Enter real face width W in meters (typical adult face width ~0.16 m)", 0.16)
D = ask_float("Enter distance D from camera when capturing (meters). Measure this with a ruler/tape.", 1.0)

# compute focal length
f = (pw_px * D) / W
print(f"\nComputed focal length f = {f:.2f} px (using W={W} m, D={D} m, pw_px={pw_px})")

# save to params.json
params = {"focal_length_px": float(f), "target_width_m": float(W)}
with open("params.json", "w") as fh:
    json.dump(params, fh, indent=2)
print("Saved params.json with focal_length_px and target_width_m in repo root.")
print("params.json contents:")
print(json.dumps(params, indent=2))
