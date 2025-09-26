# src/vision_control_approach/control.py
import numpy as np
import os

class BasicController:
    def __init__(self, detector, params: dict, img_size: int = 256):
        self.detector = detector
        self.params = dict(params)
        self.img_size = img_size
        # smoothing and state
        self.smoothing_alpha = float(self.params.get("smoothing_alpha", 0.4))
        self.min_diam_px = int(self.params.get("min_diameter_px", 6))
        self.cx_prev = None
        self.d_prev = None
        # lunge state
        self.in_lunge = False
        self.lunge_end_t = 0.0
        # load optional lookup table if provided
        self.lookup = None
        lookup_file = self.params.get("lookup_csv")
        if lookup_file and os.path.exists(lookup_file):
            try:
                import csv
                pxs = []; ds = []
                with open(lookup_file, "r") as fh:
                    rdr = csv.reader(fh)
                    for r in rdr:
                        if not r: continue
                        try:
                            px = float(r[0]); d = float(r[1])
                            pxs.append(px); ds.append(d)
                        except:
                            continue
                if len(pxs) >= 2:
                    self.lookup = (np.array(pxs), np.array(ds))
            except Exception:
                self.lookup = None
        # parameters
        self.steer_scale = float(self.params.get("steer_scale", 0.45))
        self.Kp_dist = float(self.params.get("Kp_dist", 0.7))
        self.desired_dist = float(self.params.get("desired_dist_m", 0.6))
        self.collision_threshold = float(self.params.get("collision_threshold_m", 0.45))
        self.lunge_power = float(self.params.get("lunge_power_pct", 0.85))
        self.lunge_time = float(self.params.get("lunge_time_ms", 300)) / 1000.0
        # if calibration constant 'k' present use it: z = k / diameter_px
        self.k = float(self.params.get("k", 0.0)) if self.params.get("k") else None
        # fallback focal_width method expects focal_length_px and target_width_m
        self.f_px = float(self.params.get("focal_length_px", 600.0))
        self.target_w = float(self.params.get("target_width_m", 0.12))
        # clamps
        self.min_dist_m = float(self.params.get("min_dist_m", 0.05))
        self.max_dist_m = float(self.params.get("max_dist_m", 50.0))

    def _diameter_to_distance(self, diam_px):
        if diam_px is None or diam_px <= 0:
            return None
        # if lookup available and diam in range, prefer interpolation
        if self.lookup is not None:
            pxs, ds = self.lookup
            # if diam within table range
            if diam_px >= pxs.min() and diam_px <= pxs.max():
                return float(np.interp(diam_px, pxs, ds))
        # if calibration constant k available use direct k/px
        if self.k:
            z = (self.k / float(diam_px))
            return float(np.clip(z, self.min_dist_m, self.max_dist_m))
        # fallback to pinhole: z = (W * f) / px
        if self.target_w > 0 and self.f_px > 0:
            z = (self.target_w * self.f_px) / float(diam_px)
            return float(np.clip(z, self.min_dist_m, self.max_dist_m))
        return None

    def get_command(self, coords, diam):
        cy, cx = coords
        if cx is None or cy is None or diam is None:
            return None

        # reject tiny detections
        if diam < self.min_diam_px:
            print(f"[WARN] Detected diameter too small: {diam:.1f}px < min {self.min_diam_px}px")
            return None

        # smoothing EMA
        if self.cx_prev is None:
            cx_s = float(cx); d_s = float(diam)
        else:
            a = float(self.smoothing_alpha)
            cx_s = a * float(cx) + (1.0 - a) * float(self.cx_prev)
            d_s = a * float(diam) + (1.0 - a) * float(self.d_prev)

        self.cx_prev = float(cx_s); self.d_prev = float(d_s)

        # distance (use smoothed diameter)
        z = self._diameter_to_distance(d_s)
        if z is None:
            print(f"[WARN] Could not estimate distance from diameter {d_s:.1f}px")
            return None

        # proportional steering from smoothed cx
        img_cx = self.img_size // 2
        norm = (cx_s - img_cx) / float(img_cx)
        angle = (0.5 + np.clip(norm * self.steer_scale, -self.steer_scale, self.steer_scale)) * 2 - 1

        # throttle control
        raw = self.Kp_dist * (z - self.desired_dist)
        print(f"Kp_dist: {self.Kp_dist}, z: {z}m, desired: {self.desired_dist}m => raw throttle {raw}")
        throttle = float(np.clip(raw, 0.0, 1.0))

        return throttle, angle
