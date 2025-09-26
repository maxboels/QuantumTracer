import numpy as np
import cv2
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import time

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

    def analyse_img(self, img_rgb):
        if img_rgb is None:
            return None, None, None
        img_u8 = img_rgb.astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        mask2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, None, mask
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area_px:
            return None, None, mask
        (x, y), radius = cv2.minEnclosingCircle(c)
        diameter = 2.0 * float(radius)
        center = (int(round(y)), int(round(x)))  # (row, col)
        diam = int(round(diameter))
        return center, diam, mask

    def make_debug_view(self, img_rgb, mask, center, diam,
                        label_font=cv2.FONT_HERSHEY_SIMPLEX):
        if img_rgb is None:
            return None

        # Panel 1: Original
        orig_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(orig_bgr, "Original", (10, 30), label_font, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Panel 2: Segmentation
        if mask is None:
            mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        mask_norm = (mask > 0).astype(np.uint8) * 255
        mask_color = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
        cv2.putText(mask_color, "Segmentation", (10, 30), label_font, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Panel 3: Overlay with circle + green midpoint dot
        overlay = orig_bgr.copy()
        if center is not None and diam is not None and diam > 0:
            y, x = int(center[0]), int(center[1])
            radius = int(max(1, diam // 2))
            # Circle outline (green)
            cv2.circle(overlay, (x, y), radius, (0, 255, 0), 2)
            # Midpoint dot (green)
            cv2.circle(overlay, (x, y), 6, (0, 255, 0), -1)
            # Label
            cv2.putText(overlay, f"(x={x}, y={y}), d={diam}px",
                        (10, 60), label_font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(overlay, "No detection", (10, 60),
                        label_font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(overlay, "Circle + midpoint", (10, 30),
                    label_font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Stack panels
        h = orig_bgr.shape[0]
        def resize_h(img): return cv2.resize(img, (img.shape[1], h),
                                             interpolation=cv2.INTER_AREA)
        composite = cv2.hconcat([resize_h(orig_bgr),
                                 resize_h(mask_color),
                                 resize_h(overlay)])

        # Downscale wide composite if needed
        max_width = 1600
        if composite.shape[1] > max_width:
            scale = max_width / composite.shape[1]
            composite = cv2.resize(composite,
                                   (int(composite.shape[1]*scale),
                                    int(composite.shape[0]*scale)),
                                   interpolation=cv2.INTER_AREA)
        return composite


# ----------------- MJPEG streamer code stays the same -----------------
class _StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html", "/stream.mjpg"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Age", 0)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        try:
            while True:
                frame = self.server.parent.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                self.wfile.write(b"--FRAME\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            pass

    def log_message(self, *args, **kwargs):
        return


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    def __init__(self, server_address, RequestHandlerClass, parent):
        super().__init__(server_address, RequestHandlerClass)
        self.parent = parent


class MJPEGStreamer:
    def __init__(self, host="0.0.0.0", port=8081, jpeg_quality=80):
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self._frame_bytes = None
        self._lock = threading.Lock()
        self._server = _ThreadedHTTPServer((self.host, self.port),
                                           _StreamingHandler, parent=self)
        self._thread = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        print(f"[MJPEG] Server started on http://{self.host}:{self.port}/")

    def stop(self):
        if not self._running:
            return
        self._server.shutdown()
        self._server.server_close()
        self._running = False
        print("[MJPEG] Server stopped.")

    def update(self, bgr_image):
        if bgr_image is None:
            return
        ok, buf = cv2.imencode(".jpg", bgr_image,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        with self._lock:
            self._frame_bytes = buf.tobytes()

    def get_frame(self):
        with self._lock:
            return self._frame_bytes
