"""Physio-TabPFN Web Application.

Design:
- the design:
  * Bold headline typography ("YOU'RE A..." style with prominent bold uppercase text).
  * Animated floating background SVG/mesh waves pattern.
  * Strict Black, White & Grey monochrome palette.
  * Side panel with Posture Matrix Widgets showing live counts per exercise.
  * Highest predicted class highlighted on top in large, bold characters.
  * Dynamic, live updating Form Quality % and Feedback.
  * Strict rep counter: valid reps increment only on clean form, bad reps are rejected.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import json
from pathlib import Path
import threading
import time
from typing import Optional
from urllib.parse import parse_qs, urlparse
import cv2
import numpy as np

from physio_tabPFN import (
    CameraFeed,
    PoseEstimator,
    PhysioTabPFNModel,
    QualityRepCounter,
    extract_exercise_features,
    is_full_body_visible,
)
from physio_tabPFN.env_loader import load_env

load_env()

DATASET_CSV = Path(__file__).resolve().parent / "data" / "exercise_dataset.csv"


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpeg_frame: Optional[bytes] = None
        self.predicted_exercise = "Idle"
        self.confidence = 0.0
        self.probabilities = {"Idle": 1.0, "Squats": 0.0, "Step-Ups": 0.0}
        self.latency_ms = 0.0
        self.mode = "LOCAL"
        self.active_exercise = "Squats"
        self.counter = QualityRepCounter(self.active_exercise)

    def set_active_exercise(self, name: str):
        with self.lock:
            self.active_exercise = name
            self.counter.reset(name, reset_counts=False)
            print(f"[App] Target exercise changed to: {name}")

    def reset_counter(self):
        with self.lock:
            self.counter.reset(self.active_exercise, reset_counts=True)
            print("[App] Counters reset")

    def get_status_json(self) -> str:
        with self.lock:
            summary = self.counter.get_summary()
            data = {
                "active_exercise": self.active_exercise,
                "predicted_exercise": self.predicted_exercise,
                "confidence": round(self.confidence * 100, 1),
                "latency_ms": round(self.latency_ms, 1),
                "mode": self.mode,
                "valid_reps": summary["valid_reps"],
                "bad_reps": summary["bad_reps"],
                "total_attempts": summary["total_attempts"],
                "quality_score": summary["quality_score"],
                "feedback": summary["feedback"],
                "last_rep_status": summary["last_rep_status"],
                "posture_counts": summary["posture_counts"],
                "exercise_reps": summary.get("exercise_reps", {}),
                "probabilities": {k: round(v * 100, 1) for k, v in self.probabilities.items()},
            }
            return json.dumps(data)


state = AppState()

TEMPLATE_PATH = Path(__file__).resolve().parent / "src" / "physio_tabPFN" / "web" / "templates" / "index.html"
if TEMPLATE_PATH.exists():
    HTML_PAGE = TEMPLATE_PATH.read_text(encoding="utf-8")
else:
    raise FileNotFoundError(f"Template not found at {TEMPLATE_PATH}")


class WebRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(state.get_status_json().encode("utf-8"))

        elif path == "/set_exercise":
            params = parse_qs(parsed.query)
            if "name" in params and len(params["name"]) > 0:
                exercise_name = params["name"][0]
                state.set_active_exercise(exercise_name)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        elif path == "/reset":
            state.reset_counter()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        elif path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with state.lock:
                        frame_bytes = state.latest_jpeg_frame
                    if frame_bytes is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                    time.sleep(0.033)  # ~30 FPS
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


def run_cv_inference_loop(model: PhysioTabPFNModel):
    """Background worker continuously capturing video, running TabPFN, and updating state."""
    with CameraFeed(source=0, width=1280, height=720) as cam, PoseEstimator() as estimator:
        frame_idx = 0
        while True:
            rgb_frame = cam.read_rgb()
            if rgb_frame is None:
                with state.lock:
                    if state.latest_jpeg_frame is None:
                        standby = np.ones((720, 1280, 3), dtype=np.uint8) * 255
                        cv2.putText(standby, "Camera Initializing...", (420, 360),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 60), 2)
                        _, jpg = cv2.imencode(".jpg", standby)
                        state.latest_jpeg_frame = jpg.tobytes()
                time.sleep(0.01)
                continue

            frame_idx += 1
            results = estimator.process_frame(rgb_frame)
            landmarks_np = estimator.extract_numpy_landmarks(results)

            preview_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            if landmarks_np is not None:
                body_ok, body_msg = is_full_body_visible(landmarks_np)

                if not body_ok:
                    estimator.draw_landmarks(preview_frame, results)
                    # High contrast monochrome banner on preview feed
                    h, w = preview_frame.shape[:2]
                    banner_h = 70
                    overlay = preview_frame.copy()
                    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (17, 24, 39), -1)
                    cv2.addWeighted(overlay, 0.85, preview_frame, 0.15, 0, preview_frame)
                    cv2.putText(
                        preview_frame,
                        "STEP BACK: FULL BODY REQUIRED",
                        (max(20, w // 2 - 240), h - 38),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        preview_frame,
                        "Hips, knees, and feet must be in camera frame to track reps.",
                        (max(20, w // 2 - 270), h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (209, 213, 219),
                        1,
                        cv2.LINE_AA,
                    )
                    with state.lock:
                        state.predicted_exercise = "Step Back"
                        state.confidence = 0.0
                        state.counter.feedback = body_msg
                        state.counter.update("Step Back", 0.0, None)
                else:
                    estimator.draw_landmarks(preview_frame, results)
                    features = extract_exercise_features(landmarks_np, require_full_body=True)

                    if features is not None:
                        # Query TabPFN every 3 frames
                        if frame_idx % 3 == 0:
                            t0 = time.perf_counter()
                            try:
                                raw_lbl, raw_conf, raw_probs = model.predict_single(features, min_confidence=0.0)
                                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                                with state.lock:
                                    # Temporal EMA smoothing across inference frames (65% current, 35% historical)
                                    smoothed_probs = {}
                                    for cls, p in raw_probs.items():
                                        prev_p = state.probabilities.get(cls, p)
                                        smoothed_probs[cls] = round(0.65 * p + 0.35 * prev_p, 4)

                                    top_cls = max(smoothed_probs, key=smoothed_probs.get)
                                    top_conf = smoothed_probs[top_cls]

                                    # Mark as Unknown if confidence < 60%
                                    if top_conf < 0.60:
                                        lbl = "Unknown"
                                        conf = top_conf
                                    else:
                                        lbl = top_cls
                                        conf = top_conf

                                    state.predicted_exercise = lbl
                                    state.confidence = conf
                                    state.probabilities = smoothed_probs
                                    state.latency_ms = (
                                        (0.7 * state.latency_ms + 0.3 * elapsed_ms)
                                        if state.latency_ms > 0
                                        else elapsed_ms
                                    )

                                    # Auto-synchronize active exercise to detected posture if standing
                                    if lbl in ("Squats", "Step-Ups") and conf >= 0.60:
                                        if state.counter.state == "standing" and state.active_exercise != lbl:
                                            state.active_exercise = lbl
                                            state.counter.exercise = lbl

                                    state.counter.update(lbl, conf, features)
                            except Exception as e:
                                print(f"[Model Warning]: {e}")
                                with state.lock:
                                    state.counter.update(state.predicted_exercise, state.confidence, features)
                        else:
                            with state.lock:
                                state.counter.update(state.predicted_exercise, state.confidence, features)
                    else:
                        with state.lock:
                            state.counter.update(state.predicted_exercise, state.confidence, None)
            else:
                with state.lock:
                    state.predicted_exercise = "Idle"
                    state.confidence = 1.0
                    state.counter.update("Idle", 1.0, None)

            # JPEG encode for web streaming
            ret, jpeg = cv2.imencode(".jpg", preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                with state.lock:
                    state.latest_jpeg_frame = jpeg.tobytes()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    print("=" * 65)
    print(" PHYSIO-TABPFN MINIMALIST WEB APPLICATION")
    print(" Initializing Model & In-Context Dataset...")
    print("=" * 65)

    tabpfn_model = PhysioTabPFNModel(mode="auto", N_ensemble_configurations=1)
    tabpfn_model.fit_from_csv(DATASET_CSV, max_samples_per_class=12)
    state.mode = tabpfn_model.active_mode.upper()

    cam_thread = threading.Thread(target=run_cv_inference_loop, args=(tabpfn_model,), daemon=True)
    cam_thread.start()

    server = ThreadedHTTPServer(("0.0.0.0", 5001), WebRequestHandler)
    print("\n" + "=" * 65)
    print(" 🚀 WEB SERVER READY!")
    print(" Open in your browser: http://localhost:5001")
    print("=" * 65 + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server...")
        server.server_close()


if __name__ == "__main__":
    main()
