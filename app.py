"""Physio-TabPFN Web Application.

Design:
- Inspired by the attached design:
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
        self.probabilities = {"Idle": 1.0, "Squats": 0.0, "Step-Ups": 0.0, "Pogo Hops": 0.0}
        self.latency_ms = 0.0
        self.mode = "LOCAL"
        self.active_exercise = "Squats"
        self.counter = QualityRepCounter(self.active_exercise)

    def set_active_exercise(self, name: str):
        with self.lock:
            self.active_exercise = name
            self.counter.reset(name)
            print(f"[App] Target exercise changed to: {name}")

    def reset_counter(self):
        with self.lock:
            self.counter.reset(self.active_exercise)
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
                "probabilities": {k: round(v * 100, 1) for k, v in self.probabilities.items()},
            }
            return json.dumps(data)


state = AppState()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PhysioTabPFN | Movement Analysis</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #ffffff;
      --card-bg: rgba(255, 255, 255, 0.95);
      --border: #e5e7eb;
      --border-dark: #111827;
      --text: #111827;
      --muted: #6b7280;
      --light-grey: #f3f4f6;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
      position: relative;
    }

    /* Animated Pattern Background (Subtle Monochrome Waves) */
    .bg-pattern {
      position: fixed;
      inset: 0;
      z-index: -1;
      opacity: 0.045;
      background-image: 
        radial-gradient(circle at 20% 30%, #000 1px, transparent 1px),
        radial-gradient(circle at 80% 70%, #000 1px, transparent 1px);
      background-size: 40px 40px;
      pointer-events: none;
    }
    .wave-bg {
      position: fixed;
      bottom: -80px;
      left: 0;
      width: 200%;
      height: 320px;
      z-index: -1;
      opacity: 0.035;
      background: repeat-x url('data:image/svg+xml;utf8,<svg viewBox="0 0 1200 120" xmlns="http://www.w3.org/2000/svg"><path d="M0,0 C150,90 350,-40 500,45 C650,130 900,-20 1200,40 L1200,120 L0,120 Z" fill="%23000"/></svg>');
      animation: waveFloat 22s linear infinite;
    }
    @keyframes waveFloat {
      0% { transform: translateX(0); }
      100% { transform: translateX(-50%); }
    }

    /* Header matching Poster Style */
    header {
      padding: 24px 48px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.85);
      backdrop-filter: blur(10px);
      position: sticky;
      top: 0;
      z-index: 50;
    }

    .brand-group {
      display: flex;
      flex-direction: column;
    }
    .brand-eyebrow {
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--muted);
    }
    .brand-title {
      font-family: 'Syne', sans-serif;
      font-size: 1.6rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      color: var(--text);
      line-height: 1.1;
    }

    .badge-bar {
      display: flex;
      gap: 10px;
    }
    .badge {
      font-size: 0.76rem;
      font-weight: 600;
      letter-spacing: 0.4px;
      padding: 6px 14px;
      border-radius: 6px;
      background: var(--light-grey);
      color: var(--muted);
      border: 1px solid var(--border);
    }

    /* Container Grid */
    .container {
      max-width: 1440px;
      margin: 28px auto;
      padding: 0 40px;
      display: grid;
      grid-template-columns: 1fr 420px;
      gap: 32px;
    }

    /* Left Video Showcase Card */
    .video-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.03);
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .video-wrapper {
      position: relative;
      width: 100%;
      border-radius: 14px;
      overflow: hidden;
      aspect-ratio: 16 / 9;
      background: #fafafa;
      border: 1px solid var(--border);
    }

    .video-wrapper img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }

    .pill-clean {
      position: absolute;
      top: 16px;
      left: 16px;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid var(--border);
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 0.74rem;
      font-weight: 700;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      color: var(--text);
    }

    /* Dynamic Form Quality Banner */
    .quality-banner {
      background: #f9fafb;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .banner-label {
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--muted);
    }
    .banner-value {
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--text);
      margin-top: 4px;
      transition: all 0.2s;
    }
    .score-circle {
      text-align: right;
    }
    .score-number {
      font-family: 'Syne', sans-serif;
      font-size: 2.2rem;
      font-weight: 800;
      color: var(--text);
      line-height: 1;
    }

    /* Right Panel Widgets */
    .panel {
      display: flex;
      flex-direction: column;
      gap: 22px;
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 22px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.02);
    }

    .card-heading {
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 14px;
    }

    /* Big Bold Poster Style Prediction Headline */
    .headline-card {
      background: #111827;
      color: #ffffff;
      border-radius: 18px;
      padding: 24px;
      text-align: center;
    }
    .headline-eyebrow {
      font-size: 0.7rem;
      letter-spacing: 2px;
      font-weight: 700;
      text-transform: uppercase;
      color: #9ca3af;
    }
    .headline-main {
      font-family: 'Syne', sans-serif;
      font-size: 2.4rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      text-transform: uppercase;
      margin-top: 4px;
      line-height: 1.1;
      color: #ffffff;
    }
    .headline-conf {
      font-size: 0.85rem;
      font-weight: 600;
      color: #d1d5db;
      margin-top: 6px;
    }

    /* Exercise Selector Segmented Control */
    .selector-bar {
      display: flex;
      gap: 8px;
      background: var(--light-grey);
      padding: 6px;
      border-radius: 12px;
    }
    .btn-choice {
      flex: 1;
      background: transparent;
      border: none;
      color: var(--muted);
      padding: 10px 4px;
      border-radius: 8px;
      font-family: inherit;
      font-size: 0.82rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.15s ease;
      text-align: center;
    }
    .btn-choice.active {
      background: #ffffff;
      color: #111827;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }

    /* Set Repetition Boxes */
    .rep-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 8px;
    }
    .rep-box {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      background: #fafafa;
    }
    .rep-box.valid { border-color: #111827; }
    .rep-num {
      font-family: 'Syne', sans-serif;
      font-size: 2.8rem;
      font-weight: 800;
      line-height: 1;
      color: var(--text);
    }
    .rep-label {
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      color: var(--muted);
      margin-top: 4px;
    }

    /* Multi-Widget Posture Matrix */
    .posture-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 8px;
    }
    .posture-widget {
      background: #fafafa;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .posture-name {
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--text);
    }
    .posture-badge {
      font-size: 0.85rem;
      font-family: 'Syne', sans-serif;
      font-weight: 800;
      color: var(--muted);
    }

    .btn-reset {
      width: 100%;
      background: transparent;
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 12px;
      border-radius: 10px;
      font-family: inherit;
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      cursor: pointer;
      margin-top: 14px;
      transition: all 0.2s;
    }
    .btn-reset:hover {
      border-color: #111827;
      color: #111827;
      background: #f9fafb;
    }
  </style>
</head>
<body>

  <div class="bg-pattern"></div>
  <div class="wave-bg"></div>

  <header>
    <div class="brand-group">
      <span class="brand-eyebrow">BIO-MECHANICS AI</span>
      <span class="brand-title">PHYSIO-TABPFN</span>
    </div>
    <div class="badge-bar">
      <div class="badge" id="latencyBadge">LATENCY: 0 MS</div>
      <div class="badge" id="modeBadge">TABPFN LOCAL</div>
    </div>
  </header>

  <div class="container">
    <!-- Video Showcase Left -->
    <div class="video-card">
      <div class="video-wrapper">
        <img src="/video_feed" alt="Live Stream">
        <div class="pill-clean">Clean Camera Feed (No Skeleton)</div>
      </div>
      <div class="quality-banner">
        <div>
          <div class="banner-label">Form Feedback</div>
          <div class="banner-value" id="feedbackText">Ready. Begin your set.</div>
        </div>
        <div class="score-circle">
          <div class="banner-label">Form Quality</div>
          <div class="score-number" id="qualityScore">100%</div>
        </div>
      </div>
    </div>

    <!-- Side Panel Widgets Right -->
    <div class="panel">
      <!-- Highest Prediction on Top in Large Poster Typography -->
      <div class="headline-card">
        <div class="headline-eyebrow">CURRENT POSTURE</div>
        <div class="headline-main" id="topHeadline">IDLE</div>
        <div class="headline-conf" id="topConfidence">100% Confidence</div>
      </div>

      <!-- Target Exercise Selector -->
      <div class="card">
        <div class="card-heading">Target Exercise</div>
        <div class="selector-bar">
          <button class="btn-choice active" id="btn-Squats" onclick="selectExercise('Squats')">Squats</button>
          <button class="btn-choice" id="btn-Step-Ups" onclick="selectExercise('Step-Ups')">Step-Ups</button>
          <button class="btn-choice" id="btn-Pogo-Hops" onclick="selectExercise('Pogo Hops')">Pogo Hops</button>
        </div>
      </div>

      <!-- Set Repetition Counter -->
      <div class="card">
        <div class="card-heading">Set Repetition Counter</div>
        <div class="rep-grid">
          <div class="rep-box valid">
            <div class="rep-num" id="validReps">0</div>
            <div class="rep-label">Valid Reps</div>
          </div>
          <div class="rep-box">
            <div class="rep-num" id="badReps" style="color: #6b7280;">0</div>
            <div class="rep-label">Bad Reps (Rejected)</div>
          </div>
        </div>
        <button class="btn-reset" onclick="triggerReset()">Reset Counter</button>
      </div>

      <!-- Multi-Widget Posture Tracker Matrix -->
      <div class="card">
        <div class="card-heading">Posture Matrix (Frame Detections)</div>
        <div class="posture-grid">
          <div class="posture-widget">
            <span class="posture-name">Squats</span>
            <span class="posture-badge" id="count-Squats">0</span>
          </div>
          <div class="posture-widget">
            <span class="posture-name">Step-Ups</span>
            <span class="posture-badge" id="count-Step-Ups">0</span>
          </div>
          <div class="posture-widget">
            <span class="posture-name">Pogo Hops</span>
            <span class="posture-badge" id="count-Pogo-Hops">0</span>
          </div>
          <div class="posture-widget">
            <span class="posture-name">Idle</span>
            <span class="posture-badge" id="count-Idle">0</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    async function updateDashboard() {
      try {
        const res = await fetch('/status');
        if (!res.ok) return;
        const data = await res.json();

        // 1. Move highest prediction on top in large bold character headline
        document.getElementById('topHeadline').innerText = data.predicted_exercise.toUpperCase();
        document.getElementById('topConfidence').innerText = data.confidence + '% Confidence';

        // 2. Dynamic live Form Feedback and Quality %
        document.getElementById('feedbackText').innerText = data.feedback;
        document.getElementById('qualityScore').innerText = data.quality_score + '%';

        // 3. Valid Reps and Bad Reps
        document.getElementById('validReps').innerText = data.valid_reps;
        document.getElementById('badReps').innerText = data.bad_reps;

        // 4. Latency & Mode
        document.getElementById('latencyBadge').innerText = 'LATENCY: ' + data.latency_ms + ' MS';
        document.getElementById('modeBadge').innerText = 'TABPFN ' + data.mode;

        // 5. Posture counts widget matrix
        if (data.posture_counts) {
          document.getElementById('count-Squats').innerText = data.posture_counts['Squats'] || 0;
          document.getElementById('count-Step-Ups').innerText = data.posture_counts['Step-Ups'] || 0;
          document.getElementById('count-Pogo-Hops').innerText = data.posture_counts['Pogo Hops'] || 0;
          document.getElementById('count-Idle').innerText = data.posture_counts['Idle'] || 0;
        }

        // 6. Active Exercise Selector
        document.querySelectorAll('.btn-choice').forEach(btn => {
          btn.classList.toggle('active', btn.innerText.trim() === data.active_exercise.trim());
        });

      } catch (err) {
        console.error('Fetch error:', err);
      }
    }

    async function selectExercise(name) {
      document.querySelectorAll('.btn-choice').forEach(btn => {
        btn.classList.toggle('active', btn.innerText.trim() === name.trim());
      });
      try {
        await fetch('/set_exercise?name=' + encodeURIComponent(name));
      } catch (e) {
        console.error(e);
      }
    }

    async function triggerReset() {
      try {
        await fetch('/reset');
        document.getElementById('validReps').innerText = '0';
        document.getElementById('badReps').innerText = '0';
        document.getElementById('qualityScore').innerText = '100%';
      } catch (e) {
        console.error(e);
      }
    }

    setInterval(updateDashboard, 100);
  </script>
</body>
</html>
"""


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
                time.sleep(0.01)
                continue

            frame_idx += 1
            results = estimator.process_frame(rgb_frame)
            landmarks_np = estimator.extract_numpy_landmarks(results)

            preview_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            if landmarks_np is not None:
                features = extract_exercise_features(landmarks_np)

                if features is not None:
                    # Query TabPFN every 3 frames
                    if frame_idx % 3 == 0:
                        t0 = time.perf_counter()
                        try:
                            lbl, conf, probs = model.predict_single(features)
                            elapsed_ms = (time.perf_counter() - t0) * 1000.0
                            with state.lock:
                                state.predicted_exercise = lbl
                                state.confidence = conf
                                state.probabilities = probs
                                state.latency_ms = (
                                    (0.7 * state.latency_ms + 0.3 * elapsed_ms)
                                    if state.latency_ms > 0
                                    else elapsed_ms
                                )
                                state.counter.update(lbl, conf, features)
                        except Exception as e:
                            print(f"[Model Warning]: {e}")
                    else:
                        with state.lock:
                            state.counter.update(state.predicted_exercise, state.confidence, features)
            else:
                with state.lock:
                    state.predicted_exercise = "Idle"
                    state.confidence = 1.0

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

    tabpfn_model = PhysioTabPFNModel(mode="auto")
    tabpfn_model.fit_from_csv(DATASET_CSV, max_samples_per_class=25)
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
