"""Live Real-Time Exercise Classifier using Local TabPFN (or Cloud API).

Optimizations:
- Threaded asynchronous inference: video display runs fluidly at full FPS
  without blocking on model execution.
- Auto-detects local TabPFN (PyTorch Apple Silicon / MPS) for sub-30ms latency.
"""

from pathlib import Path
import queue
import threading
import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from physio_tabPFN import (
    CameraFeed,
    PoseEstimator,
    PhysioTabPFNModel,
    extract_exercise_features,
)
from physio_tabPFN.env_loader import load_env

load_env()

DATASET_CSV = Path(__file__).resolve().parent.parent / "data" / "exercise_dataset.csv"


class AsyncInferenceWorker:
    """Runs TabPFN inference in a dedicated background thread to keep video render loop at 60 FPS."""

    def __init__(self, model: PhysioTabPFNModel) -> None:
        self.model = model
        self.input_queue: queue.Queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Thread-safe shared prediction state
        self.lock = threading.Lock()
        self.latest_label = "Idle"
        self.latest_confidence = 1.0
        self.latest_probs = {c: 0.25 for c in self.model.classes_} if self.model.classes_ is not None else {}
        self.latest_latency_ms = 0.0

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def push_features(self, features: np.ndarray) -> None:
        """Pushes latest frame features; drops old pending frames if worker is busy."""
        if not self.input_queue.full():
            self.input_queue.put_nowait(features)

    def get_results(self) -> Tuple[str, float, Dict[str, float], float]:
        with self.lock:
            return (
                self.latest_label,
                self.latest_confidence,
                dict(self.latest_probs),
                self.latest_latency_ms,
            )

    def _worker_loop(self) -> None:
        while self.running:
            try:
                features = self.input_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                lbl, conf, probs = self.model.predict_single(features)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                with self.lock:
                    self.latest_label = lbl
                    self.latest_confidence = conf
                    self.latest_probs = probs
                    # Apply exponential smoothing for displayed latency to avoid jitter
                    if self.latest_latency_ms == 0.0:
                        self.latest_latency_ms = elapsed_ms
                    else:
                        self.latest_latency_ms = 0.8 * self.latest_latency_ms + 0.2 * elapsed_ms
            except Exception as e:
                print(f"[Inference Worker] Warning: {e}")
            finally:
                self.input_queue.task_done()


def draw_prediction_hud(
    frame: np.ndarray,
    label: str,
    confidence: float,
    all_probs: dict,
    latency_ms: float,
    mode_text: str,
) -> None:
    h, w, _ = frame.shape

    # Semi-transparent top bar & right probability bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (15, 15, 15), -1)
    cv2.rectangle(overlay, (w - 320, 110), (w, 310), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Top Bar: Predicted Exercise & Confidence
    color_map = {
        "Squats": (0, 255, 255),
        "Step-Ups": (255, 180, 0),
        "Pogo Hops": (0, 255, 100),
        "Idle": (160, 160, 160),
    }
    label_color = color_map.get(label, (255, 255, 255))

    cv2.putText(
        frame,
        f"PHYSIO TABPFN ({mode_text.upper()})",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),
        1,
    )

    cv2.putText(
        frame,
        f"{label.upper()}",
        (25, 85),
        cv2.FONT_HERSHEY_DUPLEX,
        1.25,
        label_color,
        2,
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (360, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    # Latency coloring: green if fast (<50ms), orange if medium, red if slow
    lat_color = (50, 255, 50) if latency_ms < 60 else ((0, 200, 255) if latency_ms < 300 else (0, 0, 255))
    cv2.putText(
        frame,
        f"Latency: {latency_ms:.1f}ms",
        (w - 240, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        lat_color,
        2,
    )

    # Right side: Class probability breakdown bars
    cv2.putText(
        frame,
        "CLASS PROBABILITIES",
        (w - 300, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        1,
    )

    y = 170
    for cls_name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        cv2.putText(
            frame,
            f"{cls_name[:12]}",
            (w - 300, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
        )

        bar_x = w - 170
        bar_w = int(140 * prob)
        cv2.rectangle(frame, (bar_x, y - 12), (bar_x + 140, y + 2), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, y - 12), (bar_x + bar_w, y + 2), (0, 255, 200), -1)

        cv2.putText(
            frame,
            f"{prob * 100:.0f}%",
            (w - 25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
        )
        y += 32

    # Footer
    cv2.putText(
        frame,
        "Press 'q' to Quit",
        (25, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
    )


def main():
    print("=" * 65)
    print(" PHYSIO-TABPFN REAL-TIME INFERENCE")
    print(" Initializing TabPFN (preferring Local PyTorch on Apple Silicon)...")
    print("=" * 65)

    # Mode can be 'auto', 'local', or 'api'
    tabpfn_model = PhysioTabPFNModel(mode="auto")
    tabpfn_model.fit_from_csv(DATASET_CSV, max_samples_per_class=120)

    # Launch asynchronous background inference worker
    worker = AsyncInferenceWorker(tabpfn_model)
    worker.start()

    print(f"\nEngine running in [{tabpfn_model.active_mode.upper()}] mode.")
    print("Starting video stream. Press 'q' to exit.")

    try:
        with CameraFeed(source=0, width=1280, height=720) as cam, PoseEstimator() as estimator:
            for rgb_frame in cam.frames():
                results = estimator.process_frame(rgb_frame)
                landmarks_np = estimator.extract_numpy_landmarks(results)

                preview_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                if landmarks_np is not None:
                    estimator.draw_landmarks(preview_frame, results)

                    features = extract_exercise_features(landmarks_np)
                    if features is not None:
                        # Non-blocking push to background inference thread
                        worker.push_features(features)

                # Retrieve latest prediction without blocking video display
                current_label, current_confidence, all_probs, latency_ms = worker.get_results()

                draw_prediction_hud(
                    preview_frame,
                    label=current_label,
                    confidence=current_confidence,
                    all_probs=all_probs,
                    latency_ms=latency_ms,
                    mode_text=tabpfn_model.active_mode,
                )

                cv2.imshow("Physio TabPFN Real-Time Classifier", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        worker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
