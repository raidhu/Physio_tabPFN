"""Live Real-Time Exercise Classifier using TabPFN.

Uses:
- Webcam (CameraFeed)
- MediaPipe (PoseEstimator)
- Distance-invariant kinematics
- TabPFN-3.5-Plus Cloud API (PriorLabs)
"""

from pathlib import Path
import time
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


def draw_prediction_hud(
    frame: np.ndarray,
    label: str,
    confidence: float,
    all_probs: dict,
    latency_ms: float,
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

    cv2.putText(frame, "TABPFN-3.5 CLASSIFIER", (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    cv2.putText(frame, f"{label.upper()}", (25, 85),
                cv2.FONT_HERSHEY_DUPLEX, 1.25, label_color, 2)

    cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (360, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.putText(frame, f"API Latency: {latency_ms:.0f}ms", (w - 220, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Right side: Class probability breakdown bars
    cv2.putText(frame, "CLASS PROBABILITIES", (w - 300, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

    y = 170
    for cls_name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        # Text
        cv2.putText(frame, f"{cls_name[:12]}", (w - 300, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        # Bar graph
        bar_x = w - 170
        bar_w = int(140 * prob)
        cv2.rectangle(frame, (bar_x, y - 12), (bar_x + 140, y + 2), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, y - 12), (bar_x + bar_w, y + 2), (0, 255, 200), -1)

        cv2.putText(frame, f"{prob * 100:.0f}%", (w - 25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        y += 32

    # Footer
    cv2.putText(frame, "Press 'q' to Quit", (25, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)


def main():
    print("=" * 65)
    print(" PHYSIO-TABPFN REAL-TIME INFERENCE")
    print(" Loading model with TabPFN API...")
    print("=" * 65)

    tabpfn_model = PhysioTabPFNModel()
    tabpfn_model.fit_from_csv(DATASET_CSV, max_samples_per_class=120)

    # State variables for smoothing
    current_label = "Idle"
    current_confidence = 1.0
    all_probs = {c: 0.25 for c in tabpfn_model.classes_}
    latency_ms = 0.0

    # Rate-limiter: TabPFN is queried every N frames to maintain 30 FPS video
    # while running smooth API inference
    INFERENCE_INTERVAL_FRAMES = 5
    frame_counter = 0

    print("\nStarting video stream. Press 'q' to exit.")

    with CameraFeed(source=0, width=1280, height=720) as cam, PoseEstimator() as estimator:
        for rgb_frame in cam.frames():
            frame_counter += 1
            results = estimator.process_frame(rgb_frame)
            landmarks_np = estimator.extract_numpy_landmarks(results)

            preview_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            if landmarks_np is not None:
                estimator.draw_landmarks(preview_frame, results)

                # Every N frames, send kinematic feature vector to TabPFN API
                if frame_counter % INFERENCE_INTERVAL_FRAMES == 0:
                    features = extract_exercise_features(landmarks_np)
                    if features is not None:
                        t0 = time.time()
                        try:
                            current_label, current_confidence, all_probs = tabpfn_model.predict_single(features)
                            latency_ms = (time.time() - t0) * 1000.0
                        except Exception as e:
                            print(f"TabPFN API call warning: {e}")

            draw_prediction_hud(
                preview_frame,
                label=current_label,
                confidence=current_confidence,
                all_probs=all_probs,
                latency_ms=latency_ms,
            )

            cv2.imshow("Physio TabPFN Real-Time Classifier", preview_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
