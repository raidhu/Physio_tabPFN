"""Dataset Recorder for Physio-TabPFN.

Captures distance-invariant kinematic feature vectors directly using kinematics.py
and records them into a labeled CSV file for TabPFN training.

Exercises:
  [1] Squats
  [2] Step-Ups
  [3] Pogo Hops
  [0] Idle / Standing (Negative class)

Controls:
  Spacebar : 3-second countdown then START recording / STOP recording
  '1'-'3'  : Select active exercise label
  '0'      : Select 'Idle' label
  'c'      : Clear recorded frames in current session
  's'      : Save records to CSV immediately
  'q'      : Save and quit
"""

import csv
from datetime import datetime
from enum import Enum
from pathlib import Path
import time
from typing import Dict, List, Optional
import cv2
import numpy as np

from physio_tabPFN import (
    CameraFeed,
    PoseEstimator,
    extract_exercise_features,
)


class TargetExercise(str, Enum):
    SQUATS = "Squats"
    STEP_UPS = "Step-Ups"
    POGO_HOPS = "Pogo Hops"
    IDLE = "Idle"


EXERCISE_MAP = {
    ord("1"): TargetExercise.SQUATS.value,
    ord("2"): TargetExercise.STEP_UPS.value,
    ord("3"): TargetExercise.POGO_HOPS.value,
    ord("0"): TargetExercise.IDLE.value,
}

FEATURE_COLUMNS = [
    # Joint angles (degrees)
    "left_knee_angle",
    "right_knee_angle",
    "mean_knee_angle",
    "knee_angle_diff",
    "left_hip_angle",
    "right_hip_angle",
    "mean_hip_angle",
    "hip_angle_diff",
    "left_elbow_angle",
    "right_elbow_angle",
    # Distance-invariant normalized spatial metrics (divided by torso_scale)
    "trunk_inclination",
    "hip_height_ratio",
    "foot_height_diff",
    "knee_height_diff",
    "normalized_stance_width",
    "torso_scale",
]

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


class DatasetRecorder:
    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.columns = FEATURE_COLUMNS + ["label", "timestamp"]
        self.records: List[List] = []
        self.counts_by_label = {ex.value: 0 for ex in TargetExercise}

    def add_sample(self, feature_vector: np.ndarray, label: str) -> None:
        row = [float(v) for v in feature_vector]
        row.append(label)
        row.append(time.time())
        self.records.append(row)
        self.counts_by_label[label] = self.counts_by_label.get(label, 0) + 1

    def save_csv(self, filename: str = "exercise_dataset.csv") -> Optional[Path]:
        if not self.records:
            print("No records to save.")
            return None

        out_path = self.output_dir / filename
        file_exists = out_path.exists()

        with open(out_path, mode="a" if file_exists else "w", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self.columns)
            writer.writerows(self.records)

        print(f"\nSuccessfully wrote {len(self.records)} samples to: {out_path}")
        print("Session breakdown:")
        for lbl, count in self.counts_by_label.items():
            print(f"  - {lbl}: {count} frames")
        return out_path


def render_recorder_ui(
    frame: np.ndarray,
    current_label: str,
    is_recording: bool,
    session_count: int,
    counts_by_label: dict,
    countdown: int = 0,
) -> None:
    h, w, _ = frame.shape

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
    cv2.rectangle(overlay, (w - 280, 100), (w, 290), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    if countdown > 0:
        status_text = f"GET READY ({countdown}s)..."
        status_color = (0, 255, 255)
    elif is_recording:
        status_text = "● RECORDING"
        status_color = (0, 0, 255)
    else:
        status_text = "PAUSED"
        status_color = (130, 130, 130)

    cv2.putText(frame, status_text, (25, 42), cv2.FONT_HERSHEY_DUPLEX, 0.95, status_color, 2)
    cv2.putText(frame, f"LABEL: {current_label.upper()}", (350, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255), 2)
    cv2.putText(frame, "[SPACE] Start/Stop | [1] Squats | [2] Step-Ups | [3] Pogo Hops | [0] Idle | [s] Save | [q] Quit",
                (25, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    if countdown > 0:
        cv2.putText(frame, str(countdown), (w // 2 - 35, h // 2 + 35),
                    cv2.FONT_HERSHEY_DUPLEX, 4.0, (0, 255, 255), 6)

    cv2.putText(frame, "RECORDED SAMPLES", (w - 265, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    y = 160
    for lbl, cnt in counts_by_label.items():
        color = (0, 255, 255) if lbl == current_label else (180, 180, 180)
        cv2.putText(frame, f"{lbl}: {cnt}", (w - 265, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 24

    cv2.putText(frame, f"Total: {session_count}", (w - 265, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 50), 2)


def main():
    recorder = DatasetRecorder()
    current_label = TargetExercise.SQUATS.value
    is_recording = False
    countdown = 0
    countdown_start_time = 0

    print("=" * 65)
    print(" TABPFN DATASET RECORDER")
    print(" Exercises: [1] Squats  [2] Step-Ups  [3] Pogo Hops  [0] Idle")
    print(" Controls:  [SPACE] Toggle Recording  |  [s] Save  |  [c] Clear  |  [q] Quit")
    print("=" * 65)

    with CameraFeed(source=0, width=1280, height=720) as cam, PoseEstimator() as estimator:
        for rgb_frame in cam.frames():
            results = estimator.process_frame(rgb_frame)
            landmarks_np = estimator.extract_numpy_landmarks(results)

            preview_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            if countdown > 0:
                elapsed = time.time() - countdown_start_time
                if elapsed >= 1.0:
                    countdown -= 1
                    countdown_start_time = time.time()
                    if countdown == 0:
                        is_recording = True
                        print(f">> NOW RECORDING: {current_label} <<")

            if landmarks_np is not None:
                estimator.draw_landmarks(preview_frame, results)
                features = extract_exercise_features(landmarks_np)

                if features is not None and is_recording and countdown == 0:
                    recorder.add_sample(features, current_label)

            render_recorder_ui(
                preview_frame,
                current_label=current_label,
                is_recording=is_recording and (countdown == 0),
                session_count=len(recorder.records),
                counts_by_label=recorder.counts_by_label,
                countdown=countdown,
            )

            cv2.imshow("TabPFN Dataset Recorder", preview_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                if is_recording or countdown > 0:
                    is_recording = False
                    countdown = 0
                    print(f"Paused. Total session samples: {len(recorder.records)}")
                else:
                    countdown = 3
                    countdown_start_time = time.time()
                    print(f"Recording {current_label} in 3 seconds... Get into position!")
            elif key == ord("s"):
                recorder.save_csv()
            elif key == ord("c"):
                recorder.records.clear()
                for k in recorder.counts_by_label:
                    recorder.counts_by_label[k] = 0
                print("Cleared current session.")
            elif key in EXERCISE_MAP:
                current_label = EXERCISE_MAP[key]
                print(f"Selected label: {current_label}")

    if recorder.records:
        recorder.save_csv()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
