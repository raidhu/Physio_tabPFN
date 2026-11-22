"""Dataset Recorder & Movement GIF Generator for Physio-TabPFN.

Captures distance-invariant kinematic feature vectors directly using kinematics.py,
records them into a labeled CSV file for TabPFN training, and generates animated GIFs
of recognized exercise repetitions into the 'gif' folder during continuous 2-minute sets.

Exercises:
  [1] Squats
  [2] Step-Ups
  [3] Pogo Hops
  [0] Idle / Standing (Negative class)

Controls:
  Spacebar : 3-second countdown then 2-MINUTE recording set (auto-pauses after 2 min)
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
import threading
import time
from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image

from physio_tabPFN import (
    CameraFeed,
    PoseEstimator,
    PhysioTabPFNModel,
    QualityRepCounter,
    extract_exercise_features,
)
from physio_tabPFN.env_loader import load_env

load_env()

# 2 minutes continuous recording duration per set
RECORDING_DURATION_SEC = 120.0

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_GIF_DIR = Path(__file__).resolve().parent.parent / "gif"
DATASET_CSV = DEFAULT_OUTPUT_DIR / "exercise_dataset.csv"


class TargetExercise(str, Enum):
    SQUATS = "Squats"
    STEP_UPS = "Step-Ups"
    IDLE = "Idle"


EXERCISE_MAP = {
    ord("1"): TargetExercise.SQUATS.value,
    ord("2"): TargetExercise.STEP_UPS.value,
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


class GifRepetitionRecorder:
    """Buffers live frames during an exercise repetition and writes an animated GIF to gif/."""

    def __init__(self, gif_dir: Path = DEFAULT_GIF_DIR, max_buffer_frames: int = 150) -> None:
        self.gif_dir = gif_dir
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        self.max_buffer_frames = max_buffer_frames
        self.active_frames: List[np.ndarray] = []
        self.is_capturing = False
        self.last_saved_gif: Optional[str] = None
        self.last_save_time: float = 0.0

    def start_movement(self) -> None:
        """Called when a repetition trajectory starts (e.g. squat inflection / step lift)."""
        self.is_capturing = True
        self.active_frames = []

    def append_frame(self, frame: np.ndarray) -> None:
        if self.is_capturing:
            # Resize to 480x270 for smooth, lightweight GIF encoding
            small = cv2.resize(frame, (480, 270))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            self.active_frames.append(rgb)
            if len(self.active_frames) > self.max_buffer_frames:
                self.active_frames.pop(0)

    def finish_and_save(self, exercise_name: str, rep_number: int) -> Optional[Path]:
        """Saves captured movement sequence as an animated GIF asynchronously."""
        if not self.is_capturing or len(self.active_frames) < 12:
            self.is_capturing = False
            self.active_frames = []
            return None

        frames_to_save = list(self.active_frames)
        self.is_capturing = False
        self.active_frames = []

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = exercise_name.lower().replace(" ", "_").replace("-", "_")
        filename = f"{slug}_rep{rep_number}_{timestamp_str}.gif"
        out_path = self.gif_dir / filename

        # Subsample frames if rep was long, keeping GIF smooth at ~16 FPS
        step = 1 if len(frames_to_save) <= 40 else 2
        pil_images = [Image.fromarray(f) for f in frames_to_save[::step]]

        def _save_task():
            try:
                pil_images[0].save(
                    out_path,
                    save_all=True,
                    append_images=pil_images[1:],
                    duration=60,  # ~16.6 FPS
                    loop=0,
                    optimize=True,
                )
                print(f"\n>> 🎬 [GIF SAVED] -> {out_path} ({len(pil_images)} frames) <<")
            except Exception as err:
                print(f"[GIF Error]: {err}")

        threading.Thread(target=_save_task, daemon=True).start()
        self.last_saved_gif = filename
        self.last_save_time = time.time()
        return out_path

    def cancel(self) -> None:
        """Discards buffered frames when repetition is flawed or rejected."""
        self.is_capturing = False
        self.active_frames = []


def render_recorder_ui(
    frame: np.ndarray,
    current_label: str,
    predicted_label: str,
    predicted_conf: float,
    rep_count: int,
    is_recording: bool,
    session_count: int,
    counts_by_label: dict,
    countdown: int = 0,
    time_remaining: float = 0.0,
    total_duration: float = RECORDING_DURATION_SEC,
    just_completed: bool = False,
    last_saved_gif: Optional[str] = None,
    last_save_time: float = 0.0,
) -> None:
    h, w, _ = frame.shape

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
    cv2.rectangle(overlay, (w - 280, 100), (w, 320), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    if countdown > 0:
        status_text = f"GET READY ({countdown}s)..."
        status_color = (0, 255, 255)
    elif is_recording:
        mins = int(time_remaining) // 60
        secs = int(time_remaining) % 60
        status_text = f"● RECORDING [{mins:02d}:{secs:02d}]"
        status_color = (0, 0, 255)
        # Visual progress bar along bottom of top banner
        progress = max(0.0, min(1.0, 1.0 - (time_remaining / total_duration)))
        bar_width = int(w * progress)
        cv2.rectangle(frame, (0, 96), (bar_width, 100), (50, 220, 50), -1)
    elif just_completed:
        status_text = "✓ 2-MIN SET COMPLETE - PAUSED"
        status_color = (50, 255, 50)
    else:
        status_text = "PAUSED (Press [SPACE] for 2-min set)"
        status_color = (160, 160, 160)

    # Primary status and active exercise label
    cv2.putText(frame, status_text, (25, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, status_color, 2)
    cv2.putText(frame, f"LABEL: {current_label.upper()}", (w - 650, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255), 2)
    
    # Live prediction & Rep counter HUD
    pred_str = f"PRED: {predicted_label.upper()} ({int(predicted_conf * 100)}%)  |  REPS: {rep_count}"
    cv2.putText(frame, pred_str, (w - 650, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.putText(frame, "[SPACE] Start 2-min set / Pause | [1] Squats | [2] Step-Ups | [0] Idle | [s] Save | [q] Quit",
                (25, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # Big center countdown numbers
    if countdown > 0:
        cv2.putText(frame, str(countdown), (w // 2 - 35, h // 2 + 35),
                    cv2.FONT_HERSHEY_DUPLEX, 4.0, (0, 255, 255), 6)

    # Right panel: dataset counts
    cv2.putText(frame, "RECORDED SAMPLES", (w - 265, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    y = 160
    for lbl, cnt in counts_by_label.items():
        color = (0, 255, 255) if lbl == current_label else (180, 180, 180)
        cv2.putText(frame, f"{lbl}: {cnt}", (w - 265, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 24

    cv2.putText(frame, f"Total: {session_count}", (w - 265, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 50), 2)

    # Bottom banner notification when GIF is created
    if last_saved_gif and (time.time() - last_save_time) < 4.5:
        cv2.rectangle(frame, (20, h - 55), (w - 20, h - 15), (20, 20, 20), -1)
        cv2.putText(frame, f"✓ ANIMATED GIF SAVED: gif/{last_saved_gif}", (35, h - 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, (50, 255, 50), 1)


def main():
    recorder = DatasetRecorder()
    gif_recorder = GifRepetitionRecorder()
    current_label = TargetExercise.SQUATS.value
    counter = QualityRepCounter(current_label)

    is_recording = False
    countdown = 0
    countdown_start_time = 0.0
    recording_start_time = 0.0
    time_remaining = RECORDING_DURATION_SEC
    just_completed = False

    # Predicted pose tracking
    predicted_label = "Idle"
    predicted_conf = 1.0

    # Initialize TabPFN for real-time pose prediction from the loop
    tabpfn_model: Optional[PhysioTabPFNModel] = None
    if DATASET_CSV.exists() and DATASET_CSV.stat().st_size > 500:
        try:
            print("[TabPFN] Loading reference dataset for live pose prediction...")
            tabpfn_model = PhysioTabPFNModel(mode="auto", N_ensemble_configurations=1)
            tabpfn_model.fit_from_csv(DATASET_CSV, max_samples_per_class=8)
            print("[TabPFN] Live Pose Predictor initialized.")
        except Exception as e:
            print(f"[TabPFN Notice]: {e}")

    print("=" * 65)
    print(" TABPFN DATASET RECORDER & GIF GENERATOR (2-MIN CONTINUOUS LOOP)")
    print(" Exercises: [1] Squats  [2] Step-Ups  [0] Idle")
    print(" Controls:  [SPACE] Start 2-Min Set / Pause  |  [s] Save  |  [c] Clear  |  [q] Quit")
    print(f" Set Duration: {int(RECORDING_DURATION_SEC // 60)} minutes per run")
    print(f" Output GIF Directory: {DEFAULT_GIF_DIR}")
    print("=" * 65)

    frame_idx = 0

    with CameraFeed(source=0, width=1280, height=720) as cam, PoseEstimator() as estimator:
        for rgb_frame in cam.frames():
            frame_idx += 1
            results = estimator.process_frame(rgb_frame)
            landmarks_np = estimator.extract_numpy_landmarks(results)

            preview_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # 3-second countdown state
            if countdown > 0:
                elapsed = time.time() - countdown_start_time
                if elapsed >= 1.0:
                    countdown -= 1
                    countdown_start_time = time.time()
                    if countdown == 0:
                        is_recording = True
                        recording_start_time = time.time()
                        time_remaining = RECORDING_DURATION_SEC
                        just_completed = False
                        counter.reset(current_label)
                        print(f">> NOW RECORDING: {current_label} (2-minute timer started) <<")

            # 2-minute recording duration timer check
            if is_recording and countdown == 0:
                elapsed_rec = time.time() - recording_start_time
                time_remaining = max(0.0, RECORDING_DURATION_SEC - elapsed_rec)
                if elapsed_rec >= RECORDING_DURATION_SEC:
                    is_recording = False
                    just_completed = True
                    time_remaining = 0.0
                    print(f"\n>> 🎯 2 MINUTES COMPLETED for {current_label}! Paused automatically. <<")
                    recorder.save_csv()

            if landmarks_np is not None:
                estimator.draw_landmarks(preview_frame, results)
                features = extract_exercise_features(landmarks_np)

                if features is not None:
                    # 1. Predict exact particular pose using TabPFN
                    if tabpfn_model is not None and frame_idx % 3 == 0:
                        try:
                            lbl, conf, _ = tabpfn_model.predict_single(features)
                            predicted_label = lbl
                            predicted_conf = conf
                        except Exception:
                            pass

                    # 2. Add sample to dataset CSV if recording
                    if is_recording and countdown == 0:
                        recorder.add_sample(features, current_label)

                    # 3. Track repetition movement and capture animated GIF
                    prev_counter_state = counter.state
                    prev_valid_reps = counter.valid_reps
                    prev_bad_reps = counter.bad_reps
                    counter.update(predicted_label, predicted_conf, features)

                    # When movement trajectory starts (descending, stepping, or hopping)
                    if prev_counter_state == "standing" and counter.state != "standing":
                        gif_recorder.start_movement()

                    # Buffer frame for GIF while movement is ongoing
                    if gif_recorder.is_capturing:
                        gif_recorder.append_frame(preview_frame)

                    # When movement repetition completes
                    if counter.valid_reps > prev_valid_reps:
                        # Only export clean, verified repetitions
                        gif_recorder.finish_and_save(
                            exercise_name=current_label,
                            rep_number=counter.valid_reps,
                        )
                    elif counter.bad_reps > prev_bad_reps or (counter.last_rep_status == "BAD" and counter.state == "standing"):
                        # Discard flawed repetition frames
                        gif_recorder.cancel()

            render_recorder_ui(
                preview_frame,
                current_label=current_label,
                predicted_label=predicted_label,
                predicted_conf=predicted_conf,
                rep_count=counter.valid_reps + counter.bad_reps,
                is_recording=is_recording and (countdown == 0),
                session_count=len(recorder.records),
                counts_by_label=recorder.counts_by_label,
                countdown=countdown,
                time_remaining=time_remaining,
                total_duration=RECORDING_DURATION_SEC,
                just_completed=just_completed,
                last_saved_gif=gif_recorder.last_saved_gif,
                last_save_time=gif_recorder.last_save_time,
            )

            cv2.imshow("TabPFN Dataset Recorder & GIF Generator", preview_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                if is_recording or countdown > 0:
                    is_recording = False
                    countdown = 0
                    just_completed = False
                    if recorder.records:
                        recorder.save_csv()
                    print(f"Paused early and auto-saved dataset. Total session samples: {len(recorder.records)}")
                else:
                    countdown = 3
                    countdown_start_time = time.time()
                    just_completed = False
                    time_remaining = RECORDING_DURATION_SEC
                    counter.reset(current_label)
                    print(f"Starting 2-minute set for {current_label} in 3 seconds... Get into position!")
            elif key == ord("s"):
                recorder.save_csv()
            elif key == ord("c"):
                recorder.records.clear()
                for k in recorder.counts_by_label:
                    recorder.counts_by_label[k] = 0
                counter.reset(current_label)
                just_completed = False
                print("Cleared current session.")
            elif key in EXERCISE_MAP:
                current_label = EXERCISE_MAP[key]
                counter.reset(current_label)
                just_completed = False
                print(f"Selected label: {current_label}")

    if recorder.records:
        recorder.save_csv()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
