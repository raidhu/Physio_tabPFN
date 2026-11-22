#!/usr/bin/env python3
"""Builds a standardized ground-truth kinematic test set from demonstration videos (assets/*.mov).

Outputs:
    data/ground_truth_test_set.csv: 16 kinematic feature columns + label + video_file + frame_idx + rel_phase.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict

from physio_tabPFN import PoseEstimator, extract_exercise_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "ground_truth_test_set.csv"

VIDEO_MAPPING = {
    "Squats.mov": "Squats",
    "Step-Ups.mov": "Step-Ups",
    "Pogo-Hops.mov": "Pogo Hops",
}

FEATURE_COLUMNS = [
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
    "trunk_inclination",
    "hip_height_ratio",
    "foot_height_diff",
    "knee_height_diff",
    "normalized_stance_width",
    "torso_scale",
]


def extract_features_from_video(
    video_path: Path, exercise_label: str, estimator: PoseEstimator
) -> List[Dict]:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    records = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = estimator.process_frame(frame_rgb)
        landmarks = estimator.extract_numpy_landmarks(results)

        if landmarks is not None:
            features = extract_exercise_features(landmarks)
            if features is not None:
                row_dict = {col: float(val) for col, val in zip(FEATURE_COLUMNS, features)}
                row_dict["label"] = exercise_label
                row_dict["video_file"] = video_path.name
                row_dict["frame_idx"] = frame_idx
                row_dict["total_frames"] = total_frames
                row_dict["rel_phase"] = round(frame_idx / max(total_frames, 1), 3)
                records.append(row_dict)

    cap.release()
    print(f"[{video_path.name}] Extracted {len(records)}/{total_frames} valid pose frames for '{exercise_label}'")
    return records


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []

    print("=" * 60)
    print(" EXTRACTING GROUND-TRUTH TEST SET FROM ASSETS/*.MOV")
    print("=" * 60)

    with PoseEstimator() as estimator:
        for video_filename, label in VIDEO_MAPPING.items():
            video_path = ASSETS_DIR / video_filename
            if not video_path.exists():
                print(f"[Warning] Ground-truth video not found: {video_path}")
                continue
            records = extract_features_from_video(video_path, label, estimator)
            all_records.extend(records)

    if not all_records:
        print("[Error] No ground-truth features extracted.")
        return

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Successfully exported {len(df)} ground-truth test frames to:")
    print(f"  -> {OUTPUT_CSV}")
    print("\nBreakdown by exercise class:")
    for lbl, count in df["label"].value_counts().items():
        print(f"  - {lbl}: {count} frames")


if __name__ == "__main__":
    main()
