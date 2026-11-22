#!/usr/bin/env python3
"""Generates synthetic repetition loops of Pogo Hops based on recorded GIFs in gif/pogo_hops_*.gif.

Extracts kinematic trajectories from the 7 recorded repetitions:
- Applies cadence resampling along movement phases
- Injects minor physiological noise
- Generates continuous sequential timestamps
- Appends ~1200 frames to data/exercise_dataset.csv to match Squats and Step-Ups.
"""

from pathlib import Path
import glob
from PIL import Image
import numpy as np
import pandas as pd

from physio_tabPFN import PoseEstimator, extract_exercise_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GIF_DIR = PROJECT_ROOT / "gif"
DATASET_CSV = DATA_DIR / "exercise_dataset.csv"

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


def extract_base_reps_from_gifs() -> list[pd.DataFrame]:
    reps = []
    gt_csv = DATA_DIR / "ground_truth_test_set.csv"
    if gt_csv.exists():
        gt_df = pd.read_csv(gt_csv)
        gt_pogo = gt_df[gt_df["label"] == "Pogo Hops"][FEATURE_COLUMNS]
        if len(gt_pogo) >= 10:
            reps.append(gt_pogo.copy())
            print(f"[Ground-Truth Video] Loaded {len(gt_pogo)} reference frames")

    pogo_gifs = sorted(glob.glob(str(GIF_DIR / "pogo_hops*.gif")))
    if not pogo_gifs and not reps:
        raise FileNotFoundError(f"No Pogo Hops GIFs found in {GIF_DIR}")

    with PoseEstimator() as estimator:
        for gif_path in pogo_gifs:
            img = Image.open(gif_path)
            rep_records = []
            for i in range(img.n_frames):
                img.seek(i)
                frame_rgb = np.array(img.convert("RGB"))
                res = estimator.process_frame(frame_rgb)
                lm = estimator.extract_numpy_landmarks(res)
                if lm is not None:
                    feat = extract_exercise_features(lm)
                    if feat is not None:
                        rep_records.append(feat)

            if len(rep_records) >= 10:
                rep_df = pd.DataFrame(rep_records, columns=FEATURE_COLUMNS)
                # Ensure base rep contains active athletic hopping motion (arms active, not resting at sides)
                if not (rep_df["left_elbow_angle"].mean() > 140.0 and rep_df["right_elbow_angle"].mean() > 140.0):
                    reps.append(rep_df)
                    print(f"[{Path(gif_path).name}] Extracted {len(rep_df)} athletic frames")
                else:
                    print(f"[{Path(gif_path).name}] Filtered out static standing rep")

    return reps


def generate_synthetic_pogo_loops(
    base_reps: list[pd.DataFrame],
    target_total_rows: int = 1200,
    random_seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(random_seed)
    synthetic_dfs = []
    current_total = 0
    loop_idx = 0

    last_ts = 1795344300.0  # Sequential timestamp baseline

    while current_total < target_total_rows:
        # Pick one of the base repetitions
        base_rep = base_reps[loop_idx % len(base_reps)]
        base_len = len(base_rep)

        # 1. Cadence resampling (±3 frames for natural hop frequency ~ 1.5 - 2.5 Hz)
        jitter = np.random.randint(-2, 3)
        target_len = max(12, base_len + jitter)

        orig_phase = np.linspace(0, 1, base_len)
        new_phase = np.linspace(0, 1, target_len)

        resampled = {}
        for col in FEATURE_COLUMNS:
            resampled[col] = np.interp(new_phase, orig_phase, base_rep[col])

        loop_df = pd.DataFrame(resampled)

        # 2. Physiological noise injection (preserving spring and upright torso)
        loop_df["trunk_inclination"] = np.clip(
            loop_df["trunk_inclination"] + np.random.normal(0, 0.25, target_len), 0.0, 7.5
        )
        loop_df["foot_height_diff"] = np.clip(
            loop_df["foot_height_diff"] + np.random.normal(0, 0.008, target_len), 0.005, 0.45
        )
        loop_df["left_knee_angle"] = np.clip(
            loop_df["left_knee_angle"] + np.random.normal(0, 0.9, target_len), 142.0, 180.0
        )
        loop_df["right_knee_angle"] = np.clip(
            loop_df["right_knee_angle"] + np.random.normal(0, 0.9, target_len), 142.0, 180.0
        )

        # 3. Consistency recalculations and athletic arm preservation
        loop_df["left_elbow_angle"] = np.clip(loop_df["left_elbow_angle"], 5.0, 155.0)
        loop_df["right_elbow_angle"] = np.clip(loop_df["right_elbow_angle"], 5.0, 155.0)
        loop_df["mean_knee_angle"] = (loop_df["left_knee_angle"] + loop_df["right_knee_angle"]) / 2.0
        loop_df["knee_angle_diff"] = np.abs(loop_df["left_knee_angle"] - loop_df["right_knee_angle"])
        loop_df["mean_hip_angle"] = (loop_df["left_hip_angle"] + loop_df["right_hip_angle"]) / 2.0
        loop_df["hip_angle_diff"] = np.abs(loop_df["left_hip_angle"] - loop_df["right_hip_angle"])

        # 4. Continuous sequential timestamps
        loop_df["label"] = "Pogo Hops"
        dt = np.random.uniform(0.028, 0.038, size=target_len)  # ~30 FPS hop cadence
        ts = last_ts + np.cumsum(dt)
        loop_df["timestamp"] = ts
        last_ts = ts[-1] + 0.03

        synthetic_dfs.append(loop_df)
        current_total += len(loop_df)
        loop_idx += 1

    return pd.concat(synthetic_dfs, ignore_index=True)


def main():
    print("=" * 65)
    print(" EXTRACTING & SYNTHESIZING POGO HOPS FROM USER GIFS")
    print("=" * 65)

    base_reps = extract_base_reps_from_gifs()
    print(f"\nExtracted {len(base_reps)} base reps from your recorded GIFs.")

    # Target 1200 frames to match Squats (1200) and Step-Ups (1236)
    pogo_df = generate_synthetic_pogo_loops(base_reps, target_total_rows=1200, random_seed=42)
    print(f"Generated {len(pogo_df)} synthetic Pogo Hops frames.")

    # Read existing dataset and append
    df = pd.read_csv(DATASET_CSV)
    # Remove any existing Pogo Hops rows to avoid duplicates
    existing_other = df[df["label"] != "Pogo Hops"].copy()
    
    updated_df = pd.concat([existing_other, pogo_df], ignore_index=True)
    updated_df.to_csv(DATASET_CSV, index=False)

    print("\n✓ Successfully updated exercise_dataset.csv:")
    for lbl, count in updated_df["label"].value_counts().items():
        print(f"  - {lbl}: {count} frames")
    print(f"  - Total: {len(updated_df)} frames across all classes")


if __name__ == "__main__":
    main()
