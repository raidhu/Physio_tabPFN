#!/usr/bin/env python3
"""Generates synthetic repetition loops of Step-Ups based on the verified ground-truth data in exercise_dataset.csv.

Applies:
1. Temporal cadence resampling (interpolating along normalized movement phase [0, 1]).
2. Bilateral limb alternation (generating both right-lead and left-lead step-up cycles).
3. Minor physiological noise injection (micro-postural variations while respecting physical bounds).
4. Synchronized spatial and angular metric consistency (means, diffs, ratios).
5. Continuous sequential timestamps simulating a multi-rep continuous exercise set.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_CSV = DATA_DIR / "exercise_dataset.csv"


def generate_synthetic_step_ups_loops(
    base_step_ups: pd.DataFrame,
    target_total_rows: int = 1200,
    random_seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(random_seed)
    feature_cols = [c for c in base_step_ups.columns if c not in ["label", "timestamp"]]

    base_len = len(base_step_ups)
    num_loops = max(1, int(round(target_total_rows / base_len)) - 1)
    
    synthetic_dfs = [base_step_ups.copy().reset_index(drop=True)]
    last_ts = float(base_step_ups["timestamp"].max())

    for loop_idx in range(num_loops):
        # 1. Cadence variation: resample length to 44-52 frames (~2.2 - 2.8s)
        target_len = int(np.random.randint(44, 53))
        orig_phase = np.linspace(0, 1, base_len)
        new_phase = np.linspace(0, 1, target_len)

        resampled_data = {}
        for col in feature_cols:
            resampled_data[col] = np.interp(new_phase, orig_phase, base_step_ups[col])

        loop_df = pd.DataFrame(resampled_data)

        # 2. Bilateral alternation: alternate lead leg every repetition
        is_left_lead = (loop_idx % 2 == 1)
        if is_left_lead:
            loop_df["left_knee_angle"], loop_df["right_knee_angle"] = (
                loop_df["right_knee_angle"].copy(),
                loop_df["left_knee_angle"].copy(),
            )
            loop_df["left_hip_angle"], loop_df["right_hip_angle"] = (
                loop_df["right_hip_angle"].copy(),
                loop_df["left_hip_angle"].copy(),
            )
            loop_df["left_elbow_angle"], loop_df["right_elbow_angle"] = (
                loop_df["right_elbow_angle"].copy(),
                loop_df["left_elbow_angle"].copy(),
            )

        # 3. Form variation & micro-posture noise
        loop_df["trunk_inclination"] = np.clip(
            loop_df["trunk_inclination"] + np.random.normal(0, 0.35, target_len), 0.0, 9.0
        )
        loop_df["foot_height_diff"] = np.clip(
            loop_df["foot_height_diff"] + np.random.normal(0, 0.015, target_len), 0.005, 1.05
        )
        loop_df["knee_height_diff"] = np.clip(
            loop_df["knee_height_diff"] + np.random.normal(0, 0.015, target_len), 0.005, 1.05
        )
        loop_df["left_knee_angle"] = np.clip(
            loop_df["left_knee_angle"] + np.random.normal(0, 1.2, target_len), 40.0, 180.0
        )
        loop_df["right_knee_angle"] = np.clip(
            loop_df["right_knee_angle"] + np.random.normal(0, 1.2, target_len), 40.0, 180.0
        )

        # 4. Consistency recalculations
        loop_df["mean_knee_angle"] = (loop_df["left_knee_angle"] + loop_df["right_knee_angle"]) / 2.0
        loop_df["knee_angle_diff"] = np.abs(loop_df["left_knee_angle"] - loop_df["right_knee_angle"])
        loop_df["mean_hip_angle"] = (loop_df["left_hip_angle"] + loop_df["right_hip_angle"]) / 2.0
        loop_df["hip_angle_diff"] = np.abs(loop_df["left_hip_angle"] - loop_df["right_hip_angle"])

        # 5. Continuous sequential timestamps
        loop_df["label"] = "Step-Ups"
        dt = np.random.uniform(0.045, 0.055, size=target_len)
        ts = last_ts + np.cumsum(dt)
        loop_df["timestamp"] = ts
        last_ts = ts[-1] + 0.12  # Natural brief pause between reps

        synthetic_dfs.append(loop_df)

    return pd.concat(synthetic_dfs, ignore_index=True)


def main():
    if not DATASET_CSV.exists():
        print(f"[Error] {DATASET_CSV} does not exist.")
        return

    df = pd.read_csv(DATASET_CSV)
    squats_df = df[df["label"] == "Squats"].copy()
    step_ups_df = df[df["label"] == "Step-Ups"].copy()

    if len(step_ups_df) == 0:
        print("[Error] No Step-Ups data found in exercise_dataset.csv.")
        return

    print("=" * 65)
    print(" SYNTHETIC DATA LOOP GENERATOR FOR STEP-UPS")
    print(f" Current Dataset: {len(squats_df)} Squats, {len(step_ups_df)} Step-Ups")
    print("=" * 65)

    # Generate synthetic loops matching Squats volume (~1200 frames, 25 continuous cycles)
    target_count = len(squats_df) if len(squats_df) > 0 else 1200
    expanded_step_ups = generate_synthetic_step_ups_loops(
        base_step_ups=step_ups_df,
        target_total_rows=target_count,
        random_seed=42,
    )

    # Combine Squats and expanded Step-Ups
    updated_df = pd.concat([squats_df, expanded_step_ups], ignore_index=True)
    updated_df.to_csv(DATASET_CSV, index=False)

    print(f"\n✓ Successfully updated {DATASET_CSV}:")
    print(f"  - Squats:   {len(squats_df)} frames")
    print(f"  - Step-Ups: {len(expanded_step_ups)} frames (25 continuous bilateral loops)")
    print(f"  - Total:    {len(updated_df)} frames")


if __name__ == "__main__":
    main()
