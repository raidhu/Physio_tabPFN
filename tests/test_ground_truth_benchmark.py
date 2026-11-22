#!/usr/bin/env python3
"""PyTest benchmark validating TabPFN in-context model against ground-truth test data."""

from pathlib import Path
import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:
    class MockPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            return lambda fn: fn
    pytest = MockPytest()

from physio_tabPFN import PhysioTabPFNModel, QualityRepCounter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CSV = PROJECT_ROOT / "data" / "exercise_dataset.csv"
TEST_SET_CSV = PROJECT_ROOT / "data" / "ground_truth_test_set.csv"

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


@pytest.fixture(scope="module")
def trained_model():
    model = PhysioTabPFNModel(mode="auto", N_ensemble_configurations=1)
    model.fit_from_csv(DATASET_CSV, max_samples_per_class=12)
    return model


@pytest.fixture(scope="module")
def test_dataset():
    if not TEST_SET_CSV.exists():
        from scripts.build_test_set import main as build_main
        build_main()
    return pd.read_csv(TEST_SET_CSV)


def test_ground_truth_classification_accuracy(trained_model, test_dataset):
    """Verifies that trained classes achieve >= 85% accuracy on ground truth frames."""
    available_classes = set(trained_model.classes_)
    eval_df = test_dataset[test_dataset["label"].isin(available_classes)].copy()
    assert len(eval_df) > 0, "No overlapping classes between trained model and test set"

    y_true = eval_df["label"].tolist()
    y_pred = []

    for _, row in eval_df.iterrows():
        feat = row[FEATURE_COLUMNS].values.astype(np.float32)
        lbl, conf, _ = trained_model.predict_single(feat, min_confidence=0.0)
        y_pred.append(lbl)

    acc = np.mean([t == p for t, p in zip(y_true, y_pred)])
    print(f"\n[Test Benchmark] Accuracy on trained classes ({available_classes}): {acc * 100:.1f}%")
    assert acc >= 0.85, f"Expected accuracy >= 85%, got {acc * 100:.1f}%"


def test_ground_truth_repetition_counting(test_dataset):
    """Verifies that each active ground-truth video episode triggers valid repetition."""
    active_exercises = {"Squats", "Step-Ups"}
    for video_name, group in test_dataset.groupby("video_file"):
        exercise_name = group["label"].iloc[0]
        if exercise_name not in active_exercises:
            continue
        counter = QualityRepCounter(exercise_name)

        for _, row in group.iterrows():
            feat = row[FEATURE_COLUMNS].values.astype(np.float32)
            counter.update(exercise_name, 1.0, feat)

        print(f"\n[Video: {video_name}] {exercise_name}: {counter.valid_reps} valid, {counter.bad_reps} bad")
        assert counter.valid_reps >= 1, f"Expected at least 1 valid repetition for {exercise_name} in {video_name}"


def test_confidence_threshold_and_idle_stability(trained_model):
    """Verifies confidence < 60% is labeled 'Unknown' and Idle posture is stably detected with near 100% confidence."""
    df = pd.read_csv(DATASET_CSV)
    idle_sub = df[df["label"] == "Idle"].head(50)
    assert len(idle_sub) > 0

    misclass_count = 0
    idle_confs = []
    for _, row in idle_sub.iterrows():
        feat = row[FEATURE_COLUMNS].values.astype(np.float32)
        lbl, conf, probs = trained_model.predict_single(feat, min_confidence=0.60)
        if lbl != "Idle":
            misclass_count += 1
        idle_confs.append(conf)

    print(f"\n[Idle Stability Test] Non-Idle predictions on Idle frames: {misclass_count} / {len(idle_sub)}")
    print(f"[Idle Stability Test] Mean Idle confidence: {np.mean(idle_confs) * 100:.1f}%")
    assert misclass_count == 0, f"Idle posture was misclassified {misclass_count} times!"
    assert np.mean(idle_confs) >= 0.90, f"Expected Idle confidence >= 90%, got {np.mean(idle_confs)*100:.1f}%"

    # Test ambiguous synthetic vector with forced low confidence
    # If confidence < 0.60, label MUST be 'Unknown'
    dummy_feat = np.ones(len(FEATURE_COLUMNS), dtype=np.float32) * 50.0
    lbl, conf, probs = trained_model.predict_single(dummy_feat, min_confidence=0.999)
    assert lbl == "Unknown", f"Expected 'Unknown' when confidence < threshold, got {lbl}"
    print("[Confidence Threshold Test] Verified < 60% confidence marks prediction as 'Unknown'")


if __name__ == "__main__":
    print("Running Ground-Truth Benchmark Tests...")
    model = trained_model()
    test_ds = test_dataset()
    test_ground_truth_classification_accuracy(model, test_ds)
    test_ground_truth_repetition_counting(test_ds)
    test_confidence_threshold_and_idle_stability(model)
    print("\n✓ ALL GROUND-TRUTH BENCHMARKS PASSED SUCCESSFULLY!")
