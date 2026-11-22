#!/usr/bin/env python3
"""Evaluates TabPFN In-Context Model against Ground-Truth Test Set and Reference Videos.

Computes:
- Per-class Classification Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Mean & P95 Inference Latency (ms)
- Temporal Biomechanical Validation (QualityRepCounter on full video episodes)
- Structured JSON report export to data/ground_truth_eval_report.json
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from physio_tabPFN import PhysioTabPFNModel, QualityRepCounter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CSV = PROJECT_ROOT / "data" / "exercise_dataset.csv"
TEST_SET_CSV = PROJECT_ROOT / "data" / "ground_truth_test_set.csv"
REPORT_JSON = PROJECT_ROOT / "data" / "ground_truth_eval_report.json"

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


def evaluate_classification(
    model: PhysioTabPFNModel, test_df: pd.DataFrame
) -> Tuple[Dict, List[float], List[str], List[str]]:
    y_true = test_df["label"].tolist()
    y_pred = []
    confidences = []
    latencies = []

    # Warmup
    warmup_feat = test_df[FEATURE_COLUMNS].iloc[0].values.astype(np.float32)
    model.predict_single(warmup_feat)

    for _, row in test_df.iterrows():
        feat = row[FEATURE_COLUMNS].values.astype(np.float32)
        t0 = time.time()
        pred_label, conf, _ = model.predict_single(feat, min_confidence=0.0)
        latencies.append((time.time() - t0) * 1000.0)
        y_pred.append(pred_label)
        confidences.append(conf)

    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(list(set(y_true + y_pred))))
    labels_order = sorted(list(set(y_true + y_pred)))

    trained_classes = set(model.classes_)
    trained_mask = [y in trained_classes for y in y_true]
    trained_y_true = [y for y, m in zip(y_true, trained_mask) if m]
    trained_y_pred = [y for y, m in zip(y_pred, trained_mask) if m]
    trained_acc = float(accuracy_score(trained_y_true, trained_y_pred)) if trained_y_true else 0.0

    eval_results = {
        "trained_classes": list(trained_classes),
        "trained_classes_accuracy": round(trained_acc, 4),
        "overall_accuracy": round(acc, 4),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "per_class": {
            cls: {
                "precision": round(metrics.get("precision", 0.0), 3),
                "recall": round(metrics.get("recall", 0.0), 3),
                "f1_score": round(metrics.get("f1-score", 0.0), 3),
                "support": metrics.get("support", 0),
            }
            for cls, metrics in report.items()
            if isinstance(metrics, dict) and cls not in ["macro avg", "weighted avg"]
        },
        "confusion_matrix": {
            "labels": labels_order,
            "matrix": cm.tolist(),
        },
    }
    return eval_results, latencies, y_true, y_pred


def evaluate_temporal_repetitions(
    test_df: pd.DataFrame, trained_classes: Optional[set] = None
) -> Dict[str, Dict]:
    results = {}
    for video_name, group in test_df.groupby("video_file"):
        exercise_name = group["label"].iloc[0]
        if trained_classes and exercise_name not in trained_classes:
            continue
        counter = QualityRepCounter(exercise_name)
        
        for _, row in group.iterrows():
            feat = row[FEATURE_COLUMNS].values.astype(np.float32)
            # Simulate prediction as the true exercise for biomechanical testing
            counter.update(exercise_name, 1.0, feat)

        results[exercise_name] = {
            "video_file": video_name,
            "total_frames": len(group),
            "valid_reps_detected": counter.valid_reps,
            "bad_reps_detected": counter.bad_reps,
            "last_rep_status": counter.last_rep_status,
            "target_repetition_passed": (counter.valid_reps >= 1 and counter.bad_reps == 0),
            "feedback": counter.feedback,
        }
    return results


def print_scorecard(eval_results: Dict, rep_results: Dict):
    print("\n" + "=" * 70)
    print(" 📊 TABPFN GROUND-TRUTH BENCHMARK SCORECARD")
    print("=" * 70)
    print(f"Trained Classes Acc:    {eval_results['trained_classes_accuracy'] * 100:.1f}% ({', '.join(eval_results['trained_classes'])})")
    print(f"Overall Accuracy:       {eval_results['overall_accuracy'] * 100:.1f}% (all 108 test frames)")
    print(f"Mean Inference Latency: {eval_results['mean_latency_ms']:.1f} ms  (P95: {eval_results['p95_latency_ms']:.1f} ms)")
    print(f"Mean Posterior Conf:    {eval_results['mean_confidence'] * 100:.1f}%")
    print("-" * 70)

    print("\nClassification Breakdown by Exercise:")
    print(f"{'Exercise':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Frames':<8}")
    print("-" * 62)
    for cls, metrics in eval_results["per_class"].items():
        print(f"{cls:<14} | {metrics['precision']:<10.3f} | {metrics['recall']:<10.3f} | {metrics['f1_score']:<10.3f} | {metrics['support']:<8}")

    print("\nConfusion Matrix:")
    labels = eval_results["confusion_matrix"]["labels"]
    matrix = eval_results["confusion_matrix"]["matrix"]
    header = "Pred -> " + "  ".join([f"{l[:7]:>7}" for l in labels])
    print(header)
    for i, row in enumerate(matrix):
        row_str = "  ".join([f"{val:>7}" for val in row])
        print(f"{labels[i][:7]:<7} | {row_str}")

    print("\n" + "-" * 70)
    print(" 🏋️ TEMPORAL BIOMECHANICAL CYCLE VALIDATION (QualityRepCounter)")
    print("-" * 70)
    print(f"{'Exercise':<14} | {'Valid Reps':<11} | {'Bad Reps':<9} | {'Result':<10} | {'Video File':<16}")
    print("-" * 68)
    for ex_name, r in rep_results.items():
        status_str = "✓ PASS" if r["target_repetition_passed"] else "✗ FAIL"
        print(f"{ex_name:<14} | {r['valid_reps_detected']:<11} | {r['bad_reps_detected']:<9} | {status_str:<10} | {r['video_file']:<16}")
    print("=" * 70 + "\n")


def main():
    if not TEST_SET_CSV.exists():
        print(f"[Notice] {TEST_SET_CSV} not found. Building ground-truth test set first...")
        from scripts.build_test_set import main as build_main
        build_main()

    print("[TabPFN] Loading in-context model from dataset...")
    model = PhysioTabPFNModel(mode="auto", N_ensemble_configurations=1)
    model.fit_from_csv(DATASET_CSV, max_samples_per_class=12)

    test_df = pd.read_csv(TEST_SET_CSV)
    print(f"[Benchmark] Evaluating across {len(test_df)} ground-truth frames...")

    eval_results, _, _, _ = evaluate_classification(model, test_df)
    rep_results = evaluate_temporal_repetitions(test_df, trained_classes=set(model.classes_))

    print_scorecard(eval_results, rep_results)

    # Save full report
    full_report = {
        "timestamp": time.time(),
        "model_classes": list(model.classes_),
        "classification": eval_results,
        "repetition_validation": rep_results,
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"✓ Structured evaluation report saved to: {REPORT_JSON}")


if __name__ == "__main__":
    main()
