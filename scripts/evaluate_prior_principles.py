"""In-Context Evaluator and Benchmark against PriorLabs Best Practices:
1. "Features before raw data"
2. "Mind the loop rate"
3. "Choose the rows you show" (k-NN / cosine similarity context retrieval)
4. "Test on what you haven't seen" (cross-validation holdout evaluation)
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from physio_tabPFN import PhysioTabPFNModel

DATASET_CSV = Path(__file__).resolve().parent.parent / "data" / "exercise_dataset.csv"


def benchmark_holdout_evaluation(
    csv_path: Path = DATASET_CSV,
    n_splits: int = 5,
    max_train_per_class: int = 100,
) -> None:
    """Evaluates Point 4: 'Test on what you haven't seen'.

    Performs K-Fold cross-validation simulating unseen movement sequences.
    """
    print("\n" + "=" * 65)
    print(" 1. PRIORLABS PRINCIPLE: 'TEST ON WHAT YOU HAVEN'T SEEN'")
    print(f" Stratified {n_splits}-Fold Cross-Validation on: {csv_path.name}")
    print("=" * 65)

    df = pd.read_csv(csv_path)
    drop_cols = [c for c in ["label", "timestamp"] if c in df.columns]
    X = df.drop(columns=drop_cols).values.astype(np.float32)
    y = df["label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_y_true = []
    all_y_pred = []
    latencies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Balance training context
        balanced_train_idx = []
        for cls in np.unique(y_train):
            c_idx = np.where(y_train == cls)[0]
            if len(c_idx) > max_train_per_class:
                chosen = np.random.choice(c_idx, max_train_per_class, replace=False)
            else:
                chosen = c_idx
            balanced_train_idx.extend(chosen)

        X_train_sub = X_train[balanced_train_idx]
        y_train_sub = y_train[balanced_train_idx]

        model = PhysioTabPFNModel(mode="auto")
        # Fit in-memory
        model.model.fit(X_train_sub, y_train_sub)
        model.classes_ = np.array(model.model.classes_)
        model.is_fitted = True

        t0 = time.time()
        preds = model.model.predict(X_test)
        elapsed = (time.time() - t0) / len(X_test) * 1000.0  # per query row in ms
        latencies.append(elapsed)

        all_y_true.extend(y_test)
        all_y_pred.extend(preds)
        print(f" Fold {fold}/{n_splits} - Avg Query Latency: {elapsed:.2f} ms/row")

    print("\n--- Out-of-Sample Classification Report ---")
    print(classification_report(all_y_true, all_y_pred, digits=3))
    print(f"Average Single-Sample Prediction Latency: {np.mean(latencies):.2f} ms")


class ContextRetriever:
    """Implements Point 2: 'Choose the rows you show'.

    Instead of passing random static rows to TabPFN, retrieves the Top-K
    most geometrically/dynamically relevant teacher states for the incoming frame.
    """

    def __init__(self, X_bank: np.ndarray, y_bank: np.ndarray, top_k: int = 128) -> None:
        self.X_bank = X_bank.astype(np.float32)
        self.y_bank = y_bank
        self.top_k = min(top_k, len(X_bank))

        # Normalize features for cosine similarity retrieval
        norms = np.linalg.norm(self.X_bank, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.X_normalized = self.X_bank / norms

    def retrieve_context(self, query_feature: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves top_k most similar teacher examples for query_feature."""
        q = np.asarray(query_feature, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Vectorized dot-product cosine similarity
        similarities = np.dot(self.X_normalized, q)
        top_indices = np.argpartition(similarities, -self.top_k)[-self.top_k:]

        return self.X_bank[top_indices], self.y_bank[top_indices]


if __name__ == "__main__":
    benchmark_holdout_evaluation()
