"""TabPFN Client Model Wrapper for Exercise Classification.

Implements TabPFN-3.5-Plus API integration:
  from tabpfn_client import TabPFNClassifier
  model = TabPFNClassifier()
  model.fit(X_past, y_past)
  model.predict_proba(X_now)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .env_loader import load_env

# Load environment variables (.env) so TABPFN_TOKEN is exported
load_env()

try:
    from tabpfn_client import TabPFNClassifier, init as tabpfn_init
except ImportError:
    TabPFNClassifier = None
    tabpfn_init = None


class PhysioTabPFNModel:
    """Wrapper around PriorLabs TabPFN-3.5-Plus Cloud API classifier."""

    def __init__(self, token: Optional[str] = None) -> None:
        if TabPFNClassifier is None:
            raise ImportError(
                "tabpfn-client is not installed. Please install it using: "
                "pip install --upgrade tabpfn-client"
            )

        self.token = token or os.environ.get("TABPFN_TOKEN") or os.environ.get("api_key")
        if self.token:
            os.environ["TABPFN_TOKEN"] = self.token
            try:
                tabpfn_init(token=self.token)
            except TypeError:
                try:
                    tabpfn_init()
                except Exception:
                    pass
            except Exception:
                pass

        self.model = TabPFNClassifier()
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted = False

    def fit_from_csv(
        self,
        csv_path: Union[str, Path],
        max_samples_per_class: int = 150,
        random_state: int = 42,
    ) -> "PhysioTabPFNModel":
        """Loads labeled CSV, balances/subsamples data, and fits the TabPFN in-context model."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Exclude metadata columns
        drop_cols = [c for c in ["label", "timestamp"] if c in df.columns]
        X = df.drop(columns=drop_cols)
        y = df["label"].values

        self.feature_names_ = list(X.columns)

        # Subsample to keep representative, lightning-fast context for TabPFN
        # (TabPFN operates best on high-signal balanced rows)
        balanced_indices = []
        np.random.seed(random_state)
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            if len(cls_idx) > max_samples_per_class:
                selected = np.random.choice(cls_idx, size=max_samples_per_class, replace=False)
            else:
                selected = cls_idx
            balanced_indices.extend(selected)

        X_past = X.iloc[balanced_indices].values
        y_past = y[balanced_indices]

        print(f"Fitting TabPFN with {len(X_past)} high-signal rows across {len(np.unique(y_past))} classes...")
        self.model.fit(X_past, y_past)
        self.classes_ = np.array(self.model.classes_)
        self.is_fitted = True
        print(f"TabPFN Model successfully fitted on classes: {list(self.classes_)}")
        return self

    def predict_proba(self, X_now: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for one or multiple query rows."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit_from_csv first.")

        if X_now.ndim == 1:
            X_now = X_now.reshape(1, -1)

        return self.model.predict_proba(X_now)

    def predict_single(self, feature_vector: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Predicts class and returns (top_label, top_confidence, all_probabilities)."""
        probs = self.predict_proba(feature_vector)[0]
        top_idx = int(np.argmax(probs))
        top_label = str(self.classes_[top_idx])
        top_confidence = float(probs[top_idx])

        all_probs = {str(cls): float(p) for cls, p in zip(self.classes_, probs)}
        return top_label, top_confidence, all_probs
