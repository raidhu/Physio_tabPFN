"""TabPFN Model Wrapper supporting both Local PyTorch and PriorLabs Cloud API.

Provides:
- mode="local": Uses local open-source TabPFNClassifier (offline, ~15-35ms latency on Apple Silicon).
- mode="api": Uses PriorLabs Cloud TabPFN-3.5-Plus API (via tabpfn-client & TABPFN_TOKEN).
- mode="auto": Prefers local model if installed; falls back to cloud API cleanly.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .env_loader import load_env

load_env()

# Check local tabpfn
try:
    import torch
    from tabpfn import TabPFNClassifier as LocalTabPFNClassifier
    HAS_LOCAL_TABPFN = True
except ImportError:
    HAS_LOCAL_TABPFN = False
    LocalTabPFNClassifier = None

# Check cloud tabpfn-client
try:
    from tabpfn_client import TabPFNClassifier as CloudTabPFNClassifier
    from tabpfn_client import init as tabpfn_init
    HAS_CLOUD_TABPFN = True
except ImportError:
    HAS_CLOUD_TABPFN = False
    CloudTabPFNClassifier = None
    tabpfn_init = None


class PhysioTabPFNModel:
    """Unified TabPFN Classifier supporting Local PyTorch and Cloud API execution."""

    def __init__(
        self,
        mode: str = "auto",
        device: str = "auto",
        token: Optional[str] = None,
        N_ensemble_configurations: int = 1,
    ) -> None:
        """
        Args:
            mode: "auto", "local", or "api".
            device: "auto", "cpu", "mps", or "cuda" (for local mode).
            token: PriorLabs API token (for api mode).
            N_ensemble_configurations: Number of ensemble passes for local model.
                                      Default 4 gives sub-30ms inference on Apple Silicon.
        """
        self.mode = mode
        self.requested_device = device
        self.N_ensemble_configurations = N_ensemble_configurations

        # Resolve mode
        if self.mode == "auto":
            if HAS_LOCAL_TABPFN:
                self.active_mode = "local"
            elif HAS_CLOUD_TABPFN:
                self.active_mode = "api"
            else:
                raise ImportError(
                    "Neither 'tabpfn' (local) nor 'tabpfn-client' (cloud API) is installed. "
                    "Install local model with: pip install torch tabpfn"
                )
        else:
            self.active_mode = self.mode

        # Initialize chosen backend
        if self.active_mode == "local":
            if not HAS_LOCAL_TABPFN:
                raise ImportError(
                    "Local TabPFN requested but 'tabpfn' or 'torch' is not installed. "
                    "Install with: pip install torch tabpfn"
                )
            resolved_device = self._resolve_local_device(device)
            print(f"[TabPFN] Initializing LOCAL model on device: {resolved_device}")
            # Note: tabpfn 2.x/10.x uses n_estimators instead of N_ensemble_configurations
            try:
                self.model = LocalTabPFNClassifier(
                    device=resolved_device,
                    n_estimators=self.N_ensemble_configurations,
                )
            except TypeError:
                try:
                    self.model = LocalTabPFNClassifier(
                        device=resolved_device,
                        N_ensemble_configurations=self.N_ensemble_configurations,
                    )
                except TypeError:
                    self.model = LocalTabPFNClassifier(device=resolved_device)
        elif self.active_mode == "api":
            if not HAS_CLOUD_TABPFN:
                raise ImportError(
                    "Cloud API requested but 'tabpfn-client' is not installed. "
                    "Install with: pip install tabpfn-client"
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
            print("[TabPFN] Initializing CLOUD API model (PriorLabs)")
            self.model = CloudTabPFNClassifier()
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Expected 'auto', 'local', or 'api'.")

        self.classes_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted = False

    def _resolve_local_device(self, device: str) -> str:
        if device != "auto":
            return device
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def fit_from_csv(
        self,
        csv_path: Union[str, Path],
        max_samples_per_class: int = 25,
        random_state: int = 42,
    ) -> "PhysioTabPFNModel":
        """Loads labeled CSV, balances/subsamples data, and fits the TabPFN in-context model."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        drop_cols = [c for c in ["label", "timestamp"] if c in df.columns]
        X = df.drop(columns=drop_cols)
        y = df["label"].values

        self.feature_names_ = list(X.columns)

        # Balanced subsampling to ensure fast in-context fitting
        balanced_indices = []
        np.random.seed(random_state)
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            if len(cls_idx) > max_samples_per_class:
                selected = np.random.choice(cls_idx, size=max_samples_per_class, replace=False)
            else:
                selected = cls_idx
            balanced_indices.extend(selected)

        X_past = X.iloc[balanced_indices].values.astype(np.float32)
        y_past = y[balanced_indices]

        print(
            f"[TabPFN] Fitting in-context dataset: {len(X_past)} rows, "
            f"{X_past.shape[1]} features across {len(np.unique(y_past))} classes..."
        )
        self.model.fit(X_past, y_past)
        self.classes_ = np.array(self.model.classes_)
        self.is_fitted = True
        print(f"[TabPFN] Fit complete. Classes: {list(self.classes_)}")
        return self

    def predict_proba(self, X_now: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for one or multiple query rows."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit_from_csv first.")

        if X_now.ndim == 1:
            X_now = X_now.reshape(1, -1)

        X_now = np.asarray(X_now, dtype=np.float32)
        return self.model.predict_proba(X_now)

    def predict_single(self, feature_vector: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Predicts class and returns (top_label, top_confidence, all_probabilities)."""
        probs = self.predict_proba(feature_vector)[0]
        top_idx = int(np.argmax(probs))
        top_label = str(self.classes_[top_idx])
        top_confidence = float(probs[top_idx])

        all_probs = {str(cls): float(p) for cls, p in zip(self.classes_, probs)}
        return top_label, top_confidence, all_probs
