"""TabPFN Model Wrapper supporting both Local PyTorch and PriorLabs Cloud API.

Provides:
- mode="local": Uses local open-source TabPFNClassifier (offline, ~15-35ms latency on Apple Silicon).
- mode="api": Uses PriorLabs Cloud TabPFN-3.5-Plus API (via tabpfn-client & TABPFN_TOKEN).
- mode="auto": Prefers local model if installed; falls back to cloud API cleanly.
"""

import os
os.environ.setdefault("SKB_DATA_DIRECTORY", "/tmp/skrub_data")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

# Maximize multi-core CPU throughput on Apple Silicon M-series
torch.set_num_threads(8)

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
        max_samples_per_class: int = 12,
        random_state: int = 42,
    ) -> "PhysioTabPFNModel":
        """Loads labeled CSV, balances/subsamples data, and fits the TabPFN in-context model."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        drop_cols = [c for c in ["label", "timestamp"] if c in df.columns]
        X = df.drop(columns=drop_cols)
        y = df["label"].values if "label" in df.columns else np.array([])

        self.feature_names_ = list(X.columns)

        if len(df) == 0 or len(y) == 0:
            print("[TabPFN Notice] Dataset is fresh/empty. Waiting for recorded samples...")
            self.classes_ = np.array(["Squats", "Step-Ups", "Idle"])
            self.is_fitted = False
            return self

        # Phase-aligned in-context exemplar selection across movement cycles
        balanced_indices = []
        for cls in np.unique(y):
            sub = df[df["label"] == cls]
            n_samples = min(len(sub), max_samples_per_class)
            if cls == "Squats" and "mean_knee_angle" in sub.columns:
                # Spans from bottom apex (95 deg) through descent/ascent to standing (175 deg)
                sorted_sub = sub.sort_values(by="mean_knee_angle")
                ranks = np.linspace(0, len(sorted_sub) - 1, n_samples, dtype=int)
                chosen = sorted_sub.iloc[ranks].index.tolist()
            elif cls == "Step-Ups" and "foot_height_diff" in sub.columns:
                # Spans from ground level to peak elevated box contact (0.02 to 1.05)
                sorted_sub = sub.sort_values(by="foot_height_diff")
                ranks = np.linspace(0, len(sorted_sub) - 1, n_samples, dtype=int)
                chosen = sorted_sub.iloc[ranks].index.tolist()
            else:
                ranks = np.linspace(0, len(sub) - 1, n_samples, dtype=int)
                chosen = sub.iloc[ranks].index.tolist()
            balanced_indices.extend(chosen)

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
            n_queries = 1 if X_now.ndim == 1 else len(X_now)
            default_p = np.zeros((n_queries, 3), dtype=np.float32)
            default_p[:, 2] = 1.0  # 100% Idle baseline
            return default_p

        if X_now.ndim == 1:
            X_now = X_now.reshape(1, -1)

        X_now = np.asarray(X_now, dtype=np.float32)
        if HAS_LOCAL_TABPFN and torch is not None:
            with torch.inference_mode():
                return self.model.predict_proba(X_now)
        return self.model.predict_proba(X_now)

    def predict(self, X_now: np.ndarray, min_confidence: float = 0.60) -> np.ndarray:
        """Predicts class labels for query rows, returning 'Unknown' if confidence < min_confidence."""
        probs = self.predict_proba(X_now)
        top_indices = np.argmax(probs, axis=1)
        top_confs = np.max(probs, axis=1)
        preds = []
        for idx, conf in zip(top_indices, top_confs):
            if conf < min_confidence:
                preds.append("Unknown")
            else:
                preds.append(str(self.classes_[idx]))
        return np.array(preds)

    def predict_single(
        self, feature_vector: np.ndarray, min_confidence: float = 0.60
    ) -> Tuple[str, float, Dict[str, float]]:
        """Predicts class and returns (top_label, top_confidence, all_probabilities).

        If top_confidence < min_confidence (default 0.60), top_label is 'Unknown'.
        """
        if not self.is_fitted:
            return "Idle", 1.0, {"Squats": 0.0, "Step-Ups": 0.0, "Idle": 1.0}

        probs = self.predict_proba(feature_vector)[0]
        top_idx = int(np.argmax(probs))
        top_confidence = float(probs[top_idx])

        if top_confidence < min_confidence:
            top_label = "Unknown"
        else:
            top_label = str(self.classes_[top_idx])

        all_probs = {str(cls): float(p) for cls, p in zip(self.classes_, probs)}
        return top_label, top_confidence, all_probs
