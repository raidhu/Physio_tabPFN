from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import urllib.request
import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        PoseLandmarksConnections,
        RunningMode,
        drawing_styles,
        drawing_utils,
    )
except ImportError:
    mp = None
    BaseOptions = None
    vision = None
    PoseLandmarker = None


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "pose_landmarker_heavy.task"


@dataclass
class LandmarkPoint:
    """Represents a single 2D/3D pose landmark."""
    id: int
    name: str
    x: float
    y: float
    z: float
    visibility: float


class PoseEstimator:
    """Estimates human pose landmarks using MediaPipe's modern Tasks Vision API."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_poses: int = 1,
    ) -> None:
        if mp is None or PoseLandmarker is None:
            raise ImportError(
                "MediaPipe is not installed. Please install it using: pip install mediapipe"
            )

        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._ensure_model_downloaded()

        base_options = BaseOptions(model_asset_path=str(self.model_path))
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def _ensure_model_downloaded(self) -> None:
        """Downloads the pretrained pose landmarker task bundle if not present."""
        if not self.model_path.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading MediaPipe pose model to {self.model_path}...")
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, str(self.model_path))
            print("Download complete.")

    def process_frame(self, frame_rgb: np.ndarray):
        """Processes an RGB frame and returns PoseLandmarkerResult."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self._landmarker.detect(mp_image)

    def extract_landmarks(self, results) -> Optional[List[LandmarkPoint]]:
        """Extracts landmarks from the first detected person into typed LandmarkPoint items."""
        if not results or not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        first_pose = results.pose_landmarks[0]
        landmarks: List[LandmarkPoint] = []
        for idx, lm in enumerate(first_pose):
            try:
                name = vision.PoseLandmark(idx).name
            except (ValueError, AttributeError):
                name = f"LANDMARK_{idx}"

            landmarks.append(
                LandmarkPoint(
                    id=idx,
                    name=name,
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility if lm.visibility is not None else 1.0,
                )
            )
        return landmarks

    def extract_numpy_landmarks(
        self, results, include_visibility: bool = True
    ) -> Optional[np.ndarray]:
        """Extracts landmarks as a NumPy matrix of shape (33, 3) or (33, 4).
        
        Useful for feeding directly into machine learning pipelines (TabPFN, PyTorch).
        Columns: [x, y, z] or [x, y, z, visibility].
        """
        if not results or not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        first_pose = results.pose_landmarks[0]
        coords = []
        for lm in first_pose:
            vis = lm.visibility if lm.visibility is not None else 1.0
            if include_visibility:
                coords.append([lm.x, lm.y, lm.z, vis])
            else:
                coords.append([lm.x, lm.y, lm.z])

        return np.array(coords, dtype=np.float32)

    def draw_landmarks(
        self,
        frame: np.ndarray,
        results,
    ) -> np.ndarray:
        """Draws detected pose landmarks on the given image frame (in-place)."""
        if not results or not results.pose_landmarks:
            return frame

        # Draw each detected pose
        for pose_landmarks in results.pose_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                pose_landmarks,
                PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
            )
        return frame

    def close(self) -> None:
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            self._landmarker.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
