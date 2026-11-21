from typing import Generator, Optional, Union
import cv2
import numpy as np


class CameraFeed:
    """Captures video frames and returns them in RGB format."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera/video source: {self.source}")

        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read_rgb(self) -> Optional[np.ndarray]:
        """Reads a single frame and returns it in RGB format (or None if stream ends)."""
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera is not open. Call open() or use a 'with' block.")

        ret, frame_bgr = self._cap.read()
        if not ret:
            return None

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Continuously yields RGB frames until the feed ends."""
        while True:
            frame_rgb = self.read_rgb()
            if frame_rgb is None:
                break
            yield frame_rgb

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraFeed":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()