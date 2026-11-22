import cv2
from pathlib import Path
from physio_tabPFN import CameraFeed, PoseEstimator

ASSET_VIDEO = Path(__file__).resolve().parent.parent / "assets" / "Squats.mov"

def main():
    try:
        with CameraFeed(source=0, width=1280, height=720) as cam, PoseEstimator() as estimator:
            for rgb_frame in cam.frames():
                results = estimator.process_frame(rgb_frame)
                landmarks_np = estimator.extract_numpy_landmarks(results)
                preview_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                estimator.draw_landmarks(preview_frame, results)
                cv2.imshow("MediaPipe Pose Tracking (Press 'q' to quit)", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()
    except (RuntimeError, Exception) as e:
        print(f"[Camera Notice] Live camera unavailable: {e}. Testing PoseEstimator with asset video...")
        if ASSET_VIDEO.exists():
            cap = cv2.VideoCapture(str(ASSET_VIDEO))
            ret, frame = cap.read()
            cap.release()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with PoseEstimator() as estimator:
                    results = estimator.process_frame(rgb)
                    landmarks_np = estimator.extract_numpy_landmarks(results)
                    assert landmarks_np is not None, "Pose landmarks should be extracted from reference video"
                    print(f"✓ PoseEstimator verified successfully: extracted {len(landmarks_np)} landmarks.")

if __name__ == "__main__":
    main()
