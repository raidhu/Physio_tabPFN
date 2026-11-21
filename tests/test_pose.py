import cv2
from physio_tabPFN import CameraFeed, PoseEstimator

def main():
    # CameraFeed yields RGB frames directly
    with CameraFeed(source=0, width=1280, height=720) as cam, PoseEstimator() as estimator:
        for rgb_frame in cam.frames():
            # Process pose
            results = estimator.process_frame(rgb_frame)

            # Extract numeric landmarks as numpy matrix (shape: 33x4 [x, y, z, visibility])
            # Ready for downstream tabular / feature extraction for TabPFN
            landmarks_np = estimator.extract_numpy_landmarks(results)
            if landmarks_np is not None:
                # e.g., print nose coordinates (landmark 0)
                nose = landmarks_np[0]
                # print(f"Nose: x={nose[0]:.2f}, y={nose[1]:.2f}, vis={nose[3]:.2f}")

            # For preview display with OpenCV, convert to BGR and overlay landmarks
            preview_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            estimator.draw_landmarks(preview_frame, results)

            cv2.imshow("MediaPipe Pose Tracking (Press 'q' to quit)", preview_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
