import cv2
import numpy as np
from physio_tabPFN import (
    CameraFeed,
    PoseEstimator,
    compute_angle,
    compute_normalization_scale,
    extract_physio_angles,
    LandmarkIndex,
)


def draw_hud(frame: np.ndarray, angles: dict, scale: float) -> None:
    """Renders real-time kinematics stats on screen."""
    # Semi-transparent overlay box
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 240), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "KINEMATICS MONITOR", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Torso Scale: {scale:.3f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    y = 90
    for name, angle in angles.items():
        text = f"{name.replace('_', ' ').title()}: {angle:.1f} deg"
        cv2.putText(frame, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 24


def main():
    print("Starting Kinematics Feed. Press 'q' to quit.")
    with CameraFeed(source=0, width=1280, height=720) as cam, PoseEstimator() as estimator:
        for rgb_frame in cam.frames():
            results = estimator.process_frame(rgb_frame)
            landmarks_np = estimator.extract_numpy_landmarks(results)

            preview_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            if landmarks_np is not None:
                # 1. Compute normalization scale (invariant to camera distance)
                torso_scale = compute_normalization_scale(landmarks_np, method="torso", use_3d=True)

                # 2. Compute key physiotherapy joint angles (2D or 3D)
                angles = extract_physio_angles(landmarks_np, use_3d=False)

                # 3. Draw pose skeleton
                estimator.draw_landmarks(preview_frame, results)

                # 4. Display angle overlay on screen
                draw_hud(preview_frame, angles, torso_scale)

            cv2.imshow("Physio Kinematics Angle Analysis", preview_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
