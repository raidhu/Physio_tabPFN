import cv2
import numpy as np
from physio_tabPFN import (
    CameraFeed,
    PoseEstimator,
    compute_angle,
    compute_normalization_scale,
    extract_physio_angles,
    is_full_body_visible,
    extract_exercise_features,
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
    try:
        print("Starting Kinematics Feed. Press 'q' to quit.")
        with CameraFeed(source=0, width=1280, height=720) as cam, PoseEstimator() as estimator:
            for rgb_frame in cam.frames():
                results = estimator.process_frame(rgb_frame)
                landmarks_np = estimator.extract_numpy_landmarks(results)
                preview_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                if landmarks_np is not None:
                    torso_scale = compute_normalization_scale(landmarks_np, method="torso", use_3d=True)
                    angles = extract_physio_angles(landmarks_np, use_3d=False)
                    estimator.draw_landmarks(preview_frame, results)
                    draw_hud(preview_frame, angles, torso_scale)

                cv2.imshow("Physio Kinematics Angle Analysis", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()
    except (RuntimeError, Exception) as e:
        print(f"[Camera Notice] Live camera unavailable: {e}. Testing kinematics on reference video...")
        from pathlib import Path
        asset = Path(__file__).resolve().parent.parent / "assets" / "Squats.mov"
        if asset.exists():
            cap = cv2.VideoCapture(str(asset))
            ret, frame = cap.read()
            cap.release()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with PoseEstimator() as estimator:
                    res = estimator.process_frame(rgb)
                    lm = estimator.extract_numpy_landmarks(res)
                    assert lm is not None, "Landmarks must be extracted"
                    scale = compute_normalization_scale(lm, method="torso", use_3d=True)
                    angles = extract_physio_angles(lm, use_3d=False)
                    assert "left_knee" in angles and "right_knee" in angles
                    print(f"✓ Kinematics verified: scale={scale:.3f}, left_knee={angles['left_knee']:.1f}°")

                    # Verify full body visibility & gating
                    visible, msg = is_full_body_visible(lm)
                    assert visible, f"Full body should be visible in reference video frame: {msg}"
                    feat = extract_exercise_features(lm, require_full_body=True)
                    assert feat is not None and len(feat) == 16, "Features should be extracted from full body frame"

                    # Verify occluded/off-screen lower body rejection
                    occluded_lm = lm.copy()
                    occluded_lm[LandmarkIndex.LEFT_HIP :, 1] = 2.5  # Simulate off-screen joints
                    occluded_lm[LandmarkIndex.LEFT_HIP :, 3] = 0.02 # Simulate low visibility
                    occ_visible, occ_msg = is_full_body_visible(occluded_lm)
                    assert not occ_visible, "Occluded lower body must NOT pass visibility check"
                    occ_feat = extract_exercise_features(occluded_lm, require_full_body=True)
                    assert occ_feat is None, "extract_exercise_features must return None when require_full_body=True and joints are occluded"

                    # Verify seated-at-desk rejection (upper body visible, lower body collapsed/clamped)
                    desk_lm = lm.copy()
                    desk_lm[LandmarkIndex.LEFT_SHOULDER, :2] = [0.4, 0.40]
                    desk_lm[LandmarkIndex.RIGHT_SHOULDER, :2] = [0.6, 0.40]
                    desk_lm[LandmarkIndex.LEFT_HIP, :2] = [0.45, 0.88]
                    desk_lm[LandmarkIndex.RIGHT_HIP, :2] = [0.55, 0.88]
                    # Knees and ankles clamped right at bottom of desk/camera
                    desk_lm[LandmarkIndex.LEFT_KNEE, :2] = [0.45, 0.94]
                    desk_lm[LandmarkIndex.RIGHT_KNEE, :2] = [0.55, 0.94]
                    desk_lm[LandmarkIndex.LEFT_ANKLE, :2] = [0.45, 0.97]
                    desk_lm[LandmarkIndex.RIGHT_ANKLE, :2] = [0.55, 0.97]
                    desk_visible, desk_msg = is_full_body_visible(desk_lm)
                    assert not desk_visible, f"Desk sitting must be rejected: {desk_msg}"
                    assert "Step back" in desk_msg, f"Expected step back message, got: {desk_msg}"
                    desk_feat = extract_exercise_features(desk_lm, require_full_body=True)
                    assert desk_feat is None, "Features must be None for desk sitting posture"

                    print("✓ Full-body visibility gating, desk sitting rejection, and feature occlusion checks passed successfully!")


if __name__ == "__main__":
    main()
