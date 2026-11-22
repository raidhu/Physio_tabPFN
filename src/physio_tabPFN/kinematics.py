"""Kinematics computation module for pose landmark analysis.

Provides:
- 2D/3D angle calculations using the vector dot-product formula.
- Normalization factors (shoulder width, hip width, torso height) to ensure
  invariance to camera distance and user proportions.
- Landmark coordinate normalization.
- Convenient extraction of standard clinical/physio joint angles.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np


# Standard MediaPipe Pose landmark indices for quick reference
class LandmarkIndex:
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


def compute_angle(
    a: Union[np.ndarray, list, Tuple[float, ...]],
    b: Union[np.ndarray, list, Tuple[float, ...]],
    c: Union[np.ndarray, list, Tuple[float, ...]],
    use_3d: bool = False,
    degrees: bool = True,
) -> float:
    """Computes the angle at vertex B formed by points A-B-C.

    Formula:
        theta = arccos( (BA . BC) / (||BA|| * ||BC||) )

    Args:
        a: Endpoint A [x, y] or [x, y, z, ...].
        b: Vertex point B [x, y] or [x, y, z, ...] where angle is measured.
        c: Endpoint C [x, y] or [x, y, z, ...].
        use_3d: If True, uses [x, y, z]; if False, uses [x, y] only.
        degrees: If True, returns angle in degrees [0, 180]. If False, in radians [0, pi].

    Returns:
        Angle in degrees or radians. Returns 0.0 if any vector has zero length.
    """
    dim = 3 if use_3d else 2
    pt_a = np.asarray(a, dtype=np.float64)[:dim]
    pt_b = np.asarray(b, dtype=np.float64)[:dim]
    pt_c = np.asarray(c, dtype=np.float64)[:dim]

    # Vectors BA and BC
    ba = pt_a - pt_b
    bc = pt_c - pt_b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Avoid division by zero if points coincide
    if norm_ba < 1e-7 or norm_bc < 1e-7:
        return 0.0

    cosine_value = np.dot(ba, bc) / (norm_ba * norm_bc)
    # Clip to valid arccos domain [-1.0, 1.0] to prevent numerical precision NaN
    cosine_value = np.clip(cosine_value, -1.0, 1.0)

    angle = np.arccos(cosine_value)

    if degrees:
        return float(np.degrees(angle))
    return float(angle)


def compute_distance(
    p1: Union[np.ndarray, list, Tuple[float, ...]],
    p2: Union[np.ndarray, list, Tuple[float, ...]],
    use_3d: bool = False,
) -> float:
    """Computes Euclidean distance between two points."""
    dim = 3 if use_3d else 2
    vec1 = np.asarray(p1, dtype=np.float64)[:dim]
    vec2 = np.asarray(p2, dtype=np.float64)[:dim]
    return float(np.linalg.norm(vec1 - vec2))


def compute_normalization_scale(
    landmarks: np.ndarray,
    method: str = "torso",
    use_3d: bool = False,
) -> float:
    """Computes a body normalization reference length (scale factor).

    This allows landmark positions and displacement features to be invariant
    to camera distance and differences in user height/build.

    Args:
        landmarks: Array of shape (N, D) where N >= 33, containing landmark coordinates.
        method: Metric to use:
            - "torso": Average distance between shoulders and hips (recommended).
            - "shoulder": Distance between left and right shoulder.
            - "hip": Distance between left and right hip.
            - "height": Mid-shoulder to mid-ankle length.
        use_3d: Whether to compute distance in 3D.

    Returns:
        Positive scale factor. If calculation fails or is zero, returns 1.0.
    """
    if landmarks is None or len(landmarks) < 33:
        return 1.0

    left_shoulder = landmarks[LandmarkIndex.LEFT_SHOULDER]
    right_shoulder = landmarks[LandmarkIndex.RIGHT_SHOULDER]
    left_hip = landmarks[LandmarkIndex.LEFT_HIP]
    right_hip = landmarks[LandmarkIndex.RIGHT_HIP]

    if method == "shoulder":
        scale = compute_distance(left_shoulder, right_shoulder, use_3d=use_3d)
    elif method == "hip":
        scale = compute_distance(left_hip, right_hip, use_3d=use_3d)
    elif method == "torso":
        # Average of left torso (shoulder to hip) and right torso
        left_torso = compute_distance(left_shoulder, left_hip, use_3d=use_3d)
        right_torso = compute_distance(right_shoulder, right_hip, use_3d=use_3d)
        scale = (left_torso + right_torso) / 2.0
    elif method == "height":
        mid_shoulder = (left_shoulder + right_shoulder) / 2.0
        mid_ankle = (landmarks[LandmarkIndex.LEFT_ANKLE] + landmarks[LandmarkIndex.RIGHT_ANKLE]) / 2.0
        scale = compute_distance(mid_shoulder, mid_ankle, use_3d=use_3d)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Fallback if scale is too small / zero
    return float(scale) if scale > 1e-5 else 1.0


def normalize_landmarks(
    landmarks: np.ndarray,
    reference_point_index: int = LandmarkIndex.LEFT_HIP,
    normalization_method: str = "torso",
    use_3d: bool = False,
) -> np.ndarray:
    """Translates landmarks relative to a root landmark and scales by body size.

    After normalization:
    - Root landmark is placed at origin (0, 0, 0).
    - Distances are scaled by the body reference length (e.g. torso length).
    - Preserves visibility channel if present (column index >= 3).

    Returns:
        Normalized landmark array with identical shape to input.
    """
    if landmarks is None or len(landmarks) < 33:
        return landmarks

    normed = landmarks.copy()
    dim = 3 if use_3d else 2

    # Root point (e.g. mid-hip or reference landmark)
    root = landmarks[reference_point_index, :dim]

    # Center coordinates relative to root
    normed[:, :dim] -= root

    # Scale by body reference length
    scale = compute_normalization_scale(landmarks, method=normalization_method, use_3d=use_3d)
    normed[:, :dim] /= scale

    return normed


def extract_physio_angles(
    landmarks: np.ndarray,
    use_3d: bool = False,
    degrees: bool = True,
) -> Dict[str, float]:
    """Computes standard clinical / physiotherapy joint angles for exercise and posture analysis.

    Extracts:
    - Left/Right Knee flexion: Hip - Knee - Ankle
    - Left/Right Elbow flexion: Shoulder - Elbow - Wrist
    - Left/Right Shoulder abduction/elevation: Hip - Shoulder - Elbow
    - Left/Right Hip flexion: Shoulder - Hip - Knee
    - Trunk/Back incline: Mid-shoulder - Mid-hip - Vertical reference

    Returns:
        Dictionary mapping angle names to their scalar values.
    """
    if landmarks is None or len(landmarks) < 33:
        return {}

    idx = LandmarkIndex

    angles = {
        # Elbows
        "left_elbow": compute_angle(
            landmarks[idx.LEFT_SHOULDER],
            landmarks[idx.LEFT_ELBOW],
            landmarks[idx.LEFT_WRIST],
            use_3d=use_3d,
            degrees=degrees,
        ),
        "right_elbow": compute_angle(
            landmarks[idx.RIGHT_SHOULDER],
            landmarks[idx.RIGHT_ELBOW],
            landmarks[idx.RIGHT_WRIST],
            use_3d=use_3d,
            degrees=degrees,
        ),
        # Shoulders
        "left_shoulder": compute_angle(
            landmarks[idx.LEFT_HIP],
            landmarks[idx.LEFT_SHOULDER],
            landmarks[idx.LEFT_ELBOW],
            use_3d=use_3d,
            degrees=degrees,
        ),
        "right_shoulder": compute_angle(
            landmarks[idx.RIGHT_HIP],
            landmarks[idx.RIGHT_SHOULDER],
            landmarks[idx.RIGHT_ELBOW],
            use_3d=use_3d,
            degrees=degrees,
        ),
        # Knees (e.g. Squats, gait)
        "left_knee": compute_angle(
            landmarks[idx.LEFT_HIP],
            landmarks[idx.LEFT_KNEE],
            landmarks[idx.LEFT_ANKLE],
            use_3d=use_3d,
            degrees=degrees,
        ),
        "right_knee": compute_angle(
            landmarks[idx.RIGHT_HIP],
            landmarks[idx.RIGHT_KNEE],
            landmarks[idx.RIGHT_ANKLE],
            use_3d=use_3d,
            degrees=degrees,
        ),
        # Hips
        "left_hip": compute_angle(
            landmarks[idx.LEFT_SHOULDER],
            landmarks[idx.LEFT_HIP],
            landmarks[idx.LEFT_KNEE],
            use_3d=use_3d,
            degrees=degrees,
        ),
        "right_hip": compute_angle(
            landmarks[idx.RIGHT_SHOULDER],
            landmarks[idx.RIGHT_HIP],
            landmarks[idx.RIGHT_KNEE],
            use_3d=use_3d,
            degrees=degrees,
        ),
    }

    return angles


def is_full_body_visible(
    landmarks: np.ndarray,
    min_visibility: float = 0.25,
    max_y_boundary: float = 0.98,
    min_y_boundary: float = -0.10,
    min_x_boundary: float = -0.10,
    max_x_boundary: float = 1.10,
) -> Tuple[bool, str]:
    """Validates whether essential lower-body and core joints are visible and anatomically plausible.

    Checks key landmarks:
    - Shoulders (11, 12)
    - Hips (23, 24)
    - Knees (25, 26)
    - Ankles (27, 28)

    Enforces:
    1. Boundary guard bands (joints must not be clamped off-screen or at bottom edge).
    2. Visibility confidence floor.
    3. Kinetic chain integrity (at least one continuous leg from hip -> knee -> ankle).
    4. Biomechanical proportionality (leg segment length vs torso length).

    Returns:
        (is_visible: bool, feedback_message: str)
    """
    if landmarks is None or len(landmarks) < 33:
        return False, "No person detected"

    idx = LandmarkIndex
    has_vis = landmarks.shape[1] >= 4

    def joint_out(joint_idx: int, min_vis: float = min_visibility, max_coord_y: float = max_y_boundary) -> bool:
        pt = landmarks[joint_idx]
        y, vis = pt[1], pt[3] if has_vis else 1.0
        return (
            (vis < min_vis)
            or (y > max_coord_y)
            or (y < min_y_boundary)
            or (pt[0] < min_x_boundary)
            or (pt[0] > max_x_boundary)
        )

    # Shoulders (upper torso anchor)
    if joint_out(idx.LEFT_SHOULDER, min_vis=0.25, max_coord_y=0.95) and joint_out(idx.RIGHT_SHOULDER, min_vis=0.25, max_coord_y=0.95):
        return False, "Step back: upper body occluded"

    # Hips (critical for pelvic tracking and torso scale)
    if joint_out(idx.LEFT_HIP, min_vis=0.25, max_coord_y=0.97) and joint_out(idx.RIGHT_HIP, min_vis=0.25, max_coord_y=0.97):
        return False, "Step back: hips not in camera frame"

    # Knees (critical for squat depth & step-up flexion)
    if joint_out(idx.LEFT_KNEE, min_vis=0.20, max_coord_y=0.98) and joint_out(idx.RIGHT_KNEE, min_vis=0.20, max_coord_y=0.98):
        return False, "Step back: knees not in camera frame"

    # Ankles (critical for stance and foot elevation)
    if joint_out(idx.LEFT_ANKLE, min_vis=0.15, max_coord_y=0.99) and joint_out(idx.RIGHT_ANKLE, min_vis=0.15, max_coord_y=0.99):
        return False, "Step back: feet not in camera frame"

    # Continuous kinetic chain: at least one complete leg must be visible
    left_leg_valid = (
        not joint_out(idx.LEFT_HIP, min_vis=0.20, max_coord_y=0.97)
        and not joint_out(idx.LEFT_KNEE, min_vis=0.20, max_coord_y=0.98)
        and not joint_out(idx.LEFT_ANKLE, min_vis=0.15, max_coord_y=0.99)
    )
    right_leg_valid = (
        not joint_out(idx.RIGHT_HIP, min_vis=0.20, max_coord_y=0.97)
        and not joint_out(idx.RIGHT_KNEE, min_vis=0.20, max_coord_y=0.98)
        and not joint_out(idx.RIGHT_ANKLE, min_vis=0.15, max_coord_y=0.99)
    )
    if not left_leg_valid and not right_leg_valid:
        return False, "Step back: lower body occluded"

    # Biomechanical proportionality check:
    # A human torso and legs have physiological bounds. When seated at a desk, hallucinated
    # lower-body landmarks collapse against the bottom edge of the frame.
    mid_shoulder = (landmarks[idx.LEFT_SHOULDER, :2] + landmarks[idx.RIGHT_SHOULDER, :2]) / 2.0
    mid_hip = (landmarks[idx.LEFT_HIP, :2] + landmarks[idx.RIGHT_HIP, :2]) / 2.0
    torso_len = float(np.linalg.norm(mid_shoulder - mid_hip))

    if torso_len > 0.05:
        left_femur = float(np.linalg.norm(landmarks[idx.LEFT_HIP, :2] - landmarks[idx.LEFT_KNEE, :2]))
        left_tibia = float(np.linalg.norm(landmarks[idx.LEFT_KNEE, :2] - landmarks[idx.LEFT_ANKLE, :2]))
        right_femur = float(np.linalg.norm(landmarks[idx.RIGHT_HIP, :2] - landmarks[idx.RIGHT_KNEE, :2]))
        right_tibia = float(np.linalg.norm(landmarks[idx.RIGHT_KNEE, :2] - landmarks[idx.RIGHT_ANKLE, :2]))

        left_leg_len = (left_femur + left_tibia) if left_leg_valid else 0.0
        right_leg_len = (right_femur + right_tibia) if right_leg_valid else 0.0
        max_leg_len = max(left_leg_len, right_leg_len)

        # In real human anatomy, femur + tibia is at least 0.70x to 1.8x torso length even in deep squats.
        # If leg length collapses relative to torso, lower body is clipped or hallucinated.
        if max_leg_len < 0.70 * torso_len:
            return False, "Step back: lower body truncated"

    return True, "Full body visible"


def extract_exercise_features(
    landmarks: np.ndarray, require_full_body: bool = True
) -> Optional[np.ndarray]:
    """Computes the 16 distance-invariant kinematic features used for TabPFN classification.

    Features:
    - 10 joint angles (left/right/mean/diff for knees and hips, plus elbows)
    - 6 normalized spatial metrics (trunk inclination, hip-height ratio, foot height diff,
      knee height diff, normalized stance width, torso scale)

    Args:
        landmarks: MediaPipe pose landmark array of shape (33, 3) or (33, 4).
        require_full_body: If True, validates full body visibility before computing features.
                           Returns None if lower body joints are not in camera frame.
    """
    if landmarks is None or len(landmarks) < 33:
        return None

    if require_full_body:
        is_visible, _ = is_full_body_visible(landmarks)
        if not is_visible:
            return None

    idx = LandmarkIndex

    # 1. Normalization scale (torso distance)
    torso_scale = compute_normalization_scale(landmarks, method="torso", use_3d=True)
    if torso_scale < 1e-5:
        torso_scale = 1.0

    # 2. Joint angles
    angles = extract_physio_angles(landmarks, use_3d=False, degrees=True)
    lk = angles["left_knee"]
    rk = angles["right_knee"]
    lh = angles["left_hip"]
    rh = angles["right_hip"]

    # 3. Spatial positions normalized by torso
    mid_hip = (landmarks[idx.LEFT_HIP] + landmarks[idx.RIGHT_HIP]) / 2.0
    mid_shoulder = (landmarks[idx.LEFT_SHOULDER] + landmarks[idx.RIGHT_SHOULDER]) / 2.0
    mid_ankle = (landmarks[idx.LEFT_ANKLE] + landmarks[idx.RIGHT_ANKLE]) / 2.0

    # Torso tilt relative to vertical
    torso_vec = mid_shoulder[:2] - mid_hip[:2]
    torso_len = np.linalg.norm(torso_vec)
    vertical_axis = np.array([0.0, -1.0])
    if torso_len > 1e-6:
        cos_inc = np.dot(torso_vec, vertical_axis) / torso_len
        trunk_inclination = float(np.degrees(np.arccos(np.clip(cos_inc, -1.0, 1.0))))
    else:
        trunk_inclination = 0.0

    hip_height_ratio = float((mid_ankle[1] - mid_hip[1]) / torso_scale)
    foot_height_diff = float(abs(landmarks[idx.LEFT_ANKLE, 1] - landmarks[idx.RIGHT_ANKLE, 1]) / torso_scale)
    knee_height_diff = float(abs(landmarks[idx.LEFT_KNEE, 1] - landmarks[idx.RIGHT_KNEE, 1]) / torso_scale)
    stance_width = float(compute_distance(landmarks[idx.LEFT_ANKLE], landmarks[idx.RIGHT_ANKLE]) / torso_scale)

    vec = [
        lk,
        rk,
        (lk + rk) / 2.0,
        abs(lk - rk),
        lh,
        rh,
        (lh + rh) / 2.0,
        abs(lh - rh),
        angles["left_elbow"],
        angles["right_elbow"],
        trunk_inclination,
        hip_height_ratio,
        foot_height_diff,
        knee_height_diff,
        stance_width,
        torso_scale,
    ]
    return np.array(vec, dtype=np.float32)

