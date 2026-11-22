"""Quality Rep Counter for Physio-TabPFN.

Tracks and validates form for:
- Squats
- Step-Ups
- Pogo Hops

Key Rules:
- Real-time live form quality evaluation.
- Valid reps increment ONLY when form criteria are met.
- Incomplete depth or bad postures increment bad reps and are REJECTED from the valid count.
- Tracks per-posture counts and overall workout statistics.
"""

from typing import Dict, Optional, Tuple
import numpy as np


class QualityRepCounter:
    """Evaluates motion trajectories, validates repetition form, and rejects bad sets."""

    def __init__(self, exercise: str = "Squats") -> None:
        self.exercise = exercise
        self.valid_reps = 0
        self.bad_reps = 0
        self.state = "standing"  # "standing", "inflection"
        self.feedback = "Ready. Begin your set."
        self.last_rep_status: Optional[str] = None
        self.current_live_score = 100.0

        # Motion trajectory trackers
        self.min_knee_in_rep = 180.0
        self.max_foot_diff_in_rep = 0.0
        self.max_trunk_in_rep = 0.0
        self.frames_in_rep = 0

        # Per-exercise repetition trackers
        self.exercise_reps = {
            "Squats": {"valid": 0, "bad": 0},
            "Step-Ups": {"valid": 0, "bad": 0},
        }

        # Global posture classification count log
        self.posture_counts = {
            "Squats": 0,
            "Step-Ups": 0,
            "Idle": 0,
        }

    def reset(self, exercise: Optional[str] = None, reset_counts: bool = True) -> None:
        if exercise:
            self.exercise = exercise
        if reset_counts:
            self.valid_reps = 0
            self.bad_reps = 0
            for k in self.exercise_reps:
                self.exercise_reps[k] = {"valid": 0, "bad": 0}
            self.posture_counts = {k: 0 for k in self.posture_counts}
        else:
            if self.exercise in self.exercise_reps:
                self.valid_reps = self.exercise_reps[self.exercise]["valid"]
                self.bad_reps = self.exercise_reps[self.exercise]["bad"]

        self.state = "standing"
        self.feedback = f"Switched to {self.exercise}. Ready."
        self.last_rep_status = None
        self.current_live_score = 100.0
        self.min_knee_in_rep = 180.0
        self.max_foot_diff_in_rep = 0.0
        self.max_trunk_in_rep = 0.0
        self.frames_in_rep = 0

    def update(
        self,
        predicted_exercise: str,
        confidence: float,
        features: Optional[np.ndarray],
    ) -> Dict:
        """Processes each frame's kinematics."""
        if predicted_exercise in self.posture_counts:
            self.posture_counts[predicted_exercise] += 1

        if features is None or len(features) < 16:
            if self.state != "standing":
                self.state = "standing"
                self.frames_in_rep = 0
            return self.get_summary()

        # If posture is marked as Unknown or Step Back (due to occluded joints or low confidence),
        # inhibit repetition transitions and reset active rep tracker
        if predicted_exercise in ("Step Back", "Unknown"):
            if self.state != "standing":
                self.state = "standing"
                self.frames_in_rep = 0
            return self.get_summary()

        # Auto-synchronize active exercise to detected posture when in standing/ready state
        if self.state == "standing" and predicted_exercise in self.exercise_reps:
            if self.exercise != predicted_exercise:
                self.exercise = predicted_exercise
                self.valid_reps = self.exercise_reps[self.exercise]["valid"]
                self.bad_reps = self.exercise_reps[self.exercise]["bad"]
                self.min_knee_in_rep = 180.0
                self.max_foot_diff_in_rep = 0.0
                self.max_trunk_in_rep = 0.0
                self.frames_in_rep = 0

        mean_knee = float(features[2])
        knee_diff = float(features[3])
        trunk_inc = float(features[10])
        foot_diff = float(features[12])

        if self.exercise == "Squats":
            self._evaluate_squat(mean_knee, knee_diff, trunk_inc, predicted_exercise, confidence)
        elif self.exercise == "Step-Ups":
            self._evaluate_step_up(mean_knee, foot_diff, knee_diff, trunk_inc, predicted_exercise, confidence)
        elif self.exercise == "Pogo Hops":
            self._evaluate_pogo_hop(mean_knee, foot_diff, trunk_inc)

        return self.get_summary()

    def _evaluate_squat(
        self,
        mean_knee: float,
        knee_diff: float,
        trunk_inc: float,
        predicted_exercise: str = "Squats",
        confidence: float = 1.0,
    ) -> None:
        # Live dynamic quality deduction if posture tilts excessively
        live_penalty = min(trunk_inc * 1.5 + knee_diff * 0.8, 40.0)
        self.current_live_score = max(round(100.0 - live_penalty, 1), 50.0)

        # State transition: standing (> 165) -> descending (< 155)
        if self.state == "standing":
            # Rep descent requires the model to confirm active squat posture (inhibits false starts on Idle/Unknown)
            if predicted_exercise != "Squats":
                return

            if mean_knee < 155.0:
                self.state = "descending"
                self.frames_in_rep = 1
                self.min_knee_in_rep = mean_knee
                self.max_trunk_in_rep = trunk_inc
                self.feedback = "Squatting down... keep back straight."

        elif self.state == "descending":
            self.frames_in_rep += 1
            self.min_knee_in_rep = min(self.min_knee_in_rep, mean_knee)
            self.max_trunk_in_rep = max(self.max_trunk_in_rep, trunk_inc)

            # Return to standing position
            if mean_knee > 165.0:
                self.state = "standing"

                # Criteria calibrated against ground-truth demonstration (Squats.mov):
                # 1. Depth: ground truth reaches 95.6°, so knee must reach <= 130° (partial is rejected)
                # 2. Torso posture: ground truth stays <= 20.9°, so max trunk <= 28°
                # 3. Symmetry: knees roughly balanced (<= 25°)
                is_deep_enough = self.min_knee_in_rep <= 130.0
                is_back_straight = self.max_trunk_in_rep <= 28.0
                is_balanced = knee_diff <= 25.0

                if is_deep_enough and is_back_straight and is_balanced:
                    self.valid_reps += 1
                    self.exercise_reps["Squats"]["valid"] = self.valid_reps
                    self.last_rep_status = "GOOD"
                    self.feedback = f"✓ Clean Rep {self.valid_reps}! Great depth."
                else:
                    self.bad_reps += 1
                    self.exercise_reps["Squats"]["bad"] = self.bad_reps
                    self.last_rep_status = "BAD"
                    reasons = []
                    if not is_deep_enough:
                        reasons.append("Squat deeper")
                    if not is_back_straight:
                        reasons.append("Chest up")
                    if not is_balanced:
                        reasons.append("Uneven knees")
                    self.feedback = f"✗ Bad Rep: {', '.join(reasons)}"

    def _evaluate_step_up(
        self,
        mean_knee: float,
        foot_diff: float,
        knee_diff: float,
        trunk_inc: float,
        predicted_exercise: str = "Step-Ups",
        confidence: float = 1.0,
    ) -> None:
        live_penalty = min(trunk_inc * 2.0, 35.0)
        self.current_live_score = max(round(100.0 - live_penalty, 1), 60.0)

        # Trigger on unilateral leg elevation
        if self.state == "standing":
            # Rep initiation requires the model to confirm active step-up posture (inhibits false starts on Idle/Unknown)
            if predicted_exercise != "Step-Ups":
                return

            if foot_diff > 0.15 or knee_diff > 25.0 or mean_knee < 150.0:
                self.state = "stepping"
                self.frames_in_rep = 1
                self.max_foot_diff_in_rep = foot_diff
                self.max_trunk_in_rep = trunk_inc
                self.feedback = "Stepping up through heel..."

        elif self.state == "stepping":
            self.frames_in_rep += 1
            self.max_foot_diff_in_rep = max(self.max_foot_diff_in_rep, foot_diff)
            self.max_trunk_in_rep = max(self.max_trunk_in_rep, trunk_inc)

            if foot_diff < 0.08 and mean_knee > 165.0:
                self.state = "standing"
                # Calibrated against ground-truth (Step-Ups.mov reaches 0.374) and live box steps (up to 1.05)
                # Torso posture: ground truth stays <= 17.8°
                # Cadence: >= 15 frames
                has_elevation = 0.15 <= self.max_foot_diff_in_rep <= 1.10
                has_posture = self.max_trunk_in_rep <= 18.0
                has_duration = self.frames_in_rep >= 15

                if has_elevation and has_posture and has_duration:
                    self.valid_reps += 1
                    self.exercise_reps["Step-Ups"]["valid"] = self.valid_reps
                    self.last_rep_status = "GOOD"
                    self.feedback = f"✓ Good Step-Up {self.valid_reps}!"
                else:
                    self.bad_reps += 1
                    self.exercise_reps["Step-Ups"]["bad"] = self.bad_reps
                    self.last_rep_status = "BAD"
                    reasons = []
                    if self.max_foot_diff_in_rep < 0.15:
                        reasons.append("Incomplete step height")
                    elif self.max_foot_diff_in_rep > 1.10:
                        reasons.append("Avoid kicking foot too high")
                    if not has_posture:
                        reasons.append("Keep torso upright")
                    if not has_duration:
                        reasons.append("Maintain steady pace (too fast)")
                    self.feedback = f"✗ Bad Rep: {', '.join(reasons)}"

    def _evaluate_pogo_hop(
        self, mean_knee: float, foot_diff: float, trunk_inc: float
    ) -> None:
        live_penalty = min(abs(mean_knee - 170.0) * 1.0, 30.0)
        self.current_live_score = max(round(100.0 - live_penalty, 1), 65.0)

        if self.state == "standing":
            if foot_diff > 0.04 or mean_knee < 160.0:
                self.state = "hop"
                self.frames_in_rep = 1
                self.min_knee_in_rep = mean_knee
                self.max_trunk_in_rep = trunk_inc
                self.feedback = "Hop! Spring off the ankles."

        elif self.state == "hop":
            self.frames_in_rep += 1
            self.min_knee_in_rep = min(self.min_knee_in_rep, mean_knee)
            self.max_trunk_in_rep = max(self.max_trunk_in_rep, trunk_inc)

            if foot_diff < 0.02 and mean_knee > 168.0:
                self.state = "standing"
                # Calibrated against ground-truth demonstration (Pogo-Hops.mov):
                # 1. Knee angle: ground truth stays >= 147.7° (stiff spring off ankles, no deep knee squat)
                # 2. Torso posture: ground truth stays <= 6.3°
                has_knee_spring = self.min_knee_in_rep >= 142.0
                has_posture = self.max_trunk_in_rep <= 12.0

                if has_knee_spring and has_posture:
                    self.valid_reps += 1
                    self.last_rep_status = "GOOD"
                    self.feedback = f"✓ Crisp Hop {self.valid_reps}!"
                else:
                    self.bad_reps += 1
                    self.last_rep_status = "BAD"
                    reasons = []
                    if not has_knee_spring:
                        reasons.append("Knees collapsed too deep")
                    if not has_posture:
                        reasons.append("Keep torso upright")
                    self.feedback = f"✗ Bad Rep: {', '.join(reasons)}"

    def get_summary(self) -> Dict:
        active_valid = self.exercise_reps.get(self.exercise, {}).get("valid", self.valid_reps)
        active_bad = self.exercise_reps.get(self.exercise, {}).get("bad", self.bad_reps)
        total = active_valid + active_bad
        # Calculate dynamic form quality percentage
        if total > 0:
            hist_score = (active_valid / total) * 100.0
            overall_quality = round(0.7 * hist_score + 0.3 * self.current_live_score, 1)
        else:
            overall_quality = round(self.current_live_score, 1)

        return {
            "exercise": self.exercise,
            "valid_reps": active_valid,
            "bad_reps": active_bad,
            "total_attempts": total,
            "quality_score": overall_quality,
            "feedback": self.feedback,
            "last_rep_status": self.last_rep_status,
            "posture_counts": dict(self.posture_counts),
            "exercise_reps": {k: dict(v) for k, v in self.exercise_reps.items()},
        }
