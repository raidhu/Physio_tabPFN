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

        # Global posture classification count log
        self.posture_counts = {
            "Squats": 0,
            "Step-Ups": 0,
            "Pogo Hops": 0,
            "Idle": 0,
        }

    def reset(self, exercise: Optional[str] = None) -> None:
        if exercise:
            self.exercise = exercise
        self.valid_reps = 0
        self.bad_reps = 0
        self.state = "standing"
        self.feedback = f"Switched to {self.exercise}. Ready."
        self.last_rep_status = None
        self.current_live_score = 100.0
        self.min_knee_in_rep = 180.0
        self.max_foot_diff_in_rep = 0.0
        self.max_trunk_in_rep = 0.0
        self.posture_counts = {k: 0 for k in self.posture_counts}

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
            return self.get_summary()

        mean_knee = float(features[2])
        knee_diff = float(features[3])
        trunk_inc = float(features[10])
        foot_diff = float(features[12])

        if self.exercise == "Squats":
            self._evaluate_squat(mean_knee, knee_diff, trunk_inc)
        elif self.exercise == "Step-Ups":
            self._evaluate_step_up(mean_knee, foot_diff, knee_diff, trunk_inc)
        elif self.exercise == "Pogo Hops":
            self._evaluate_pogo_hop(mean_knee, foot_diff, trunk_inc)

        return self.get_summary()

    def _evaluate_squat(self, mean_knee: float, knee_diff: float, trunk_inc: float) -> None:
        # Live dynamic quality deduction if posture tilts excessively
        live_penalty = min(trunk_inc * 1.5 + knee_diff * 0.8, 40.0)
        self.current_live_score = max(round(100.0 - live_penalty, 1), 50.0)

        # State transition: standing (> 165) -> descending (< 150)
        if self.state == "standing":
            if mean_knee < 155.0:
                self.state = "descending"
                self.min_knee_in_rep = mean_knee
                self.max_trunk_in_rep = trunk_inc
                self.feedback = "Squatting down... keep back straight."

        elif self.state == "descending":
            self.min_knee_in_rep = min(self.min_knee_in_rep, mean_knee)
            self.max_trunk_in_rep = max(self.max_trunk_in_rep, trunk_inc)

            # Return to standing position
            if mean_knee > 165.0:
                self.state = "standing"

                # Criteria:
                # 1. Depth: knee must reach below 135 deg (partial is rejected, 90 deg ideal)
                # 2. Torso posture: torso shouldn't bend past 35 deg forward
                # 3. Symmetry: knees roughly balanced
                is_deep_enough = self.min_knee_in_rep <= 135.0
                is_back_straight = self.max_trunk_in_rep <= 35.0
                is_balanced = knee_diff <= 25.0

                if is_deep_enough and is_back_straight and is_balanced:
                    self.valid_reps += 1
                    self.last_rep_status = "GOOD"
                    self.feedback = f"✓ Clean Rep {self.valid_reps}! Great depth."
                else:
                    self.bad_reps += 1
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
        self, mean_knee: float, foot_diff: float, knee_diff: float, trunk_inc: float
    ) -> None:
        live_penalty = min(trunk_inc * 2.0, 35.0)
        self.current_live_score = max(round(100.0 - live_penalty, 1), 60.0)

        # Trigger on unilateral leg elevation
        if self.state == "standing":
            if foot_diff > 0.15 or knee_diff > 25.0 or mean_knee < 150.0:
                self.state = "stepping"
                self.max_foot_diff_in_rep = foot_diff
                self.max_trunk_in_rep = trunk_inc
                self.feedback = "Stepping up through heel..."

        elif self.state == "stepping":
            self.max_foot_diff_in_rep = max(self.max_foot_diff_in_rep, foot_diff)
            self.max_trunk_in_rep = max(self.max_trunk_in_rep, trunk_inc)

            if foot_diff < 0.08 and mean_knee > 165.0:
                self.state = "standing"
                has_elevation = self.max_foot_diff_in_rep >= 0.12
                has_posture = self.max_trunk_in_rep <= 25.0

                if has_elevation and has_posture:
                    self.valid_reps += 1
                    self.last_rep_status = "GOOD"
                    self.feedback = f"✓ Good Step-Up {self.valid_reps}!"
                else:
                    self.bad_reps += 1
                    self.last_rep_status = "BAD"
                    self.feedback = "✗ Bad Rep: Incomplete step height"

    def _evaluate_pogo_hop(
        self, mean_knee: float, foot_diff: float, trunk_inc: float
    ) -> None:
        live_penalty = min(abs(mean_knee - 170.0) * 1.0, 30.0)
        self.current_live_score = max(round(100.0 - live_penalty, 1), 65.0)

        if self.state == "standing":
            if foot_diff > 0.04 or mean_knee < 160.0:
                self.state = "hop"
                self.feedback = "Hop! Spring off the ankles."

        elif self.state == "hop":
            if foot_diff < 0.02 and mean_knee > 168.0:
                self.state = "standing"
                if mean_knee >= 145.0:
                    self.valid_reps += 1
                    self.last_rep_status = "GOOD"
                    self.feedback = f"✓ Crisp Hop {self.valid_reps}!"
                else:
                    self.bad_reps += 1
                    self.last_rep_status = "BAD"
                    self.feedback = "✗ Bad Rep: Knees collapsed too deep"

    def get_summary(self) -> Dict:
        total = self.valid_reps + self.bad_reps
        # Calculate dynamic form quality percentage
        if total > 0:
            hist_score = (self.valid_reps / total) * 100.0
            overall_quality = round(0.7 * hist_score + 0.3 * self.current_live_score, 1)
        else:
            overall_quality = round(self.current_live_score, 1)

        return {
            "exercise": self.exercise,
            "valid_reps": self.valid_reps,
            "bad_reps": self.bad_reps,
            "total_attempts": total,
            "quality_score": overall_quality,
            "feedback": self.feedback,
            "last_rep_status": self.last_rep_status,
            "posture_counts": dict(self.posture_counts),
        }
