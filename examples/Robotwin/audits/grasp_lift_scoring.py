"""Physics-time, object-specific grasp scoring, independent of any policy input."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class LiftHoldScore:
    clearance_m: float = .05
    hold_s: float = 1.
    target: int = 0
    elapsed_s: float = 0.
    success: bool = False
    success_arm: int | None = None
    success_attempt: int | None = None
    success_time_s: float | None = None
    starts: np.ndarray = field(default_factory=lambda: np.full((3, 2), np.nan))
    attempt_ids: np.ndarray = field(default_factory=lambda: np.full((3, 2), -1))
    max_hold_s: np.ndarray = field(default_factory=lambda: np.zeros((3, 2)))
    max_clearance_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    lifted_and_held: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=bool))

    def update(self, dt, clearances, finger_contacts, arm_attempts):
        heights = np.asarray(clearances, dtype=float)
        contacts = np.asarray(finger_contacts, dtype=bool)
        attempts = np.asarray(arm_attempts, dtype=int)
        if not np.isfinite(dt) or dt <= 0 or heights.shape != (3,) or not np.isfinite(heights).all():
            raise ValueError('Invalid physics time or object clearance')
        if contacts.shape != (3, 2, 2) or attempts.shape != (2,):
            raise ValueError('Expected per-object, per-arm, per-finger contacts and two attempt IDs')
        self.elapsed_s += dt
        self.max_clearance_m = np.maximum(self.max_clearance_m, heights)
        qualified = (heights[:, None] >= self.clearance_m) & contacts.all(-1) & (attempts[None] > 0)
        for block, arm in np.ndindex(3, 2):
            if not qualified[block, arm]:
                self.starts[block, arm] = np.nan
                continue
            if np.isnan(self.starts[block, arm]) or self.attempt_ids[block, arm] != attempts[arm]:
                self.starts[block, arm] = self.elapsed_s
                self.attempt_ids[block, arm] = attempts[arm]
            duration = self.elapsed_s - self.starts[block, arm]
            self.max_hold_s[block, arm] = max(self.max_hold_s[block, arm], duration)
            if duration + 1e-9 >= self.hold_s:
                self.lifted_and_held[block] = True
                if block == self.target and not self.success:
                    self.success = True
                    self.success_arm = arm
                    self.success_attempt = int(attempts[arm])
                    self.success_time_s = self.elapsed_s

    def result(self):
        return dict(success=self.success, first_attempt_success=self.success and self.success_attempt == 1,
                    success_arm=self.success_arm, success_attempt=self.success_attempt,
                    success_time_s=self.success_time_s, elapsed_sim_s=self.elapsed_s,
                    max_clearance_m=self.max_clearance_m.tolist(),
                    max_continuous_bilateral_hold_s=self.max_hold_s.tolist(),
                    lifted_and_held=self.lifted_and_held.tolist(),
                    wrong_object_held=bool(np.delete(self.lifted_and_held, self.target).any()))


class AttemptCounter:
    """Count closing command groups; simultaneous two-arm closure is one attempt."""
    def __init__(self, limit=2):
        self.limit = limit
        self.total = 0
        self.armed = np.ones(2, dtype=bool)
        self.arm_attempts = np.zeros(2, dtype=int)
        self.events = []

    def command(self, grips, step, sim_s):
        grips = np.asarray(grips, dtype=float)
        if grips.shape != (2,) or not np.isfinite(grips).all():
            raise ValueError('Invalid gripper commands')
        armed = self.armed | (grips > .8)
        closing = armed & (grips < .2)
        if self.total + int(closing.any()) > self.limit:
            return False
        self.armed = armed
        if closing.any():
            self.total += 1
        for arm in np.flatnonzero(closing):
            self.arm_attempts[arm] = self.total
            self.armed[arm] = False
            self.events.append(dict(attempt=self.total, arm=int(arm), action_step=int(step), sim_s=float(sim_s)))
        return True
