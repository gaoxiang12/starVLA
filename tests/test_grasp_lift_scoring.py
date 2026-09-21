import unittest
import numpy as np
from examples.Robotwin.audits.grasp_lift_scoring import LiftHoldScore, AttemptCounter


class GraspScoringTests(unittest.TestCase):
    def contacts(self, block=0, arm=0):
        c = np.zeros((3, 2, 2), dtype=bool)
        c[block, arm] = True
        return c

    def test_airborne_without_finger_contact_is_not_a_grasp(self):
        score = LiftHoldScore()
        for _ in range(600):
            score.update(.004, [.08, 0, 0], np.zeros((3, 2, 2)), [1, 0])
        self.assertFalse(score.success)

    def test_requires_full_physics_second(self):
        score = LiftHoldScore()
        for _ in range(250):
            score.update(.004, [.05, 0, 0], self.contacts(), [1, 0])
        self.assertFalse(score.success)
        score.update(.004, [.05, 0, 0], self.contacts(), [1, 0])
        self.assertTrue(score.result()['first_attempt_success'])

    def test_wrong_object_is_reported_without_target_success(self):
        score = LiftHoldScore()
        for _ in range(260):
            score.update(.004, [0, .07, 0], self.contacts(1), [1, 0])
        self.assertTrue(score.result()['wrong_object_held'])
        self.assertFalse(score.success)

    def test_contact_loss_or_arm_switch_resets_hold(self):
        score = LiftHoldScore()
        for arm in [0, 1]:
            for _ in range(200):
                score.update(.004, [.07, 0, 0], self.contacts(0, arm), [1, 2])
        self.assertFalse(score.success)
        for _ in range(51):
            score.update(.004, [.07, 0, 0], self.contacts(0, 1), [1, 2])
        self.assertTrue(score.success)
        self.assertFalse(score.result()['first_attempt_success'])

    def test_contact_on_one_finger_of_each_arm_is_not_bilateral_grasp(self):
        score = LiftHoldScore()
        c = np.zeros((3, 2, 2), bool)
        c[0, :, 0] = True
        for _ in range(260):
            score.update(.004, [.07, 0, 0], c, [1, 2])
        self.assertFalse(score.success)

    def test_attempt_hysteresis_and_third_attempt_refusal(self):
        counter = AttemptCounter()
        for step, grips in enumerate([[0, 1], [0, 1], [.5, 1], [0, 1], [1, 1], [0, 1]]):
            self.assertTrue(counter.command(grips, step, step*.1))
        self.assertEqual(counter.total, 2)
        self.assertFalse(counter.command([0, 0], 6, .6))
        self.assertEqual(counter.total, 2)

    def test_simultaneous_arms_share_first_attempt(self):
        counter = AttemptCounter()
        self.assertTrue(counter.command([0, 0], 0, 0))
        self.assertEqual(counter.total, 1)
        np.testing.assert_array_equal(counter.arm_attempts, [1, 1])


if __name__ == '__main__':
    unittest.main()
