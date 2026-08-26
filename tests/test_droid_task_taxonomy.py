import unittest

from examples.UnifiedPretrain.data_tools.build_droid_task_taxonomy import (
    classify_task,
)
from starVLA.task_language import canonical_droid_task, resolve_task_language


class DroidTaskTaxonomyTest(unittest.TestCase):
    def test_empty_and_no_action(self):
        self.assertEqual(classify_task(""), "")
        self.assertEqual(classify_task("No action."), "no action")
        self.assertEqual(classify_task("Not action"), "no action")

    def test_transfer_aliases_share_one_label(self):
        expected = "place object in container"
        self.assertEqual(classify_task("Put the marker in the cup"), expected)
        self.assertEqual(classify_task("Place the pen inside the mug."), expected)
        self.assertEqual(canonical_droid_task("place a toy into the bin"), expected)
        self.assertEqual(
            resolve_task_language(
                "place a toy into the bin", "droid_lerobot", "droid_taxonomy"
            ),
            expected,
        )

    def test_direction_and_device_tasks(self):
        self.assertEqual(classify_task("Move the cup to the left"), "move object left")
        self.assertEqual(classify_task("Turn off the light"), "switch device off")
        self.assertEqual(classify_task("Press the remote"), "press control")

    def test_open_adjective_does_not_override_close(self):
        self.assertEqual(
            classify_task("Close the open drawer"), "close container or door"
        )

    def test_common_skills(self):
        self.assertEqual(classify_task("Fold the blue towel"), "fold object")
        self.assertEqual(classify_task("Pour the cup into the bowl"), "pour contents")
        self.assertEqual(
            classify_task("Put the rubber band around the jar"), "wrap object"
        )


if __name__ == "__main__":
    unittest.main()
