import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import yaml

from examples.Robotwin.eval_files import robotwin_eval_runner


class RobotwinEvalRunnerTest(unittest.TestCase):
    def test_injects_checkpoint_and_overrides_rollout_budget(self):
        recorded = {}
        evaluator = SimpleNamespace()
        evaluator.yaml = yaml

        def original_eval_policy(*args, **kwargs):
            recorded["test_num"] = kwargs["test_num"]
            return 123, 2

        evaluator.eval_policy = original_eval_policy
        evaluator.parse_args_and_config = lambda: {"task_name": "test_task"}

        def evaluator_main(config):
            recorded["config"] = config
            recorded["task_config"] = evaluator.yaml.load(
                "eval_video_log: true\nrender_freq: 0\n", Loader=yaml.FullLoader
            )
            recorded["result"] = evaluator.eval_policy(
                "task", object(), {}, object(), 0, test_num=100
            )

        evaluator.main = evaluator_main
        test_render = ModuleType("test_render")
        test_render.Sapien_TEST = lambda: recorded.setdefault("render_checked", True)

        env = {
            "ROBOTWIN_TEST_NUM": "4",
            "ROBOTWIN_POLICY_CKPT_PATH": "/checkpoints/model.pt",
            "ROBOTWIN_EVAL_VIDEO_LOG": "0",
        }
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.object(
                robotwin_eval_runner,
                "_load_robotwin_evaluator",
                return_value=evaluator,
            ),
            mock.patch.dict(sys.modules, {"test_render": test_render}),
        ):
            robotwin_eval_runner.main()

        self.assertTrue(recorded["render_checked"])
        self.assertEqual(recorded["config"]["policy_ckpt_path"], "/checkpoints/model.pt")
        self.assertEqual(recorded["test_num"], 4)
        self.assertEqual(recorded["result"], (123, 50.0))
        self.assertFalse(recorded["task_config"]["eval_video_log"])

    def test_rejects_invalid_video_boolean(self):
        env = {
            "ROBOTWIN_TEST_NUM": "1",
            "ROBOTWIN_POLICY_CKPT_PATH": "/checkpoints/model.pt",
            "ROBOTWIN_EVAL_VIDEO_LOG": "sometimes",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with self.assertRaisesRegex(SystemExit, "ROBOTWIN_EVAL_VIDEO_LOG"):
                robotwin_eval_runner.main()


if __name__ == "__main__":
    unittest.main()
