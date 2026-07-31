"""Run RoboTwin's evaluator with a configurable rollout budget.

RoboTwin 2.0 hard-codes 100 valid rollouts in ``script/eval_policy.py``.
That is correct for final benchmark reporting, but unnecessarily expensive for
training-time smoke tests and checkpoint selection.  This adapter leaves the
third-party checkout untouched and only overrides the ``test_num`` argument
passed to RoboTwin's ``eval_policy`` function.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw_value!r}") from exc
    if value <= 0:
        raise SystemExit(f"{name} must be positive, got {value}")
    return value


def _load_robotwin_evaluator(robotwin_root: Path) -> ModuleType:
    eval_script = robotwin_root / "script" / "eval_policy.py"
    if not eval_script.is_file():
        raise SystemExit(f"RoboTwin evaluator not found: {eval_script}")

    # Match the import paths that Python supplies when eval_policy.py is run
    # directly from the RoboTwin repository root.
    sys.path.insert(0, str(robotwin_root / "script"))
    sys.path.insert(0, str(robotwin_root))

    spec = importlib.util.spec_from_file_location(
        "_starvla_robotwin_eval_policy", eval_script
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot import RoboTwin evaluator: {eval_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    robotwin_root = Path(os.environ.get("ROBOTWIN_PATH", os.getcwd())).resolve()
    episodes = _positive_int_env("ROBOTWIN_TEST_NUM", 100)
    module = _load_robotwin_evaluator(robotwin_root)

    original_eval_policy = module.eval_policy

    def eval_policy_with_budget(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        mutable_args = list(args)
        if len(mutable_args) >= 6:
            mutable_args[5] = episodes
        else:
            kwargs["test_num"] = episodes
        next_seed, successes = original_eval_policy(*mutable_args, **kwargs)

        # RoboTwin's main() divides this value by its hard-coded 100 when it
        # writes _result.txt. Rescale only that return value so the saved rate
        # remains correct; live counters and logs retain their true counts.
        result_successes = float(successes) * 100.0 / episodes
        return next_seed, result_successes

    module.eval_policy = eval_policy_with_budget

    from test_render import Sapien_TEST

    Sapien_TEST()
    module.main(module.parse_args_and_config())


if __name__ == "__main__":
    main()
