"""Read live LiLa-WAM logs without treating partial progress as acceptance."""
import argparse
import json
from pathlib import Path
import re

from examples.Robotwin.eval_files.run_lila_benchmark import ANSI, COUNTER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    status = json.loads((args.output / "status.json").read_text())
    stage = "full" if (args.output / "full/protocol.json").exists() else "smoke"
    protocol_path = args.output / stage / "protocol.json"
    if not protocol_path.exists():
        print(json.dumps(status, indent=2))
        return
    protocol = json.loads(protocol_path.read_text())
    rows = []
    for task in protocol["tasks"]:
        root = args.output / stage / task
        state = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
        log = ANSI.sub("", (root / "eval.log").read_text(errors="replace")) if (root / "eval.log").exists() else ""
        counters = COUNTER.findall(log)
        successes, trials, seed = map(int, counters[-1]) if counters else (0, 0, None)
        steps = re.findall(r"step:\s*(\d+)\s*/\s*(\d+)", log)
        rows.append(dict(task=task, state=state.get("state", "pending"), successes=successes, trials=trials,
                         target=protocol["episodes_per_task"], last_completed_seed=seed,
                         latest_step=steps[-1] if steps else None, error=state.get("error")))
    summary_path = args.output / "full/summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    print(json.dumps(dict(supervisor=status, stage=stage,
                          completed_episode_successes=sum(row["successes"] for row in rows),
                          completed_episodes=sum(row["trials"] for row in rows),
                          requested_episodes=len(rows) * protocol["episodes_per_task"],
                          passed_90_percent=summary.get("passed_90_percent", False),
                          results=rows), indent=2))


if __name__ == "__main__":
    main()
