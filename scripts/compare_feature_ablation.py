#!/usr/bin/env python3
"""Compare closed-loop screening results without inventing episode identities.

Only evaluation comparability is checked here. A caller must separately verify
the training/feature manifest before attributing a change to feature extraction.
All conclusions remain preliminary evidence from a single training seed.
"""

import argparse
import ast
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile


class ComparisonError(ValueError):
    """A result is corrupt or its evaluation protocol is incompatible."""


@dataclass
class Evaluation:
    path: Path
    protocol: dict
    successes: int
    episodes: int
    expected_episodes: int
    complete: bool
    groups: dict = field(default_factory=dict)
    # (suite, task, actual environment seed, official init-state index or None)
    outcomes: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    identity_provenance: list = field(default_factory=list)


def _json(path):
    try:
        result = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ComparisonError(f"Expected a JSON object: {path}")
    return result


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ComparisonError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _required(mapping, keys, description):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ComparisonError(f"Missing {description} fields: {', '.join(missing)}")
    return {key: mapping[key] for key in keys}


def _counts(row, episode_key, description):
    n = _integer(row.get(episode_key), f"{description}.{episode_key}")
    s = _integer(row.get("successes"), f"{description}.successes")
    if s > n:
        raise ComparisonError(f"{description}: successes exceed episodes")
    return s, n


def _compare_protocols(first, second, allow_missing=False):
    keys = set(first) | set(second)
    missing = [key for key in sorted(keys) if key not in first or key not in second]
    compared = keys - set(missing) if allow_missing else keys
    differences = [key for key in sorted(compared) if first.get(key) != second.get(key)]
    if differences:
        detail = "; ".join(f"{k}: {first.get(k)!r} != {second.get(k)!r}" for k in differences)
        raise ComparisonError(f"Incompatible evaluation protocols: {detail}")
    return missing


def exact_mcnemar(gains, losses):
    """Exact two-sided binomial test on discordant paired Bernoulli outcomes."""
    n = gains + losses
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(gains, losses) + 1))
    return min(1.0, 2 * tail / (1 << n))


def _snapshot_source(summary_path, plan):
    relative = Path("source_snapshot/examples/LIBERO/eval_files/eval_libero.py")
    candidates = [parent / relative for parent in summary_path.parents]
    checkpoint = plan.get("checkpoint")
    if checkpoint:
        candidates.append(Path(checkpoint).parent.parent / relative)
    return next((path for path in candidates if path.is_file()), None)


def _fixed_libero_identity_source(path):
    """Recognize the archived evaluator's direct, ordered official-state loop.

    This is deliberately conservative: a future evaluator should explicitly log
    its episode seed and init_state_index instead of relying on this contract.
    The current workspace source is never used as evidence for historical logs.
    """
    if path is None:
        return None
    source = path.read_bytes()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    fn = functions.get("eval_libero")
    env_fn = functions.get("_get_libero_env")
    if fn is None or env_fn is None:
        return None
    loops = [node for node in ast.walk(fn) if isinstance(node, ast.For)
             and ast.unparse(node.target) == "episode_idx"]
    if len(loops) != 1:
        return None
    loop = loops[0]
    iterator = loop.iter
    if isinstance(iterator, ast.Call) and ast.unparse(iterator.func) == "tqdm.tqdm" and len(iterator.args) == 1:
        iterator = iterator.args[0]
    if ast.unparse(iterator) != "range(args.num_trials_per_task)":
        return None
    statements = [ast.unparse(node) for node in loop.body]
    if "obs = env.set_init_state(initial_states[episode_idx])" not in statements:
        return None
    if "task_episodes += 1" not in statements:
        return None
    if not any("Starting episode {task_episodes + 1}" in statement for statement in statements):
        return None
    if not any(isinstance(node, ast.Assign) and len(node.targets) == 1
               and isinstance(node.targets[0], ast.Tuple)
               and [ast.unparse(item) for item in node.targets[0].elts] == ["task_episodes", "task_successes"]
               and ast.literal_eval(node.value) == (0, 0)
               for node in ast.walk(fn)
               if isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple)
               and all(isinstance(item, ast.Constant) for item in node.value.elts)):
        return None
    # No control-flow block may skip/reorder an iteration before completion;
    # nested while-loop control is unrelated to the outer episode identity.
    for node in loop.body:
        if isinstance(node, ast.While):
            continue
        if any(isinstance(child, (ast.Break, ast.Continue, ast.Return, ast.For))
               for child in ast.walk(node)):
            return None
    calls = [ast.unparse(node) for node in ast.walk(fn) if isinstance(node, ast.Call)]
    if calls.count("env.set_init_state(initial_states[episode_idx])") != 1:
        return None
    if "task_suite.get_task_init_states(task_id)" not in calls:
        return None
    if "_get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)" not in calls:
        return None
    if not any(isinstance(node, ast.Call) and ast.unparse(node) == "env.seed(seed)"
               for node in ast.walk(env_fn)):
        return None
    return {"method": "archived_fixed_official_init_state_loop",
            "source": str(path), "source_sha256": hashlib.sha256(source).hexdigest(),
            "init_state_index_rule": "episode number minus one", "seed_rule": "logged Arguments.seed"}


def _log_arguments(content, path):
    position = content.find("Arguments:")
    if position < 0:
        raise ComparisonError(f"LIBERO log has no logged Arguments: {path}")
    brace = content.find("{", position)
    try:
        arguments, _ = json.JSONDecoder().raw_decode(content[brace:])
    except json.JSONDecodeError as exc:
        raise ComparisonError(f"Invalid logged Arguments: {path}") from exc
    if not isinstance(arguments, dict):
        raise ComparisonError(f"Invalid logged Arguments: {path}")
    return arguments


LIBERO_ARGUMENT_FIELDS = ("seed", "num_steps_wait", "unnorm_key", "post_process_action",
                         "execute_horizon", "temporal_action_ensemble", "adaptive_ensemble_alpha")
LIBERO_META_FIELDS = ("action_chunk_size", "action_chunk_sizes", "action_specs", "visual_context_length",
                      "task_language_mode", "task_language_modes", "action_keys", "state_keys", "spatial_ablation")
INIT_INDEX = re.compile(r"\b(?:init(?:ial)?[_ ]state[_ ]index|init[_ ]index)\s*[:=]\s*(\d+)", re.I)
EPISODE_SEED = re.compile(r"\b(?:episode[_ ]|env[_ ])?seed\s*[:=]\s*(\d+)", re.I)


def _libero_log(path, row, plan, source_provenance):
    content = path.read_text(errors="replace")
    args = _log_arguments(content, path)
    protocol = _required(args, LIBERO_ARGUMENT_FIELDS, "LIBERO log Arguments")
    if args.get("task_suite_name") != row["suite"] or args.get("start_task") != row["task_id"] or args.get("max_tasks") != 1:
        raise ComparisonError(f"LIBERO task/log identity mismatch: {path}")
    if args["seed"] != plan["seed"] or args["execute_horizon"] != plan["execute_horizon"] or args["unnorm_key"] != plan["unnorm_key"]:
        raise ComparisonError(f"LIBERO Arguments disagree with summary plan: {path}")
    declared_trials = _integer(args.get("num_trials_per_task"), "logged num_trials_per_task", 1)
    requested = plan["trials_per_task"]
    prefix = row.get("counted_first_episodes") if row.get("reused") else None
    if declared_trials != requested and not (prefix == requested and declared_trials >= requested):
        raise ComparisonError(f"LIBERO logged trial budget disagrees with plan: {path}")
    # Deployment metadata permits feature input changes, but action semantics,
    # context, language resolution and evaluation perturbations must match.
    match = re.search(r"server_meta:\s*(\{.*\})\s*\*\*\*", content)
    if match:
        try:
            metadata = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError) as exc:
            raise ComparisonError(f"Invalid server metadata: {path}") from exc
        protocol["server_action_metadata"] = {k: metadata[k] for k in LIBERO_META_FIELDS if k in metadata}
    starts = list(re.finditer(r"Starting episode (\d+)\.\.\.", content))
    if [int(match.group(1)) for match in starts] != list(range(1, len(starts) + 1)):
        raise ComparisonError(f"LIBERO episode sequence repeats or skips an episode: {path}")
    if len(starts) > declared_trials:
        raise ComparisonError(f"Too many LIBERO episode starts: {path}")
    outcomes, identities, flags = [], [], []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(content)
        segment = content[start.end():end]
        successes = re.findall(r"\bSuccess:\s*(True|False)\b", segment)
        if not successes:
            if index != len(starts) - 1:
                raise ComparisonError(f"Missing LIBERO outcome before another episode: {path}")
            break
        if len(successes) != 1:
            raise ComparisonError(f"Duplicate LIBERO episode outcome: {path}")
        outcomes.append(successes[0] == "True")
        explicit = INIT_INDEX.search(segment)
        logged_seed = EPISODE_SEED.search(segment)
        if explicit and (logged_seed or source_provenance):
            init_index = int(explicit.group(1))
            seed = int(logged_seed.group(1)) if logged_seed else args["seed"]
            if source_provenance and (seed, init_index) != (args["seed"], index):
                raise ComparisonError(f"Explicit LIBERO identity disagrees with archived evaluator: {path}")
            flags.append("explicit_init_state_index")
            identities.append((seed, init_index))
        elif source_provenance:
            identities.append((args["seed"], index))
            flags.append("archived_fixed_official_init_state_loop")
        else:
            identities.append(None)
            flags.append("unverified_episode_identity")
    if prefix is not None:
        outcomes, identities, flags = outcomes[:prefix], identities[:prefix], flags[:prefix]
    if len(outcomes) != row["episodes"] or sum(outcomes) != row["successes"]:
        raise ComparisonError(f"LIBERO log outcomes disagree with summary counts: {path}")
    counters = [(int(n), int(s)) for n, s in re.findall(
        r"# episodes completed so far:\s*(\d+).*?# successes:\s*(\d+)", content, re.S)]
    if prefix is not None:
        counters = counters[:prefix]
    if len(counters) != len(outcomes) or any(n != i + 1 or s != sum(outcomes[:i + 1]) for i, (n, s) in enumerate(counters)):
        raise ComparisonError(f"LIBERO episode counters disagree with logged outcomes: {path}")
    final_counts = re.findall(r"\bTotal episodes:\s*(\d+)", content)
    if len(final_counts) != 1 or int(final_counts[0]) != declared_trials:
        flags.append("incomplete_log_final_summary")
    identity_rows = [(identity, success) for identity, success in zip(identities, outcomes) if identity is not None]
    return protocol, identity_rows, sorted(set(flags))


def load_libero(path):
    path = Path(path).resolve()
    summary = _json(path)
    plan = summary.get("plan") or _json(path.parent / "plan.json")
    protocol = _required(plan, ("seed", "execute_horizon", "unnorm_key", "trials_per_task", "suites",
                                "total_tasks", "total_episodes", "libero_commit", "precision"), "LIBERO plan")
    requested = _integer(plan["trials_per_task"], "trials_per_task", 1)
    expected_tasks = _integer(plan["total_tasks"], "total_tasks", 1)
    expected = _integer(plan["total_episodes"], "total_episodes", 1)
    if expected != expected_tasks * requested:
        raise ComparisonError("LIBERO plan total episode count disagrees with its task/trial budget")
    suites = plan["suites"]
    if not isinstance(suites, list) or not suites or len(set(suites)) != len(suites):
        raise ComparisonError("LIBERO suite list must be nonempty and unique")
    protocol["suites"] = sorted(suites)
    s, n = _counts(summary, "episodes", "LIBERO summary")
    result = Evaluation(path, protocol, s, n, expected, summary.get("status") == "complete")
    provenance = _fixed_libero_identity_source(_snapshot_source(path, plan))
    if provenance:
        result.identity_provenance.append(provenance)
        protocol["evaluator_source_sha256"] = provenance["source_sha256"]
    tasks = summary.get("tasks", [])
    seen = set()
    log_protocol = None
    for row in tasks:
        suite = row.get("suite")
        task = _integer(row.get("task_id"), "LIBERO task_id")
        if suite not in suites or (suite, task) in seen:
            raise ComparisonError(f"Invalid or duplicate LIBERO task: {suite}/{task}")
        seen.add((suite, task))
        rs, rn = _counts(row, "episodes", f"{suite}/{task}")
        if rn > requested:
            raise ComparisonError(f"LIBERO task exceeds trial budget: {suite}/{task}")
        result.groups[(suite, str(task))] = (rs, rn)
        if row.get("exit_code") != 0 or rn != requested:
            result.issues.append(f"Incomplete LIBERO task {suite}/{task}")
        reported_log = Path(row.get("log", f"{suite}_task{task:02d}.log"))
        local_log = path.parent / reported_log.name
        log = local_log if local_log.is_file() else reported_log
        if not log.is_file():
            result.issues.append(f"Missing LIBERO task log: {log}")
            continue
        current, identities, methods = _libero_log(log, row, plan, provenance)
        if log_protocol is not None:
            _compare_protocols(log_protocol, current)
        log_protocol = current
        for identity, success in identities:
            key = (suite, str(task), *identity)
            if key in result.outcomes:
                raise ComparisonError(f"Duplicate LIBERO official episode identity: {key}")
            result.outcomes[key] = success
        if "unverified_episode_identity" in methods:
            result.issues.append(f"Unverified LIBERO init-state index/seed in {log}")
        if "incomplete_log_final_summary" in methods:
            result.issues.append(f"LIBERO task log has no valid final completion count: {log}")
        if "explicit_init_state_index" in methods:
            result.identity_provenance.append({"method": "explicit_init_state_index", "log": str(log)})
    if log_protocol:
        protocol["logged_evaluation_arguments"] = log_protocol
    if len(seen) != expected_tasks:
        result.issues.append(f"Only {len(seen)}/{expected_tasks} LIBERO tasks recorded")
    if expected_tasks == 10 * len(suites) and seen != {(suite, task) for suite in suites for task in range(10)}:
        result.issues.append("LIBERO standard suite task IDs are incomplete")
    if sum(v[0] for v in result.groups.values()) != s or sum(v[1] for v in result.groups.values()) != n:
        raise ComparisonError("LIBERO summary totals disagree with per-task counts")
    for suite, row in summary.get("suites", {}).items():
        ss, sn = _counts(row, "episodes", f"LIBERO suite {suite}")
        counted = [v for k, v in result.groups.items() if k[0] == suite]
        if (ss, sn) != (sum(v[0] for v in counted), sum(v[1] for v in counted)):
            raise ComparisonError(f"LIBERO suite counts disagree with task counts: {suite}")
    if n != expected:
        result.issues.append(f"Only {n}/{expected} LIBERO episodes completed")
    result.complete = result.complete and not result.issues
    return result


ROBOTWIN_PROTOCOL_FIELDS = ("task_config", "episodes_per_task", "seed", "execute_horizon", "action_chunk_size",
                            "smooth_actions", "task_success", "expert_filter", "simulator_image_channel_order",
                            "policy_image_channel_order", "swap_rb_before_normalization", "robotwin_commit")
ROBOTWIN_EVAL_SOURCES = ("gawm_hdf5_server.py", "gawm_hdf5_interface.py", "robotwin_eval_runner.py",
                        "eval_policy.py", "demo_clean.yml", "run_robotwin_hdf5_clean_eval.py", "environment.json")


def load_robotwin(path):
    path = Path(path).resolve()
    summary = _json(path)
    recorded = summary.get("protocol") or _json(path.parent / "protocol.json")
    protocol = _required(recorded, ROBOTWIN_PROTOCOL_FIELDS, "RoboTwin protocol")
    tasks = recorded.get("tasks")
    if not isinstance(tasks, list) or not tasks or len(tasks) != len(set(tasks)):
        raise ComparisonError("RoboTwin protocol task list must be nonempty and unique")
    protocol["tasks"] = sorted(tasks)
    if "environment" in recorded:
        protocol["environment"] = recorded["environment"]
    hashes = {}
    for filename, digest in recorded.get("source_sha256", {}).items():
        basename = Path(filename).name
        if basename in ROBOTWIN_EVAL_SOURCES:
            if basename in hashes and hashes[basename] != digest:
                raise ComparisonError(f"Ambiguous recorded evaluator source hash: {basename}")
            hashes[basename] = digest
    if hashes:
        protocol["evaluation_source_sha256"] = hashes
    requested = _integer(recorded["episodes_per_task"], "episodes_per_task", 1)
    s, n = _counts(summary, "trials", "RoboTwin summary")
    result = Evaluation(path, protocol, s, n, len(tasks) * requested, summary.get("state") == "complete")
    seen = set()
    complete_task_count = 0
    for row in summary.get("results", []):
        task = row.get("task")
        if task not in tasks or task in seen:
            raise ComparisonError(f"Unknown or duplicate RoboTwin task: {task}")
        seen.add(task)
        episodes = row.get("episodes", [])
        if row.get("state") != "complete":
            result.issues.append(f"Incomplete RoboTwin task {task}")
            # The existing evaluator's summary counts only complete tasks.
            continue
        complete_task_count += 1
        rs, rn = _counts(row, "trials", f"RoboTwin task {task}")
        if len(episodes) != rn or rn > requested:
            raise ComparisonError(f"RoboTwin task count disagrees with episodes: {task}")
        if rn != requested:
            result.issues.append(f"Only {rn}/{requested} episodes for RoboTwin task {task}")
        successes = 0
        for episode in episodes:
            seed = _integer(episode.get("seed"), f"{task}.episode.seed")
            success = episode.get("success")
            if not isinstance(success, bool):
                raise ComparisonError(f"Expected Boolean outcome for {task}, seed {seed}")
            key = ("robotwin", task, seed, None)
            if key in result.outcomes:
                raise ComparisonError(f"Duplicate RoboTwin (task, seed): {task}, {seed}")
            result.outcomes[key] = success
            successes += success
        if successes != rs:
            raise ComparisonError(f"RoboTwin success count disagrees with episodes: {task}")
        result.groups[("robotwin", task)] = (rs, rn)
    if sum(v[0] for v in result.groups.values()) != s or sum(v[1] for v in result.groups.values()) != n:
        raise ComparisonError("RoboTwin summary totals disagree with completed per-task episodes")
    if summary.get("total_tasks") != len(tasks) or summary.get("completed_tasks") != complete_task_count:
        raise ComparisonError("RoboTwin summary task counts disagree with protocol/results")
    if summary.get("sources_unchanged") is False:
        result.issues.append("RoboTwin evaluator sources changed during evaluation")
    if summary.get("failed_tasks") or len(seen) != len(tasks) or complete_task_count != len(tasks) or n != result.expected_episodes:
        result.issues.append("RoboTwin evaluation did not finish its complete task/episode budget")
    result.identity_provenance.append({"method": "recorded_actual_task_and_episode_seed"})
    result.complete = result.complete and not result.issues
    return result


def _rate(successes, episodes):
    return successes / episodes if episodes else None


def _paired(first, second, first_count, second_count):
    common = first.keys() & second.keys()
    gains = sum(second[key] and not first[key] for key in common)
    losses = sum(first[key] and not second[key] for key in common)
    baseline_successes = sum(first[key] for key in common)
    candidate_successes = sum(second[key] for key in common)
    return {"intersection_episodes": len(common), "baseline_only_episodes": first_count - len(common),
            "candidate_only_episodes": second_count - len(common),
            "baseline_coverage": len(common) / first_count if first_count else 0.0,
            "candidate_coverage": len(common) / second_count if second_count else 0.0,
            "baseline_successes_on_intersection": baseline_successes,
            "candidate_successes_on_intersection": candidate_successes,
            "gains": gains, "losses": losses,
            "delta_percentage_points_on_intersection": 100 * (gains - losses) / len(common) if common else None,
            "mcnemar_exact_two_sided_p": exact_mcnemar(gains, losses) if common else None}


def _stats(first, second, groups):
    first_s = sum(first.groups.get(group, (0, 0))[0] for group in groups)
    first_n = sum(first.groups.get(group, (0, 0))[1] for group in groups)
    second_s = sum(second.groups.get(group, (0, 0))[0] for group in groups)
    second_n = sum(second.groups.get(group, (0, 0))[1] for group in groups)
    first_outcomes = {k: v for k, v in first.outcomes.items() if k[:2] in groups}
    second_outcomes = {k: v for k, v in second.outcomes.items() if k[:2] in groups}
    first_rate, second_rate = _rate(first_s, first_n), _rate(second_s, second_n)
    return {"baseline": {"successes": first_s, "episodes": first_n, "success_rate": first_rate},
            "candidate": {"successes": second_s, "episodes": second_n, "success_rate": second_rate},
            "delta_percentage_points": 100 * (second_rate - first_rate) if first_rate is not None and second_rate is not None else None,
            "paired": _paired(first_outcomes, second_outcomes, first_n, second_n)}


def compare_results(benchmark, baseline, candidate):
    loader = {"libero": load_libero, "robotwin": load_robotwin}.get(benchmark)
    if loader is None:
        raise ComparisonError(f"Unknown benchmark: {benchmark}")
    first, second = loader(baseline), loader(candidate)
    missing_protocol = _compare_protocols(first.protocol, second.protocol, allow_missing=True)
    if first.complete and second.complete and first.groups.keys() != second.groups.keys():
        raise ComparisonError("Complete evaluations use different task sets")
    groups = first.groups.keys() | second.groups.keys()
    overall = _stats(first, second, groups)
    paired = overall["paired"]
    complete = first.complete and second.complete
    fully_paired = paired["intersection_episodes"] == first.episodes == second.episodes and first.episodes > 0
    positive = (complete and not missing_protocol and fully_paired and paired["gains"] > paired["losses"]
                and paired["mcnemar_exact_two_sided_p"] < 0.05)
    status = "preliminary_positive_signal" if positive else "no_demonstrated_improvement"
    if not complete or missing_protocol:
        status = "incomplete_or_unverified"
    elif not fully_paired:
        status = "preliminary_partial_episode_overlap"
    result = {"benchmark": benchmark, "baseline_summary": str(first.path), "candidate_summary": str(second.path),
              **overall, "complete": complete, "protocol_comparable": not missing_protocol,
              "unverified_protocol_fields": missing_protocol, "fully_paired": fully_paired,
              "paired_screening_signal": positive, "effective": False, "evidence_status": status,
              "evidence_scope": "preliminary_single_training_seed",
              "feature_attribution_verified": False,
              "limitations": ["A single training seed and screening rollouts cannot establish a stable feature-extraction benefit.",
                              "Training/data/world-model equality must be verified separately using the experiment manifest.",
                              "Exact McNemar tests use only matching actual episode identities; unmatched episodes are excluded.",
                              "Per-task/suite p-values are descriptive and have no multiple-comparison correction."],
              "validation": {"baseline": {"complete": first.complete, "expected_episodes": first.expected_episodes,
                                            "verified_episode_identities": len(first.outcomes), "issues": first.issues,
                                            "identity_provenance": first.identity_provenance},
                             "candidate": {"complete": second.complete, "expected_episodes": second.expected_episodes,
                                             "verified_episode_identities": len(second.outcomes), "issues": second.issues,
                                             "identity_provenance": second.identity_provenance}},
              "protocol": first.protocol,
              "suites": {suite: _stats(first, second, {g for g in groups if g[0] == suite})
                         for suite in sorted({g[0] for g in groups})},
              "tasks": [{"suite": suite, "task": int(task) if benchmark == "libero" else task,
                         **_stats(first, second, {(suite, task)})} for suite, task in sorted(groups)]}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("libero", "robotwin"), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() in (args.baseline.resolve(), args.candidate.resolve()):
        parser.error("Output must not overwrite an input summary")
    try:
        report = compare_results(args.benchmark, args.baseline, args.candidate)
    except ComparisonError as exc:
        parser.exit(2, f"Comparison rejected: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, prefix=args.output.name + ".",
                                     suffix=".tmp", delete=False) as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary = Path(handle.name)
    try:
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "evidence_status": report["evidence_status"],
                      "baseline": report["baseline"], "candidate": report["candidate"],
                      "delta_percentage_points": report["delta_percentage_points"], "paired": report["paired"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
