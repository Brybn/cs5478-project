"""Monitor confirmation: a random-interruption control matched to a trigger's call budget, and a second fresh cohort.

calibrate: choose the random-interruption rate whose mean calls best match the selected trigger on the
           reserved scheduler-validation groups (no test groups are used).
control:   run the calibrated random control on the original test groups.
confirm:   run fixed K15/K10/K8, the frozen selected triggers and the calibrated random control on new groups.
pool:      pool test + confirmation groups and compute paired differences (pooled.json).
sweep:     rerun extra fixed chunk lengths on the pooled test + confirmation groups (calls-vs-time frontier);
           K15 is rerun and must reproduce the saved rollouts exactly.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from adaptive_action_chunking.artifacts import digest
from adaptive_action_chunking.monitor import FIXED_CHUNKS, _reset_seeds, reset_seed_sources, rollout_jobs
from adaptive_action_chunking.evaluate import _dynamics_cases, _dynamics_summary, _motion_outcomes

DATASET = Path("data/monitor_level")
PREDICTOR = Path("checkpoints/pickcube/monitor_level/predictor.pt")
SELECTION = Path("results/monitor_level/selection.json")
TEST_RESULTS = Path("results/monitor_level/test.json")
RANDOM_THRESHOLDS = [0.99, 0.98, 0.97, 0.95, 0.93, 0.92, 0.90, 0.87, 0.85, 0.80]  # replan if u > threshold, p = 1 - threshold
TARGET_VARIANTS = ["acceleration_calls15"]
LIGHT = ("states", "positions", "actions", "plans", "monitor_decisions", "trace", "policy_call_seconds")


def run(tasks, policy, workers, horizon, predictor):
    episodes = []
    for index, (task, episode) in enumerate(rollout_jobs(tasks, policy, workers, predictor, horizon=horizon), 1):
        episode.update({key: task[key] for key in ("group_id", "episode_id", "force_n", "variant")})
        episode.update(inference_seconds=episode["policy_seconds"] + episode["monitor_seconds"])
        episode.update(_motion_outcomes(task, episode))
        episode.update(start_step=task["start_step"], duration_steps=task["duration_steps"])
        episodes.append({k: v for k, v in episode.items() if k not in LIGHT})
        if index % 100 == 0:
            print(f"{index}/{len(tasks)}", flush=True)
    return episodes


def load_controls(output_dir):
    saved = json.loads((Path(output_dir) / "calibration.json").read_text())
    if "controls" in saved:
        return [{"variant": c["control_variant"], "kind": "random", "threshold": c["threshold"]}
                for c in saved["controls"].values()]
    return [{"variant": "random_control", "kind": "random", "threshold": saved["chosen"]["threshold"]}]


def make_tasks(groups, forces, methods):
    cases = _dynamics_cases(groups, forces)
    return [{**case, **method, "chunk_length": method.get("chunk_length", 15), "record_timing": True}
            for case in cases for method in methods]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "confirm", "control", "pool", "sweep"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/monitor_confirmation"))
    parser.add_argument("--groups", type=int, default=40)
    parser.add_argument("--seed", type=int, default=339001)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--predictor", type=Path, default=PREDICTOR)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--test-results", type=Path, default=TEST_RESULTS)
    parser.add_argument("--start-range", type=int, nargs=2, default=(8, 18), metavar=("LO", "HI"),
                        help="confirm: pulse start step drawn uniformly from LO..HI inclusive")
    parser.add_argument("--duration", type=int, default=2, help="confirm: pulse duration in steps")
    parser.add_argument("--targets", nargs="+", default=TARGET_VARIANTS,
                        help="Selected variants whose validation call counts the random controls match")
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 4, 6, 15],
                        help="sweep: fixed chunk lengths rerun on the pooled test + confirmation groups")
    args = parser.parse_args()
    meta = json.loads((args.dataset_dir / "metadata.json").read_text())
    selection = json.loads(args.selection.read_text())
    policy, horizon, forces = Path(meta["policy"]), meta["horizon"], meta["force_levels_n"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: next(item for item in selection["selected"] if item["variant"] == name) for name in args.targets}
    single = len(targets) == 1

    def control_name(name):
        return "random_control" if single else f"random_control_{name}"

    if args.stage == "calibrate":
        output = args.output_dir / "calibration.json"
        if output.exists():
            raise SystemExit(f"{output} exists")
        groups = [g for g in meta["groups"] if g["partition"] == "scheduler_validation"]
        methods = [{"variant": f"random_{t:g}", "kind": "random", "threshold": t} for t in RANDOM_THRESHOLDS]
        episodes = run(make_tasks(groups, forces, methods), policy, args.workers, horizon, args.predictor)
        calls = {m["variant"]: float(np.mean([e["policy_calls"] for e in episodes if e["variant"] == m["variant"]]))
                 for m in methods}
        chosen = {}
        for name, target in targets.items():
            best = min(methods, key=lambda m: abs(calls[m["variant"]] - target["validation_mean_calls"]))
            chosen[name] = {"target": target, **best, "control_variant": control_name(name)}
        saved = {"validation_calls": calls, "controls": chosen,
                 "rule": "Closest mean validation calls to each selected trigger; no outcome used"}
        if single:  # keep the original single-control layout
            saved.update(target=targets[args.targets[0]], chosen=chosen[args.targets[0]])
        output.write_text(json.dumps(saved, indent=1))
        print(json.dumps({"calls": calls, "chosen": chosen}, indent=1))
        return
    if args.stage == "control":
        output = args.output_dir / "control_on_test.json"
        if output.exists():
            raise SystemExit(f"{output} exists")
        controls = load_controls(args.output_dir)
        groups = [g for g in meta["groups"] if g["partition"] == "test"]
        episodes = run(make_tasks(groups, forces, controls), policy, args.workers, horizon, args.predictor)
        output.write_text(json.dumps({"complete": True, "episodes": episodes}))
        return
    if args.stage == "sweep":
        output = args.output_dir / "fixed_k_sweep.json"
        if output.exists():
            raise SystemExit(f"{output} exists")
        test = json.loads(args.test_results.read_text())["episodes"]
        second = json.loads((args.output_dir / "confirmation.json").read_text())["episodes"]
        plan = json.loads((args.output_dir / "plan.json").read_text())
        groups = [g for g in meta["groups"] if g["partition"] == "test"] + plan["groups"]
        methods = [{"variant": f"fixed_k{c}", "kind": "fixed", "chunk_length": c, "threshold": None}
                   for c in args.chunks]
        episodes = run(make_tasks(groups, forces, methods), policy, args.workers, horizon, args.predictor)
        # K15 reruns must reproduce the saved study rollouts exactly (same resets, pulses and policy noise).
        saved = {e["episode_id"]: e for e in [*test, *second] if e["variant"] == "fixed_k15"}
        checked = [e for e in episodes if e["variant"] == "fixed_k15"]
        keys = ("policy_calls", "success_steps", "capped_confirmation_seconds", "success_at_end")
        mismatched = [e["episode_id"] for e in checked if any(e[k] != saved[e["episode_id"]][k] for k in keys
                                                               if k in saved[e["episode_id"]])]
        bootstrap = {"samples": 5000, "seed": 339002}
        result = {"complete": True, "groups": len(groups), "chunks": args.chunks,
                  "k15_reproduction": {"checked": len(checked), "mismatched": mismatched},
                  "episodes": [{k: v for k, v in e.items() if k != "success_steps"} for e in episodes]}
        result["summary"] = _dynamics_summary(result["episodes"], bootstrap)[0]
        output.write_text(json.dumps(result))
        print(f"done; K15 reproduction mismatches {len(mismatched)}/{len(checked)}")
        return
    if args.stage == "pool":
        test = json.loads(args.test_results.read_text())["episodes"]
        control = json.loads((args.output_dir / "control_on_test.json").read_text())["episodes"]
        second = json.loads((args.output_dir / "confirmation.json").read_text())["episodes"]
        keep = {"group_id", "episode_id", "force_n", "variant", "success_at_end", "success_once", "policy_calls",
                "inference_seconds", "policy_seconds", "monitor_seconds", "capped_confirmation_seconds",
                "capped_recovery_seconds", "recovery_confirmed", "confirmed_by_60", "confirmed_by_100",
                "last_ten_success", "confirmed_then_failed"}
        pooled = [{k: e[k] for k in keep} for e in [*test, *control, *second]]
        names = {e["variant"] for e in second}
        pooled = [e for e in pooled if e["variant"] in names]
        bootstrap = {"samples": 5000, "seed": 339002}
        result = {"groups": len({e["group_id"] for e in pooled}), "episodes_per_variant": {
            v: sum(e["variant"] == v for e in pooled) for v in sorted(names)}}
        result["summary"], result["paired_vs_k15"] = _dynamics_summary(pooled, bootstrap)
        for base in ("fixed_k10", *[c["variant"] for c in load_controls(args.output_dir)]):
            result[f"paired_vs_{base}"] = _dynamics_summary(pooled, bootstrap, base)[1]
        (args.output_dir / "pooled.json").write_text(json.dumps(result, indent=1))
        print("pooled", result["groups"], "groups")
        return
    output = args.output_dir / "confirmation.json"
    if output.exists():
        raise SystemExit(f"{output} exists")
    controls = load_controls(args.output_dir)
    seen = set()
    for path in reset_seed_sources():
        seen.update(_reset_seeds(json.loads(path.read_text())) if path.suffix == ".json" else [])
    rng, groups = np.random.default_rng(args.seed), []
    for _ in range(args.groups):
        reset = int(rng.integers(10000, 2**31))
        while reset in seen:
            reset = int(rng.integers(10000, 2**31))
        seen.add(reset)
        groups.append({"group_id": f"reset_{reset}", "reset_seed": reset, "policy_seed": int(rng.integers(0, 2**31)),
                       "start_step": int(rng.integers(args.start_range[0], args.start_range[1] + 1)),
                       "duration_steps": args.duration, "angle": float(rng.uniform(0, 2 * np.pi))})
    methods = [{"variant": f"fixed_k{c}", "kind": "fixed", "chunk_length": c, "threshold": None} for c in FIXED_CHUNKS]
    methods += [{"variant": s["variant"], "kind": s["kind"], "threshold": s["threshold"]}
                for s in [*selection["selected"], *selection.get("exploratory_selected", [])] if s["threshold"] is not None]
    methods.extend(controls)
    plan = {"seed": args.seed, "groups": groups, "forces": forces, "methods": methods, "horizon": horizon,
            "policy_sha256": digest(policy), "predictor_sha256": digest(args.predictor),
            "selection_sha256": digest(args.selection), "bootstrap": {"samples": 5000, "seed": args.seed + 1}}
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=1))
    episodes = run(make_tasks(groups, forces, methods), policy, args.workers, horizon, args.predictor)
    raw = args.output_dir / "episodes.json"
    raw.write_text(json.dumps(episodes))  # keep the expensive rollouts even if a summary below fails
    result = {"complete": True, "plan_sha256": digest(args.output_dir / "plan.json"), "episodes": episodes}
    result["summary"], result["paired_vs_k15"] = _dynamics_summary(episodes, plan["bootstrap"])
    for chunk in FIXED_CHUNKS[1:]:
        result[f"paired_vs_k{chunk}"] = _dynamics_summary(episodes, plan["bootstrap"], f"fixed_k{chunk}")[1]
    for control in controls:
        result[f"paired_vs_{control['variant']}"] = _dynamics_summary(episodes, plan["bootstrap"], control["variant"])[1]
    output.write_text(json.dumps(result))
    print("done")


if __name__ == "__main__":
    main()
