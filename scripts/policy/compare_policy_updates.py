"""Lineage benchmark: compare a fine-tuned policy with its initializer (and a retained reference) under DDIM10.

Fixed K8/K15, 150 steps, forces 0-0.5 N on fresh reset groups; every rollout is replayed and audited.
Produced results/policy_level_eval, whose cases the sampler table and reproduction check re-run.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
import json
import multiprocessing as mp
from pathlib import Path
import shutil

import numpy as np
from scipy.signal import find_peaks
import torch

from adaptive_action_chunking.artifacts import digest
from adaptive_action_chunking.monitor import _reset_seeds, reset_seed_sources, source_hashes, write_json
from adaptive_action_chunking.env import extract_policy_state, make_env
from adaptive_action_chunking.evaluate import _dynamics_cases, _motion_outcomes, run_episode
from adaptive_action_chunking.policy import load_policy


WORKER = {}
FORCES = [0., .2, .3, .4, .5]


def plan_cases(groups, seed):
    excluded, sources = set(), {}
    for path in reset_seed_sources():
        values = ([json.loads(line) for line in path.read_text().splitlines() if line]
                  if path.suffix == ".jsonl" else [json.loads(path.read_text())])
        seeds = set().union(*(set(_reset_seeds(value)) for value in values))
        excluded.update(seeds)
        sources[str(path)] = {"sha256": digest(path), "seed_count": len(seeds)}
    rng = np.random.default_rng(seed)
    planned, seen = [], set(excluded)
    for _ in range(groups):
        reset = int(rng.integers(10000, 2**31))
        while reset in seen:
            reset = int(rng.integers(10000, 2**31))
        seen.add(reset)
        planned.append({"group_id": f"reset_{reset}", "reset_seed": reset,
                        "policy_seed": int(rng.integers(0, 2**31)),
                        "start_step": int(rng.integers(8, 19)),
                        "angle": float(rng.uniform(0, 2 * np.pi))})
    return _dynamics_cases(planned, [0., .4]), sources, len(excluded)


def outcomes(case, episode, horizon):
    states = np.asarray(episode["states"], dtype=np.float32)[:horizon + 1]
    actions = np.asarray(episode["actions"], dtype=np.float32)[:horizon]
    closes = np.flatnonzero(actions[:, 3] < -.5)
    stop = int(closes[0]) if len(closes) else horizon
    height = states[:stop + 1, 21]
    peaks, _ = find_peaks(height, prominence=.005)
    troughs, _ = find_peaks(-height, prominence=.005)
    success = [step for step in episode["success_steps"] if step <= horizon]
    prefix = {**episode, "steps": horizon, "success_steps": success,
              "success_at_end": horizon in success}
    calls = sum(step < horizon for step in episode["replan_steps"])
    return {"horizon": horizon, "success_at_end": horizon in success,
            "policy_calls": calls, "policy_seconds": float(sum(episode["policy_call_seconds"][:calls])),
            "first_close_seconds": stop / 20,
            "preclose_height_peaks_5mm": len(peaks),
            "preclose_height_extrema_5mm": len(peaks) + len(troughs),
            "upward_preclose_mm": float(np.maximum(np.diff(height), 0).sum() * 1000),
            "grasp_loss_count": int(((states[:-1, 18] > .5) & (states[1:, 18] <= .5)).sum()),
            **_motion_outcomes(case, prefix)}


def audit(env, case, episode):
    """Replay the recorded actions and check states, success flags, pushes and plans exactly."""
    obs, _ = env.reset(seed=case["reset_seed"])
    expected = np.asarray(episode["states"], dtype=np.float32)
    actions = np.asarray(episode["actions"], dtype=np.float32)
    assert actions.shape == (150, 4) and np.isfinite(actions).all() and np.abs(actions).max() <= 1.000001
    np.testing.assert_array_equal(extract_policy_state(obs), expected[0])
    successes = []
    for index, action in enumerate(actions):
        obs, _, _, truncated, info = env.step(action)
        state = extract_policy_state(obs)
        np.testing.assert_array_equal(state, expected[index + 1])
        official = bool(np.linalg.norm(state[29:32] - state[26:29]) <= .025
                        and np.abs(state[9:16]).max() <= .2)
        assert official == bool(info["success"].item())
        assert bool(truncated.item()) == (index == 149)
        if official:
            successes.append(index + 1)
    assert successes == episode["success_steps"] and episode["success_at_end"] == (150 in successes)
    forces = list(range(case["start_step"], case["start_step"] + 2)) if case["force_n"] else []
    assert episode["applied_force_steps"] == sorted(env.unwrapped.applied_force_steps) == forces
    assert episode["applied_force_substeps"] == env.unwrapped.applied_force_substeps == 5 * len(forces)
    assert episode["replan_steps"] == list(range(0, 150, episode["chunk_length"]))
    for planned in episode["plans"]:
        start = planned["step"]
        count = min(episode["chunk_length"], 150 - start)
        np.testing.assert_array_equal(actions[start:start + count], np.asarray(planned["actions"], dtype=np.float32)[:count])


def checkpoint_audit(training_dir, retained):
    """Check a fine-tuning record against its checkpoints; return the policies to compare."""
    path = training_dir / "training.json"
    record = json.loads(path.read_text())
    assert record["complete"]
    paths = {"start": Path(record["initial_checkpoint"]), "updated": training_dir / "best.pt"}
    if retained is not None:
        paths["retained"] = Path(retained)
    assert digest(paths["start"]) == record["initial_checkpoint_sha256"]
    winner = max(record["evaluations"], key=lambda e: (e["rollout"]["mean_final_success"], -e["combined_validation_mse"]))
    assert winner["iteration"] == record["best_iteration"]
    best = torch.load(paths["updated"], map_location="cpu", weights_only=True)
    final = torch.load(training_dir / "last.pt", map_location="cpu", weights_only=True)
    assert best["iteration"] == record["best_iteration"] and best["validation"] == winner
    assert final["iteration"] == record["config"]["total_iters"]
    for checkpoint in (best, final):
        assert checkpoint["config"] == record["config"]
        assert checkpoint["initial_checkpoint_sha256"] == record["initial_checkpoint_sha256"]
        assert {int(state["step"]) for state in checkpoint["optimizer"]["state"].values()} == (
            {checkpoint["iteration"]} if checkpoint["iteration"] else set())
    same = all(torch.equal(v, final["ema_agent"][k]) for k, v in best["ema_agent"].items())
    if not same:
        paths["final"] = training_dir / "last.pt"
    start = torch.load(paths["start"], map_location="cpu", weights_only=True)["ema_agent"]
    changed = sum(not torch.equal(v, start[k]) for k, v in best["ema_agent"].items())
    assert all(torch.isfinite(v).all() for v in best["ema_agent"].values())
    return paths, {"complete": True, "training_record_sha256": digest(path),
        "training_initialization": record["initialization"],
        "selection_horizon": record["config"].get("selection_horizon", 100),
        "best_iteration": record["best_iteration"], "final_iteration": final["iteration"],
        "best_last_ema_identical": same, "changed_parameter_tensors": changed,
        "last_checkpoint_sha256": digest(training_dir / "last.pt"),
        "training_evaluations": record["evaluations"],
        "note": "Selection cases are inspected; comparisons use fresh reset groups."}


def initialize(paths, output):
    torch.set_num_threads(1)
    env = make_env(max_episode_steps=150, disturbance={"start_step": 0, "duration_steps": 0, "force_xy": [0., 0.]})
    policies = {name: load_policy(path, device="cpu", sampler="ddim", inference_steps=10) for name, path in paths.items()}
    obs, _ = env.reset(seed=1)
    history = np.stack([extract_policy_state(obs)] * 2)
    for policy in policies.values():
        policy.sample(history, torch.Generator().manual_seed(0), torch.device("cpu"))
    WORKER.update(env=env, policies=policies, output=Path(output))


def behavior(case, episode, horizon):
    result = outcomes(case, episode, horizon)
    states = np.asarray(episode["states"], dtype=np.float32)[:horizon+1]
    held = states[:, 18] > .5
    grasped = np.flatnonzero(held)
    raised = held & (states[:, 31] >= states[0, 31]+.025)
    lifted = bool(np.any(np.convolve(raised.astype(int), np.ones(5, dtype=int), "valid") == 5))
    distance = float(np.linalg.norm(states[-1, 29:32]-states[-1, 26:29]))
    return {**result, "grasped_once": bool(held.any()),
        "last_ten_held_success": bool(result["last_ten_success"] and held[-10:].all()),
        "first_grasp_seconds": int(grasped[0])/20 if len(grasped) else horizon/20,
        "preclose_bob": result["preclose_height_peaks_5mm"] >= 1,
        "repeated_preclose_bobs": result["preclose_height_peaks_5mm"] >= 2,
        "held_at_horizon": bool(held[-1]), "final_goal_distance_mm": distance*1000,
        "outcome": ("success" if result["success_at_end"] else "no_sustained_lift" if not lifted
                    else "within_goal_not_static" if distance <= .025 else "lifted_outside_goal")}


def rollout(task):
    case, name, k, repeat = task
    env, policy = WORKER["env"], WORKER["policies"][name]
    env.unwrapped.disturbance = {key: case[key] for key in ("start_step", "duration_steps", "force_xy")}
    def execute():
        return run_episode(env, policy, k, case["reset_seed"], torch.device("cpu"), horizon=150,
                           policy_seed=case["policy_seed"], record_timing=True)
    episode = execute()
    episode.update(**case, checkpoint=name, chunk_length=k)
    audit(env, case, episode)
    if repeat:
        repeated = execute()
        for key in ("states", "actions", "plans", "success_steps", "replan_steps"):
            assert repeated[key] == episode[key]
    metrics = [{**case, "checkpoint": name, "chunk_length": k, **behavior(case, episode, h)} for h in (100, 150)]
    trace = WORKER["output"] / "rollouts" / f"{case['episode_id']}_{name}_k{k}.npz"
    states = np.asarray(episode.pop("states"), dtype=np.float32)
    actions = np.asarray(episode.pop("actions"), dtype=np.float32)
    plans = episode.pop("plans")
    np.savez_compressed(trace, states=states, actions=actions,
                        plan_steps=np.asarray([p["step"] for p in plans]),
                        plans=np.asarray([p["actions"] for p in plans], dtype=np.float32))
    episode.update(trace=str(trace), trace_sha256=digest(trace), initial_state=states[0].tolist(), repeat_exact=repeat)
    return episode, metrics


def summarize(rows, names, seed):
    groups = sorted({r["group_id"] for r in rows})
    draws = np.random.default_rng(seed+1).integers(len(groups), size=(5000, len(groups)))
    metrics = ("success_at_end", "capped_confirmation_seconds", "capped_recovery_seconds",
        "first_grasp_seconds", "first_close_seconds", "preclose_bob", "repeated_preclose_bobs",
        "preclose_height_extrema_5mm", "upward_preclose_mm", "last_ten_success", "last_ten_held_success", "grasp_loss_count")
    pairs = [("updated", "start")]
    if "retained" in names:
        pairs += [("updated", "retained")]
    if "final" in names:
        pairs += [("final", "start"), ("updated", "final")]
    if "control" in names:
        pairs += [("updated", "control"), ("control", "start")]
    reports = {}
    for horizon in (100, 150):
        reports[str(horizon)] = {}
        for k in (8, 15):
            reports[str(horizon)][str(k)] = {}
            conditions = {"overall": lambda f: f <= .4, "disturbed": lambda f: 0 < f <= .4,
                          **{f"force_{f:g}": lambda actual, force=f: actual == force for f in FORCES}}
            for condition, includes in conditions.items():
                selected = [r for r in rows if r["horizon"] == horizon and r["chunk_length"] == k and includes(r["force_n"])]
                summaries = {}
                for name in names:
                    records = [r for r in selected if r["checkpoint"] == name]
                    summaries[name] = {"episodes": len(records), "successes": sum(r["success_at_end"] for r in records),
                        "failure_types": dict(Counter(r["outcome"] for r in records if not r["success_at_end"])),
                        **{metric: float(np.mean([r[metric] for r in records])) for metric in metrics}}
                effects = {}
                for treatment, reference in pairs:
                    right = {r["episode_id"]: r for r in selected if r["checkpoint"] == reference}
                    left = [r for r in selected if r["checkpoint"] == treatment]
                    effect = {}
                    for metric in metrics:
                        vector = np.array([np.mean([float(r[metric])-float(right[r["episode_id"]][metric])
                                            for r in left if r["group_id"] == group]) for group in groups])
                        effect[metric] = {"difference": float(vector.mean()),
                                         "ci95": np.quantile(vector[draws].mean(1), [.025, .975]).tolist()}
                    delta = [int(r["success_at_end"])-int(right[r["episode_id"]]["success_at_end"]) for r in left]
                    effects[f"{treatment}_vs_{reference}"] = {"metrics": effect, "wins": delta.count(1),
                        "losses": delta.count(-1), "ties": delta.count(0)}
                reports[str(horizon)][str(k)][condition] = {"policies": summaries, "effects": effects}
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=Path("checkpoints/pickcube/stage5_level"))
    parser.add_argument("--retained", type=Path, default=Path("checkpoints/pickcube/stage3_closing/last.pt"),
                        help="extra reference policy evaluated on the same cases")
    parser.add_argument("--output-dir", type=Path, default=Path("results/policy_level_eval"))
    parser.add_argument("--groups", type=int, default=30)
    parser.add_argument("--seed", type=int, default=199001)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan", type=Path, help="Reuse exact saved cases instead of generating a history-dependent plan")
    args = parser.parse_args(); output = args.output_dir
    if output.exists() or min(args.groups, args.workers) < 1 or args.seed < 0:
        parser.error("Use a fresh output directory, positive groups/workers and nonnegative seed")
    torch.set_num_threads(1)
    paths, checkpoint_report = checkpoint_audit(args.training_dir, args.retained)
    if args.plan:
        previous = json.loads(args.plan.read_text())
        assert previous["checkpoints"] == {n: {"path": str(p), "sha256": digest(p)} for n, p in paths.items()}
        cases, overlaps, excluded = previous["cases"], previous["overlap_sources"], previous["excluded_seed_count"]
        seed, groups = previous["seed"], previous["groups"]
    else:
        base_cases, overlaps, excluded = plan_cases(args.groups, args.seed)
        cases = _dynamics_cases([c for c in base_cases if c["force_n"] == 0], FORCES)
        seed, groups = args.seed, args.groups
    code = {**source_hashes(), str(Path(__file__)): digest(__file__)}
    preserved = {str(p): digest(p) for folder in ("data", "checkpoints", "results")
                 for p in Path(folder).rglob("*") if p.is_file()}
    plan = {"planned_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed, "groups": groups,
        "cases": cases, "checkpoints": {n: {"path": str(p), "sha256": digest(p)} for n, p in paths.items()},
        "sampler": "ddim", "inference_steps": 10, "device": "cpu", "workers": args.workers,
        "horizon": 150, "endpoints": [100, 150], "chunk_lengths": [8, 15], "force_levels_n": FORCES,
        "primary": "Updated minus initializer K8/150 final success, equal-weight forces 0/0.2/0.3/0.4; 0.5 N is a separate secondary stress test.",
        "secondary": "Final checkpoint/retained comparisons when present, K15, nominal/by-force success, capped grasp/close/confirmation/recovery times, pre-close bobbing and grasp losses.",
        "policy_labels": {"start": "Initializer", "updated": f"Fine-tuned ({checkpoint_report['best_iteration']:,})",
                          "retained": "Retained reference", "final": f"Final ({checkpoint_report['final_iteration']:,})"},
        "selection_horizon": checkpoint_report["selection_horizon"],
        "sampling": "Common SeedSequence([policy_seed,absolute_step]) noise, explicit DDIM10/eta0 for every checkpoint, CPU physics and inference.",
        "statistics": "5000 whole-reset-group bootstrap draws, seed+1; secondary intervals pointwise; no outcome-dependent extension or automatic promotion.",
        "overlap_sources": overlaps, "excluded_seed_count": excluded, "code_sha256": code,
        "preserved_sha256": preserved, "reused_plan_sha256": digest(args.plan) if args.plan else None,
        "versions": {p: version(p) for p in ("torch", "numpy", "mani-skill", "diffusers", "scipy")}}
    (output / "rollouts").mkdir(parents=True)
    for name in code:
        path = Path(name); target = output / "source" / path.resolve().relative_to(Path.cwd())
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    write_json(output / "checkpoint_audit.json", checkpoint_report)
    write_json(output / "plan.json", plan)
    result = {"complete": False, "plan_sha256": digest(output / "plan.json"), "episodes": [], "metrics": [],
        "checks": {"physics_actions_replayed": 0, "exact_repeats": 0, "initial_states_matched": True}}
    write_json(output / "results.json", result)
    first = cases[0]["group_id"]
    tasks = [(case, name, k, case["group_id"] == first and case["force_n"] in (0., .4, .5))
             for case in cases for name in paths for k in (8, 15)]
    print(f"Planned {groups} fresh groups, {len(tasks)} rollouts; excluded {excluded} prior seeds", flush=True)
    initial = {}
    try:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=initialize,
                                 initargs=({n: str(p) for n, p in paths.items()}, str(output))) as pool:
            for episode, metrics in pool.map(rollout, tasks):
                assert initial.setdefault(episode["group_id"], episode["initial_state"]) == episode["initial_state"]
                result["episodes"].append(episode); result["metrics"].extend(metrics)
                result["checks"]["physics_actions_replayed"] += 150
                result["checks"]["exact_repeats"] += int(episode["repeat_exact"])
                if len(result["episodes"]) % 30 == 0:
                    write_json(output / "results.json", result)
                    print(f"Audited {len(result['episodes'])}/{len(tasks)} rollouts", flush=True)
        assert all(digest(p) == sha for p, sha in {**preserved, **code}.items())
        result.update(complete=True, reports=summarize(result["metrics"], list(paths), seed), limitations=[
            "Thirty reset groups and one training/noise seed give preliminary uncertainty; secondary intervals are pointwise.",
            "100/150 endpoints share trajectories. CPU inference differs from CUDA checkpoint selection.",
            "Teacher, state coverage, sampling mixture and training initialization/budget may change together; no isolated causal attribution.",
            "Prior cleanup removed some old plans; exclusion covers surviving recorded history only."])
    finally:
        write_json(output / "results.json", result)
    write_json(output / "manifest.json", {"complete": True, "files_sha256": {
        str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    print(json.dumps(result["reports"]["150"]["8"]["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
