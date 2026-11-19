"""Collect expert queries; --descent or --level selects a DDIM10 correction round."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import shutil

import h5py
import numpy as np
import torch

from adaptive_action_chunking.artifacts import digest
from adaptive_action_chunking.corrections.approach import APPROACH_CONTROLLER
from adaptive_action_chunking.monitor import _reset_seeds, reset_seed_sources, source_hashes, write_json
from adaptive_action_chunking.env import extract_policy_state, make_env
from adaptive_action_chunking.evaluate import run_episode
from adaptive_action_chunking.corrections.pickup import PICKUP_CONTROLLER
from adaptive_action_chunking.corrections.smooth import DescentTeacher, LevelApproachTeacher, SmoothPickupTeacher, SMOOTH_CONTROLLER
from adaptive_action_chunking.policy import load_policy


POLICY = Path("checkpoints/pickcube/stage3_closing/last.pt")
LEVEL_POLICY = Path("checkpoints/pickcube/stage4_descent/best.pt")
SEED, HORIZON, CHUNK = 90312, 150, 8
COUNTS = {"train": {"approach": 8, "close": 4, "lift": 4},
          "validation": {"approach": 2, "close": 1, "lift": 1}}
WORKER = {}


def choose_query(states, kind, min_step=8, chunk_length=CHUNK):
    """Select the first eligible actual boundary, independently of source success."""
    for query in range(chunk_length, 81 if kind in ("approach", "descend", "level") else 97, chunk_length):
        if query < min_step:
            continue
        state = states[query]
        if state[31] < .015 or np.any(np.abs(state[29:31]) > .3):
            continue
        teacher_class = LevelApproachTeacher if kind == "level" else DescentTeacher if kind == "descend" else SmoothPickupTeacher
        teacher = teacher_class(state, float(states[0, 31]))
        if kind == "level":
            eligible = (teacher.phase == "approach" and teacher.level_approach
                        and .003 < np.linalg.norm(state[19:21]-state[29:31]) <= .015
                        and .011 <= state[21]-state[31] <= .030)
        elif kind == "descend":
            eligible = (teacher.phase == "descend" and np.linalg.norm(state[19:21]-state[29:31]) <= .003
                        and .011 <= state[21]-state[31] <= .027)
        elif kind == "approach":
            eligible = teacher.phase == "approach" and np.linalg.norm(state[19:21]-state[29:31]) >= .008
        elif kind == "close":
            eligible = (teacher.phase == "close" and state[7:9].sum() >= .04
                        and state[31] <= states[0, 31]+.015)
        elif kind == "lift":
            eligible = teacher.phase == "lift" and state[18] > .5 and state[31] < states[0, 31]+.025
        else:
            raise ValueError(kind)
        if eligible:
            return query, {"query_phase": teacher.phase,
                "query_xy_mm": float(np.linalg.norm(state[19:21]-state[29:31])*1000),
                "prior_grasp_losses": int(((states[:query, 18] > .5) & (states[1:query+1, 18] <= .5)).sum())}
    return None, {}


def target_coverage(observations, actions, reset_cube_z):
    closed = actions[:15, 3] < -.5
    raised = (observations[:16, 18] > .5) & (observations[:16, 31] >= reset_cube_z+.025)
    lifted = len(raised) == 16 and np.any(np.convolve(raised.astype(int), np.ones(5, dtype=int), "valid") == 5)
    closes = np.flatnonzero(closed)
    stop = int(closes[0]) if len(closes) else min(15, len(actions))
    held_heights = observations[1:16,31][observations[1:16,18] > .5]
    return {"first8_closing_targets": int((actions[:8,3] < -.5).sum()),
            "first8_observed_grasp": bool((observations[1:9,18] > .5).any()),
            "first15_observed_grasp": bool((observations[1:16,18] > .5).any()),
            "first15_held_rise_m": float(held_heights.max()-reset_cube_z) if len(held_heights) else 0.,
            "preclose_upward_mm": float(np.maximum(np.diff(observations[:stop+1,21]),0).sum()*1000),
            "preclose_upward_commands": int((actions[:stop,2] > 1e-6).sum()),
            "first15_closing_targets": int(closed.sum()),
            "first15_verified_lift": bool(lifted),
            "first15_all_closed": bool(len(closed) == 15 and closed.all())}


def target_qualifies(kind, coverage, chunk_length=CHUNK):
    if kind in ("descend", "level"):
        block = chunk_length if kind == "level" else 8
        rise = .005 if kind == "level" else .025
        return (coverage[f"first{block}_closing_targets"] > 0 and coverage[f"first{block}_observed_grasp"]
                and coverage["first15_held_rise_m"] >= rise and coverage["preclose_upward_mm"] <= 2.
                and (kind != "level" or coverage["preclose_upward_commands"] == 0))
    if kind == "approach":
        return coverage["first15_closing_targets"] > 0
    return coverage["first15_all_closed"] and coverage["first15_verified_lift"]


def export_query(handle, trajectory, arrays):
    group = handle.create_group(trajectory)
    group.attrs["conditioning"] = "learner_takeover"
    group["obs"] = arrays["correction_obs"][:16]
    group["actions"] = arrays["correction_actions"][:15]
    group["context_obs"] = arrays["context_obs"]
    group["action_is_expert"] = np.ones(15, dtype=bool)


def initialize(descent=False, level=False):
    torch.set_num_threads(1)
    WORKER.update(env=make_env(max_episode_steps=HORIZON,
        disturbance={"start_step": 0, "duration_steps": 0, "force_xy": [0., 0.]}),
        policy=load_policy(LEVEL_POLICY if level else POLICY, device="cpu", sampler="ddim" if descent or level else "ddpm",
                           inference_steps=10 if descent or level else 100))


def collect_case(case):
    env = WORKER["env"]
    env.unwrapped.disturbance = {k: case[k] for k in ("start_step", "duration_steps", "force_xy")}
    chunk = case.get("chunk_length", CHUNK)
    source = run_episode(env, WORKER["policy"], chunk, case["reset_seed"], torch.device("cpu"),
        horizon=HORIZON, policy_seed=case["policy_seed"], collect_positions=True)
    states, actions = np.asarray(source["states"], dtype=np.float32), np.asarray(source["actions"], dtype=np.float32)
    assert states.shape == (151, 42) and actions.shape == (150, 4)
    flags = np.zeros(HORIZON, dtype=bool)
    flags[np.asarray(source["success_steps"], dtype=int)-1] = True
    query, details = choose_query(states, case["kind"],
        case["start_step"]+case["duration_steps"] if case["kind"] in ("descend", "level") else 8,
        chunk_length=chunk)
    arrays = {"policy_obs": states, "policy_actions": actions, "policy_success": flags,
        "policy_plan_steps": np.asarray(source["replan_steps"]),
        "policy_plans": np.asarray([p["actions"] for p in source["plans"]], dtype=np.float32)}
    row = {**case, **details, "query_step": query, "qualified": False, "reason": "no_eligible_query",
        "source_success_at_100": bool(flags[99]), "source_success_at_150": bool(flags[-1])}
    if query is None:
        return row, arrays
    assert query in source["replan_steps"]
    obs, _ = env.reset(seed=case["reset_seed"])
    np.testing.assert_array_equal(extract_policy_state(obs), states[0])
    for step in range(query):
        obs, _, _, _, _ = env.step(actions[step])
        np.testing.assert_array_equal(extract_policy_state(obs), states[step+1])
    state = states[query]
    teacher_class = LevelApproachTeacher if case["kind"] == "level" else DescentTeacher if case["kind"] == "descend" else SmoothPickupTeacher
    teacher = teacher_class(state, float(states[0, 31]))
    observations, corrections, success = [state.copy()], [], []
    for step in range(query, HORIZON):
        action = teacher.action(env, state)
        if teacher.failed:
            break
        obs, _, _, truncated, info = env.step(action)
        state = extract_policy_state(obs)
        observations.append(state.copy()); corrections.append(action.copy()); success.append(bool(info["success"].item()))
        assert bool(truncated.item()) == (step == HORIZON-1)
    observations, corrections = np.asarray(observations), np.asarray(corrections)
    coverage = target_coverage(observations, corrections, float(states[0, 31]))
    distance = np.linalg.norm(observations[:, 29:32]-observations[:, 26:29], axis=1)
    qualified = (len(corrections) == HORIZON-query and teacher.verified and teacher.attempts == 1
        and target_qualifies(case["kind"], coverage, chunk) and all(success[-10:])
        and np.all(observations[-10:, 18] > .5) and np.all(distance[-10:] <= .010))
    arrays.update(correction_obs=observations, correction_actions=corrections,
        correction_success=np.asarray(success, dtype=bool), context_obs=states[query-1:query+1])
    row.update(**coverage, qualified=bool(qualified),
        reason="qualified" if qualified else "teacher_attempts_exhausted" if teacher.failed else "teacher_target_not_qualified",
        prefix_replay_exact=True, teacher_verified=teacher.verified, teacher_attempts=teacher.attempts,
        teacher_events=teacher.events, teacher_final_distance_mm=float(distance[-1]*1000), correction_steps=len(corrections))
    expected = list(range(case["start_step"], case["start_step"]+2)) if case["force_n"] else []
    assert sorted(env.unwrapped.applied_force_steps) == expected
    assert env.unwrapped.applied_force_substeps == len(expected)*5
    return row, arrays


def stratum(row):
    key = row["split"], row["kind"], row.get("stratum_force", row["force_n"])
    return (*key, row["chunk_length"]) if row["kind"] == "level" else key


def stratum_name(key):
    split, kind, force = key[:3]
    name = f"{split}_{kind}_{force:g}" if force is not None else f"{split}_{kind}_mixed_forces"
    return f"{name}_k{key[3]}" if len(key) == 4 else name


def make_plan(descent=False, level=False):
    excluded, scanned = set(), {}
    for path in reset_seed_sources():
        records = [json.loads(line) for line in path.read_text().splitlines() if line] if path.suffix == ".jsonl" else [json.loads(path.read_text())]
        seeds = set().union(*(set(_reset_seeds(r)) for r in records))
        excluded.update(seeds); scanned[str(path)] = {"sha256": digest(path), "seed_count": len(seeds)}
    seed = 90314 if level else 90313 if descent else SEED
    counts = {"train": {"descend": 32}, "validation": {"descend": 8}} if descent else COUNTS
    rng, seen, cases, targets = np.random.default_rng(seed), set(excluded), [], {}
    if level:
        strata = [(split, "level", force, chunk, count, 256)
            for split, forces in {"train": {0.: 8, .4: 4, .5: 4}, "validation": {0.: 2, .4: 1, .5: 1}}.items()
            for force, count in forces.items() for chunk in (8, 15)]
    else:
        strata = [(split, kind, force, CHUNK, count, 1024 if descent else 256)
            for split, stages in counts.items() for kind, count in stages.items()
            for force in ((None,) if descent else (0., .4))]
    for split, kind, stratum_force, chunk, count, candidates in strata:
        key = (split, kind, stratum_force, chunk) if level else (split, kind, stratum_force)
        targets[key] = count
        for index in range(candidates):
            force = (0., .4)[index%2] if descent else stratum_force
            reset = int(rng.integers(10000, 2**31))
            while reset in seen:
                reset = int(rng.integers(10000, 2**31))
            seen.add(reset)
            angle = float(rng.uniform(0, 2*np.pi))
            cases.append({"episode_id": f"{'level' if level else 'descent' if descent else 'smooth'}_{split}_{reset}",
                "split": split, "kind": kind, "stratum_force": stratum_force, "chunk_length": chunk,
                "reset_seed": reset, "policy_seed": int(rng.integers(0, 2**31)), "force_n": force,
                "force_xy": [float(force*np.cos(angle)), float(force*np.sin(angle))],
                "start_step": int(rng.integers(8, 19)), "duration_steps": 2, "candidate_index": index})
    code = {**source_hashes(), **{p: digest(p) for p in (str(Path(__file__)),
        "src/adaptive_action_chunking/corrections/pickup.py", "src/adaptive_action_chunking/corrections/approach.py",
        "src/adaptive_action_chunking/corrections/smooth.py", "src/adaptive_action_chunking/corrections/collect.py",
        "src/adaptive_action_chunking/finetune.py")}}
    plan = {"complete": False, "mode": "fresh_collection", "collection_kind": "smooth_learner_approach_close_lift_queries",
        "planned_at_utc": datetime.now(timezone.utc).isoformat(), "collection_seed": seed,
        "policy": str(LEVEL_POLICY if level else POLICY), "policy_sha256": digest(LEVEL_POLICY if level else POLICY),
        "source_horizon": HORIZON, "chunk_length": [8, 15] if level else CHUNK,
        "expert_target_actions": 15, "teacher": "smooth_pickup", "controller": PICKUP_CONTROLLER,
        "approach_controller": APPROACH_CONTROLLER, "smooth_controller": SMOOTH_CONTROLLER,
        "cases": cases, "targets_by_stratum": {stratum_name(k): n for k,n in targets.items()},
        "selection": "First actual K8 boundary8..80 approach or8..96 close/lift. Approach: smooth teacher phase approach and XY>=8mm. Close: phase close, fingers>=40mm, cube near table. Lift: phase lift, held cube below reset+25mm. No source-failure requirement; one query per fresh reset.",
        "acceptance": "First-attempt teacher verified lift and final ten held official successes within10mm by absolute150. Approach15 includes a closing command. Close/lift15 all closed and contain five held observations at reset+25mm.",
        "candidate_order": "Round-robin live strata, batches of workers; export first qualified cases in planned order until quotas, record surplus without exporting.",
        "excluded_reset_seed_count": len(excluded), "overlap_sources": scanned, "code_sha256": code,
        "preserved_sha256": {str(p): digest(p) for folder in ("data", "checkpoints", "results") for p in Path(folder).rglob("*") if p.is_file()}}
    plan["sampling"] = {"sampler": "ddim" if descent or level else "ddpm", "inference_steps": 10 if descent or level else 100, "device": "cpu"}
    if descent:
        plan.update(collection_kind="aligned_descent_queries", teacher="descent_pickup",
            selection="First K8 boundary8..80 after pulse: unheld/open, XY<=3mm, tool11..27mm above cube, descent teacher phase descend. No source success filter. Candidate forces alternate0/0.4 N; quotas by split, not force, because aligned disturbed states can be rare.",
            acceptance="First8 actions contain closure and observed grasp; first15 contain held rise>=25mm and pre-close upward travel<=2mm. Full branch: first-attempt verified lift, last ten held official successes within10mm at absolute150.")
    if level:
        plan.update(collection_kind="level_approach_queries", teacher="level_pickup",
            selection="First actual K8/K15 boundary8..80 after pulse: open/unheld, XY>3..15mm, height11..30mm, level teacher phase approach. No source success filter. Split/force/chunk quotas: train16 nominal/8 at0.4/8 at0.5; validation4/2/2; half each atK8/K15.",
            acceptance="Closure and observed grasp within the executed K8/K15 block; first15 held rise>=5mm, pre-close upward travel<=2mm and no upward commands. Full branch: first-attempt verified >=25mm lift, last ten held official successes within10mm at150.")
    return plan, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--descent", action="store_true", help="Collect the DDIM10 direct-descent round (32 train/8 validation)")
    modes.add_argument("--level", action="store_true", help="Collect level alignment at 0/0.4/0.5 N and K8/K15 (32 train/8 validation)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); output = args.output_dir or Path("data/recovery_level" if args.level else "data/recovery_descent" if args.descent else "data/recovery_smooth")
    if output.exists() or args.workers < 1:
        parser.error("Use a fresh output directory and positive workers")
    plan, targets = make_plan(args.descent, args.level)
    output.mkdir(parents=True); (output/"rollouts").mkdir()
    for name in plan["code_sha256"]:
        target = output/"source"/Path(name).resolve().relative_to(Path.cwd())
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(name,target)
    write_json(output/"plan.json", plan); write_json(output/"metadata.json", plan)
    queues = {key: [c for c in plan["cases"] if stratum(c) == key] for key in targets}
    counts, used, accepted, attempts, repeats = Counter(), Counter(), [], [], {}
    files = {split: h5py.File(output/f"{split}.h5", "x") for split in COUNTS}
    print(f"Reserved {len(plan['cases'])} fresh groups, excluding {plan['excluded_reset_seed_count']}; target32 train/8 validation", flush=True)
    try:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=initialize, initargs=(args.descent, args.level)) as pool:
            while True:
                live = [k for k in targets if counts[k] < targets[k] and used[k] < len(queues[k])]
                if not live:
                    break
                batch = []
                while len(batch) < args.workers and live:
                    for key in list(live):
                        batch.append(queues[key][used[key]]); used[key] += 1
                        if used[key] == len(queues[key]): live.remove(key)
                        if len(batch) == args.workers: break
                for row, arrays in pool.map(collect_case, batch):
                    key = stratum(row)
                    raw = output/"rollouts"/f"{row['episode_id']}.npz"; np.savez_compressed(raw, **arrays)
                    exported = row["qualified"] and counts[key] < targets[key]
                    if row["qualified"] and not exported: row["reason"] = "quota_filled"
                    row.update(accepted=bool(exported), raw_trace=str(raw), raw_trace_sha256=digest(raw))
                    if exported:
                        trajectory = f"traj_{sum(e['split'] == row['split'] for e in accepted)}"
                        export_query(files[row["split"]], trajectory, arrays)
                        row.update(trajectory=trajectory, episode_seed=row["reset_seed"], teacher=plan["teacher"],
                            pickup_events=[e for e in row["teacher_events"] if e["step"] <= 15])
                        if counts[key] == 0:
                            repeats[key] = (row, next(c for c in batch if c["episode_id"] == row["episode_id"]))
                        counts[key] += 1; accepted.append(row)
                    attempts.append(row)
                write_json(output/"attempts.json", {"attempts": attempts})
                print(f"Sources {len(attempts)}; exported {len(accepted)}/40", flush=True)
            for (previous, case), (row, arrays) in zip(repeats.values(), pool.map(collect_case, [c for _,c in repeats.values()])):
                with np.load(previous["raw_trace"]) as original:
                    assert set(original.files) == set(arrays)
                    for key, value in arrays.items(): np.testing.assert_array_equal(value, original[key])
                for key in ("query_step", "qualified", "teacher_events"): assert row[key] == previous[key]
    finally:
        for handle in files.values(): handle.close()
    complete = all(counts[k] == n for k,n in targets.items())
    for split in COUNTS:
        write_json(output/f"{split}.json", {"complete": complete, "split": split,
            "episodes": [r for r in accepted if r["split"] == split]})
    for path, sha in {**plan["preserved_sha256"], **plan["code_sha256"]}.items(): assert digest(path) == sha, path
    plan.update(complete=complete, completed_at_utc=datetime.now(timezone.utc).isoformat(), attempts=attempts,
        accepted=accepted, accepted_counts={s:sum(r["split"] == s for r in accepted) for s in COUNTS},
        accepted_by_stratum={stratum_name(k): counts[k] for k in targets}, exact_repeats=len(repeats),
        accepted_by_force={s:{f"{f:g}":sum(r["split"]==s and r["force_n"]==f for r in accepted) for f in ((0.,.4,.5) if args.level else (0.,.4))} for s in COUNTS},
        artifact_sha256={f"{s}.h5": digest(output/f"{s}.h5") for s in COUNTS})
    write_json(output/"metadata.json", plan)
    print(json.dumps({"complete": complete, "accepted_counts": plan["accepted_counts"], "sources": len(attempts)}, indent=2))
    if not complete:
        raise RuntimeError("Candidate pool exhausted; partial data remain excluded from training")


if __name__ == "__main__":
    main()
