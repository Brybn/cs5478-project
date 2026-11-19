"""Collect expert corrections from policy-generated, grasped near-goal states."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/robots2-matplotlib")

import h5py
import numpy as np
import torch

from ..artifacts import ROOT, digest, resolve_artifact
from ..policy import load_policy
from ..monitor import DynamicsPredictor, MotionTrigger, _reset_seeds, reset_seed_sources
from ..env import extract_policy_state, make_env
from ..evaluate import run_episode


CONTROLLER = {
    "gain": 0.5, "max_step_m": 0.008, "deadband_m": 0.004,
    "accept_distance_m": 0.010, "success_hold_steps": 10,
    "min_steps": 20, "max_steps": 60, "gripper_action": -1.0,
}
SOURCE_HORIZON = 100
FORCES = (0.0, 0.2, 0.3, 0.4)
CANDIDATES_PER_TARGET = 10
_WORKER = {}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def distance_mm(states):
    states = np.asarray(states)
    return np.linalg.norm(states[..., 29:32] - states[..., 26:29], axis=-1) * 1000


def choose_takeover(source):
    """Prefer a confirmed twenty-step stall; otherwise correct a near-goal state.

    The fallback is an intervention during ordinary motion, not an observed
    failure. Its provenance remains explicit in every saved trajectory.
    """
    states = np.asarray(source["states"])
    distances = distance_mm(states)
    for step in range(40, len(states)):
        tail = states[step - 20:step + 1]
        d = distances[step - 20:step + 1]
        travel = np.linalg.norm(np.diff(tail[:, 19:22], axis=0), axis=1).sum() * 1000
        if (np.all(tail[:, 18] > .5) and np.all(d > 25) and np.all(d <= 75)
                and np.ptp(d) <= 5 and d[0] - d[-1] <= 2 and travel <= 15):
            return step, "observed_stall"
    candidates = [step for step in range(40, len(states))
                  if np.all(states[step - 4:step + 1, 18] > .5)
                  and 30 <= distances[step] <= 75]
    return (candidates[-1], "near_goal_intervention") if candidates else (None, None)


def corrective_action(env, state):
    """Translate toward the cube's goal, preserving its current grasp offset."""
    return cartesian_action(env, state[26:29] - state[29:32])


def cartesian_action(env, error, gripper=-1.0, max_step_m=.008, deadband_m=.004, gain=.5):
    """Convert a world-frame XYZ correction into normalized root-frame actions."""
    from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

    arm = env.unwrapped.agent.controller.controllers["arm"]
    config = arm.config
    if config.frame != "root_translation" or not config.use_delta or config.use_target:
        raise ValueError("The correction controller requires instantaneous root-frame delta-position control")
    error = np.asarray(error, dtype=np.float32)
    delta = gain * error
    if np.linalg.norm(error) <= deadband_m:
        delta[:] = 0
    else:
        delta *= min(1.0, max_step_m / max(float(np.linalg.norm(delta)), 1e-12))
    rotation = quaternion_to_matrix(arm.root_link.pose.q)[0].detach().cpu().numpy()
    local_delta = rotation.T @ delta
    low = np.broadcast_to(np.asarray(config.pos_lower), (3,))
    high = np.broadcast_to(np.asarray(config.pos_upper), (3,))
    command = 2 * (local_delta - low) / (high - low) - 1 if config.normalize_action else local_delta
    action = np.asarray([*command, gripper], dtype=np.float32)
    if not np.isfinite(action).all() or np.any(np.abs(action) > 1 + 1e-6):
        raise ValueError("Correction action is non-finite or exceeds the normalized action bounds")
    return np.clip(action, -1, 1)


def _video_frame(env, state, phase, step, force):
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import get_data_path

    scene = np.asarray(env.render()).squeeze()
    panel = Image.new("RGB", (scene.shape[1], scene.shape[0] + 96), (20, 25, 35))
    panel.paste(Image.fromarray(scene), (0, 96))
    draw = ImageDraw.Draw(panel)
    font_path = str(Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf")
    title = ImageFont.truetype(font_path, 20)
    font = ImageFont.truetype(font_path, 15)
    draw.text((12, 8), phase, font=title, fill=(255, 180, 65) if phase.startswith("SCRIPTED") else "white")
    draw.text((12, 39), f"Step {step} | prior force {force:g} N | goal error {distance_mm(state):.1f} mm", font=font, fill="white")
    draw.text((12, 65), "Correction demonstration; policy weights unchanged", font=font, fill=(175, 185, 200))
    return np.asarray(panel)


def _init_worker(policy_path, predictor_path=None, render=False, teacher="placement"):
    torch.set_num_threads(1)
    options = {}
    if render:
        from mani_skill.utils import sapien_utils
        options = {"human_render_camera_configs": {
            "width": 512, "height": 512, "fov": 1., "shader_pack": "default",
            "pose": sapien_utils.look_at([.6, .7, .6], [-.1, 0., .2])}, "enable_shadow": True}
    if teacher == "pickup":
        from .pickup import PICKUP_CONTROLLER
        max_steps = PICKUP_CONTROLLER["max_steps"]
    else:
        max_steps = CONTROLLER["max_steps"]
    env = make_env(render=render, max_episode_steps=SOURCE_HORIZON + max_steps + 1,
                   disturbance={"start_step": 0, "duration_steps": 0, "force_xy": [0., 0.]}, **options)
    _WORKER.update(env=env, policy=load_policy(policy_path, device="cpu"),
                   predictor=DynamicsPredictor.load(predictor_path) if predictor_path else None)


def _collect_one(task):
    case, method = task["case"], task["method"]
    env = _WORKER["env"]
    env.unwrapped.disturbance = {key: case[key] for key in ("start_step", "duration_steps", "force_xy")}
    trigger = (MotionTrigger(method["kind"], method["threshold"], _WORKER["predictor"])
               if method["kind"] != "fixed" else None)
    source = run_episode(env, _WORKER["policy"], method["chunk_length"], case["reset_seed"],
                         torch.device("cpu"), horizon=SOURCE_HORIZON, policy_seed=case["policy_seed"],
                         motion_trigger=trigger, collect_positions=True)
    if source["steps"] != SOURCE_HORIZON:
        raise RuntimeError("Source episode ended before its planned horizon")
    expected_force = list(range(case["start_step"], case["start_step"] + case["duration_steps"])) if case["force_n"] else []
    if source["applied_force_steps"] != expected_force:
        raise RuntimeError("The source force pulse did not match its plan")
    expected = task.get("expected_source")
    if expected is not None:
        for key in ("actions", "positions", "plans", "monitor_decisions", "success_steps"):
            if source[key] != expected[key]:
                raise RuntimeError(f"Pilot differs from the saved validation episode: {key}")
        if source["states"][0] != expected["initial_state"]:
            raise RuntimeError("Pilot reset differs from the saved validation episode")
    teacher_name = task.get("teacher", "placement")
    pickup = teacher_name == "pickup"
    controller, teacher = CONTROLLER, None
    if pickup:
        from .pickup import PICKUP_CONTROLLER, PickupTeacher, choose_pickup_takeover
        controller = PICKUP_CONTROLLER
        takeover, kind = choose_pickup_takeover(source, case.get("target_group"))
    else:
        takeover, kind = ((SOURCE_HORIZON, "observed_validation_failure") if expected is not None else choose_takeover(source))
    if "takeover_step" in task:
        takeover, kind = task["takeover_step"], "pilot_fixed_handoff"
        if not pickup or not 1 <= takeover < SOURCE_HORIZON:
            raise ValueError("Fixed pilot handoffs require a pickup teacher and a step before the horizon")
    result = {"case": case, "method": method, "source": source, "takeover_step": takeover,
              "takeover_kind": kind, "accepted": False, "reason": "no_grasped_near_goal_state"}
    result["teacher"] = teacher_name
    if takeover is None:
        if pickup:
            result["reason"] = "no_failed_pickup_state"
        return result
    state = np.asarray(source["states"][takeover], dtype=np.float32)
    if pickup:
        teacher = PickupTeacher(state, reset_cube_z=float(source["states"][0][31]))
    if not pickup and (state[18] < .5 or not 25 < distance_mm(state) <= 75):
        result["reason"] = "takeover_outside_grasped_near_goal_scope"
        return result

    # Replay the source prefix exactly, then branch into the expert correction.
    obs, _ = env.reset(seed=case["reset_seed"])
    state = extract_policy_state(obs)
    if not np.array_equal(state, np.asarray(source["states"][0], dtype=np.float32)):
        raise RuntimeError("Replaying the prefix changed its initial state")
    writer = None
    if task.get("video"):
        import imageio.v2 as imageio
        writer = imageio.get_writer(task["video"], fps=env.unwrapped.control_freq,
                                    codec="libx264", quality=8)
    try:
        for index, action in enumerate(source["actions"][:takeover]):
            obs, _, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
            state = extract_policy_state(obs)
            if not np.array_equal(state, np.asarray(source["states"][index + 1], dtype=np.float32)):
                raise RuntimeError(f"Source-prefix replay differs at step {index + 1}")
            if writer is not None and index + 1 >= takeover - 10:
                writer.append_data(_video_frame(env, state, "POLICY PREFIX", index + 1, case["force_n"]))
        context = np.asarray([source["states"][max(0, takeover - 1)], source["states"][takeover]], dtype=np.float32)
        observations, actions, success_flags, distances = [state.copy()], [], [], [float(distance_mm(state))]
        hold = 0
        horizon_success = None
        result["reason"] = "correction_timeout"
        for index in range(controller["max_steps"]):
            action = teacher.action(env, state) if pickup else corrective_action(env, state)
            if pickup and teacher.failed:
                result["reason"] = getattr(teacher, "failure_reason", "pickup_attempts_exhausted")
                break
            obs, _, terminated, truncated, info = env.step(action)
            state = extract_policy_state(obs)
            actions.append(action)
            observations.append(state.copy())
            distances.append(float(distance_mm(state)))
            success = bool(info["success"].item())
            success_flags.append(success)
            held = bool(info["is_grasped"].item())
            verified = teacher.verified if pickup else True
            if pickup and takeover + len(actions) == SOURCE_HORIZON:
                horizon_success = bool(verified and held and success)
            hold = hold + 1 if verified and held and success and distances[-1] <= controller["accept_distance_m"] * 1000 else 0
            if writer is not None:
                writer.append_data(_video_frame(env, state, "SCRIPTED CORRECTION", takeover + index + 1, case["force_n"]))
            if not held and not pickup:
                result["reason"] = "grasp_lost_during_correction"
                break
            if (hold >= controller["success_hold_steps"] and len(actions) >= controller["min_steps"]
                    and (not pickup or takeover + len(actions) >= SOURCE_HORIZON)):
                result.update(accepted=not pickup or bool(horizon_success),
                              reason="sustained_success" if not pickup or horizon_success else "pickup_exceeds_source_horizon")
                break
            if bool(truncated.item()) or (bool(terminated.item()) and not success):
                result["reason"] = "environment_ended"
                break
        result.update(prefix_replay_exact=True, context_obs=context,
                      correction_obs=np.asarray(observations, dtype=np.float32),
                      correction_actions=np.asarray(actions, dtype=np.float32),
                      correction_success=np.asarray(success_flags, dtype=bool),
                      correction_distances_mm=np.asarray(distances),
                      success_hold_steps=hold, video=task.get("video"))
        if pickup:
            result.update(pickup_attempts=teacher.attempts, pickup_verified=teacher.verified,
                          pickup_events=teacher.events, success_at_source_horizon=horizon_success)
    finally:
        if writer is not None:
            writer.close()
    return result


def _preserved_sources():
    paths = ["data/split/train.h5", "data/split/validation.h5", "checkpoints/pickcube/stage0_nominal/best.pt"]
    return {name: digest(name) for name in paths if (ROOT / name).is_file()}


def _collection_code():
    paths = (Path(__file__), ROOT / "scripts/corrections/collect_recovery_demos.py",
             ROOT / "src/adaptive_action_chunking/corrections/pickup.py",
             ROOT / "src/adaptive_action_chunking/policy.py",
             ROOT / "src/adaptive_action_chunking/env.py",
             ROOT / "src/adaptive_action_chunking/evaluate.py")
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def collection_forces(forces, train_count, validation_count, workers):
    """Validate the collection design before reserving groups or creating files."""
    forces = tuple(float(force) for force in forces)
    if not forces or not all(np.isfinite(force) and force >= 0 for force in forces) or len(set(forces)) != len(forces):
        raise ValueError("Force levels must be unique, finite, nonnegative magnitudes in N")
    if (min(train_count, validation_count, workers) < 1
            or train_count % len(forces) or validation_count % len(forces)):
        raise ValueError("Positive train/validation counts must be multiples of the number of force levels; "
                         "workers must be positive")
    return forces


def _make_plan(output, train_count, validation_count, seed, policy_path, forces=FORCES,
               candidates_per_target=CANDIDATES_PER_TARGET, teacher="placement"):
    from .pickup import PICKUP_GROUPS

    previous = reset_seed_sources()
    excluded, scanned = set(), {}
    for path in previous:
        values = ([json.loads(line) for line in path.read_text().splitlines() if line]
                  if path.suffix == ".jsonl" else [json.loads(path.read_text())])
        excluded.update(value for record in values for value in _reset_seeds(record))
        scanned[str(path)] = digest(path)
    rng = np.random.default_rng(seed)
    seen, cases = set(excluded), []
    for split, target in (("train", train_count), ("validation", validation_count)):
        for force in forces:
            for index in range(candidates_per_target * (target // len(forces))):
                reset = int(rng.integers(10000, 2**31))
                while reset in seen:
                    reset = int(rng.integers(10000, 2**31))
                seen.add(reset)
                angle = float(rng.uniform(0, 2 * np.pi))
                cases.append({"episode_id": f"recovery_{split}_{reset}", "split": split,
                              "reset_seed": reset, "policy_seed": int(rng.integers(0, 2**31)),
                              "force_n": force, "force_xy": [force * np.cos(angle), force * np.sin(angle)],
                              "start_step": int(rng.integers(8, 19)), "duration_steps": 2,
                              "chunk_length": (15, 8)[(index // len(PICKUP_GROUPS) if teacher == "pickup" else index) % 2],
                              "candidate_index": index,
                              **({"target_group": PICKUP_GROUPS[index % len(PICKUP_GROUPS)]} if teacher == "pickup" else {})})
    return {"complete": False, "mode": "fresh_collection", "planned_at_utc": datetime.now(timezone.utc).isoformat(),
            "collection_seed": seed, "policy": str(policy_path), "policy_sha256": digest(policy_path),
            "targets": {"train": train_count, "validation": validation_count}, "cases": cases,
            "forces_n": list(forces),
            "candidates_per_target": candidates_per_target,
            "source_horizon": SOURCE_HORIZON, "controller": CONTROLLER,
            "code_sha256": _collection_code(),
            "selection": "Prefer a grasped 20-step near-goal stall; otherwise use the latest grasped state 30-75 mm from goal after step 40",
            "excluded_reset_seed_count": len(excluded), "overlap_sources_sha256": scanned,
            "preserved_sha256": _preserved_sources(), "attempts": [], "accepted": []}


def _save_attempt(result, output, files, plan):
    case, source = result["case"], result["source"]
    identifier = f"{case['episode_id']}_{result['method']['variant']}"
    raw_path = output / "rollouts" / f"{identifier}.npz"
    arrays = {"policy_obs": np.asarray(source["states"], dtype=np.float32),
              "policy_actions": np.asarray(source["actions"], dtype=np.float32)}
    for key in ("context_obs", "correction_obs", "correction_actions", "correction_success", "correction_distances_mm"):
        if key in result:
            arrays[key] = result[key]
    np.savez_compressed(raw_path, **arrays)
    row = {**case, "source_variant": result["method"]["variant"],
           "accepted": result["accepted"], "reason": result["reason"],
           "source_success_at_end": source["success_at_end"], "source_success_once": source["success_once"],
           "takeover_step": result["takeover_step"], "takeover_kind": result["takeover_kind"],
           "prefix_replay_exact": result.get("prefix_replay_exact", False),
           "raw_trace": str(raw_path), "raw_trace_sha256": digest(raw_path), "video": result.get("video")}
    if "correction_actions" in result:
        row.update(correction_steps=len(result["correction_actions"]),
                   start_distance_mm=float(result["correction_distances_mm"][0]),
                   final_distance_mm=float(result["correction_distances_mm"][-1]),
                   success_hold_steps=result["success_hold_steps"],
                   correction_end_step=result["takeover_step"] + len(result["correction_actions"]),
                   completed_within_source_horizon=result["takeover_step"] + len(result["correction_actions"]) <= SOURCE_HORIZON)
    row["teacher"] = result.get("teacher", "placement")
    pickup = row["teacher"] == "pickup"
    if pickup:
        from .pickup import PICKUP_CONTROLLER
        row.update({key: result[key] for key in ("pickup_attempts", "pickup_verified", "pickup_events", "success_at_source_horizon") if key in result})
    if result["accepted"]:
        obs, actions = result["correction_obs"], result["correction_actions"]
        if (obs.shape != (len(actions) + 1, 42) or actions.shape[1:] != (4,)
                or not np.isfinite(obs).all() or not np.isfinite(actions).all()
                or (not pickup and not np.all(obs[:, 18] > .5)) or np.any(np.abs(actions) > 1 + 1e-6)
                or (not pickup and not np.all(actions[:, 3] == CONTROLLER["gripper_action"]))
                or len(actions) < CONTROLLER["min_steps"]
                or not np.array_equal(result["context_obs"][-1], obs[0])
                or not np.all(result["correction_success"][-CONTROLLER["success_hold_steps"]:])
                or not np.all(obs[-CONTROLLER["success_hold_steps"]:, 18] > .5)
                or np.any(distance_mm(obs[-CONTROLLER["success_hold_steps"]:]) > CONTROLLER["accept_distance_m"] * 1000)):
            raise RuntimeError("Accepted correction failed export checks")
        if pickup:
            lifts = [event for event in result.get("pickup_events", []) if event.get("verified_lift")]
            if not result.get("pickup_verified") or not lifts:
                raise RuntimeError("Pickup correction has no verified lift")
            lift = lifts[-1]
            step, count = lift["step"], PICKUP_CONTROLLER["verify_steps"]
            if (not count - 1 <= step < len(obs) or not np.all(obs[step - count + 1:, 18] > .5)
                    or not np.all(obs[step - count + 1:step + 1, 31] >= lift["start_cube_z"] + PICKUP_CONTROLLER["verify_rise_m"] - 1e-6)):
                raise RuntimeError("Pickup correction failed sustained lift verification")
            horizon = SOURCE_HORIZON - result["takeover_step"]
            if (not 1 <= horizon < len(obs) or not result["correction_success"][horizon - 1]
                    or obs[horizon, 18] <= .5 or not any(event["step"] <= horizon for event in lifts)):
                raise RuntimeError("Pickup correction failed the original rollout horizon")
        handle = files[case["split"]]
        name = f"traj_{len(handle)}"
        group = handle.create_group(name)
        group.create_dataset("obs", data=obs, compression="gzip")
        group.create_dataset("actions", data=actions, compression="gzip")
        group.create_dataset("context_obs", data=result["context_obs"])
        group.create_dataset("action_is_expert", data=np.ones(len(actions), dtype=bool))
        group.attrs["reset_seed"] = case["reset_seed"]
        group.attrs["takeover_step"] = result["takeover_step"]
        group.attrs["takeover_kind"] = result["takeover_kind"]
        group.attrs["teacher"] = row["teacher"]
        row["trajectory"] = name
        row["episode_seed"] = case["reset_seed"]
        plan["accepted"].append(row)
        handle.flush()
    plan["attempts"].append(row)
    print(f"{case['split']} force={case['force_n']:g} {case['reset_seed']}: {result['reason']}"
          + (f" ({row['start_distance_mm']:.1f} -> {row['final_distance_mm']:.1f} mm, {result['takeover_kind']})"
             if "final_distance_mm" in row else ""), flush=True)
    write_json(output / "metadata.json", plan)


def _pickup_pilot_episodes(source, name):
    """Choose distinct development cases for the expanded handoff behaviours."""
    episodes = [e for e in source["episodes"] if e["checkpoint"] == name]
    failed = [e for e in episodes if not e["success_at_end"]]
    def reopens(e):
        a, g = np.asarray(e["actions"]), np.asarray(e["grasped"])
        return bool(np.any((a[:-1, 3] < -.5) & (a[1:, 3] > .5) & ~g[1:-1]))
    def no_lift(e):
        p = np.asarray(e["positions"])
        return all(e["grasped"][-21:]) and np.max(p[-21:, 2]) < p[0, 2] + .025
    def weak_contact(e):
        p, g = np.asarray(e["positions"]), np.asarray(e["grasped"])
        return any(g[t-4:t+1].all() and p[t, 2] < p[0, 2] + .025
                   and np.linalg.norm(p[t, 3:5] - p[t, :2]) > .025 for t in range(40, 61))
    roles = [
        ("retry", [e for e in failed if e["force_n"] == 0 and reopens(e)]),
        ("lift", [e for e in failed if no_lift(e)]),
        ("lift", [e for e in failed if weak_contact(e)]),
        ("retry", [e for e in failed if not e["grasped"][-1] and any(e["grasped"][:50])]),
        ("place", [e for e in failed if e["grasped"][-1] and e["goal_distances_m"][-1] > .025 and not no_lift(e)]),
        ("place", [e for e in failed if e["grasped"][-1] and e["goal_distances_m"][-1] <= .025]),
        ("successful", [e for e in episodes if e["success_at_end"] and e["force_n"] > 0]),
    ]
    selected, used = [], set()
    for group, eligible in roles:
        episode = next((e for e in sorted(eligible, key=lambda e: e["chunk_length"] != 15)
                        if e["group_id"] not in used), None)
        if episode is None:
            raise ValueError("Pilot needs distinct reopening, held-lift, weak-contact, lost-grasp, placement, settling and successful cases")
        selected.append((group, episode))
        used.add(episode["group_id"])
    return selected


def collect(output, policy_path, train_count=32, validation_count=8, seed=90271, workers=8,
            pilot_results=None, pilot_case=None, record_video=False, forces=FORCES, teacher="placement"):
    output, policy_path = Path(output).resolve(), resolve_artifact(policy_path)
    if output.exists():
        raise ValueError(f"Choose a fresh output directory: {output}")
    forces = collection_forces(forces, train_count, validation_count, workers)
    if teacher not in ("placement", "pickup"):
        raise ValueError("Teacher must be placement or pickup")
    from .pickup import PICKUP_GROUPS
    if teacher == "pickup" and not pilot_results and any(count % (len(forces) * len(PICKUP_GROUPS)) for count in (train_count, validation_count)):
        raise ValueError("Pickup counts must balance all four target groups at each force")
    predictor_path = None
    if pilot_results and teacher == "pickup":
        source = json.loads(Path(pilot_results).read_text())
        if not source["complete"] or source["plan"].get("comparison") != "explicit":
            raise ValueError("Pickup pilot requires a completed explicit policy comparison")
        names = [name for name, entry in source["plan"]["checkpoints"].items() if entry["sha256"] == digest(policy_path)]
        if len(names) != 1:
            raise ValueError("Pickup pilot policy must match exactly one saved comparison policy")
        tasks = []
        for group, episode in _pickup_pilot_episodes(source, names[0]):
            case = next(case for case in source["plan"]["cases"] if case["episode_id"] == episode["episode_id"])
            tasks.append({"case": {**case, "split": "pilot", "chunk_length": episode["chunk_length"], "target_group": group},
                          "method": {"variant": episode["variant"], "kind": "fixed", "threshold": None,
                                     "chunk_length": episode["chunk_length"]}, "expected_source": episode})
        plan = {"complete": False, "mode": "evaluation_pilot_only", "policy": str(policy_path),
                "policy_sha256": digest(policy_path), "code_sha256": _collection_code(),
                "source": str(pilot_results), "source_sha256": digest(pilot_results),
                "cases": [task["case"] for task in tasks], "targets": {"pilot": len(tasks)},
                "preserved_sha256": _preserved_sources(), "attempts": [], "accepted": []}
    elif pilot_results:
        source = json.loads(Path(pilot_results).read_text())
        if not source["complete"] or source["partition"] != "scheduler_validation":
            raise ValueError("Pilot replay must use completed scheduler-validation data")
        if digest(policy_path) != source["policy_sha256"]:
            raise ValueError("Pilot policy differs from the saved validation policy")
        predictor_path = resolve_artifact(source["predictor"], source["predictor_sha256"])
        case = next(c for c in source["cases"] if c["episode_id"] == pilot_case)
        variants = ("fixed_k15", "fixed_k8", "prediction_error_4")
        methods = [next(m for m in source["methods"] if m["variant"] == name) for name in variants]
        case = {**case, "split": "pilot"}
        tasks = [{"case": case, "method": method,
                  "expected_source": next(e for e in source["episodes"] if e["episode_id"] == pilot_case and e["variant"] == method["variant"])}
                 for method in methods]
        plan = {"complete": False, "mode": "validation_pilot_only", "policy": str(policy_path),
                "policy_sha256": digest(policy_path), "controller": CONTROLLER,
                "code_sha256": _collection_code(),
                "source": str(pilot_results), "source_sha256": digest(pilot_results),
                "cases": [case], "targets": {"pilot": len(tasks)},
                "preserved_sha256": _preserved_sources(), "attempts": [], "accepted": []}
    else:
        plan = _make_plan(output, train_count, validation_count, seed, policy_path, forces,
                          candidates_per_target=50 if teacher == "pickup" else CANDIDATES_PER_TARGET, teacher=teacher)
        tasks = [{"case": case, "method": {"variant": f"fixed_k{case['chunk_length']}",
                  "kind": "fixed", "chunk_length": case["chunk_length"], "threshold": None}}
                 for case in sorted(plan["cases"], key=lambda c: (c["candidate_index"], c["split"], c["force_n"]))]
    plan["teacher"] = teacher
    if teacher == "pickup":
        from .pickup import PICKUP_CONTROLLER
        plan["controller"] = PICKUP_CONTROLLER
        plan["target_groups"] = list(PICKUP_GROUPS)
        plan["selection"] = "Balance retry, held-lift and placement interventions on failed sources with full expert sequences on successful disturbed sources"
    for task in tasks:
        task["teacher"] = teacher
    output.mkdir(parents=True)
    (output / "rollouts").mkdir()
    if record_video:
        (output / "videos").mkdir()
        for task in tasks:
            task["video"] = str(output / "videos" / f"{task['case']['episode_id']}_{task['method']['variant']}.mp4")
    write_json(output / "metadata.json", plan)
    files = {split: h5py.File(output / f"{split}.h5", "x") for split in plan["targets"]}
    try:
        counts = Counter()
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker,
                                 initargs=(str(policy_path), str(predictor_path) if predictor_path else None, record_video, teacher)) as pool:
            pending = tasks
            while pending:
                batch, remaining, reserved = [], [], Counter()
                for task in pending:
                    case = task["case"]
                    quota = (plan["targets"][case["split"]] if pilot_results else
                             plan["targets"][case["split"]] // (len(forces) * (len(PICKUP_GROUPS) if teacher == "pickup" else 1)))
                    bucket = (case["split"], case["force_n"], case.get("target_group"))
                    if counts[bucket] >= quota:
                        continue
                    if len(batch) < workers and counts[bucket] + reserved[bucket] < quota:
                        batch.append(task)
                        reserved[bucket] += 1
                    else:
                        remaining.append(task)
                pending = remaining
                if not batch:
                    break
                for result in pool.map(_collect_one, batch):
                    case = result["case"]
                    _save_attempt(result, output, files, plan)
                    counts[case["split"], case["force_n"], case.get("target_group")] += int(result["accepted"])
        accepted_counts = Counter(row["split"] for row in plan["accepted"])
        plan["accepted_counts"] = dict(accepted_counts)
        plan["takeover_counts"] = dict(Counter(row["takeover_kind"] for row in plan["accepted"]))
        plan["complete"] = all(accepted_counts[split] == target for split, target in plan["targets"].items())
        plan["all_attempts_complete"] = True
        for split, handle in files.items():
            handle.flush()
            write_json(output / f"{split}.json", {
                "format": "expert correction segments; obs[T+1,42], actions[T,4]",
                "context_obs": "Actual observations at takeover-1 and takeover; preserve for two-frame conditioning",
                "action_provenance": "All HDF5 actions are scripted expert corrections; policy prefixes are separate NPZ traces",
                "episodes": [row for row in plan["accepted"] if row["split"] == split]})
    finally:
        for handle in files.values():
            handle.close()
        plan["artifact_sha256"] = {str(path): digest(path) for path in output.glob("*.h5")}
        write_json(output / "metadata.json", plan)
    changed = [name for name, value in plan["preserved_sha256"].items() if digest(name) != value]
    if changed:
        raise RuntimeError(f"Preserved scientific artifacts changed: {changed}")
    print(json.dumps({key: plan[key] for key in ("complete", "accepted_counts", "takeover_counts")}, indent=2), flush=True)
    if not plan["complete"]:
        raise RuntimeError("Candidate pool exhausted before all quotas were met; partial data and attempts are preserved")
    return plan
