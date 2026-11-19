"""Roll out a policy with fixed or event-triggered chunking, summarise matched cohorts, select triggers."""

from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from mani_skill.utils.common import to_numpy

from .env import extract_monitor_positions, extract_policy_state, make_env
from .artifacts import resolve_artifact
from .policy import BaselinePolicy, load_policy


def _scalar(value) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sample(policy, history, policy_seed, step, device):
    """Common sampling noise for policy calls at the same absolute control step."""
    call_seed = int(np.random.SeedSequence([policy_seed, step]).generate_state(1)[0])
    generator = torch.Generator(device=device).manual_seed(call_seed)
    return policy.sample(np.stack(history), generator, device)


def run_episode(
    env,
    policy: BaselinePolicy,
    chunk_length: int,
    seed: int,
    device: torch.device,
    horizon: int = 100,
    video_path: Path | None = None,
    policy_seed: int | None = None,
    motion_trigger=None,
    collect_positions: bool = False,
    record_timing: bool = False,
) -> dict:
    """Execute fixed K-step chunks, or K15 chunks cut short by `motion_trigger`; check success at the horizon.

    With `policy_seed`, the diffusion noise of a call depends only on (policy_seed, control step), so schedules that
    call the policy at the same step draw the same noise.
    """
    if motion_trigger is not None and chunk_length != policy.config.max_execution_horizon:
        raise ValueError("Triggered replanning uses the full 15-action plan")
    obs, _ = env.reset(seed=seed)
    state = extract_policy_state(obs)
    history = deque(
        (state.copy() for _ in range(policy.config.obs_horizon)),
        maxlen=policy.config.obs_horizon,
    )
    positions = extract_monitor_positions(obs)
    motion_history = deque([positions.copy()], maxlen=3)
    prediction = None
    generator = torch.Generator(device=device).manual_seed(seed)
    chunk = np.empty((0, policy.act_dim), dtype=np.float32)
    chunk_age = 0
    replan_steps = []
    policy_seconds = 0.0
    success_once = False
    success_at_end = False
    first_success_step = None
    termination = "horizon"
    writer = None
    trace = ({"policy_seed": policy_seed, "states": [state.tolist()], "actions": [],
              "plans": [], "monitor_decisions": [], "monitor_seconds": 0.0,
              "success_steps": [], "monitor_replans": 0}
             if policy_seed is not None or motion_trigger is not None or collect_positions or record_timing else None)
    if record_timing:
        trace["policy_call_seconds"] = []
    if trace is not None and (collect_positions or motion_trigger is not None):
        trace["positions"] = [positions.tolist()]
    if trace is not None:
        # Fixed 42D policy layout: grasp flag 18, goal XYZ 26:29, cube XYZ 29:32.
        trace["grasped"] = [bool(state[18] > .5)]
        trace["goal_distances_m"] = [float(np.linalg.norm(state[29:32] - state[26:29]))]

    try:
        if video_path is not None:
            import imageio.v2 as imageio

            encoding = ({"codec": "ffv1", "pixelformat": "bgr0"} if video_path.suffix == ".mkv"
                        else {"quality": 9})
            writer = imageio.get_writer(str(video_path), fps=env.unwrapped.control_freq, **encoding)
            writer.append_data(to_numpy(env.render()).squeeze())

        for step in range(horizon):
            reason = "exhausted" if chunk_age >= len(chunk) else "fixed_chunk" if chunk_age >= chunk_length else None
            if reason is None and motion_trigger is not None:
                started = time.perf_counter()
                score = motion_trigger.score(motion_history, prediction)
                trigger = (chunk_age >= 2 and score is not None and
                           motion_trigger.threshold is not None and score > motion_trigger.threshold)
                trace["monitor_seconds"] += time.perf_counter() - started
                trace["monitor_decisions"].append({
                    "step": step, "chunk_age": chunk_age, "score": score,
                    "eligible": chunk_age >= 2, "replan": trigger,
                    "prediction": prediction.tolist() if prediction is not None else None,
                })
                if trigger:
                    reason = "monitor"
                    trace["monitor_replans"] += 1
            if reason is not None:
                _synchronize(device)
                started = time.perf_counter()
                chunk = (_sample(policy, history, policy_seed, step, device) if policy_seed is not None else
                         policy.sample(np.stack(history), generator, device))
                _synchronize(device)
                elapsed = time.perf_counter() - started
                policy_seconds += elapsed
                if record_timing:
                    trace["policy_call_seconds"].append(elapsed)
                replan_steps.append(step)
                chunk_age = 0
                if trace is not None:
                    trace["plans"].append({"step": step, "reason": reason, "actions": chunk.tolist()})

            action = chunk[chunk_age].astype(np.float32, copy=False)
            if motion_trigger is not None:
                started = time.perf_counter()
                prediction = motion_trigger.predict_next(motion_history, action)
                trace["monitor_seconds"] += time.perf_counter() - started
            obs, _, terminated, truncated, info = env.step(action)
            chunk_age += 1
            state = extract_policy_state(obs)
            history.append(state)
            positions = extract_monitor_positions(obs)
            motion_history.append(positions)
            success_at_end = _scalar(info.get("success", False))
            success_once |= success_at_end
            if success_at_end and first_success_step is None:
                first_success_step = step + 1
            if trace is not None:
                trace["states"].append(state.tolist())
                trace["actions"].append(action.tolist())
                trace["grasped"].append(bool(state[18] > .5))
                trace["goal_distances_m"].append(float(np.linalg.norm(state[29:32] - state[26:29])))
                if "positions" in trace:
                    trace["positions"].append(positions.tolist())
                if success_at_end:
                    trace["success_steps"].append(step + 1)
            if writer is not None:
                writer.append_data(to_numpy(env.render()).squeeze())
            if _scalar(info.get("fail", False)) or (_scalar(terminated) and not success_at_end):
                termination = "failure"
                break
            if _scalar(truncated):
                termination = "horizon" if step + 1 == horizon else "early_truncation"
                break
    finally:
        if writer is not None:
            writer.close()

    if trace is not None:
        trace["applied_force_steps"] = sorted(getattr(env.unwrapped, "applied_force_steps", []))
        trace["applied_force_substeps"] = getattr(env.unwrapped, "applied_force_substeps", 0)
    return {
        "seed": seed,
        "steps": step + 1,
        "termination": termination,
        "success_once": success_once,
        "success_at_end": success_at_end,
        "first_success_step": first_success_step,
        "policy_calls": len(replan_steps),
        "policy_seconds": policy_seconds,
        "replan_steps": replan_steps,
        **(trace or {}),
        **({"video": str(video_path)} if video_path is not None else {}),
    }


def evaluate_policy(
    policy_path: Path,
    output: Path,
    episodes: int = 10,
    chunk_length: int = 8,
    seed: int = 1000,
    horizon: int = 100,
    device: str = "cpu",
    record_video: bool = False,
) -> dict:
    """Save run settings, episode results and aggregate metrics in one JSON file."""
    if episodes < 1 or horizon < 1 or seed < 0:
        raise ValueError("Episodes and horizon must be positive; seed must be nonnegative")
    inference_device = torch.device(device)
    policy = load_policy(policy_path, device=inference_device)
    if not 1 <= chunk_length <= policy.config.max_execution_horizon:
        raise ValueError(f"Chunk length must be between 1 and {policy.config.max_execution_horizon}")

    result = {
        "environment": "PickCube-v1",
        "condition": "nominal",
        "policy": str(policy_path),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "device": str(inference_device),
        "sampler": policy.config.sampler,
        "inference_steps": len(policy.noise_scheduler.timesteps),
        "observation_horizon": policy.config.obs_horizon,
        "prediction_horizon": policy.config.pred_horizon,
        "chunk_length": chunk_length,
        "episode_horizon": horizon,
        "seed": seed,
        "requested_episodes": episodes,
        "complete": False,
        "episodes": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")

    env = None
    try:
        video_dir = output.with_suffix("") if record_video else None
        if video_dir is not None:
            video_dir.mkdir(exist_ok=False)
        env = make_env(render=record_video, max_episode_steps=horizon)
        # Warm up once, outside the measured rollouts, with an independent RNG.
        obs, _ = env.reset(seed=seed)
        state = extract_policy_state(obs)
        policy.sample(
            np.stack([state] * policy.config.obs_horizon),
            torch.Generator(device=inference_device).manual_seed(0),
            inference_device,
        )
        for episode in range(episodes):
            video_path = video_dir / f"episode_{episode:03d}.mp4" if video_dir is not None else None
            record = run_episode(
                env, policy, chunk_length, seed + episode, inference_device,
                horizon=horizon, video_path=video_path,
            )
            result["episodes"].append(record)
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                f"Episode {episode + 1}/{episodes}: "
                f"success_once={record['success_once']} "
                f"success_at_end={record['success_at_end']} "
                f"calls={record['policy_calls']}",
                flush=True,
            )
        result["summary"] = {
            key: float(np.mean([record[key] for record in result["episodes"]]))
            for key in ("success_once", "success_at_end", "policy_calls", "policy_seconds")
        }
        result["complete"] = True
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if env is not None:
            env.close()
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output}", flush=True)
    return result


def _dynamics_cases(groups, forces):
    return [{**group, "episode_id": f"{group['group_id']}_force_{force:g}", "force_n": force,
             "duration_steps": group.get("duration_steps", 2),
             "force_xy": [force * np.cos(group["angle"]), force * np.sin(group["angle"])]}
            for group in groups for force in forces]


def _motion_outcomes(task, episode):
    """Keep failed and transiently successful trials in capped timing metrics."""
    horizon = episode["steps"]
    hz = episode.get("control_freq_hz", 20)
    success = set(episode["success_steps"])
    origin = task["start_step"] + task["duration_steps"] if task["force_n"] else 0

    def confirmation(start):
        return next((step for step in range(start + 10, horizon + 1)
                     if all(t in success for t in range(step - 9, step + 1))), None)

    confirmed, recovered = confirmation(0), confirmation(origin)
    return {"success_confirmation_step": confirmed, "recovery_confirmation_step": recovered,
            "recovery_origin_step": origin,
            "capped_confirmation_seconds": (confirmed if confirmed is not None else horizon) / hz,
            "capped_recovery_seconds": ((recovered if recovered is not None else horizon) - origin) / hz,
            "recovery_confirmed": recovered is not None,
            "confirmed_by_60": confirmed is not None and confirmed <= 60,
            "confirmed_by_100": confirmed is not None and confirmed <= 100,
            "last_ten_success": horizon >= 10 and all(t in success for t in range(horizon - 9, horizon + 1)),
            "confirmed_then_failed": confirmed is not None and not episode["success_at_end"]}


def _dynamics_summary(episodes, bootstrap, baseline_variant="fixed_k15"):
    """Bootstrap reset groups, retaining all force conditions in each draw."""
    metrics = ("success_at_end", "success_once", "policy_calls", "inference_seconds")
    metrics += tuple(key for key in ("capped_confirmation_seconds", "capped_recovery_seconds",
                                     "recovery_confirmed", "confirmed_by_60", "confirmed_by_100",
                                     "last_ten_success", "confirmed_then_failed")
                     if all(key in episode for episode in episodes))
    groups = sorted({episode["group_id"] for episode in episodes})
    indices = np.random.default_rng(bootstrap["seed"]).integers(0, len(groups), (bootstrap["samples"], len(groups)))
    result = {}
    for name in dict.fromkeys(episode["variant"] for episode in episodes):
        records = [episode for episode in episodes if episode["variant"] == name]
        summary = {"episodes": len(records), "success_at_end_count": sum(e["success_at_end"] for e in records),
                   "success_once_count": sum(e["success_once"] for e in records), "ci95": {}}
        for metric in metrics:
            values = np.array([np.mean([e[metric] for e in records if e["group_id"] == group]) for group in groups])
            summary[metric] = float(values.mean())
            summary["ci95"][metric] = np.quantile(values[indices].mean(1), [0.025, 0.975]).tolist()
        summary["policy_seconds"] = float(np.mean([e["policy_seconds"] for e in records]))
        summary["monitor_seconds"] = float(np.mean([e["monitor_seconds"] for e in records]))
        result[name] = summary
    baseline = {e["episode_id"]: e for e in episodes if e["variant"] == baseline_variant}
    paired = {}
    for name in result:
        if name == baseline_variant:
            continue
        records = [e for e in episodes if e["variant"] == name]
        differences = np.array([int(e["success_at_end"]) - int(baseline[e["episode_id"]]["success_at_end"]) for e in records])
        comparison = {"wins": int(np.sum(differences > 0)), "losses": int(np.sum(differences < 0)),
                      "ties": int(np.sum(differences == 0)), "ci95": {}}
        for metric in metrics:
            values = np.array([np.mean([float(e[metric]) - float(baseline[e["episode_id"]][metric])
                                       for e in records if e["group_id"] == group]) for group in groups])
            comparison[metric + "_difference"] = float(values.mean())
            comparison["ci95"][metric + "_difference"] = np.quantile(values[indices].mean(1), [0.025, 0.975]).tolist()
        paired[name] = comparison
    return result, paired


def _audit_motion_episode(task, episode, predictor):
    """Reconstruct action alignment, scores, cooldown and mandatory replans."""
    from .monitor import MotionTrigger

    positions = np.asarray(episode["positions"], dtype=np.float32)
    actions = np.asarray(episode["actions"], dtype=np.float32)
    plans = {plan["step"]: plan for plan in episode["plans"]}
    decisions = {decision["step"]: decision for decision in episode["monitor_decisions"]}
    trigger = MotionTrigger(task["kind"], task["threshold"], predictor) if task["kind"] != "fixed" else None
    chunk, age, checked = None, 0, 0
    for step in range(episode["steps"]):
        mandatory = chunk is None or age >= len(chunk) or age >= task["chunk_length"]
        replan = False
        if trigger is not None and not mandatory:
            history = positions[max(0, step - 2):step + 1]
            prediction = (predictor.predict(positions[step - 2], positions[step - 1], actions[step - 1])
                          if task["kind"] == "prediction_error" and step >= 2 else None)
            score = trigger.score(history, prediction)
            replan = age >= 2 and score is not None and task["threshold"] is not None and score > task["threshold"]
            decision = decisions[step]
            if (decision["chunk_age"] != age or decision["eligible"] != (age >= 2) or decision["replan"] != replan or
                    ((score is None) != (decision["score"] is None)) or
                    (score is not None and not np.isclose(score, decision["score"], rtol=1e-6, atol=1e-7))):
                raise RuntimeError("Saved motion decision differs from the reconstructed trigger")
            if prediction is not None and not np.array_equal(prediction, np.asarray(decision["prediction"], dtype=np.float32)):
                raise RuntimeError("Prediction was not aligned with the preceding executed action")
            checked += 1
        if mandatory or replan:
            plan = plans[step]
            if (plan["reason"] == "monitor") != replan:
                raise RuntimeError("A replan violated the trigger or mandatory schedule")
            chunk, age = np.asarray(plan["actions"], dtype=np.float32), 0
        elif step in plans:
            raise RuntimeError("Unexpected policy call")
        if not np.array_equal(actions[step], chunk[age]):
            raise RuntimeError("Executed action differs from its saved plan")
        age += 1
    return checked


def _select_motion_thresholds(result, grid, selection_rule):
    from .monitor import FIXED_CHUNKS

    chosen, bands = [], []
    for kind, thresholds in grid.items():
        candidates = []
        for threshold in thresholds:
            variant = "fixed_k15" if threshold is None else f"{kind}_{threshold:g}"
            candidates.append({"kind": kind, "threshold": threshold, "source_variant": variant,
                               "reproduces_k15": threshold is None, **result["summary"][variant]})
        for target in (len(range(0, result.get("horizon", 100), chunk)) for chunk in FIXED_CHUNKS):
            matching = [candidate for candidate in candidates if 0.9 * target <= candidate["policy_calls"] <= 1.1 * target]
            bands.append({"kind": kind, "target_calls": target, "matching_candidates": len(matching)})
            if matching:
                best = min(matching, key=lambda c: (c.get("capped_confirmation_seconds", 0), -c["success_at_end_count"], c["policy_calls"],
                                                   c["threshold"] is not None, -(c["threshold"] or 0)))
                chosen.append({"variant": f"{kind}_calls{target}", "target_calls": target,
                               **{key: best[key] for key in ("kind", "threshold", "source_variant", "reproduces_k15")},
                               "validation_success": best["success_at_end_count"],
                               "validation_mean_calls": best["policy_calls"]})
    out = {"selected": chosen, "bands": bands, "rule": selection_rule}
    learned = [m for m in result["methods"] if m["kind"] == "prediction_error" and m["threshold"] is not None]
    if learned:
        best = min(learned, key=lambda m: (result["summary"][m["variant"]].get("capped_confirmation_seconds", 0),
                                           -result["summary"][m["variant"]]["success_at_end_count"],
                                           result["summary"][m["variant"]]["policy_calls"]))
        if not any(c["kind"] == "prediction_error" and c["threshold"] == best["threshold"] for c in chosen):
            metrics = result["summary"][best["variant"]]
            out["exploratory_selected"] = [{
                "variant": "prediction_error_exploratory", "kind": "prediction_error",
                "threshold": best["threshold"], "source_variant": best["variant"],
                "reproduces_k15": False, "exploratory": True,
                "validation_success": metrics["success_at_end_count"],
                "validation_mean_calls": metrics["policy_calls"],
            }]
            out["exploratory_rule"] = (
                "Lowest active learned validation time to goal; tie-break by success, then fewer mean policy calls. "
                "Primary selection unchanged."
            )
            out["exploratory_reason"] = (
                "Report an active learned point separately from the primary call-band selection."
            )
    return out


def _plot_dynamics_comparison(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    summaries = result["summary"]
    methods = {method["variant"]: method for method in result["methods"]}
    for name, summary in summaries.items():
        if methods[name].get("reproduces_k15"):
            continue
        if name == "fixed_k15":
            label = "Fixed K15 / disabled triggers"
        elif name.startswith("fixed_k"):
            label = f"Fixed K{name[7:]}"
        else:
            method = methods[name]
            title = "Prediction error" if method["kind"] == "prediction_error" else "Acceleration"
            unit = " m" if method["kind"] == "acceleration" else ""
            label = f"{title} (threshold {method['threshold']:g}{unit})"
            if method.get("exploratory"):
                label += " [exploratory]"
        for axis, metric, scale in ((axes[0], "capped_confirmation_seconds", 1), (axes[1], "success_at_end", 100)):
            lower, upper = summary["ci95"][metric]
            axis.errorbar(summary["policy_calls"], scale * summary[metric],
                          yerr=[[scale * (summary[metric] - lower)], [scale * (upper - summary[metric])]],
                          fmt="o", capsize=4, label=label)
    axes[0].set(xlabel="Mean diffusion-policy calls per episode", ylabel="Time to goal (s, capped; lower is better)",
                title="Time to goal vs sampling cost; group-bootstrap 95% intervals")
    axes[1].set(xlabel="Mean diffusion-policy calls per episode", ylabel="Final success (%)",
                title="Final success vs sampling cost", ylim=(0, 105))
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="lower left")
    groups = len({case["group_id"] for case in result["cases"]})
    fig.suptitle(f"Fresh test: {groups} reset groups × {len(result['by_force'])} forces; "
                 f"{result['horizon']} steps, frozen policy and thresholds")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def compare_dynamics(policy_path, predictor_path, dataset_dir, output_dir, stage="validate", workers=4):
    """Select thresholds on reserved validation groups, then run the frozen test."""
    from datetime import datetime, timezone
    from .monitor import FIXED_CHUNKS, DynamicsPredictor, digest, rollout_jobs, source_hashes, write_json

    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    meta = json.loads((dataset_dir / "metadata.json").read_text())
    horizon = meta["horizon"]
    checkpoint = torch.load(predictor_path, map_location="cpu", weights_only=True)
    if not meta["complete"] or digest(policy_path) != checkpoint["policy_sha256"] or meta["transitions_sha256"] != checkpoint["transitions_sha256"]:
        raise ValueError("Use the completed fresh dataset and its frozen policy/predictor")
    if digest(dataset_dir / "transitions.npz") != meta["transitions_sha256"]:
        raise ValueError("Transition dataset has changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / ("validation.json" if stage == "validate" else "test.json")
    selection_path = output_dir / "selection.json"
    if output.exists() or (stage == "validate" and selection_path.exists()):
        raise ValueError(f"Choose a fresh results directory: {output_dir}")
    methods = [{"variant": f"fixed_k{chunk}", "kind": "fixed", "chunk_length": chunk, "threshold": None}
               for chunk in FIXED_CHUNKS]
    if stage == "validate":
        methods.extend({"variant": f"{kind}_{threshold:g}", "kind": kind, "threshold": threshold, "chunk_length": 15}
                       for kind, thresholds in meta["threshold_grid"].items() for threshold in thresholds if threshold is not None)
        partition = "scheduler_validation"
    else:
        selection = json.loads(selection_path.read_text())
        if (selection["predictor_sha256"] != digest(predictor_path) or
                selection["validation_sha256"] != digest(output_dir / "validation.json") or
                selection["metadata_sha256"] != digest(dataset_dir / "metadata.json")):
            raise ValueError("Frozen selection no longer matches its predictor, validation or partitions")
        methods.extend({**point, "chunk_length": 15}
                       for point in [*selection["selected"], *selection.get("exploratory_selected", [])])
        partition = "test"
    groups = [group for group in meta["groups"] if group["partition"] == partition]
    cases = _dynamics_cases(groups, meta["force_levels_n"])
    tasks = [{**case, **method, "record_timing": horizon != 100,
              "repeat": case["group_id"] == groups[0]["group_id"] and case["force_n"] in (0, 0.4)}
             for case in cases for method in methods]
    result = {"complete": False, "stage": stage, "partition": partition,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "policy": str(policy_path), "policy_sha256": digest(policy_path),
              "predictor": str(predictor_path), "predictor_sha256": digest(predictor_path),
              "metadata_sha256": digest(dataset_dir / "metadata.json"), "code_sha256": source_hashes(),
              "device": "cpu", "horizon": horizon, "methods": methods, "cases": cases,
              "sampling": meta["sampling"], "bootstrap": meta["bootstrap"],
              "episodes": [], "checks": {"motion_decisions": 0, "exact_repeats": 0}}
    if stage == "test":
        result.update(selection=str(selection_path), selection_sha256=digest(selection_path))
    write_json(output, result)
    predictor = DynamicsPredictor.load(predictor_path)
    torch.set_num_threads(1)
    initial_states = {}
    started = time.perf_counter()
    try:
        for task, episode in rollout_jobs(tasks, policy_path, workers, predictor_path, horizon=horizon):
            initial = episode["states"][0]
            if horizon == 100:
                episode.pop("states")
            if initial_states.setdefault(task["episode_id"], initial) != initial:
                raise RuntimeError("Compared methods started from different states")
            result["checks"]["motion_decisions"] += _audit_motion_episode(task, episode, predictor)
            result["checks"]["exact_repeats"] += int(episode.get("repeat_exact", False))
            episode.update({key: task[key] for key in ("group_id", "episode_id", "force_n", "variant")})
            episode.update(initial_state=initial, inference_seconds=episode["policy_seconds"] + episode["monitor_seconds"])
            episode.update(_motion_outcomes(task, episode))
            result["episodes"].append(episode)
            if len(result["episodes"]) % 20 == 0:
                print(f"{stage}: {len(result['episodes'])}/{len(tasks)} rollouts complete", flush=True)
        result["summary"], result["paired_vs_k15"] = _dynamics_summary(result["episodes"], meta["bootstrap"])
        for chunk in FIXED_CHUNKS[1:]:
            _, result[f"paired_vs_k{chunk}"] = _dynamics_summary(result["episodes"], meta["bootstrap"], f"fixed_k{chunk}")
        result["by_force"] = {}
        for force in meta["force_levels_n"]:
            summary, paired = _dynamics_summary([e for e in result["episodes"] if e["force_n"] == force], meta["bootstrap"])
            result["by_force"][f"{force:g}"] = {"summary": summary, "paired_vs_k15": paired}
            for chunk in FIXED_CHUNKS[1:]:
                _, result["by_force"][f"{force:g}"][f"paired_vs_k{chunk}"] = _dynamics_summary(
                    [e for e in result["episodes"] if e["force_n"] == force], meta["bootstrap"], f"fixed_k{chunk}")
        result["checks"].update(initial_states_matched=True, force_schedules_checked=True,
                                 fixed_call_counts_checked=True, partitions_disjoint=True)
        result["complete"] = True
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(output, result)
    if stage == "validate":
        now = datetime.now(timezone.utc).isoformat()
        selection = _select_motion_thresholds(result, meta["threshold_grid"], meta["selection_rule"])
        selection.update(selected_at_utc=now,
                         predictor_sha256=digest(predictor_path), validation_sha256=digest(output),
                         metadata_sha256=digest(dataset_dir / "metadata.json"),
                         disabled_candidate="Reuse fixed K15 success and call counts; final test measures monitor overhead directly")
        if selection.get("exploratory_selected"):
            selection["exploratory_selected_at_utc"] = now
        write_json(selection_path, selection)
        print(json.dumps(selection, indent=2), flush=True)
    else:
        changed = [path for path, expected in meta["preserved_sha256"].items() if digest(path, expected) != expected]
        if changed:
            raise RuntimeError(f"Previous artifacts changed: {changed}")
        result["checks"]["preserved_artifacts"] = len(meta["preserved_sha256"])
        write_json(output, result)
        _plot_dynamics_comparison(result, output_dir / "comparison.png")
        print(json.dumps(result["summary"], indent=2), flush=True)
    return result


def _checkpoint_loss_probe(checkpoints, demo_dir="data/split"):
    """Compare EMA denoising error with fixed noise on both demonstration splits."""
    from .train import DemoDataset

    datasets = {split: DemoDataset(Path(demo_dir) / f"{split}.h5") for split in ("train", "validation")}
    result = {"noise_seed": 8641, "draws_per_window": 4, "batch_size": 64, "weights": "EMA",
              "window_counts": {split: len(dataset) for split, dataset in datasets.items()},
              "initial_grasp_flags": {}, "checkpoints": {}}
    for split, dataset in datasets.items():
        flags = [float(obs[0, 18]) for obs in dataset.trajectories["observations"]]
        result["initial_grasp_flags"][split] = {str(flag): flags.count(flag) for flag in sorted(set(flags))}
    for name, path in checkpoints.items():
        policy = load_policy(path, device=torch.device("cpu"))
        result["checkpoints"][name] = {}
        for split, dataset in datasets.items():
            generator = torch.Generator().manual_seed(8641)
            totals = np.zeros(len(dataset.trajectories["observations"]))
            counts = np.zeros_like(totals)
            for start in range(0, len(dataset), 64):
                indices = list(range(start, min(start + 64, len(dataset))))
                batch = [dataset[index] for index in indices]
                observations = policy.observation_normalizer(
                    torch.stack([item["observations"] for item in batch])).flatten(start_dim=1)
                actions = torch.stack([item["actions"] for item in batch])
                losses = torch.zeros(len(indices))
                with torch.inference_mode():
                    for _ in range(4):
                        noise = torch.randn(actions.shape, generator=generator)
                        steps = torch.randint(0, 100, (len(indices),), generator=generator)
                        noisy = policy.noise_scheduler.add_noise(actions, noise, steps)
                        predicted = policy.noise_pred_net(noisy, steps, global_cond=observations)
                        losses += ((predicted - noise) ** 2).mean(dim=(1, 2)) / 4
                for index, loss in zip(indices, losses.tolist()):
                    trajectory = dataset.slices[index][0]
                    totals[trajectory] += loss
                    counts[trajectory] += 1
            result["checkpoints"][name][split] = {
                "window_mean": float(totals.sum() / counts.sum()),
                "episode_losses": (totals / counts).tolist(),
            }
            print(f"Checkpoint {name}: {split} denoising MSE={totals.sum() / counts.sum():.6f}", flush=True)
    return result


def _checkpoint_summaries(episodes, bootstrap, baseline="25000"):
    summary, _ = _dynamics_summary(episodes, bootstrap, f"{baseline}_k15")
    paired = {}
    for k in (8, 15):
        records = [episode for episode in episodes if episode["chunk_length"] == k]
        _, paired[str(k)] = _dynamics_summary(records, bootstrap, f"{baseline}_k{k}")
    return {"summary": summary, f"paired_vs_{baseline}_same_chunk": paired}


def _plot_checkpoint_comparison(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    names = list(result["plan"]["checkpoints"])
    labels = {name: name.replace("_", " ").title() for name in names}
    updates = list(range(len(names)))
    for split in ("train", "validation"):
        axes[0].plot(updates,
                     [result["denoising"]["checkpoints"][name][split]["window_mean"] for name in names],
                     marker="o", label="Training demos" if split == "train" else "Held-out demos")
    axes[0].set(xlabel="Policy", ylabel="Denoising MSE", title="Fixed-noise EMA loss check")
    axes[0].set_xticks(updates, [labels[name] for name in names])
    axes[0].legend()
    forces = [float(force) for force in result["by_force"]]
    for axis, k in zip(axes[1:], (15, 8)):
        for name in names:
            summaries = [result["by_force"][f"{force:g}"]["summary"][f"{name}_k{k}"] for force in forces]
            means = np.array([summary["success_at_end"] for summary in summaries])
            intervals = np.array([summary["ci95"]["success_at_end"] for summary in summaries])
            axis.errorbar(forces, 100 * means, yerr=100 * np.stack([means - intervals[:, 0], intervals[:, 1] - means]),
                          marker="o", capsize=3,
                          label=labels[name])
        axis.set(xlabel="Horizontal force (N)", ylabel="Final success (%)", title=f"Fixed K{k}: fresh matched resets", ylim=(0, 105))
        axis.set_xticks(forces)
        axis.legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"Policy checkpoint audit: {result['plan']['groups']} reset groups × four forces; group-bootstrap 95% intervals")
    fig.savefig(output, dpi=180)
    fig.savefig(Path(output).with_suffix(".svg"))
    plt.close(fig)


def compare_policy_checkpoints(output_dir, checkpoints, groups=30, seed=7501, workers=4, baseline=None,
                               demo_dir="data/split"):
    """Compare named policy checkpoints on fresh matched K8/K15 force scenarios."""
    from datetime import datetime, timezone
    from .monitor import _reset_seeds, digest, reset_seed_sources, rollout_jobs, source_hashes, write_json

    if groups < 1 or workers < 1 or seed < 0:
        raise ValueError("Require positive groups/workers and a nonnegative seed")
    if (len(checkpoints) < 2 or any(not name or not all(c.isalnum() or c in "_-" for c in name)
                                                for name in checkpoints)):
        raise ValueError("Supply at least two policies with distinct alphanumeric/underscore/hyphen names")
    torch.set_num_threads(1)
    output_dir = Path(output_dir)
    checkpoints = {name: resolve_artifact(path) for name, path in checkpoints.items()}
    baseline = baseline or next(iter(checkpoints))
    if baseline not in checkpoints:
        raise ValueError("The baseline must name one of the supplied checkpoints")
    demo_dir = Path(demo_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = reset_seed_sources()
    seen, scanned = set(), {}
    for path in sources:
        records = [json.loads(line) for line in path.read_text().splitlines() if line] if path.suffix == ".jsonl" else [json.loads(path.read_text())]
        excluded = set().union(*(set(_reset_seeds(record)) for record in records))
        seen.update(excluded)
        scanned[str(path)] = {"sha256": digest(path), "seed_count": len(excluded)}
    excluded_count = len(seen)
    rng = np.random.default_rng(seed)
    planned = []
    for _ in range(groups):
        reset = int(rng.integers(10000, 2**31))
        while reset in seen:
            reset = int(rng.integers(10000, 2**31))
        seen.add(reset)
        planned.append({"group_id": f"reset_{reset}", "reset_seed": reset,
                        "policy_seed": int(rng.integers(0, 2**31)), "start_step": int(rng.integers(8, 19)),
                        "angle": float(rng.uniform(0, 2 * np.pi))})
    cases = _dynamics_cases(planned, [0.0, 0.2, 0.3, 0.4])
    preserved = {str(path): digest(path) for folder in ("data", "checkpoints", "results")
                 for path in Path(folder).rglob("*") if path.is_file() and output_dir not in path.parents}
    plan = {"planned_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed, "groups": groups,
            "checkpoints": {name: {"path": str(path), "sha256": digest(path)} for name, path in checkpoints.items()},
            "current_policy_sha256": digest(checkpoints[baseline]),
            "chunk_lengths": [8, 15], "horizon": 100, "weights": "EMA", "device": "cpu", "workers": workers,
            "force_levels_n": [0.0, 0.2, 0.3, 0.4], "force_duration_steps": 2, "force_onset": "uniform integer 8..18",
            "sampling": "SeedSequence([policy_seed, absolute_control_step]); common noise at shared call steps",
            "comparison": "explicit", "baseline": baseline,
            "cohort_role": "Held-out comparison of fixed candidates; any subsequent policy tuning needs a new final cohort",
            "trace_fields": ["actions", "positions", "grasped", "goal_distances_m", "success_steps"],
            "primary_endpoint": "Final success against the original policy at the same chunk length, overall and by force",
            "bootstrap": {"samples": 5000, "seed": seed + 1, "unit": "complete reset group, retaining force conditions"},
            "cases": cases, "excluded_seed_count": excluded_count, "overlap_sources": scanned,
            "preserved_sha256": preserved, "code_sha256": source_hashes(),
            "demo_sha256": {split: digest(demo_dir / f"{split}.h5") for split in ("train", "validation")}}
    write_json(output_dir / "plan.json", plan)
    print(f"Planned {groups} fresh groups, excluding {excluded_count} prior seeds", flush=True)
    result = {"complete": False, "plan": plan, "episodes": [], "checks": {"exact_repeats": 0, "actions_checked": 0}}
    output = output_dir / "results.json"
    write_json(output, result)
    tasks = [{**case, "checkpoint": name, "variant": f"{name}_k{k}", "chunk_length": k,
              "kind": "fixed", "threshold": None,
              "repeat": case["group_id"] == planned[0]["group_id"] and case["force_n"] in (0, 0.4)}
             for case in cases for name in checkpoints for k in (8, 15)]
    initial_states = {}
    started = time.perf_counter()
    try:
        result["denoising"] = _checkpoint_loss_probe(checkpoints, demo_dir)
        for task, episode in rollout_jobs(tasks, checkpoints, workers):
            initial = episode.pop("states")[0]
            if initial_states.setdefault(task["episode_id"], initial) != initial:
                raise RuntimeError("Checkpoint comparisons started from different states")
            _audit_motion_episode(task, episode, None)
            result["checks"]["actions_checked"] += episode["steps"]
            result["checks"]["exact_repeats"] += int(episode.get("repeat_exact", False))
            episode.update({key: task[key] for key in ("group_id", "episode_id", "force_n", "variant", "checkpoint", "chunk_length")})
            episode.update(initial_state=initial, inference_seconds=episode["policy_seconds"] + episode["monitor_seconds"])
            result["episodes"].append(episode)
            if len(result["episodes"]) % 20 == 0:
                write_json(output, result)
                print(f"Checkpoint audit: {len(result['episodes'])}/{len(tasks)} rollouts complete", flush=True)
        result.update(_checkpoint_summaries(result["episodes"], plan["bootstrap"], baseline))
        result["by_force"] = {f"{force:g}": _checkpoint_summaries([e for e in result["episodes"] if e["force_n"] == force], plan["bootstrap"], baseline)
                              for force in plan["force_levels_n"]}
        result["by_condition"] = {condition: _checkpoint_summaries([e for e in result["episodes"] if (e["force_n"] > 0) == disturbed], plan["bootstrap"], baseline)
                                  for condition, disturbed in (("nominal", False), ("disturbed", True))}
        changed = [path for path, expected in preserved.items() if digest(path, expected) != expected]
        if changed:
            raise RuntimeError(f"Previous artifacts changed: {changed}")
        result["checks"].update(initial_states_matched=True, force_schedules_checked=True,
                                 fixed_call_counts_checked=True, fresh_reset_groups=groups, preserved_artifacts=len(preserved))
        result["complete"] = True
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(output, result)
    _plot_checkpoint_comparison(result, output_dir / "comparison.png")
    print(json.dumps(result["summary"], indent=2), flush=True)
    return result
