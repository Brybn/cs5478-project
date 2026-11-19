"""Fresh transition collection, one-step prediction and motion triggers."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
import multiprocessing as mp
from pathlib import Path
import time
import zlib

import numpy as np
import torch
from torch import nn

from .artifacts import digest, resolve_artifact


PARTITIONS = {"train": 40, "predictor_validation": 20, "scheduler_validation": 30, "test": 40}
THRESHOLDS = {"prediction_error": [0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, None],
              "acceleration": [0.0003, 0.0005, 0.001, 0.0015, 0.002, 0.003, None]}
FORCES = [0.0, 0.3, 0.4, 0.5]
TASK_FORCES = {"PickCube": FORCES, "StackCube": [0.0, 0.2, 0.3, 0.4],  # StackCube fails for >= 0.6 N
               "LiftPegUpright": [0.0, 2.0, 3.0, 3.5]}  # 0.60 kg peg; 4 N leaves ~25% success
FIXED_CHUNKS = (15, 10, 8)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes():
    paths = [Path(__file__), Path(__file__).with_name("env.py"),
             Path(__file__).with_name("evaluate.py"), Path(__file__).with_name("policy.py"),
             Path(__file__).with_name("artifacts.py"), Path("scripts/monitor/study.py"),
             Path("pyproject.toml"), Path("uv.lock")]
    return {str(path): digest(path) for path in paths}


class DynamicsMLP(nn.Module):
    def __init__(self, input_dim=16):  # 2 x 6 positions + action (4-D, or 7-D with rotation -> 19)
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128),
                                 nn.ReLU(), nn.Linear(128, 6))

    def forward(self, x):
        return self.net(x)


class DynamicsPredictor:
    def __init__(self, model, input_mean, input_std, target_mean, target_std, residual_scale):
        self.model = model.eval()
        self.input_mean = np.asarray(input_mean, dtype=np.float32)
        self.input_std = np.asarray(input_std, dtype=np.float32)
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)
        self.residual_scale = np.maximum(np.asarray(residual_scale, dtype=np.float32), 1e-4)

    @classmethod
    def load(cls, path):
        saved = torch.load(resolve_artifact(path), map_location="cpu", weights_only=True)
        model = DynamicsMLP(len(saved["input_mean"]))
        model.load_state_dict(saved["model"])
        return cls(model, *(saved[key] for key in (
            "input_mean", "input_std", "target_mean", "target_std", "residual_scale")))

    @torch.no_grad()
    def predict(self, previous, current, action):
        inputs = np.concatenate([previous, current, action]).astype(np.float32)
        delta = self.model(torch.from_numpy((inputs - self.input_mean) / self.input_std)).numpy()
        return current + delta * self.target_std + self.target_mean

    def score(self, predicted, actual):
        return float(np.sqrt(np.mean(((actual - predicted) / self.residual_scale) ** 2)))


class MotionTrigger:
    """Score only observed motion and the preceding action's prediction."""

    def __init__(self, kind, threshold, predictor=None):
        self.kind, self.threshold, self.predictor = kind, threshold, predictor

    def score(self, history, prediction):
        if self.kind == "random":
            # Deterministic pseudo-random draw from the observed state: control with timing independent of dynamics.
            return zlib.crc32(np.asarray(history[-1], dtype=np.float32).tobytes()) / 2**32
        if self.kind == "prediction_error":
            return None if prediction is None else self.predictor.score(prediction, history[-1])
        if len(history) < 3:
            return None
        return float(np.linalg.norm(history[-1][:3] - 2 * history[-2][:3] + history[-3][:3]))

    def predict_next(self, history, action):
        if self.kind == "prediction_error" and len(history) >= 2:
            return self.predictor.predict(history[-2], history[-1], action)
        return None


def _reset_seeds(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("reset_seed", "seed") and isinstance(item, int):
                yield item
            elif key in ("reset_seeds", "new_reset_seeds", "excluded_reset_seeds"):
                yield from item
            else:
                yield from _reset_seeds(item)
    elif isinstance(value, list):
        for item in value:
            yield from _reset_seeds(item)


# Union of every seed the original project used before its 13 Nov 2026 purge (local only; a fresh clone has none).
SEED_REGISTRY = Path("data/reset_seed_registry.json")


def reset_seed_sources():
    """Include policy selection and previous dynamics partitions in new seed checks."""
    return sorted({*([SEED_REGISTRY] if SEED_REGISTRY.exists() else []), *Path("data").rglob("metadata.json"),
                   *Path("data").glob("**/split*/*.json"), *Path("results").rglob("*.json"),
                   *Path("checkpoints").rglob("training.json")})


def task_name():
    from .env import task

    return task()[0]


def plan_collection(policy_path, dataset_dir, seed=6401, horizon=100):
    """Write all partitions and the finite threshold grid before any rollout."""
    if horizon < 1:
        raise ValueError("Episode horizon must be positive")
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=False)
    sources = reset_seed_sources()
    excluded, scanned = set(), {}
    for path in sources:
        if path.suffix == ".jsonl":
            seeds = set().union(*(set(_reset_seeds(json.loads(line))) for line in path.read_text().splitlines() if line))
        else:
            seeds = set(_reset_seeds(json.loads(path.read_text())))
        excluded.update(seeds)
        scanned[str(path)] = {"sha256": digest(path), "seed_count": len(seeds)}
    rng = np.random.default_rng(seed)
    groups, seen = [], set(excluded)
    for partition, count in PARTITIONS.items():
        for index in range(count):
            reset = int(rng.integers(10000, 2**31))
            while reset in seen:
                reset = int(rng.integers(10000, 2**31))
            seen.add(reset)
            groups.append({"group_id": f"reset_{reset}", "partition": partition,
                           "reset_seed": reset, "policy_seed": int(rng.integers(0, 2**31)),
                           "start_step": int(rng.integers(8, 19)),
                           "angle": float(rng.uniform(0, 2 * np.pi))})
    preserved = [path for folder in ("checkpoints", "results", "data")
                 for path in Path(folder).rglob("*") if path.is_file() and dataset_dir not in path.parents]
    call_targets = [len(range(0, horizon, chunk)) for chunk in FIXED_CHUNKS]
    metadata = {
        "complete": False, "planned_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": str(policy_path), "policy_sha256": digest(policy_path), "device": "cpu",
        "seed": seed, "partitions": PARTITIONS, "groups": groups, "horizon": horizon,
        "collection_chunks": [8, 15], "collection_force_n": 0,
        "sampling": "SeedSequence([policy_seed, absolute_control_step]); common noise at shared steps",
        "force_levels_n": TASK_FORCES[task_name()], "force_duration_steps": 2, "force_onset": "uniform integer 8..18",
        "threshold_grid": THRESHOLDS, "max_chunk_length": 15, "min_replan_interval": 2,
        "call_targets": call_targets,
        "selection_rule": f"For each trigger and target {'/'.join(map(str, call_targets))} calls (fixed K15/K10/K8), require mean calls in [0.9*target, 1.1*target]; minimize mean capped time to goal, then maximize final success, then minimize calls, then prefer disabled/higher threshold. Omit a band if no candidate fits.",
        "primary": "Mean capped time to goal (first ten consecutive official-success steps from reset, capped at the horizon), against policy calls. Report paired differences against fixed K15, K10 and K8; call-band points compare with their corresponding fixed schedule. Success by 60/100 steps, recovery time after the pulse and final success are secondary; final-ten held success is a guard. Active learned point outside bands stays exploratory.",
        "timing": "Same definition as primary: first ten consecutive official-success steps, capped at the deadline if absent. Recovery latency starts after the two-step pulse (nominal starts at reset); require ten success states after that origin. Keep later failures and final-ten success separate. Simulation time excludes inference.",
        "bootstrap": {"unit": "complete reset group with all four forces", "samples": 5000, "seed": seed + 1},
        "overlap_sources": scanned, "excluded_seed_count": len(excluded),
        "overlap_with_previous": [], "code_sha256": source_hashes(),
        "preserved_sha256": {str(path): digest(path) for path in preserved},
        "versions": {name: version(name) for name in ("torch", "numpy", "mani-skill", "diffusers")},
    }
    write_json(dataset_dir / "metadata.json", metadata)
    print(f"Planned {len(groups)} disjoint groups; checked {len(sources)} previous files ({len(excluded)} seeds)", flush=True)
    return metadata


_WORKER = {}


def _init_worker(policy_path, predictor_path, horizon=100):
    from .env import extract_policy_state, make_env
    from .policy import load_policy

    torch.set_num_threads(1)
    device = torch.device("cpu")
    paths = policy_path if isinstance(policy_path, dict) else {"default": policy_path}
    policies = {name: load_policy(path, device=device) for name, path in paths.items()}
    env = make_env(max_episode_steps=horizon,
                   disturbance={"start_step": 0, "duration_steps": 0, "force_xy": [0.0, 0.0]})
    obs, _ = env.reset(seed=1)
    state = extract_policy_state(obs)
    for policy in policies.values():
        policy.sample(np.stack([state] * 2), torch.Generator().manual_seed(0), device)
    _WORKER.update(env=env, policy=next(iter(policies.values())), policies=policies, device=device, horizon=horizon,
                   predictor=DynamicsPredictor.load(predictor_path) if predictor_path else None)


def _rollout_job(task):
    from .evaluate import run_episode

    env = _WORKER["env"]
    horizon = _WORKER.get("horizon", 100)
    env.unwrapped.disturbance = {key: task[key] for key in ("start_step", "duration_steps", "force_xy")}
    trigger = (MotionTrigger(task["kind"], task["threshold"], _WORKER["predictor"])
               if task["kind"] != "fixed" else None)
    policy = _WORKER["policies"][task["checkpoint"]] if "checkpoint" in task else _WORKER["policy"]
    def run():
        return run_episode(env, policy, task["chunk_length"], task["reset_seed"],
                           _WORKER["device"], horizon=horizon, policy_seed=task["policy_seed"],
                           motion_trigger=trigger, collect_positions=True,
                           record_timing=task.get("record_timing", False))
    episode = run()
    expected = list(range(task["start_step"], task["start_step"] + task["duration_steps"])) if any(task["force_xy"]) else []
    substeps = env.unwrapped.sim_freq // env.unwrapped.control_freq
    if (episode["steps"] != horizon or episode["applied_force_steps"] != expected or
            episode["applied_force_substeps"] != substeps * len(expected)):
        raise RuntimeError("Rollout horizon or force schedule differs from the plan")
    if task.get("repeat"):
        repeated = run()
        keys = [key for key in episode if key not in ("policy_seconds", "monitor_seconds", "policy_call_seconds")]
        if any(episode[key] != repeated[key] for key in keys):
            raise RuntimeError("Repeated rollout changed its actions, states or decisions")
        episode["repeat_exact"] = True
    if task["kind"] == "fixed" and episode["policy_calls"] != len(range(0, horizon, task["chunk_length"])):
        raise RuntimeError("Fixed scheduler made an incorrect number of policy calls")
    if task.get("record_timing"):
        episode["control_freq_hz"] = env.unwrapped.control_freq
    return task, episode


def rollout_jobs(tasks, policy_path, workers=4, predictor_path=None, horizon=100):
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker, initargs=(policy_path, predictor_path, horizon)) as pool:
        yield from pool.map(_rollout_job, tasks, chunksize=1)


def collect_transitions(policy_path, dataset_dir, workers=4, seed=6401):
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else plan_collection(policy_path, dataset_dir, seed)
    if meta["complete"] or (dataset_dir / "transitions.npz").exists():
        raise ValueError("Transition dataset already exists")
    if digest(policy_path) != meta["policy_sha256"]:
        raise ValueError("Policy differs from the collection plan")
    horizon = meta["horizon"]
    tasks = [{**group, "kind": "fixed", "threshold": None, "chunk_length": k,
              "start_step": 0, "duration_steps": 0, "force_xy": [0.0, 0.0],
              "repeat": group == meta["groups"][0]}
             for group in meta["groups"] if group["partition"] in ("train", "predictor_validation")
             for k in meta["collection_chunks"]]
    columns = {key: [] for key in ("previous", "current", "action", "next", "reset_seed", "step", "chunk_length", "partition")}
    summaries = []
    started = time.perf_counter()
    for task, episode in rollout_jobs(tasks, policy_path, workers, horizon=horizon):
        positions = np.asarray(episode["positions"], dtype=np.float32)
        states = np.asarray(episode["states"], dtype=np.float32)
        from .env import MONITOR_STATE_SLICES, task as _task

        (c0, c1), (t0, t1) = MONITOR_STATE_SLICES[_task()[0]]
        if not np.array_equal(positions, np.concatenate([states[:, c0:c1], states[:, t0:t1]], axis=1)):
            raise RuntimeError("Monitor position extraction differs from policy observation order")
        current, following = positions[:-1], positions[1:]
        previous = np.concatenate([positions[:1], positions[:-2]])
        for key, values in {"previous": previous, "current": current,
                            "action": np.asarray(episode["actions"], dtype=np.float32), "next": following,
                            "reset_seed": np.full(horizon, task["reset_seed"], dtype=np.int64),
                            "step": np.arange(horizon), "chunk_length": np.full(horizon, task["chunk_length"]),
                            "partition": np.full(horizon, task["partition"])}.items():
            columns[key].append(values)
        summaries.append({**{key: task[key] for key in ("group_id", "partition", "chunk_length")},
                          **{key: episode[key] for key in ("steps", "success_once", "success_at_end", "policy_calls")}})
        print(f"Collected {len(summaries)}/{len(tasks)}: {task['partition']} K{task['chunk_length']}", flush=True)
    arrays = {key: np.concatenate(values) for key, values in columns.items()}
    np.savez_compressed(dataset_dir / "transitions.npz", **arrays)
    meta.update(complete=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                transitions_sha256=digest(dataset_dir / "transitions.npz"),
                transition_counts={split: int(np.sum(arrays["partition"] == split)) for split in ("train", "predictor_validation")},
                reset_history="At step 0 previous=current; every later previous is the preceding observation",
                position_order=["cube_x", "cube_y", "cube_z", "tcp_x", "tcp_y", "tcp_z"],
                episodes=summaries, repeat_checks=2, alignment_checked=True,
                collection_code_sha256=source_hashes(), elapsed_seconds=time.perf_counter() - started)
    write_json(metadata_path, meta)
    print(f"Saved {len(arrays['step'])} fresh transitions to {dataset_dir}", flush=True)


def _prediction_metrics(target, prediction, steps, horizon=100):
    masks = {"overall": np.ones(len(steps), dtype=bool), "early_0_24": steps < 25,
             "middle_25_59": (steps >= 25) & (steps < 60), f"late_60_{horizon-1}": steps >= 60,
             "moving_over_1mm": np.linalg.norm(target, axis=1) > 0.001}
    result = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        result[name] = {"transitions": int(mask.sum())}
        for label, estimate in (("predictor", prediction), ("unchanged", np.zeros_like(target))):
            error = target[mask] - estimate[mask]
            result[name][label] = {"cube_rmse_mm": float(1000 * np.sqrt(np.mean(error[:, :3] ** 2))),
                                  "gripper_rmse_mm": float(1000 * np.sqrt(np.mean(error[:, 3:] ** 2)))}
    return result


def _plot_training(history, best_epoch, metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    for key, label in (("train_mse", "Training"), ("validation_mse", "Held-out")):
        axes[0].plot(epochs, [row[key] for row in history], label=label)
    axes[0].axvline(best_epoch, color="gray", linestyle="--", label=f"Selected epoch {best_epoch}")
    axes[0].set(xlabel="Epoch", ylabel="Normalized prediction MSE", yscale="log", title="Fresh predictor training")
    axes[0].legend()
    values = [("overall", "cube_rmse_mm"), ("overall", "gripper_rmse_mm"),
              ("moving_over_1mm", "cube_rmse_mm"), ("moving_over_1mm", "gripper_rmse_mm")]
    x = np.arange(len(values))
    for offset, method, label in ((-0.18, "predictor", "Learned predictor"), (0.18, "unchanged", "Positions unchanged")):
        bars = axes[1].bar(x + offset, [metrics[group][method][key] for group, key in values],
                          width=0.36, label=label)
        axes[1].bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    axes[1].set_xticks(x, ["Cube\nall steps", "Tool\nall steps", "Cube\nmoving", "Tool\nmoving"])
    axes[1].set(ylabel="Per-coordinate RMSE (mm)", title="Held-out prediction error")
    axes[1].margins(y=0.18)
    axes[1].legend(loc="upper left")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def train_dynamics(dataset_dir, output_dir, seed=4278):
    """Fit training normalization and select the lowest validation MSE in 100 epochs."""
    torch.set_num_threads(1)
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    meta = json.loads((dataset_dir / "metadata.json").read_text())
    if not meta["complete"] or digest(dataset_dir / "transitions.npz") != meta["transitions_sha256"]:
        raise ValueError("Collection is incomplete or transitions have changed")
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(dataset_dir / "transitions.npz") as data:
        inputs = np.concatenate([data["previous"], data["current"], data["action"]], axis=1)
        targets = data["next"] - data["current"]
        masks = {split: data["partition"] == split for split in ("train", "predictor_validation")}
        steps = data["step"].copy()
    train_x, train_y = inputs[masks["train"]], targets[masks["train"]]
    mean_x, std_x = train_x.mean(0), np.maximum(train_x.std(0), 1e-6)
    mean_y, std_y = train_y.mean(0), np.maximum(train_y.std(0), 1e-6)
    normalized_x = torch.from_numpy((inputs - mean_x) / std_x)
    normalized_y = torch.from_numpy((targets - mean_y) / std_y)
    x, y = normalized_x[masks["train"]], normalized_y[masks["train"]]
    vx, vy = normalized_x[masks["predictor_validation"]], normalized_y[masks["predictor_validation"]]
    torch.manual_seed(seed)
    model = DynamicsMLP(inputs.shape[1])
    initial = deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00001)
    generator = torch.Generator().manual_seed(seed)
    history, best_loss, best_epoch, best_weights = [], float("inf"), None, None
    started = time.perf_counter()
    for epoch in range(1, 101):
        model.train()
        total = 0.0
        for indices in torch.randperm(len(x), generator=generator).split(256):
            loss = torch.mean((model(x[indices]) - y[indices]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        model.eval()
        with torch.no_grad():
            validation_loss = float(torch.mean((model(vx) - vy) ** 2))
        history.append({"epoch": epoch, "train_mse": total / len(x), "validation_mse": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch, best_weights = validation_loss, epoch, deepcopy(model.state_dict())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/100: validation MSE={validation_loss:.6f}", flush=True)
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        predictions = model(normalized_x).numpy() * std_y + mean_y
    validation_mask = masks["predictor_validation"]
    scales = np.maximum(np.sqrt(np.mean((targets[validation_mask] - predictions[validation_mask]) ** 2, axis=0)), 1e-4)
    config = {"architecture": [int(inputs.shape[1]), 128, 128, 6], "activation": "ReLU", "optimizer": "AdamW",
              "lr": 0.001, "weight_decay": 0.00001, "batch_size": 256, "epochs": 100, "seed": seed}
    checkpoint = {"model": best_weights, "input_mean": mean_x.tolist(), "input_std": std_x.tolist(),
                  "target_mean": mean_y.tolist(), "target_std": std_y.tolist(), "residual_scale": scales.tolist(),
                  "config": config, "best_epoch": best_epoch, "policy_sha256": meta["policy_sha256"],
                  "transitions_sha256": meta["transitions_sha256"], "metadata_sha256": digest(dataset_dir / "metadata.json")}
    path = output_dir / "predictor.pt"
    torch.save(checkpoint, path)
    loaded = DynamicsPredictor.load(path)
    reloaded = np.stack([loaded.predict(row[:6], row[6:12], row[12:]) for row in inputs[:32]])
    batch_positions = inputs[:32, 6:12] + predictions[:32]
    reload_error = float(np.max(np.abs(reloaded - batch_positions)))
    if reload_error > 1e-6:
        raise RuntimeError("Reloaded predictor differs from training predictions")
    metrics = {split: _prediction_metrics(targets[mask], predictions[mask], steps[mask], meta["horizon"])
               for split, mask in masks.items()}
    write_json(output_dir / "training.json", {
        "complete": True, "config": config, "dataset": str(dataset_dir),
        "policy_sha256": meta["policy_sha256"], "transitions_sha256": meta["transitions_sha256"],
        "best_epoch": best_epoch, "best_validation_mse": best_loss, "history": history,
        "initialization": "Fresh torch.manual_seed(seed); no predictor checkpoint loaded",
        "weights_changed": any(not torch.equal(initial[key], best_weights[key]) for key in initial),
        "normalization": "Training transitions only; population standard deviation with 1e-6 floor",
        "residual_scale_metres": scales.tolist(), "reload_max_abs_error_metres": reload_error,
        "rmse_definition": "Root mean squared error per Cartesian coordinate, aggregated over XYZ",
        "metrics": metrics, "code_sha256": source_hashes(), "elapsed_seconds": time.perf_counter() - started,
    })
    _plot_training(history, best_epoch, metrics["predictor_validation"], output_dir / "training.png")
    print(json.dumps({"best_epoch": best_epoch, "validation": metrics["predictor_validation"]}, indent=2), flush=True)
    return metrics
