"""Fine-tune the existing policy with nominal demonstrations and expert corrections."""

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/robots2-matplotlib")

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
from tqdm import trange

from .artifacts import digest, resolve_artifact
from .policy import PolicyConfig, load_policy
from .env import task
from .train import DemoDataset, DiffusionPolicyAgent, evaluate_policy, selection_cases


GRASP_PHASES = ("align", "descend", "close", "lift")


class RecoveryDataset(DemoDataset):
    """Keep actual takeover history and supervise only recorded expert actions."""

    def __init__(self, h5_path):
        super().__init__(h5_path, obs_horizon=2, pred_horizon=16)
        self.path = Path(h5_path)
        self.context = []
        self.learner_conditioning = []
        query_only = []
        with h5py.File(h5_path, "r") as handle:
            self.trajectory_keys = sorted(handle, key=lambda name: int(name.split("_")[-1]))
            for index, key in enumerate(self.trajectory_keys):
                group = handle[key]
                obs = self.trajectories["observations"][index]
                actions = self.trajectories["actions"][index]
                context = torch.as_tensor(group["context_obs"][:], dtype=torch.float32)
                expert = group["action_is_expert"][:]
                conditioning = group.attrs.get("conditioning", "expert_states")
                if conditioning not in ("expert_states", "learner_takeover", "learner_continuation"):
                    raise ValueError(f"Unknown recovery conditioning: {h5_path}:{key}")
                learner = conditioning != "expert_states"
                if conditioning == "learner_takeover" and len(actions) != 15:
                    raise ValueError("Learner takeover chunks must contain fifteen expert actions")
                if conditioning == "learner_continuation" and len(actions) < 15:
                    raise ValueError("Learner continuations must contain at least fifteen expert actions")
                if (obs.shape != (len(actions) + 1, task()[3]) or actions.ndim != 2
                        or context.shape != (2, task()[3]) or not torch.equal(context[1], obs[0])
                        or expert.shape != (len(actions),) or not np.all(expert == 1)
                        or not all(torch.isfinite(value).all() for value in (obs, actions, context))):
                    raise ValueError(f"Invalid expert correction segment: {h5_path}:{key}")
                self.context.append(context)
                self.learner_conditioning.append(learner)
                query_only.append(conditioning == "learner_takeover")
        self.slices = [s for s in self.slices if not query_only[s[0]] or s[1] == -1]

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        trajectory, start, end = self.slices[index]
        length = len(self.trajectories["actions"][trajectory])
        if start < 0:
            sample["observations"] = self.context[trajectory].clone()
            sample["actions"] = sample["actions"].clone()
            sample["actions"][0] = 0  # Placeholder preceding expert takeover; excluded from loss.
        times = torch.arange(start, end)
        sample["action_mask"] = (times >= 0) & (times < length)
        return sample


def grasp_window_indices(recovery):
    """Select current conditioning states in recorded expert grasp phases."""
    datasets = list(recovery.datasets) if isinstance(recovery, ConcatDataset) else [recovery]
    indices, sources, offset = [], [], 0
    counts = dict.fromkeys(GRASP_PHASES, 0)
    for dataset in datasets:
        episodes = json.loads(dataset.path.with_suffix(".json").read_text())["episodes"]
        entries = {entry["trajectory"]: entry for entry in episodes}
        if len(entries) != len(episodes) or set(entries) != set(dataset.trajectory_keys):
            raise ValueError(f"Grasp phase metadata does not match trajectories: {dataset.path}")
        phases = []
        for trajectory, key in enumerate(dataset.trajectory_keys):
            length = len(dataset.trajectories["actions"][trajectory])
            labels = np.full(length, "other", dtype=object)
            entry = entries[key]
            if entry.get("teacher") == "pickup":
                previous = -1
                for event in entry["pickup_events"]:
                    if "phase" not in event:
                        continue
                    step, phase = event["step"], event["phase"]
                    if (type(step) is not int or not 0 <= step <= length or step < previous
                            or phase not in (*GRASP_PHASES, "retract", "place")):
                        raise ValueError(f"Invalid expert phase event: {dataset.path}:{key}")
                    labels[step:] = phase
                    previous = step
            phases.append(labels)
        before = len(indices)
        for index, (trajectory, start, _) in enumerate(dataset.slices):
            phase = phases[trajectory][start + dataset.obs_horizon - 1]
            if phase in counts:
                indices.append(offset + index)
                counts[phase] += 1
        sources.append({"path": str(dataset.path), "grasp_windows": len(indices) - before})
        offset += len(dataset)
    if not indices:
        raise ValueError("No grasp windows with recorded pickup-teacher phases")
    return indices, {"phases": list(GRASP_PHASES), "phase_counts": counts, "sources": sources,
                     "eligible_windows": len(indices),
                     "indices_sha256": sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
                     "rule": "Current conditioning observation in align/descend/close/lift; retract/place excluded"}


def learner_window_indices(recovery, latest_only=False):
    """Locate explicit learner-conditioned queries without changing old datasets."""
    datasets = list(recovery.datasets) if isinstance(recovery, ConcatDataset) else [recovery]
    indices, sources, offset = [], [], 0
    for number, dataset in enumerate(datasets):
        selected = [offset+i for i, (trajectory, start, _) in enumerate(dataset.slices)
                    if start == -1 and dataset.learner_conditioning[trajectory]
                    and (not latest_only or number == len(datasets)-1)]
        indices.extend(selected)
        sources.append({"path": str(dataset.path), "learner_windows": len(selected)})
        offset += len(dataset)
    if latest_only and not indices:
        raise ValueError("The last recovery directory must contain learner takeover queries")
    return indices, {"eligible_windows": len(indices), "sources": sources,
                     "indices_sha256": sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
                     "rule": ("Last recovery directory's learner takeover queries; earlier rounds remain in pooled recovery"
                              if latest_only else "One true learner context with fifteen expert targets per query; full continuation windows stay pooled when present")}


def fit_observation_stats(nominal, recovery, recovery_fraction, batch_size=256):
    """Fit z-scores on training conditioning windows, weighted by the data mixture."""
    moments = []
    for dataset in (nominal, recovery):
        total, squared, count = 0, 0, 0
        for start in range(0, len(dataset), batch_size):
            obs = torch.stack([dataset[i]["observations"]
                               for i in range(start, min(start + batch_size, len(dataset)))]).double()
            total = total + obs.sum((0, 1))
            squared = squared + obs.square().sum((0, 1))
            count += obs.shape[0] * obs.shape[1]
        moments.append((total / count, squared / count))
    mean = (1 - recovery_fraction) * moments[0][0] + recovery_fraction * moments[1][0]
    second = (1 - recovery_fraction) * moments[0][1] + recovery_fraction * moments[1][1]
    std = (second - mean.square()).clamp_min(0).sqrt().clamp_min(1e-3)
    return mean.float(), std.float()


def stack_samples(samples, device):
    return {"observations": torch.stack([s["observations"] for s in samples]).to(device),
            "actions": torch.stack([s["actions"] for s in samples]).to(device),
            "action_mask": torch.stack([s.get("action_mask", torch.ones(len(s["actions"]), dtype=torch.bool))
                                         for s in samples]).to(device)}


def mixed_batch(nominal, recovery, batch_size, recovery_fraction, generator, device,
                balance_recovery_sources=False, grasp_indices=None, learner_indices=None):
    count = round(batch_size * recovery_fraction)
    if not 0 <= recovery_fraction <= 1 or (0 < recovery_fraction < 1 and count in (0, batch_size)):
        raise ValueError("Choose a recovery fraction and batch size that provide both sources")
    recovery_sources = list(recovery.datasets) if balance_recovery_sources and isinstance(recovery, ConcatDataset) else [recovery]
    if count % len(recovery_sources):
        raise ValueError("Recovery samples per batch must divide evenly across balanced sources")
    draws = [(nominal, batch_size - count, False, False)]
    if grasp_indices is not None and learner_indices is not None:
        raise ValueError("Choose one focused recovery pool")
    focused = grasp_indices if grasp_indices is not None else learner_indices
    if focused is not None:
        if balance_recovery_sources or count < 2 or count % 2 or not len(focused):
            raise ValueError("Focused sampling requires an even recovery count and pooled recovery sources")
        draws += [(Subset(recovery, focused), count // 2, True, True),
                  (recovery, count // 2, True, False)]
    else:
        draws += [(dataset, count // len(recovery_sources), True, False) for dataset in recovery_sources]
    samples, sources, grasp_draws = [], [], []
    for dataset, size, is_recovery, is_grasp_draw in draws:
        for index in torch.randint(len(dataset), (size,), generator=generator).tolist():
            samples.append(dataset[index])
            sources.append(is_recovery)
            grasp_draws.append(is_grasp_draw)
    order = torch.randperm(batch_size, generator=generator).tolist()
    batch = stack_samples([samples[index] for index in order], device)
    batch["source_is_recovery"] = torch.tensor([sources[index] for index in order])
    if grasp_indices is not None:
        batch["sampled_from_grasp_pool"] = torch.tensor([grasp_draws[index] for index in order])
    if learner_indices is not None:
        batch["sampled_from_learner_pool"] = torch.tensor([grasp_draws[index] for index in order])
    return batch


@torch.inference_mode()
def validation_loss(agent, dataset, seed, draws, batch_size=64):
    agent.eval()
    device = next(agent.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batch = stack_samples(samples, device)
        for _ in range(draws):
            loss = agent.compute_loss(batch["observations"], batch["actions"], generator,
                                      action_mask=batch["action_mask"])
            total += float(loss) * len(samples) / draws
    return total / len(dataset)


def dataset_info(path, dataset):
    path = Path(path).resolve()
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    episodes = metadata["episodes"]
    seeds = [entry["episode_seed"] for entry in episodes]
    if len(episodes) != len(dataset.trajectories["actions"]) or len(set(seeds)) != len(seeds):
        raise ValueError(f"Trajectory metadata or reset groups are inconsistent: {path}")
    if not len(dataset):
        raise ValueError(f"No training windows: {path}")
    return {"path": str(path), "sha256": digest(path), "metadata_sha256": digest(metadata_path),
            "episodes": len(episodes), "windows": len(dataset), "reset_seeds": seeds}


def check_split_separation(demos):
    training = set(demos["nominal_train"]["reset_seeds"]) | set(demos["recovery_train"]["reset_seeds"])
    validation = set(demos["nominal_validation"]["reset_seeds"]) | set(demos["recovery_validation"]["reset_seeds"])
    if training & validation:
        raise ValueError("Training and validation reset groups overlap across data sources")
    for split in ("train", "validation"):
        seeds = [seed for name, info in demos.items() if name.endswith("_" + split) for seed in info["reset_seeds"]]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate reset groups within {split} data across sources or correction rounds")


def recovery_datasets(paths):
    """Aggregate expert windows across rounds, retaining each source's provenance."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not paths:
        raise ValueError("At least one recovery dataset is required")
    datasets, sources = [], []
    for path in paths:
        path = Path(path)
        collection_path = path.parent / "metadata.json"
        if collection_path.exists():
            collection = json.loads(collection_path.read_text())
            if collection.get("mode") == "fresh_collection" and not collection.get("complete"):
                raise ValueError(f"Recovery collection is incomplete: {collection_path}")
        dataset = RecoveryDataset(path)
        sources.append(dataset_info(path, dataset))
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0], sources[0]
    return ConcatDataset(datasets), {
        "sources": sources, "episodes": sum(source["episodes"] for source in sources),
        "windows": sum(source["windows"] for source in sources),
        "reset_seeds": [seed for source in sources for seed in source["reset_seeds"]],
    }


def fine_tune_policy(
    init_checkpoint="checkpoints/pickcube/stage0_nominal/best.pt",
    nominal_train="data/split/train.h5", nominal_validation="data/split/validation.h5",
    recovery_train="data/recovery/train.h5", recovery_validation="data/recovery/validation.h5",
    output_dir="checkpoints/pickcube/stage1_placement", total_iters=5000, batch_size=32,
    recovery_fraction=.25, lr=1e-5, warmup_steps=100, eval_freq=1000,
    num_eval_episodes=25, validation_draws=4, validation_seed=8641,
    eval_seed=12001, seed=1, cuda=True,
    balance_recovery_sources=False, grasp_focused=False, normalize_observations=False,
    sampler="ddim", inference_steps=None, focus_latest_queries=False, selection_horizon=100,
    selection_plan=None,
):
    """Fit mixed demonstrations; init_checkpoint=None starts a fresh policy."""
    if inference_steps is None:
        inference_steps = 10 if sampler == "ddim" else 100
    PolicyConfig(sampler=sampler, inference_steps=inference_steps)
    if selection_horizon < 10:
        raise ValueError("Choose a selection horizon of at least 10 steps")
    if min(total_iters, batch_size, eval_freq, validation_draws) < 1 or num_eval_episodes < 0:
        raise ValueError("Training counts must be positive; evaluation episodes may be zero")
    if not np.isfinite(lr) or lr <= 0 or not np.isfinite(recovery_fraction) or not 0 <= recovery_fraction <= 1:
        raise ValueError("Use a positive finite learning rate and recovery fraction in [0, 1]")
    if not 0 <= warmup_steps < total_iters or min(seed, validation_seed, eval_seed) < 0:
        raise ValueError("Warmup must be shorter than training; seeds must be nonnegative")
    recovery_per_batch = round(batch_size * recovery_fraction)
    if 0 < recovery_fraction < 1 and recovery_per_batch in (0, batch_size):
        raise ValueError("The batch size is too small for the requested mixture")
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"Choose a fresh output directory: {output}")
    checkpoint_path = resolve_artifact(init_checkpoint) if init_checkpoint is not None else None
    initial_hash = digest(checkpoint_path) if checkpoint_path is not None else None
    paths = {"nominal_train": nominal_train, "nominal_validation": nominal_validation,
             "recovery_train": recovery_train, "recovery_validation": recovery_validation}
    datasets, demos, metadata_paths = {}, {}, []
    for name, path in paths.items():
        if name.startswith("recovery"):
            datasets[name], demos[name] = recovery_datasets(path)
            source_paths = [source["path"] for source in demos[name].get("sources", [demos[name]])]
        else:
            datasets[name] = DemoDataset(path)
            demos[name] = dataset_info(path, datasets[name])
            source_paths = [path]
        metadata_paths.extend(Path(source).with_suffix(".json") for source in source_paths)
    check_split_separation(demos)
    source_count = len(demos["recovery_train"].get("sources", [demos["recovery_train"]]))
    if balance_recovery_sources and recovery_per_batch % source_count:
        raise ValueError("Recovery samples per batch must divide evenly across balanced sources")
    grasp_indices, grasp_info = None, None
    learner_indices, learner_info = learner_window_indices(datasets["recovery_train"], focus_latest_queries)
    learner_validation, _ = learner_window_indices(datasets["recovery_validation"], focus_latest_queries)
    if learner_indices and (grasp_focused or balance_recovery_sources or recovery_per_batch < 2 or recovery_per_batch % 2):
        raise ValueError("Learner queries use half the recovery draws; require an even recovery count without other balancing flags")
    if grasp_focused:
        if balance_recovery_sources or recovery_per_batch < 2 or recovery_per_batch % 2:
            raise ValueError("Grasp sampling requires an even recovery count and pooled recovery sources")
        grasp_indices, grasp_info = grasp_window_indices(datasets["recovery_train"])
    obs_dim, act_dim = task()[3], datasets["nominal_train"][0]["actions"].shape[-1]  # 4 (XYZ+gripper) or 7 (pose)
    if any(item["observations"].shape != (2, obs_dim) or item["actions"].shape != (16, act_dim)
           for dataset in datasets.values() for item in (dataset[0],)):
        raise ValueError(f"Expected two {obs_dim}D observations and sixteen {act_dim}D actions in every dataset")
    if selection_plan is None:
        cases, excluded_count = selection_cases(metadata_paths, num_eval_episodes, eval_seed)
    else:
        prepared = json.loads(Path(selection_plan).read_text())
        cases, excluded_count = prepared['selection_cases'], prepared['excluded_selection_seed_count']
        used = set().union(*(set(info['reset_seeds']) for info in demos.values()))
        if len(cases) != num_eval_episodes or used & {c['reset_seed'] for c in cases}:
            raise ValueError('Shared selection cases must have the requested count and exclude demonstrations')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    agent = DiffusionPolicyAgent(obs_dim=obs_dim, act_dim=act_dim).to(device)
    if checkpoint_path is not None:
        initial_policy = load_policy(checkpoint_path, device=device)
        agent.observation_normalizer.set_stats(initial_policy.observation_normalizer.stats)
        agent.load_state_dict(initial_policy.state_dict(), strict=True)
        del initial_policy
    if normalize_observations:
        stats = fit_observation_stats(datasets["nominal_train"], datasets["recovery_train"], recovery_fraction)
        agent.observation_normalizer.set_stats(tuple(value.to(device) for value in stats))
    ema_agent = deepcopy(agent)
    ema = EMAModel(parameters=agent.parameters(), power=.75)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, betas=(.95, .999), weight_decay=1e-6)
    schedule = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=warmup_steps,
                             num_training_steps=total_iters)
    batch_generator = torch.Generator().manual_seed(seed)
    config = {"total_iters": total_iters, "batch_size": batch_size, "recovery_fraction": recovery_fraction,
              "recovery_samples_per_batch": recovery_per_batch, "lr": lr, "warmup_steps": warmup_steps,
              "eval_freq": eval_freq, "num_eval_episodes": num_eval_episodes,
              "selection_horizon": selection_horizon,
              "sampler": sampler, "inference_steps": inference_steps, "training_noise_steps": 100,
              "validation_draws": validation_draws, "validation_seed": validation_seed,
              "eval_seed": eval_seed, "seed": seed, "device": str(device),
              "optimizer": "AdamW", "betas": [.95, .999], "weight_decay": 1e-6,
              "lr_schedule": "cosine", "ema_power": .75, "obs_horizon": 2, "pred_horizon": 16,
              "sampling": ("Fixed nominal count; half recovery draws from learner takeover queries, half from all pooled recovery windows"
                           if learner_indices else
                           "Fixed nominal count; half recovery draws from grasp phases, half from all pooled recovery windows"
                           if grasp_focused else
                           "Fixed counts per batch, equal recovery-directory counts, uniform windows within each directory"
                           if balance_recovery_sources else "Fixed counts per batch, uniform windows with replacement within each source"),
              "balance_recovery_sources": balance_recovery_sources,
              "focus_latest_queries": focus_latest_queries,
              "recovery_directory_samples_per_batch": [recovery_per_batch // source_count] * source_count if balance_recovery_sources else None,
              "grasp_sampling": ({**grasp_info, "dedicated_samples_per_batch": recovery_per_batch // 2,
                                   "pooled_samples_per_batch": recovery_per_batch // 2} if grasp_focused else None),
              "learner_sampling": ({**learner_info, "dedicated_samples_per_batch": recovery_per_batch // 2,
                                     "pooled_samples_per_batch": recovery_per_batch // 2} if learner_indices else None),
              "observation_normalization": ({"mean": agent.observation_normalizer.mean.cpu().tolist(),
                                             "std": agent.observation_normalizer.std.cpu().tolist(),
                                             "fitted_on_current_training_data": normalize_observations,
                                             "method": "Training conditioning-window z-score; std floor 1e-3; nominal/recovery mixture weighting"}
                                            if agent.observation_normalizer.stats is not None else None),
              "recovery_mask": "Only recorded expert actions; pre-takeover and synthetic tail targets excluded",
              "validation_combination": "Equal mean of nominal and recovery held-out denoising MSE"}
    rule = ("Highest mean nominal final-ten success at step 150 across K8/K15; ties: final success, then lowest combined validation loss"
            if cases and selection_horizon == 150 else
            "Highest mean nominal step-100 success at K8/K15; ties: lowest combined validation loss"
            if cases else "Lowest combined nominal/recovery validation loss; rollout selection disabled")
    record = {"complete": False, "stage": "recovery_fine_tuning" if checkpoint_path is not None else "mixed_data_training",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "initial_checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
              "initial_checkpoint_sha256": initial_hash,
              "initialization": ("Load selected policy weights (EMA when available); fresh optimizer, LR schedule and EMA"
                                 if checkpoint_path is not None else "Random initialization; fresh optimizer, LR schedule and EMA"),
              "config": config, "demos": demos, "selection_rule": rule, "selection_cases": cases,
              "excluded_previous_reset_seeds": excluded_count, "evaluations": []}
    if selection_plan is not None:
        record['selection_plan'] = {'path': str(selection_plan), 'sha256': digest(selection_plan)}
    if normalize_observations:
        record["initialization"] += ("; fitted observation normalization changes conditioning before update 0"
                                     if checkpoint_path is not None else "; train-only observation normalization enabled")
    record["code_sha256"] = {str(path): digest(path) for path in (
        Path(__file__), Path(__file__).with_name("train.py"), Path(__file__).with_name("policy.py"),
        Path("scripts/policy/finetune_policy.py"), Path("uv.lock"))}
    output.mkdir(parents=True, exist_ok=False)
    mode = "Fine-tuning" if checkpoint_path is not None else "Training from scratch"
    print(f"{mode} on {device}: {batch_size - recovery_per_batch} nominal + {recovery_per_batch} recovery windows per batch", flush=True)
    if grasp_focused:
        print(f"Recovery draws: {recovery_per_batch // 2} grasp-focused + {recovery_per_batch // 2} pooled; "
              f"{len(grasp_indices)} eligible training windows", flush=True)
    if learner_indices:
        print(f"Recovery draws: {recovery_per_batch // 2} learner queries + {recovery_per_batch // 2} pooled; "
              f"{len(learner_indices)} training queries", flush=True)
    if normalize_observations:
        print("Training-only observation normalization enabled"
              + ("; initial rollout behavior may change" if checkpoint_path is not None else ""), flush=True)
    losses, best_key, best_iteration = [], None, None

    def evaluate_and_save(iteration):
        nonlocal best_key, best_iteration
        ema.copy_to(ema_agent.parameters())
        nominal_loss = validation_loss(ema_agent, datasets["nominal_validation"], validation_seed, validation_draws)
        recovery_loss = validation_loss(ema_agent, datasets["recovery_validation"], validation_seed, validation_draws)
        combined = .5 * (nominal_loss + recovery_loss)
        rollout = evaluate_policy(ema_agent, cases, sampler=sampler, inference_steps=inference_steps,
                                  horizon=selection_horizon) if cases else None
        key = ((rollout["mean_last_ten_success"], rollout["mean_final_success"], -combined)
               if rollout and selection_horizon == 150 else
               (rollout["mean_final_success"], -combined) if rollout else (-combined,))
        selected = best_key is None or key > best_key
        if selected:
            best_key, best_iteration = key, iteration
        row = {"iteration": iteration, "nominal_validation_mse": nominal_loss,
               "recovery_validation_mse": recovery_loss, "combined_validation_mse": combined,
               "rollout": rollout, "selected_best": selected}
        if learner_validation:
            row["learner_validation_mse"] = validation_loss(
                ema_agent, Subset(datasets["recovery_validation"], learner_validation), validation_seed, validation_draws)
        record["evaluations"].append(row)
        record["best_iteration"] = best_iteration
        checkpoint = {"agent": agent.state_dict(), "ema_agent": ema_agent.state_dict(),
                      "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                      "lr_scheduler": schedule.state_dict(), "iteration": iteration,
                      "config": config, "demos": demos, "selection_rule": rule,
                      "initial_checkpoint_sha256": initial_hash, "validation": row}
        torch.save(checkpoint, output / "last.pt")
        if selected:
            torch.save(checkpoint, output / "best.pt")
        np.save(output / "loss.npy", np.asarray(losses))
        (output / "training.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Update {iteration}: nominal/recovery validation MSE {nominal_loss:.6f}/{recovery_loss:.6f}"
              + (f"; nominal rollout success {rollout['mean_final_success']:.1%}" if rollout else "")
              + (f"; final-ten success {rollout['mean_last_ten_success']:.1%}"
                 if rollout and selection_horizon == 150 else "")
              + (" [best]" if selected else ""), flush=True)

    evaluate_and_save(0)  # Keep the starting policy if fine-tuning does not improve selection metrics.
    progress = trange(1, total_iters + 1, desc="Fine-tuning diffusion policy")
    for iteration in progress:
        agent.train()
        batch = mixed_batch(datasets["nominal_train"], datasets["recovery_train"], batch_size,
                            recovery_fraction, batch_generator, device, balance_recovery_sources, grasp_indices,
                            learner_indices or None)
        loss = agent.compute_loss(batch["observations"], batch["actions"], action_mask=batch["action_mask"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        ema.step(agent.parameters())
        losses.append(float(loss.detach()))
        progress.set_postfix(loss=f"{losses[-1]:.5f}")
        if iteration % eval_freq == 0 or iteration == total_iters:
            evaluate_and_save(iteration)
    if checkpoint_path is not None and digest(checkpoint_path) != initial_hash:
        raise RuntimeError("Initial checkpoint changed during fine-tuning")
    for info in demos.values():
        for source in info.get("sources", [info]):
            if (digest(source["path"]) != source["sha256"]
                    or digest(Path(source["path"]).with_suffix(".json")) != source["metadata_sha256"]):
                raise RuntimeError("Input demonstrations changed during fine-tuning")
    record.update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat())
    (output / "training.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {output / 'best.pt'}; selected update {best_iteration} (0 means starting weights)", flush=True)
    return record
