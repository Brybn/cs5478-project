"""Train the PickCube diffusion policy from replayed demonstration episodes."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import random

from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .env import extract_policy_state, make_env
from .policy import BaselinePolicy, ConditionalUnet1D, ObservationNormalizer, PolicyConfig


class DemoDataset(Dataset):
    """Dataset of (obs_history, action_chunk) slices loaded from an HDF5 demo file."""

    def __init__(
        self,
        h5_path: str | Path,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        num_traj: int | None = None,
        device: torch.device | None = None,
    ):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.device = device or torch.device("cpu")

        with h5py.File(h5_path, "r") as f:
            keys = sorted(f.keys(), key=lambda x: int(x.split("_")[-1]))
            if num_traj is not None:
                keys = keys[:num_traj]

            observations = [torch.tensor(f[k]["obs"][:], dtype=torch.float32, device=self.device) for k in keys]
            actions = [torch.tensor(f[k]["actions"][:], dtype=torch.float32, device=self.device) for k in keys]

        self.trajectories = {"observations": observations, "actions": actions}
        self.slices = []

        pad_before = obs_horizon - 1
        pad_after = pred_horizon - obs_horizon
        # Pad past the end with zero arm deltas (hold still): XYZ, plus rotation for 7-D pose actions.
        pad_action_arm = torch.zeros(actions[0].shape[-1] - 1, device=self.device)

        for traj_idx, act in enumerate(actions):
            length = len(act)
            for start in range(-pad_before, length - pred_horizon + pad_after):
                self.slices.append((traj_idx, start, start + pred_horizon))

        self.pad_action_arm = pad_action_arm

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        traj_idx, start, end = self.slices[index]
        obs = self.trajectories["observations"][traj_idx]
        acts = self.trajectories["actions"][traj_idx]
        length = len(acts)

        obs_seq = obs[max(0, start) : start + self.obs_horizon]
        act_seq = acts[max(0, start) : end]

        if start < 0:
            obs_seq = torch.cat([obs_seq[0:1].repeat(-start, 1), obs_seq], dim=0)
            act_seq = torch.cat([act_seq[0:1].repeat(-start, 1), act_seq], dim=0)

        if end > length:
            gripper = act_seq[-1:, -1]
            pad_action = torch.cat([self.pad_action_arm, gripper], dim=-1).unsqueeze(0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - length, 1)], dim=0)

        return {"observations": obs_seq, "actions": act_seq}


class DiffusionPolicyAgent(nn.Module):
    """Wraps ConditionalUnet1D with forward training loss."""

    def __init__(
        self,
        obs_dim: int = 42,
        act_dim: int = 4,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        diffusion_step_embed_dim: int = 64,
        unet_dims: tuple[int, ...] = (64, 128, 256),
        n_groups: int = 8,
        observation_stats=None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.observation_normalizer = ObservationNormalizer(obs_dim, observation_stats)

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=act_dim,
            global_cond_dim=obs_horizon * obs_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_dims,
            n_groups=n_groups,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def compute_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor, generator=None,
                     action_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = obs_seq.shape[0]
        obs_cond = self.observation_normalizer(obs_seq).flatten(start_dim=1)
        noise = torch.randn((b, self.pred_horizon, self.act_dim), device=obs_seq.device, generator=generator)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (b,), device=obs_seq.device, generator=generator
        ).long()

        noisy_act = self.noise_scheduler.add_noise(act_seq, noise, timesteps)
        pred = self.noise_pred_net(noisy_act, timesteps, global_cond=obs_cond)
        return denoising_mse(pred, noise, action_mask)


def denoising_mse(prediction, target, action_mask=None):
    """Average each sequence over its supervised actions, then average sequences."""
    if action_mask is None:
        return F.mse_loss(prediction, target)
    if action_mask.shape != prediction.shape[:2]:
        raise ValueError("Action mask must have shape (batch, prediction horizon)")
    mask = action_mask.to(device=prediction.device, dtype=prediction.dtype)
    if not torch.all((mask == 0) | (mask == 1)) or torch.any(mask.sum(1) == 0):
        raise ValueError("Action masks must be binary with at least one supervised action per sequence")
    errors = (prediction - target).square() * mask[..., None]
    return (errors.sum((1, 2)) / (mask.sum(1) * prediction.shape[-1])).mean()


@torch.inference_mode()
def dataset_loss(agent, dataset, seed=8641, draws=4, batch_size=64):
    """Fixed noise and timesteps make EMA losses comparable across checkpoints."""
    agent.eval()
    device = next(agent.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        observations = torch.stack([item["observations"] for item in batch])
        actions = torch.stack([item["actions"] for item in batch])
        for _ in range(draws):
            total += agent.compute_loss(observations, actions, generator).item() * len(batch) / draws
    return total / len(dataset)


def evaluate_policy(agent, cases, chunk_lengths=(8, 15), sampler="ddpm", inference_steps=None, horizon=100):
    """Use the shared rollout implementation without changing training's RNG."""
    from .evaluate import run_episode

    device = next(agent.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    records = []
    with torch.random.fork_rng(devices=devices):
        policy = BaselinePolicy(agent.obs_dim, agent.act_dim,
                                PolicyConfig(obs_horizon=agent.obs_horizon, pred_horizon=agent.pred_horizon,
                                             sampler=sampler, inference_steps=inference_steps),
                                observation_stats=agent.observation_normalizer.stats).to(device)
        policy.load_state_dict(agent.state_dict())
        policy.eval()
        env = make_env(max_episode_steps=horizon)
        try:
            obs, _ = env.reset(seed=cases[0]["reset_seed"])
            state = extract_policy_state(obs)
            policy.sample(np.stack([state] * agent.obs_horizon), torch.Generator(device=device).manual_seed(0), device)
            for case in cases:
                for k in chunk_lengths:
                    episode = run_episode(env, policy, k, case["reset_seed"], device,
                                          policy_seed=case["policy_seed"], horizon=horizon)
                    if episode["steps"] != horizon:
                        raise RuntimeError(f"Training validation must reach step {horizon}")
                    successes = set(episode["success_steps"])
                    records.append({**case, "chunk_length": k,
                                    "last_ten_success": all(step in successes for step in range(horizon-9, horizon+1)),
                                    "success_at_100": 100 in successes,
                                    **{key: episode[key] for key in ("steps", "success_once", "success_at_end",
                                                                     "first_success_step", "policy_calls")}})
        finally:
            env.close()
    return {"mean_final_success": float(np.mean([record["success_at_end"] for record in records])),
            "mean_last_ten_success": float(np.mean([record["last_ten_success"] for record in records])),
            "mean_success_at_100": float(np.mean([record["success_at_100"] for record in records])),
            "by_chunk": {str(k): {key: float(np.mean([record[key] for record in records if record["chunk_length"] == k]))
                                   for key in ("success_at_end", "success_once", "policy_calls", "last_ten_success", "success_at_100")}
                         for k in chunk_lengths},
            "horizon": horizon,
            "sampler": sampler, "inference_steps": len(policy.noise_scheduler.timesteps),
            "episodes": records}


def selection_cases(metadata_paths, count, seed):
    """Reserve resets independently of demonstrations and previous experiments."""
    from .monitor import _reset_seeds, reset_seed_sources

    sources = set(metadata_paths) | set(reset_seed_sources())
    excluded = set()
    for path in sorted(sources):
        values = [json.loads(line) for line in path.read_text().splitlines() if line] if path.suffix == ".jsonl" else [json.loads(path.read_text())]
        for value in values:
            excluded.update(_reset_seeds(value))
    excluded_count = len(excluded)
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(count):
        reset = int(rng.integers(10000, 2**31))
        while reset in excluded:
            reset = int(rng.integers(10000, 2**31))
        excluded.add(reset)
        cases.append({"reset_seed": reset, "policy_seed": int(rng.integers(0, 2**31))})
    return cases, excluded_count


def train_policy(
    demo_path: str | Path = "data/split/train.h5",
    total_iters: int = 30000,
    batch_size: int = 32,
    lr: float = 1e-4,
    obs_horizon: int = 2,
    pred_horizon: int = 16,
    eval_freq: int = 5000,
    num_eval_episodes: int = 25,
    output_dir: str | Path = "checkpoints/pickcube/stage0_nominal",
    seed: int = 1,
    cuda: bool = True,
    validation_path: str | Path | None = None,
    validation_seed: int = 8641,
    validation_draws: int = 4,
    eval_seed: int = 9601,
    eval_chunk_lengths: tuple[int, ...] = (8, 15),
    eval_horizon: int = 100,
) -> None:
    """Train with held-out demo loss and independent full-length rollout selection."""
    if min(total_iters, eval_freq, num_eval_episodes, validation_draws) < 1:
        raise ValueError("Iterations, evaluation frequency, episode count and loss draws must be positive")
    demo_path = Path(demo_path)
    validation_path = Path(validation_path) if validation_path else demo_path.with_name("validation.h5")
    metadata_paths = [path.with_suffix(".json") for path in (demo_path, validation_path)]
    metadata = [json.loads(path.read_text()) for path in metadata_paths]
    demo_seeds = [{episode["episode_seed"] for episode in value["episodes"]} for value in metadata]
    if demo_seeds[0] & demo_seeds[1]:
        raise ValueError("Training and validation demonstrations must use disjoint reset seeds")
    cases, excluded_count = selection_cases(metadata_paths, num_eval_episodes, eval_seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=False)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    print(f"Using device: {device}")

    print(f"Loading demos from: {demo_path}")
    dataset = DemoDataset(demo_path, obs_horizon=obs_horizon, pred_horizon=pred_horizon, device=device)
    validation = DemoDataset(validation_path, obs_horizon=obs_horizon, pred_horizon=pred_horizon, device=device)
    print(f"Dataset slices: {len(dataset)} train, {len(validation)} validation")
    if not len(dataset) or not len(validation):
        raise ValueError("Both demonstration splits need usable action chunks")

    obs_dim = dataset[0]["observations"].shape[-1]
    act_dim = dataset[0]["actions"].shape[-1]
    print(f"Observations: {obs_dim}")
    print(f"Actions: {act_dim}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    agent = DiffusionPolicyAgent(
        obs_dim=obs_dim, act_dim=act_dim, obs_horizon=obs_horizon, pred_horizon=pred_horizon
    ).to(device)

    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = DiffusionPolicyAgent(
        obs_dim=obs_dim, act_dim=act_dim, obs_horizon=obs_horizon, pred_horizon=pred_horizon
    ).to(device)

    optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, betas=(0.95, 0.999), weight_decay=1e-6)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=500, num_training_steps=total_iters)

    config = {"seed": seed, "total_iters": total_iters, "batch_size": batch_size, "lr": lr,
              "obs_horizon": obs_horizon, "pred_horizon": pred_horizon, "eval_freq": eval_freq,
              "num_eval_episodes": num_eval_episodes, "eval_seed": eval_seed,
              "eval_chunk_lengths": list(eval_chunk_lengths), "eval_horizon": eval_horizon, "validation_seed": validation_seed,
              "validation_draws": validation_draws, "validation_batch_size": 64, "device": str(device),
              "optimizer": "AdamW", "betas": [0.95, 0.999], "weight_decay": 1e-6,
              "lr_schedule": "cosine", "warmup_steps": 500, "ema_power": 0.75}
    demos = {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "metadata_sha256": hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest(),
                    "episodes": len(value["episodes"]), "windows": len(split)}
             for name, path, value, split in zip(("train", "validation"), (demo_path, validation_path),
                                                metadata, (dataset, validation))}
    selection_rule = "Highest mean step-100 success across chunk lengths; ties: lower held-out EMA loss, then earliest update"
    record = {"complete": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "demos": demos, "selection_rule": selection_rule,
              "selection_cases": cases, "excluded_previous_reset_seeds": excluded_count, "evaluations": []}
    record_path = out_path / "training.json"
    record_path.write_text(json.dumps(record, indent=2))

    iteration = 0
    pbar = tqdm(total=total_iters, desc="Training Diffusion Policy")
    best_key = None
    losses = []
    while iteration < total_iters:
        for batch in loader:
            if iteration >= total_iters:
                break

            agent.train()
            loss = agent.compute_loss(batch["observations"], batch["actions"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ema.step(agent.parameters())

            iteration += 1
            pbar.update(1)
            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if iteration % eval_freq == 0 or iteration == total_iters:
                ema.copy_to(ema_agent.parameters())
                train_loss = dataset_loss(ema_agent, dataset, validation_seed, validation_draws)
                validation_loss = dataset_loss(ema_agent, validation, validation_seed, validation_draws)
                rollout = evaluate_policy(ema_agent, cases, eval_chunk_lengths, horizon=eval_horizon)
                success_rate = rollout["mean_final_success"]
                key = (success_rate, -validation_loss)
                selected = best_key is None or key > best_key
                record["evaluations"].append({"iteration": iteration, "ema_train_loss": train_loss,
                                              "ema_validation_loss": validation_loss,
                                              "rollout": rollout, "selected_best": selected})
                print(f"\n[Iter {iteration}] Final success: {success_rate:.1%}; "
                      f"EMA loss train/validation: {train_loss:.6f}/{validation_loss:.6f}")
                checkpoint = {"agent": agent.state_dict(), "ema_agent": ema_agent.state_dict(),
                              "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                              "lr_scheduler": lr_scheduler.state_dict(), "iteration": iteration,
                              "success_rate": success_rate, "success_metric": "mean_step_100_success",
                              "validation_loss": validation_loss, "rollout": rollout,
                              "config": config, "demos": demos, "selection_rule": selection_rule}
                torch.save(checkpoint, out_path / f"step_{iteration:06}.pt")
                if selected:
                    best_key = key
                    record["best_iteration"] = iteration
                    torch.save(checkpoint, out_path / "best.pt")
                np.save(out_path / "loss.npy", np.array(losses))
                record_path.write_text(json.dumps(record, indent=2))

    pbar.close()
    record.update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat())
    record_path.write_text(json.dumps(record, indent=2))
    print(f"\nTraining complete. Best final success: {best_key[0]:.1%} at update {record['best_iteration']}")
