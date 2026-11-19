"""Diffusion policy (conditional 1-D UNet adapted from ManiSkill's baseline) with DDPM or DDIM sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from .artifacts import resolve_artifact


# 1D UNet Architecture (adapted from Diffusion Policy baseline)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(x.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sample = sample.moveaxis(-1, -2)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


# Policy Wrapper
@dataclass(frozen=True)
class PolicyConfig:
    obs_horizon: int = 2
    pred_horizon: int = 16
    diffusion_steps: int = 100
    diffusion_step_embed_dim: int = 64
    unet_dims: tuple[int, ...] = (64, 128, 256)
    n_groups: int = 8
    sampler: str = "ddpm"
    inference_steps: int | None = None

    def __post_init__(self):
        if self.sampler not in ("ddpm", "ddim"):
            raise ValueError("Sampler must be ddpm or ddim")
        if self.inference_steps is not None and (
            type(self.inference_steps) is not int or not 1 <= self.inference_steps <= self.diffusion_steps
        ):
            raise ValueError("Inference steps must be an integer between 1 and the training noise levels")

    @property
    def max_execution_horizon(self) -> int:
        return self.pred_horizon - self.obs_horizon + 1


class ObservationNormalizer(nn.Module):
    """Frozen training statistics; legacy policies retain identity conditioning."""

    def __init__(self, obs_dim, stats=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.register_buffer("mean", None)
        self.register_buffer("std", None)
        self.set_stats(stats)

    @property
    def stats(self):
        return None if self.mean is None else (self.mean, self.std)

    def set_stats(self, stats):
        if stats is None:
            self.mean = self.std = None
            return
        mean, std = (torch.as_tensor(value, dtype=torch.float32).detach().clone() for value in stats)
        if (mean.shape != (self.obs_dim,) or std.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or not (std > 0).all()):
            raise ValueError("Observation statistics must be finite feature vectors with positive scales")
        self.mean, self.std = mean, std

    def forward(self, observations):
        return observations if self.mean is None else (observations - self.mean) / self.std


class BaselinePolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, config: PolicyConfig | None = None,
                 observation_stats=None):
        super().__init__()
        config = config or PolicyConfig()
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.observation_normalizer = ObservationNormalizer(obs_dim, observation_stats)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=act_dim,
            global_cond_dim=obs_dim * config.obs_horizon,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.unet_dims,
            n_groups=config.n_groups,
        )
        scheduler = DDIMScheduler if config.sampler == "ddim" else DDPMScheduler
        self.noise_scheduler = scheduler(
            num_train_timesteps=config.diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
            **({"timestep_spacing": "trailing"} if config.sampler == "ddim" else {}),
        )
        self.noise_scheduler.set_timesteps(config.inference_steps or config.diffusion_steps)

    @torch.no_grad()
    def sample(self, history: np.ndarray, generator: torch.Generator, device: torch.device) -> np.ndarray:
        config = self.config
        if history.shape != (config.obs_horizon, self.obs_dim):
            raise ValueError(f"History shape must be ({config.obs_horizon}, {self.obs_dim})")
        obs = torch.as_tensor(history[None], dtype=torch.float32, device=device)
        cond = self.observation_normalizer(obs).flatten(start_dim=1)
        actions = torch.randn(
            (1, config.pred_horizon, self.act_dim), generator=generator, device=device
        )
        start = config.obs_horizon - 1
        for timestep in self.noise_scheduler.timesteps:
            residual = self.noise_pred_net(
                sample=actions, timestep=timestep, global_cond=cond
            )
            actions = self.noise_scheduler.step(
                model_output=residual,
                timestep=timestep,
                sample=actions,
                generator=generator,
                **({"eta": 0.0} if config.sampler == "ddim" else {}),
            ).prev_sample
        return actions[0, start:].detach().cpu().numpy()


def load_policy(
    checkpoint: str | Path, obs_dim: int | None = None, act_dim: int = 4, device: torch.device | None = None,
    *, sampler: str | None = None, inference_steps: int | None = None,
) -> BaselinePolicy:
    """Load a trained BaselinePolicy checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(resolve_artifact(checkpoint), map_location=device, weights_only=True)
    weights = data.get("ema_agent", data) if isinstance(data, dict) else data
    sampling = data.get("config", {}) if isinstance(data, dict) else {}
    saved_sampler = sampling.get("sampler", "ddpm")
    sampler = saved_sampler if sampler is None else sampler
    if inference_steps is None and sampler == saved_sampler:
        inference_steps = sampling.get("inference_steps")
    keys = ("observation_normalizer.mean", "observation_normalizer.std")
    if any(key in weights for key in keys) and not all(key in weights for key in keys):
        raise ValueError("Checkpoint contains incomplete observation statistics")
    stats = tuple(weights[key] for key in keys) if all(key in weights for key in keys) else None
    config = PolicyConfig(sampler=sampler, inference_steps=inference_steps)
    if stats is not None:
        obs_dim = int(stats[0].shape[0])
    elif obs_dim is None:
        from .env import task

        obs_dim = task()[3]
    act_dim = int(weights["noise_pred_net.final_conv.1.bias"].shape[0])  # 4 for XYZ+gripper, 7 with rotation
    model = BaselinePolicy(obs_dim, act_dim, config, observation_stats=stats).to(device)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model
