"""Task setup (PickCube, StackCube, LiftPegUpright), controlled pushes and the policy observation."""

from __future__ import annotations

import os

import gymnasium as gym
import mani_skill.envs
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.lift_peg_upright import LiftPegUprightEnv
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.common import flatten_state_dict
from mani_skill.utils.registration import register_env
import numpy as np


class PushMixin:
    """Push the task object with a known horizontal force at absolute control-step indices.

    `disturbance` = {"start_step", "duration_steps", "force_xy"}; the force acts on every physics substep of the
    selected control steps, and the applied steps/substeps are recorded for auditing.
    """

    pushed_object = "cube"

    def __init__(self, *args, disturbance: dict, **kwargs):
        self.disturbance = disturbance
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx, options):
        self.applied_force_steps = set()
        self.applied_force_substeps = 0
        super()._initialize_episode(env_idx, options)

    def _before_simulation_step(self):
        step = int(self._elapsed_steps.item())
        force_xy = self.disturbance["force_xy"]
        start = self.disturbance["start_step"]
        duration = self.disturbance["duration_steps"]
        if any(force_xy) and start <= step < start + duration:
            getattr(self, self.pushed_object).apply_force(np.asarray([*force_xy, 0.0], dtype=np.float32))
            self.applied_force_steps.add(step)
            self.applied_force_substeps += 1


@register_env("PushedPickCube-v1", max_episode_steps=100)
class PushedPickCubeEnv(PushMixin, PickCubeEnv):
    pushed_object = "cube"


@register_env("PushedStackCube-v1", max_episode_steps=100)
class PushedStackCubeEnv(PushMixin, StackCubeEnv):
    pushed_object = "cubeA"  # the cube that is picked up and placed


@register_env("PushedLiftPegUpright-v1", max_episode_steps=100)
class PushedLiftPegUprightEnv(PushMixin, LiftPegUprightEnv):
    pushed_object = "peg"


# Task selection for the shared pipeline: ROBOTS_TASK=StackCube|LiftPegUpright switches env/state/monitor features.
# name: (env id, disturbed env id, state size, disturbed-object pose key)
TASKS = {"PickCube": ("PickCube-v1", "PushedPickCube-v1", 42, "obj_pose"),
         "StackCube": ("StackCube-v1", "PushedStackCube-v1", 48, "cubeA_pose"),
         "LiftPegUpright": ("LiftPegUpright-v1", "PushedLiftPegUpright-v1", 32, "obj_pose")}
# Position-only control suffices for the cube tasks; standing the peg up needs wrist rotation (7-D actions).
CONTROL_MODES = {"LiftPegUpright": "pd_ee_delta_pose"}


def task():
    name = os.environ.get("ROBOTS_TASK", "PickCube")
    if name not in TASKS:
        raise ValueError(f"Unknown ROBOTS_TASK {name!r}; use one of {sorted(TASKS)}")
    return name, *TASKS[name]


# State-vector slices of (disturbed cube XYZ, tool XYZ); used to check monitor positions against the policy input.
MONITOR_STATE_SLICES = {"PickCube": ((29, 32), (19, 22)), "StackCube": ((25, 28), (18, 21)),
                        "LiftPegUpright": ((25, 28), (18, 21))}


def make_env(
    render: bool = False,
    sim_backend: str = "physx_cpu",
    render_backend: str | None = None,
    max_episode_steps: int = 100,
    disturbance: dict | None = None,
    human_render_camera_configs: dict | None = None,
    enable_shadow: bool = False,
) -> gym.Env:
    """Create the selected task (default PickCube); delta end-effector control (see CONTROL_MODES)."""
    if type(max_episode_steps) is not int or max_episode_steps < 1:
        raise ValueError("Episode limit must be a positive integer")
    return gym.make(
        task()[2] if disturbance is not None else task()[1],
        obs_mode="state_dict",
        control_mode=CONTROL_MODES.get(task()[0], "pd_ee_delta_pos"),
        sim_backend=sim_backend,
        render_mode="rgb_array" if render else None,
        render_backend=render_backend or "cpu",
        max_episode_steps=max_episode_steps,
        reconfiguration_freq=1,
        human_render_camera_configs=human_render_camera_configs or {},
        enable_shadow=enable_shadow,
        **({"disturbance": disturbance} if disturbance is not None else {}),
    )


def extract_policy_state(obs: dict) -> np.ndarray:
    """Keep ManiSkill's flattening order used by the replayed demonstrations."""
    state = flatten_state_dict(obs, use_torch=True)
    state = np.asarray(state.detach().cpu(), dtype=np.float32).reshape(-1)
    name, _, _, dim, _ = task()
    if state.shape != (dim,):
        raise ValueError(f"Expected {dim} {name} state features, got {state.shape}")
    return state


def extract_monitor_positions(obs: dict) -> np.ndarray:
    """Disturbed cube XYZ followed by tool-centre XYZ, in metres."""
    return np.concatenate([
        np.asarray(obs["extra"][key].detach().cpu(), dtype=np.float32).reshape(-1)[:3]
        for key in (task()[4], "tcp_pose")
    ])
