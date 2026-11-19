"""Run ManiSkill replay with software rendering and fresh PickCube scenes.

Pass the normal replay_trajectory CLI arguments after this script name. CPU
physics and control-mode conversion use the official replay implementation.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
import sys

# In WSL2, point Vulkan to Mesa software rasterizer before importing sapien/mani_skill
if "microsoft" in platform.release().lower():
    mesa_icd = Path("/usr/share/vulkan/icd.d/lvp_icd.json")
    if mesa_icd.exists():
        os.environ.setdefault("VK_ICD_FILENAMES", str(mesa_icd))

import multiprocessing as mp
import gymnasium as gym
from mani_skill.trajectory import replay_trajectory


original_make = gym.make


def cpu_render_make(*args, **kwargs):
    kwargs.setdefault("render_backend", "cpu")
    if (args[0] if args else kwargs.get("id")) == "PickCube-v1":
        kwargs.setdefault("reconfiguration_freq", 1)
    return original_make(*args, **kwargs)


gym.make = cpu_render_make

if __name__ == "__main__":
    mp.set_start_method("spawn")
    argv = sys.argv[1:] or [
        "--traj-path", "data/demos/PickCube-v1/motionplanning/trajectory.h5",
        "--use-first-env-state",
        "-c", "pd_ee_delta_pos",
        "-o", "state",
        "--save-traj",
        "-b", "physx_cpu",
        "--count", "100",
    ]
    replay_trajectory.main(replay_trajectory.parse_args(argv))
