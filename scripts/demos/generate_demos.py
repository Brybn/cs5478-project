"""Generate motion-planner demonstrations with ManiSkill's bundled Panda solutions, on CPU (WSL-safe).

Pass the normal `mani_skill.examples.motionplanning.panda.run` arguments, e.g.
  uv run --frozen --no-sync python scripts/demos/generate_demos.py -e LiftPegUpright-v1 -n 1000 --only-count-success \
      -b cpu --num-procs 12 --record-dir data/liftpeg_full --traj-name trajectory
"""

from __future__ import annotations

import os
from pathlib import Path
import platform

if "microsoft" in platform.release().lower() and Path("/usr/share/vulkan/icd.d/lvp_icd.json").exists():
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json")

import multiprocessing as mp

import gymnasium as gym
from mani_skill.examples.motionplanning.panda import run

original_make = gym.make


def cpu_render_make(*args, **kwargs):
    kwargs.setdefault("render_backend", "cpu")
    return original_make(*args, **kwargs)


gym.make = cpu_render_make
run.gym.make = cpu_render_make

if __name__ == "__main__":
    mp.set_start_method("spawn")
    run.main(run.parse_args())
