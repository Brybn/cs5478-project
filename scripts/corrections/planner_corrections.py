"""Planner-teacher corrections for LiftPegUpright.

Two steps in two venvs, because ManiSkill's planner needs mplib 0.1.1 + numpy<2:

  states (main venv, ROBOTS_TASK set): roll the policy out with a push and stop `delay` steps after it;
      save the full simulator state and the policy's last two observations (takeover context).
  teach  (.venv-mp): restore each state, run ManiSkill's planner solution from there (no reset), convert its
      joint-position actions to pd_ee_delta_pose, and keep successful continuations as correction segments in
      the finetune_policy.py format (conditioning "expert_states").
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ENV_IDS = {"LiftPegUpright": "LiftPegUpright-v1"}


def states(args):
    import torch
    from adaptive_action_chunking.monitor import _reset_seeds, reset_seed_sources
    from adaptive_action_chunking.env import make_env, task
    from adaptive_action_chunking.evaluate import run_episode
    from adaptive_action_chunking.policy import load_policy

    args.output_dir.mkdir(parents=True, exist_ok=False)
    seen = set()
    for path in reset_seed_sources():
        seen.update(_reset_seeds(json.loads(path.read_text())) if path.suffix == ".json" else [])
    rng = np.random.default_rng(args.seed)
    device = torch.device("cpu")
    torch.set_num_threads(4)
    policy = load_policy(args.policy, device=device)
    env = make_env(max_episode_steps=args.onset[1] + 2 + args.delay[1] + 1,
                   disturbance={"start_step": 0, "duration_steps": 0, "force_xy": [0.0, 0.0]})
    rows, sim_states = [], []
    for index in range(args.count):
        reset = int(rng.integers(10000, 2**31))
        while reset in seen:
            reset = int(rng.integers(10000, 2**31))
        seen.add(reset)
        force = args.forces[index % len(args.forces)]
        angle = float(rng.uniform(0, 2 * np.pi))
        row = {"candidate_index": index, "reset_seed": reset, "policy_seed": int(rng.integers(0, 2**31)),
               "start_step": int(rng.integers(args.onset[0], args.onset[1] + 1)), "duration_steps": 2,
               "force_n": force, "force_xy": [force * np.cos(angle), force * np.sin(angle)],
               "chunk_length": args.chunks[(index // len(args.forces)) % len(args.chunks)],
               "delay": int(rng.integers(args.delay[0], args.delay[1] + 1)),
               "split": "validation" if rng.random() < args.validation_fraction else "train"}
        row["takeover_step"] = row["start_step"] + 2 + row["delay"]
        env.unwrapped.disturbance = {key: row[key] for key in ("start_step", "duration_steps", "force_xy")}
        episode = run_episode(env, policy, row["chunk_length"], reset, device, horizon=row["takeover_step"],
                              policy_seed=row["policy_seed"])
        if episode["steps"] != row["takeover_step"]:
            continue  # ended early (task failure flag); nothing to correct from
        context = episode["states"][-2:]
        row["context_obs"] = context
        rows.append(row)
        sim_states.append(env.unwrapped.get_state().cpu().numpy()[0])
    np.save(args.output_dir / "sim_states.npy", np.stack(sim_states))
    (args.output_dir / "metadata.json").write_text(json.dumps(
        {"task": task()[0], "policy": str(args.policy), "seed": args.seed, "forces": args.forces,
         "onset": args.onset, "delay": args.delay, "chunks": args.chunks, "candidates": rows}, indent=1))
    print(f"{len(rows)} takeover states saved")


def teach(args):
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json")
    import gymnasium as gym
    import h5py
    import mani_skill.envs  # noqa: F401
    from mani_skill.examples.motionplanning.panda.solutions import lift_peg_upright
    from mani_skill.trajectory.utils.actions.conversion import from_pd_joint_pos

    plan = json.loads((args.states_dir / "metadata.json").read_text())
    sim_states = np.load(args.states_dir / "sim_states.npy")
    env_id = ENV_IDS[plan["task"]]
    solve = {"LiftPegUpright-v1": lift_peg_upright.solve}[env_id]
    kwargs = dict(render_backend="cpu", sim_backend="physx_cpu", max_episode_steps=10_000)
    joint = gym.make(env_id, obs_mode="none", control_mode="pd_joint_pos", **kwargs)
    pose = gym.make(env_id, obs_mode="state", control_mode="pd_ee_delta_pose", **kwargs)
    real_reset = joint.reset

    def restore(env, row, sim_state):
        env.reset(seed=row["reset_seed"])
        env.unwrapped.set_state(sim_state[None])
        return env.unwrapped.get_obs()

    results = {"train": [], "validation": []}
    log = []
    for row, sim_state in zip(plan["candidates"], sim_states):
        restore(joint, row, sim_state)
        joint_actions = []
        step = joint.step
        joint.step = lambda action: (joint_actions.append(np.asarray(action, dtype=np.float32)), step(action))[1]
        joint.reset = lambda *a, **k: None  # solve() resets first; start from the restored state instead
        try:
            status = solve(joint, seed=None)
        except RuntimeError:  # mplib occasionally fails to time-parameterize a path
            status = -1
        finally:
            joint.step, joint.reset = step, real_reset
        planned = status != -1 and bool(joint.unwrapped.evaluate()["success"].item())
        entry = {"candidate_index": row["candidate_index"], "planner_success": planned, "joint_steps": len(joint_actions)}
        if planned:
            restore(joint, row, sim_state)
            obs0 = np.asarray(restore(pose, row, sim_state)).reshape(-1)
            entry["context_error"] = float(np.max(np.abs(obs0 - np.asarray(row["context_obs"][-1]))))
            obs, actions, success = [obs0], [], []
            pose_step = pose.step

            def record(action):
                out = pose_step(action)
                actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
                obs.append(np.asarray(out[0]).reshape(-1))
                success.append(bool(out[4]["success"].item()))
                return out
            pose.step = record
            try:
                from_pd_joint_pos("pd_ee_delta_pose", joint_actions, joint, pose)
            finally:
                pose.step = pose_step
            entry.update(converted_steps=len(actions), converted_success=bool(success and success[-1]))
            if entry["converted_success"] and entry["context_error"] < 1e-4 and len(actions) <= args.max_steps:
                obs = np.stack(obs).astype(np.float32)
                obs[0] = np.asarray(row["context_obs"][-1], dtype=np.float32)  # exact learner state at takeover
                results[row["split"]].append((row, obs, np.stack(actions)))
        log.append(entry)
        print(entry, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split, items in results.items():
        episodes = []
        with h5py.File(args.output_dir / f"{split}.h5", "w") as handle:
            for index, (row, obs, actions) in enumerate(items):
                group = handle.create_group(f"traj_{index}")
                group.attrs["conditioning"] = "expert_states"
                group["obs"], group["actions"] = obs, actions
                group["context_obs"] = np.asarray(row["context_obs"], dtype=np.float32)
                group["action_is_expert"] = np.ones(len(actions), dtype=np.uint8)
                episodes.append({**{k: v for k, v in row.items() if k != "context_obs"}, "trajectory": f"traj_{index}",
                                 "episode_seed": row["reset_seed"], "teacher": "maniskill_planner",
                                 "steps": len(actions)})
        (args.output_dir / f"{split}.json").write_text(json.dumps({"complete": True, "split": split, "episodes": episodes}, indent=1))
    kept = {split: len(items) for split, items in results.items()}
    (args.output_dir / "metadata.json").write_text(json.dumps(
        {"mode": "planner_takeover", "complete": True, "states_dir": str(args.states_dir), "env_id": env_id,
         "control_mode": "pd_ee_delta_pose", "max_steps": args.max_steps, "kept": kept, "attempts": log,
         **{k: plan[k] for k in ("policy", "seed", "forces", "onset", "delay", "chunks")}}, indent=1))
    print(f"planner success {sum(e['planner_success'] for e in log)}/{len(log)}; kept {kept}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("states")
    s.add_argument("--policy", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    s.add_argument("--count", type=int, default=400)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--forces", type=float, nargs="+", required=True)
    s.add_argument("--onset", type=int, nargs=2, default=[8, 18])
    s.add_argument("--delay", type=int, nargs=2, default=[2, 10], help="takeover steps after the push ends")
    s.add_argument("--chunks", type=int, nargs="+", default=[8, 15])
    s.add_argument("--validation-fraction", type=float, default=0.15)
    t = sub.add_parser("teach")
    t.add_argument("--states-dir", type=Path, required=True)
    t.add_argument("--output-dir", type=Path, required=True)
    t.add_argument("--max-steps", type=int, default=250, help="drop longer planner continuations (detours)")
    args = parser.parse_args()
    states(args) if args.command == "states" else teach(args)


if __name__ == "__main__":
    main()
