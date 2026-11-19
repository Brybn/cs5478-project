"""Headroom pilot: how much does a smaller fixed chunk length K shorten time to goal?

Development data only (not a test cohort). Writes per-episode timing for fixed K
on fresh reset groups; the reset seeds are recorded so later plans exclude them.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from adaptive_action_chunking.monitor import _reset_seeds, reset_seed_sources, rollout_jobs


def confirmation(success, start, horizon):
    return next((t for t in range(start + 10, horizon + 1) if all(u in success for u in range(t - 9, t + 1))), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, nargs="+", default=[Path("checkpoints/pickcube/stage5_level/best.pt")],
                        help="several policies run on the same groups (rows carry the policy index)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/monitor_headroom"))
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--seed", type=int, default=329001)
    parser.add_argument("--chunks", type=int, nargs="+", default=[2, 4, 6, 8, 10, 15])
    parser.add_argument("--forces", type=float, nargs="+", default=[0.0, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--onset", type=int, nargs=2, default=[8, 18], help="push start step range (inclusive)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seen = set()
    for path in reset_seed_sources():
        seen.update(_reset_seeds(json.loads(path.read_text())) if path.suffix == ".json" else [])
    rng, tasks, groups = np.random.default_rng(args.seed), [], []
    for _ in range(args.groups):
        reset = int(rng.integers(10000, 2**31))
        while reset in seen:
            reset = int(rng.integers(10000, 2**31))
        seen.add(reset)
        group = {"group_id": f"reset_{reset}", "reset_seed": reset, "policy_seed": int(rng.integers(0, 2**31)),
                 "start_step": int(rng.integers(args.onset[0], args.onset[1] + 1)), "angle": float(rng.uniform(0, 2 * np.pi))}
        groups.append(group)
        for force in args.forces:
            for chunk in args.chunks:
                for p in range(len(args.policy)):
                    tasks.append({**group, "episode_id": f"{group['group_id']}_f{force:g}_k{chunk}_p{p}",
                                  "force_n": force, "checkpoint": str(p),
                                  "duration_steps": 2, "kind": "fixed", "threshold": None, "chunk_length": chunk,
                                  "force_xy": [force * np.cos(group["angle"]), force * np.sin(group["angle"])],
                                  "record_timing": True})
    (args.output_dir / "plan.json").write_text(json.dumps(
        {"seed": args.seed, "policy": [str(p) for p in args.policy], "horizon": args.horizon, "chunks": args.chunks,
         "forces": args.forces, "onset": args.onset, "groups": groups}, indent=1))
    rows = []
    for index, (task, episode) in enumerate(rollout_jobs(tasks, {str(i): p for i, p in enumerate(args.policy)}, args.workers, horizon=args.horizon), 1):
        success = set(episode["success_steps"])
        origin = task["start_step"] + task["duration_steps"] if task["force_n"] else 0
        confirmed = confirmation(success, 0, args.horizon)
        recovered = confirmation(success, origin, args.horizon)
        rows.append({"group_id": task["group_id"], "reset_seed": task["reset_seed"], "force_n": task["force_n"],
                     "chunk": task["chunk_length"], "policy": int(task["checkpoint"]), "calls": episode["policy_calls"],
                     "confirmed_step": confirmed, "recovered_step": recovered, "origin": origin,
                     "success_at_end": bool(episode["success_at_end"])})
        if index % 50 == 0:
            print(f"{index}/{len(tasks)}", flush=True)
    (args.output_dir / "results.json").write_text(json.dumps({"complete": True, "rows": rows}))
    print("done")


if __name__ == "__main__":
    main()
