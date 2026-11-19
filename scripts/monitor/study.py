"""Collect and train a dynamics monitor, then validate and test its triggers."""

import argparse
import json
from pathlib import Path

from adaptive_action_chunking.artifacts import resolve_artifact
from adaptive_action_chunking.monitor import collect_transitions, plan_collection, train_dynamics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="run", choices=("run", "plan", "collect", "train", "validate", "test"),
                        help="Default run collects and trains; other stages reuse the selected dataset")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/monitor"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/pickcube/monitor"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/monitor"))
    parser.add_argument("--collection-seed", type=int, default=9701)
    parser.add_argument("--training-seed", type=int, default=4278)
    parser.add_argument("--horizon", type=int, default=None,
                        help="Episode length for a new plan (default 100); later stages use its recorded horizon")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage in ("run", "plan"):
        args.policy = args.policy or Path("checkpoints/pickcube/stage0_nominal/best.pt")
        horizon = args.horizon if args.horizon is not None else 100
        if horizon < 1:
            parser.error("--horizon must be positive")
    else:
        meta = json.loads((args.dataset_dir / "metadata.json").read_text())
        if args.horizon is not None and args.horizon != meta["horizon"]:
            parser.error("--horizon differs from the recorded collection plan")
        args.policy = resolve_artifact(args.policy or meta["policy"], meta["policy_sha256"])
    if args.stage in ("run", "plan"):
        print(f"Monitor collection policy: {args.policy}", flush=True)
        plan_collection(args.policy, args.dataset_dir, args.collection_seed, horizon=horizon)
        if args.stage == "plan":
            return
    if args.stage in ("run", "collect"):
        collect_transitions(args.policy, args.dataset_dir, args.workers, args.collection_seed)
    if args.stage in ("run", "train"):
        train_dynamics(args.dataset_dir, args.output_dir, args.training_seed)
    elif args.stage in ("validate", "test"):
        from adaptive_action_chunking.evaluate import compare_dynamics

        compare_dynamics(args.policy, args.output_dir / "predictor.pt", args.dataset_dir,
                         args.results_dir, args.stage, args.workers)


if __name__ == "__main__":
    main()
