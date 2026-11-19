"""Evaluate a single policy on nominal rollouts or compare named policies on matched disturbance scenarios."""

import argparse
from pathlib import Path

import torch

from adaptive_action_chunking.evaluate import compare_policy_checkpoints, evaluate_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", dest="policy_path", type=Path, default=Path("checkpoints/pickcube/stage0_nominal/best.pt"))
    parser.add_argument("--output", "--checkpoint-results", dest="output", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--chunk-length", type=int, default=8)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--compare-policies",
        nargs="+",
        metavar="NAME=PATH",
        help="Compare explicitly named policies on matched K8/K15 force scenarios",
    )
    parser.add_argument("--baseline", help="Baseline name for --compare-policies (default: first policy)")
    parser.add_argument("--groups", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(1)

    if args.compare_policies:
        checkpoints = {}
        for item in args.compare_policies:
            name, separator, path = item.partition("=")
            if not separator or not path or name in checkpoints:
                parser.error("Use distinct NAME=PATH entries for --compare-policies")
            checkpoints[name] = Path(path)
        if len(checkpoints) < 2:
            parser.error("Supply at least two policies for --compare-policies")
        if args.groups < 1:
            parser.error("Groups must be positive")
        if args.device != "cpu":
            parser.error("Policy comparison uses CPU inference and fixed K8/K15")
        seed = 19001 if args.seed is None else args.seed
        output = args.output or Path("results/policy_comparison")
        compare_policy_checkpoints(output, checkpoints, groups=args.groups, seed=seed,
                                   workers=args.workers, baseline=args.baseline)
        return

    if args.baseline:
        parser.error("--baseline requires --compare-policies")
    seed = 1000 if args.seed is None else args.seed
    output = args.output or Path("results/nominal.json")
    if min(args.episodes, args.horizon) < 1 or seed < 0:
        parser.error("Episodes and horizon must be positive; seed must be nonnegative")
    if not 1 <= args.chunk_length <= 15:
        parser.error("Chunk length must be between 1 and 15")
    if not args.policy_path.is_file():
        parser.error(f"Policy checkpoint not found: {args.policy_path}")
    if output.exists():
        parser.error(f"Choose a fresh output path: {output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")

    evaluate_policy(
        policy_path=args.policy_path,
        output=output,
        episodes=args.episodes,
        chunk_length=args.chunk_length,
        seed=seed,
        horizon=args.horizon,
        device=args.device,
        record_video=args.record_video,
    )


if __name__ == "__main__":
    main()
