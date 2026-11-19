"""Collect scripted corrective demonstrations for recovery fine-tuning.

Collect 32 training + 8 validation corrections from fresh, disjoint reset groups:
    .venv/bin/python scripts/corrections/collect_recovery_demos.py --workers 8

Accepted corrections are balanced over the requested source force levels.
"""

import argparse
from pathlib import Path

from adaptive_action_chunking.corrections.collect import FORCES, collect


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", type=Path, default=Path("checkpoints/pickcube/stage0_nominal/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/recovery"))
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=90271)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--forces", type=float, nargs="+", default=list(FORCES),
                        help="Source pulse magnitudes in N; counts must be divisible by the number of levels")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--teacher", choices=("placement", "pickup"), default="placement")
    parser.add_argument("--pilot-results", type=Path,
                        help="For the pickup teacher, replay representative saved comparison cases as debugging examples")
    args = parser.parse_args()
    if args.pilot_results and args.teacher != "pickup":
        parser.error("--pilot-results requires --teacher pickup")
    if args.output_dir.exists():
        parser.error(f"Choose a fresh output directory: {args.output_dir}")
    collect(
        args.output_dir,
        args.policy,
        args.train_count,
        args.validation_count,
        args.seed,
        args.workers,
        record_video=args.record_video,
        forces=args.forces,
        teacher=args.teacher,
        pilot_results=args.pilot_results,
    )


if __name__ == "__main__":
    main()
