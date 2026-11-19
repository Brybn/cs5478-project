"""Fine-tune the official policy on mixed nominal and corrective demonstrations.

Default: 5,000 updates, learning rate 1e-5, batches with 75% original and 25%
recovery windows. Keeps separate validation losses and nominal K8/K15 rollout
selection with DDIM10. The initial policy is also a checkpoint-selection candidate.

    uv run --frozen --no-sync python scripts/policy/finetune_policy.py
    uv run --frozen --no-sync python scripts/policy/finetune_policy.py --recovery-fraction 0.5 --lr 1e-5

Always choose a fresh output directory. Model training runs only when invoked.
Only best.pt and last.pt weights are retained; training.json records every evaluation.
Tagged learner-query datasets automatically receive half the recovery draws;
the other half uses all accumulated recovery windows.
--focus-latest-queries dedicates those query slots to the last --recovery-dir.
Use --from-scratch for a new mixed-data policy without a checkpoint initializer.
Its defaults are 30,000 updates, lr 1e-4, 500-step warmup and checks every 5,000 updates.
"""

import argparse
from pathlib import Path

from adaptive_action_chunking.finetune import fine_tune_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--init-checkpoint", type=Path, default=Path("checkpoints/pickcube/stage0_nominal/best.pt"))
    initialization.add_argument("--from-scratch", action="store_true",
                                help="Initialize a fresh mixed-data policy instead of loading a checkpoint")
    parser.add_argument("--nominal-dir", type=Path, default=Path("data/split"))
    parser.add_argument("--recovery-dir", type=Path, nargs="+", default=[Path("data/recovery")],
                        help="One or more correction round directories; train/validation partitions stay separate")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/pickcube/stage1_placement"))
    parser.add_argument("--total-iters", type=int, help="Default: 5,000 for fine-tuning; 30,000 from scratch")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recovery-fraction", type=float, default=.25)
    parser.add_argument("--balance-recovery-sources", action="store_true",
                        help="Give each correction directory equal batch counts, independent of its dataset size")
    parser.add_argument("--grasp-focused", action="store_true",
                        help="Draw half the recovery samples from recorded alignment/closure/lift phases")
    parser.add_argument("--focus-latest-queries", action="store_true",
                        help="Use the last correction directory for dedicated query slots; keep earlier rounds in pooled recovery")
    parser.add_argument("--normalize-observations", action="store_true",
                        help="Fit training-only observation z-scores and save them for policy inference")
    parser.add_argument("--lr", type=float, help="Default: 1e-5 for fine-tuning; 1e-4 from scratch")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddim",
                        help="Rollout-evaluation sampler; training retains 100 noise levels")
    parser.add_argument("--inference-steps", type=int,
                        help="Denoising passes per plan (default: 10 for DDIM, 100 for DDPM)")
    parser.add_argument("--selection-horizon", type=int, default=100,
                        help="150 selects nominal final-ten stability, then endpoint success; any other value selects endpoint success")
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Choose a fresh output directory: {args.output_dir}")
    fine_tune_policy(
        init_checkpoint=None if args.from_scratch else args.init_checkpoint,
        nominal_train=args.nominal_dir / "train.h5",
        nominal_validation=args.nominal_dir / "validation.h5",
        recovery_train=[directory / "train.h5" for directory in args.recovery_dir],
        recovery_validation=[directory / "validation.h5" for directory in args.recovery_dir],
        output_dir=args.output_dir,
        total_iters=args.total_iters if args.total_iters is not None else 30000 if args.from_scratch else 5000,
        batch_size=args.batch_size,
        recovery_fraction=args.recovery_fraction,
        balance_recovery_sources=args.balance_recovery_sources,
        grasp_focused=args.grasp_focused,
        focus_latest_queries=args.focus_latest_queries,
        normalize_observations=args.normalize_observations,
        lr=args.lr if args.lr is not None else 1e-4 if args.from_scratch else 1e-5,
        warmup_steps=500 if args.from_scratch else 100,
        eval_freq=5000 if args.from_scratch else 1000,
        seed=args.seed,
        sampler=args.sampler,
        inference_steps=args.inference_steps,
        selection_horizon=args.selection_horizon,
        cuda=args.cuda,
    )


if __name__ == "__main__":
    main()
