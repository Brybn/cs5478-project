"""Write an inference copy of a trained checkpoint that samples with 10-step DDIM instead of 100-step DDPM.

The EMA weights are unchanged; only the stored sampler config differs. `train_policy.py` saves DDPM100 sampling in
best.pt, and every monitor study uses DDIM10 (about ten times faster per call).

  uv run --frozen --no-sync python scripts/policy/ddim_copy.py checkpoints/stackcube/policy_120k/best.pt
  # -> checkpoints/stackcube/policy_120k/best_ddim10.pt
"""

import argparse
from pathlib import Path

import torch

from adaptive_action_chunking.artifacts import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="default: <checkpoint stem>_ddim10.pt beside the source")
    parser.add_argument("--inference-steps", type=int, default=10)
    args = parser.parse_args()
    output = args.output or args.checkpoint.with_name(f"{args.checkpoint.stem}_ddim{args.inference_steps}.pt")
    if output.exists():
        parser.error(f"{output} exists")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = {**source["config"], "sampler": "ddim", "inference_steps": args.inference_steps,
              "training_noise_steps": 100}
    torch.save({"config": config, "ema_agent": source["ema_agent"], "iteration": source["iteration"],
                "source_checkpoint": str(args.checkpoint), "source_checkpoint_sha256": digest(args.checkpoint),
                "note": f"Inference copy of {args.checkpoint.name} (EMA weights, update {source['iteration']}) "
                        f"with DDIM{args.inference_steps} sampling"}, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
