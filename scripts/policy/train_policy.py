"""Train a PickCube diffusion policy from state-based demonstrations."""

import argparse
from pathlib import Path

from adaptive_action_chunking.train import train_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-path", type=Path, default=Path("data/split/train.h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/pickcube/stage0_nominal"))
    parser.add_argument("--total-iters", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-horizon", type=int, default=100, help="Selection rollout length (StackCube: 150)")
    parser.add_argument("--num-eval-episodes", type=int, default=25)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    args = parser.parse_args()
    train_policy(**vars(args))


if __name__ == "__main__":
    main()
