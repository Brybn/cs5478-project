"""Split replayed episodes and verify PickCube reset observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def split_demos(
    source_path: Path,
    output_dir: Path,
    count: int = 100,
    train_count: int = 80,
    seed: int = 4278,
    max_steps: int | None = None,
) -> tuple[Path, Path]:
    if not 0 < train_count < count:
        raise ValueError("Require 0 < train_count < count")
    metadata_path = source_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=False)
    env = None
    if metadata["env_info"]["env_id"] == "PickCube-v1":
        from adaptive_action_chunking.env import extract_policy_state, make_env

        env = make_env()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    with h5py.File(source_path, "r") as source:
        keys = sorted(
            (key for key in source if key.startswith("traj_")),
            key=lambda key: int(key.split("_")[-1]),
        )
        if max_steps is not None:  # drop rare planner detours before taking the first `count`
            keys = [key for key in keys if len(source[key]["actions"]) <= max_steps]
        keys = keys[:count]
        if len(keys) < count:
            raise ValueError(f"Requested {count} demonstrations; found {len(keys)}")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(keys))
        groups = {
            "train": [keys[i] for i in order[:train_count]],
            "validation": [keys[i] for i in order[train_count:]],
        }
        outputs = []
        episodes_by_id = {
            int(item["episode_id"]): item
            for item in metadata.get("episodes", [])
            if "episode_id" in item
        }
        for split, selected in groups.items():
            target_path = output_dir / f"{split}.h5"
            new_metadata = {
                key: value for key, value in metadata.items() if key != "episodes"
            }
            new_metadata["episodes"] = []
            new_metadata["source_file"] = str(source_path.resolve())
            new_metadata["source_sha256"] = source_hash
            new_metadata["split_seed"] = seed
            new_metadata["max_steps"] = max_steps
            corrections = []
            max_reset_error = 0.0
            with h5py.File(target_path, "w") as target:
                for attr, value in source.attrs.items():
                    target.attrs[attr] = value
                for new_id, key in enumerate(selected):
                    source.copy(key, target, name=f"traj_{new_id}")
                    original_id = int(key.split("_")[-1])
                    entry = dict(
                        episodes_by_id.get(original_id, {"episode_id": original_id})
                    )
                    entry["original_episode_id"] = original_id
                    entry["episode_id"] = new_id
                    if env is not None:
                        live, _ = env.reset(**entry["reset_kwargs"])
                        live = extract_policy_state(live)
                        saved = target[f"traj_{new_id}"]["obs"][0]
                        error = float(np.max(np.abs(np.delete(saved - live, 18))))
                        if error > 1e-6:
                            env.close()
                            raise ValueError(f"Reset state differs for episode {original_id}: {error}")
                        max_reset_error = max(max_reset_error, error)
                        if saved[18] != live[18]:
                            target[f"traj_{new_id}"]["obs"][0, 18] = live[18]
                            corrections.append({"episode_id": new_id, "original_episode_id": original_id,
                                                "seed": entry["episode_seed"], "old": float(saved[18]), "new": float(live[18])})
                    new_metadata["episodes"].append(entry)
            if env is not None:
                new_metadata["reset_observation_check"] = {
                    "verified_episodes": len(selected), "feature_index": 18, "feature": "is_grasped",
                    "max_other_feature_error": max_reset_error, "corrections": corrections,
                    "method": "Compare against fresh reconfigured live resets; change only the initial grasp flag",
                }
                print(f"{split}: verified {len(selected)} resets, corrected {len(corrections)} grasp flags")
            target_path.with_suffix(".json").write_text(
                json.dumps(new_metadata, indent=2)
            )
            outputs.append(target_path)
    if env is not None:
        env.close()
    return outputs[0], outputs[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/split"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=4278)
    parser.add_argument("--max-steps", type=int, help="skip demonstrations with more actions")
    args = parser.parse_args()
    train, validation = split_demos(
        args.source, args.output_dir, args.count, args.train_count, args.seed, args.max_steps
    )
    print(f"Training: {train}\nValidation: {validation}")


if __name__ == "__main__":
    main()
