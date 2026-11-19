"""Final success of every retained PickCube policy under DDPM100 and DDIM10, on the same scenes.

Scenes: the closing confirmation's 100 reset groups x forces 0/0.2/0.3/0.4 N (results/policy_closing_confirmation),
K8 and K15, 150 steps, CPU inference, policy noise drawn per absolute step, so every cell is paired with every
other. DDIM10 runs for every policy. DDPM100 comes from the saved confirmation for stages 2 and 3 (re-verified by
reproduce_results.py); add more policies with --ddpm (about 18 min per policy on 11 workers).

Usage from the repository root:
  uv run --frozen --no-sync python scripts/analysis/sampler_table.py               # DDIM10 for all, ~35 min
  uv run --frozen --no-sync python scripts/analysis/sampler_table.py --ddpm stage0 stage5  # also DDPM100 for stages 0 and 5
Writes results/sampler_table/sampler_table.{md,json}; finished runs are cached in results/sampler_table/sampler_runs/.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from reproduce_results import LINEAGE, POOL, R, run  # noqa: E402

OUT = Path("results/sampler_table")
CACHE = OUT / "sampler_runs"
SAMPLERS = {"ddpm100": ("ddpm", None), "ddim10": ("ddim", 10)}
STAGES = {"stage0": "0 nominal", "stage1": "1 placement, then pickup", "stage2": "2 more pickup",
          "stage3": "3 grasp closing", "stage4": "4 direct descent", "stage5": "5 level alignment (monitor study)",
          "seed2_nominal": "second policy, nominal", "seed2_finetuned": "second policy, one recovery stage"}
SAVED_DDPM = {"stage2": "retained", "stage3": "closing"}  # labels in the saved closing confirmation
KEEP = ("episode_id", "group_id", "force_n", "variant", "success_at_end", "capped_confirmation_seconds")


def cached(policy, sampler):
    path = CACHE / f"{policy}_{sampler}.json"
    if path.exists():
        return json.loads(path.read_text())
    if sampler == "ddpm100" and policy in SAVED_DDPM:
        from adaptive_action_chunking.evaluate import _motion_outcomes
        root = R / "policy_closing_confirmation"
        cases = {c["episode_id"]: c for c in json.loads((root / "comparison/plan.json").read_text())["cases"]}
        rows = []
        for e in json.loads((root / "horizon150/results.json").read_text())["episodes"]:
            if e["checkpoint"] == SAVED_DDPM[policy]:
                e = {**e, **_motion_outcomes(cases[e["episode_id"]], e), "variant": f"k{e['chunk_length']}"}
                rows.append({k: e[k] for k in KEEP})
        return rows
    return None


def summarize(rows, k):
    rows = [e for e in rows if e["variant"] == f"k{k}"]
    groups = sorted({e["group_id"] for e in rows})
    per_group = np.array([np.mean([e["success_at_end"] for e in rows if e["group_id"] == g]) for g in groups])
    draws = np.random.default_rng(169002).integers(len(groups), size=(5000, len(groups)))
    return {"successes": int(sum(e["success_at_end"] for e in rows)), "episodes": len(rows),
            "rate": float(per_group.mean()), "ci95": np.quantile(per_group[draws].mean(1), [.025, .975]).tolist(),
            "by_force": {f"{f:g}": int(sum(e["success_at_end"] for e in rows if e["force_n"] == f))
                         for f in sorted({e["force_n"] for e in rows})},
            "mean_time_to_goal_s": float(np.mean([e["capped_confirmation_seconds"] for e in rows]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ddpm", nargs="*", default=[], metavar="POLICY", help="also run DDPM100 for these policies")
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    if unknown := set(args.ddpm) - set(LINEAGE):
        parser.error(f"unknown policy {sorted(unknown)}; choose from {' '.join(LINEAGE)}")
    plan = json.loads((R / "policy_closing_confirmation/comparison/plan.json").read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    if missing := [p for p in LINEAGE if not LINEAGE[p].exists()]:
        print(f"skipping policies without a checkpoint: {' '.join(missing)}", flush=True)
    wanted = [(p, s) for p, s in [(p, "ddim10") for p in LINEAGE] + [(p, "ddpm100") for p in args.ddpm]
              if LINEAGE[p].exists()]
    for sampler in SAMPLERS:
        todo = [p for p, s in wanted if s == sampler and cached(p, sampler) is None]
        if not todo:
            continue
        name, steps = SAMPLERS[sampler]
        tasks = [{**case, "policy": p, "variant": f"k{k}", "kind": "fixed", "threshold": None, "chunk_length": k}
                 for case in plan["cases"] for p in todo for k in (8, 15)]
        spec = dict(name=sampler, task="PickCube", horizon=150, predictor=None, saved={}, tasks=tasks,
                    policies={p: (str(LINEAGE[p]), name, steps) for p in todo})
        print(f"{sampler}: {len(tasks)} rollouts for {' '.join(todo)}", flush=True)
        episodes = run(spec, args.workers)
        for p in todo:
            (CACHE / f"{p}_{sampler}.json").write_text(json.dumps([{k: e[k] for k in KEEP} for e in episodes
                                                                  if e["policy"] == p]))
    if POOL:
        POOL.pop("pool").shutdown()

    table = {p: {s: {k: summarize(rows, k) for k in (8, 15)} for s in SAMPLERS if (rows := cached(p, s))}
             for p in LINEAGE}
    cell = lambda p, s, k: (f"{table[p][s][k]['successes']}/{table[p][s][k]['episodes']} "  # noqa: E731
                            f"({100 * table[p][s][k]['rate']:.1f}%)" if s in table[p] else "-")
    lines = ["| Stage | DDPM100 K8 | DDIM10 K8 | DDPM100 K15 | DDIM10 K15 |", "| --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {STAGES[p]} | {cell(p, 'ddpm100', 8)} | {cell(p, 'ddim10', 8)} | {cell(p, 'ddpm100', 15)} | "
              f"{cell(p, 'ddim10', 15)} |" for p in LINEAGE]
    note = (f"Final success at 150 steps; {len(plan['cases']) // 4} reset groups x 0/0.2/0.3/0.4 N (closing-confirmation "
            "scenes), same resets, pushes and noise seeds in every cell. 95% CIs "
            "(whole-group bootstrap) are in sampler_table.json.")
    (OUT / "sampler_table.md").write_text("\n".join(lines) + "\n\n" + note + "\n")
    (OUT / "sampler_table.json").write_text(json.dumps({"scenes": "results/policy_closing_confirmation/comparison/plan.json",
                                                        "horizon": 150, "note": note, "table": table}, indent=1))
    print("\n".join(lines) + "\n\n" + note)


if __name__ == "__main__":
    main()
