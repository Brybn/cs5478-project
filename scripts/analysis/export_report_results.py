"""Export the results behind the report into a small, tracked folder.

Writes <output-dir>/summaries/ (the result files of docs/workflow.md section 9, mirrored from results/, with
per-episode lists removed) and <output-dir>/episodes.csv (one row per monitor-study episode in every cohort of
Tables 2-3 and Figures 2-4). Each cohort's row counts and mean time to goal are checked against the saved
summaries before anything is written.

    uv run --frozen --no-sync python scripts/analysis/export_report_results.py
"""
import argparse
import csv
import json
from pathlib import Path

R = Path("results")
# cohort: (test file, confirmation folder, fresh-group file, has fixed-K sweep)
MAIN = {
    "pickcube": ("monitor_level/test.json", "monitor_confirmation", "confirmation.json", True),
    "pickcube_seed2": ("monitor_seed2/test.json", "monitor_seed2_confirmation", "confirmation.json", False),
    "stackcube": ("monitor_stack/test.json", "monitor_stack_confirmation", "episodes.json", True),
    "liftpeg": ("monitor_liftpeg/test.json", "monitor_liftpeg_confirmation", "episodes.json", False),
}
VARIANTS = {
    "pickcube_long_pulse": "monitor_long_pulse",
    "pickcube_grasp_phase_a": "monitor_grasp_phase",
    "pickcube_grasp_phase_b": "monitor_grasp_phase_b",
    "stackcube_carry": "monitor_stack_carry",
    "stackcube_long_pulse": "monitor_stack_long_pulse",
}
SUMMARIES = (
    [f"{conf}/{name}" for _, conf, _, _ in MAIN.values() for name in ("pooled.json", "calibration.json")]
    + [f"{test.split('/')[0]}/selection.json" for test, _, _, _ in MAIN.values()]
    + [f"{conf}/fixed_k_sweep.json" for _, conf, _, sweep in MAIN.values() if sweep]
    + [f"{folder}/confirmation.json" for folder in VARIANTS.values()]
    + ["trigger_comparison.json", "latency_sensitivity.json", "sampler_table/sampler_table.json",
       "policy_closing_confirmation/summary.json"]
    + [f"{pilot}/{name}" for pilot in ("liftpeg_force_pilot", "liftpeg_recovery_pilot", "liftpeg_recovery_round2_pilot")
       for name in ("plan.json", "results.json")]
)
COLUMNS = ["cohort", "split", "group_id", "reset_seed", "policy_seed", "force_n", "pulse_start_step",
           "pulse_steps", "variant", "policy_calls", "trigger_replans", "time_to_goal_s", "goal_step",
           "success_at_end", "call_steps"]


def load(path):
    return json.loads((R / path).read_text())


def episodes(path):
    data = load(path)
    return data["episodes"] if isinstance(data, dict) else data


def row(cohort, split, e, pulse):
    start, steps = pulse[e["group_id"]]
    return [cohort, split, e["group_id"], e["seed"], e["policy_seed"], e["force_n"], start, steps, e["variant"],
            e["policy_calls"], e["monitor_replans"], e["capped_confirmation_seconds"],
            "" if e["success_confirmation_step"] is None else e["success_confirmation_step"],
            int(e["success_at_end"]), " ".join(map(str, e["replan_steps"]))]


def check(rows, summary, counts=None):
    for variant, stats in summary.items():
        times = [r[11] for r in rows if r[8] == variant]
        expected = counts[variant] if counts else stats["episodes"]
        if len(times) != expected or abs(sum(times) / len(times) - stats["capped_confirmation_seconds"]) > 1e-9:
            raise RuntimeError(f"{rows[0][0]} {variant}: {len(times)} rows do not match the saved summary")


def collect():
    out = []
    for cohort, (test, conf, fresh, sweep) in MAIN.items():
        pooled = load(f"{conf}/pooled.json")
        reported = set(pooled["episodes_per_variant"])
        controls, fresh_eps = episodes(f"{conf}/control_on_test.json"), episodes(f"{conf}/{fresh}")
        pulse = {e["group_id"]: (e["start_step"], e["duration_steps"]) for e in controls + fresh_eps}
        test_eps = [e for e in episodes(test) if e["variant"] in reported]
        test_groups = {e["group_id"] for e in test_eps}
        rows = ([row(cohort, "test", e, pulse) for e in test_eps + controls]
                + [row(cohort, "fresh", e, pulse) for e in fresh_eps if e["variant"] in reported])
        check(rows, pooled["summary"], pooled["episodes_per_variant"])
        if sweep:
            data = load(f"{conf}/fixed_k_sweep.json")
            extra = [row(cohort, "test" if e["group_id"] in test_groups else "fresh", e, pulse)
                     for e in data["episodes"] if e["variant"] != "fixed_k15"]  # K15 rerun duplicates the cohort
            check(extra, {k: v for k, v in data["summary"].items() if k != "fixed_k15"})
            rows += extra
        out += rows
    for cohort, folder in VARIANTS.items():
        data = load(f"{folder}/confirmation.json")
        pulse = {e["group_id"]: (e["start_step"], e["duration_steps"]) for e in data["episodes"]}
        rows = [row(cohort, "fresh", e, pulse) for e in data["episodes"]]
        check(rows, data["summary"])
        out += rows
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("report_results"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"{args.output_dir} exists; remove it first")
    rows = collect()
    for path in SUMMARIES:
        data = load(path)
        if isinstance(data, dict):
            data.pop("episodes", None)
        target = args.output_dir / "summaries" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=1) + "\n")
    with open(args.output_dir / "episodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print(f"{len(rows)} episodes and {len(SUMMARIES)} summary files -> {args.output_dir}")


if __name__ == "__main__":
    main()
