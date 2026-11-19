"""Paired summaries (time to goal vs K15 / random control) for the PickCube grasp-phase and 5-step-pulse cohorts.

Reads results/monitor_{grasp_phase,grasp_phase_b,long_pulse}/confirmation.json; prints only (no files written).
"""

import json

import numpy as np

from adaptive_action_chunking.evaluate import _dynamics_summary

TIME = "capped_confirmation_seconds"
VARIANTS = ("acceleration_calls15", "acceleration_calls19", "prediction_error_calls15", "prediction_error_calls19")


def load(folder):
    return json.load(open(f"results/{folder}/confirmation.json"))["episodes"]


def report(name, episodes):
    for base in ("fixed_k15", "random_control"):
        paired = _dynamics_summary(episodes, {"samples": 5000, "seed": 7}, base)[1]
        for variant in VARIANTS:
            x = paired[variant]
            low, high = x["ci95"][f"{TIME}_difference"]
            print(f"{name:12s} {variant:26s} vs {base:14s} {x[f'{TIME}_difference']:+.3f} [{low:+.3f},{high:+.3f}] "
                  f"calls {x['policy_calls_difference']:+.1f} succ {x['success_at_end_difference']:+.3f}")


def main():
    grasp_a, grasp_b, long_pulse = load("monitor_grasp_phase"), load("monitor_grasp_phase_b"), load("monitor_long_pulse")
    report("grasp_b", grasp_b)
    report("grasp_pool", grasp_a + grasp_b)
    report("long_pulse", long_pulse)
    shown = ("fixed_k15", "acceleration_calls15", "random_control")
    print("long_pulse by force: mean time to goal (s) and final success for", ", ".join(shown))
    for force in sorted({x["force_n"] for x in long_pulse}):
        rows = {v: [x for x in long_pulse if x["force_n"] == force and x["variant"] == v] for v in shown}
        print(force, [round(float(np.mean([x[TIME] for x in rows[v]])), 2) for v in shown],
              [round(float(np.mean([x["success_at_end"] for x in rows[v]])), 2) for v in shown])


if __name__ == "__main__":
    main()
