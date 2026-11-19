"""Blocking-inference sensitivity of the headline time-to-goal comparisons (no new rollouts).

Worst-case accounting: the robot waits while every policy call runs, so
time to goal = (goal step / 20 Hz) + c * (policy calls made before the goal step), capped episodes use all calls.
Sweeps a per-call latency c and reports paired differences (whole-reset-group bootstrap) and the break-even c.
The measured per-call cost in the study (CPU, one thread) marks the operating point for each task.
Also splits the trigger's gain over K15 (c = 0) into pairs where both episodes reach the goal and pairs where only
one does (the other scores the horizon).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
HZ = 20
COHORTS = {
    "PickCube (80 groups)": {
        "files": ["monitor_level/test.json", "monitor_confirmation/control_on_test.json",
                  "monitor_confirmation/confirmation.json"],
        "trigger": "acceleration_calls15", "random": "random_control", "horizon": 150},
    "StackCube (100 groups)": {
        "files": ["monitor_stack/test.json", "monitor_stack_confirmation/control_on_test.json",
                  "monitor_stack_confirmation/episodes.json"],
        "trigger": "acceleration_calls14", "random": "random_control_acceleration_calls14", "horizon": 200},
    "LiftPegUpright (100 groups)": {
        "files": ["monitor_liftpeg/test.json", "monitor_liftpeg_confirmation/control_on_test.json",
                  "monitor_liftpeg_confirmation/episodes.json"],
        "trigger": "acceleration_calls25", "random": "random_control", "horizon": 250},
}
GRID = np.round(np.arange(0, 0.301, 0.01), 3)
BOOT = {"samples": 5000, "seed": 4501}


def episodes(files, variants):
    out = []
    for f in files:
        data = json.loads((R / f).read_text())
        for e in data["episodes"] if isinstance(data, dict) else data:
            if e["variant"] in variants:
                goal = e["success_confirmation_step"]
                out.append({"id": e["episode_id"], "group": e["group_id"], "variant": e["variant"],
                            "goal": goal if goal is not None else e["steps"], "reached": goal is not None,
                            "replans": e["replan_steps"], "calls": e["policy_calls"],
                            "per_call": e["policy_seconds"] / e["policy_calls"]})
    return out


def paired(eps, variant, base, c, indices, groups):
    t = {(e["id"], e["variant"]): e["goal"] / HZ + c * sum(r < e["goal"] for r in e["replans"]) for e in eps}
    per_group = {}
    for e in eps:
        if e["variant"] == variant:
            per_group.setdefault(e["group"], []).append(t[e["id"], variant] - t[e["id"], base])
    values = np.array([np.mean(per_group[g]) for g in groups])
    return float(values.mean()), np.quantile(values[indices].mean(1), [.025, .975]).tolist()


def gain_split(eps, variant, base):
    """Mean paired difference (c = 0) from pairs where both reach the goal and where exactly one does."""
    by = {(e["id"], e["variant"]): e for e in eps}
    pairs = [(by[i, variant], by[i, base]) for i, v in by if v == variant]
    split = {"both_reached": 0.0, "one_reached": 0.0, "neither": 0.0}
    for a, b in pairs:
        key = "both_reached" if a["reached"] and b["reached"] else "neither" if not (a["reached"] or b["reached"]) else "one_reached"
        split[key] += (a["goal"] - b["goal"]) / HZ / len(pairs)
    return {"pairs": len(pairs), **split}


def main():
    result, fig = {}, plt.figure(figsize=(5.25 * len(COHORTS), 4))
    axes = fig.subplots(1, len(COHORTS), sharey=True)
    for ax, (name, c) in zip(axes, COHORTS.items()):
        eps = episodes(c["files"], {"fixed_k15", c["trigger"], c["random"]})
        groups = sorted({e["group"] for e in eps})
        indices = np.random.default_rng(BOOT["seed"]).integers(0, len(groups), (BOOT["samples"], len(groups)))
        measured = float(np.median([e["per_call"] for e in eps]))
        entry = {"measured_seconds_per_call": measured, "episodes": len(eps), "groups": len(groups), "curves": {},
                 "gain_split_vs_k15_s": gain_split(eps, c["trigger"], "fixed_k15")}
        for base, label, colour in (("fixed_k15", "vs fixed K15", "#2a6fb0"), (c["random"], "vs call-matched random", "#c8641e")):
            rows = [(cost, *paired(eps, c["trigger"], base, cost, indices, groups)) for cost in GRID]
            mean = np.array([r[1] for r in rows]); lo = np.array([r[2][0] for r in rows]); hi = np.array([r[2][1] for r in rows])
            crossing = next((float(cost) for cost, m in zip(GRID, mean) if m >= 0), None)
            at_measured = paired(eps, c["trigger"], base, measured, indices, groups)
            entry["curves"][base] = {"latency_s": GRID.tolist(), "difference_s": mean.tolist(),
                                     "ci95": list(zip(lo.tolist(), hi.tolist())), "break_even_latency_s": crossing,
                                     "at_measured_latency": {"difference_s": at_measured[0], "ci95": at_measured[1]}}
            ax.plot(GRID, mean, color=colour, lw=1.8, label=label)
            ax.fill_between(GRID, lo, hi, color=colour, alpha=.18, lw=0)
        ax.axhline(0, color="#888", lw=1)
        if measured <= GRID[-1]:
            ax.axvline(measured, color="#333", ls=":", lw=1)
            ax.text(measured, ax.get_ylim()[0], f" measured CPU\n {measured:.2f} s/call", fontsize=8, va="bottom")
        else:
            ax.text(GRID[-1], ax.get_ylim()[0], f"measured CPU {measured:.2f} s/call\n(off scale) →",
                    fontsize=8, ha="right", va="bottom")
        ax.set_title(name, fontsize=11); ax.set_xlabel("per-call inference latency, robot waiting (s)")
        ax.grid(alpha=.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        result[name] = entry
        print(name, f"measured {measured:.3f} s/call",
              {b: (v["break_even_latency_s"], round(v["at_measured_latency"]["difference_s"], 3),
                   [round(x, 2) for x in v["at_measured_latency"]["ci95"]]) for b, v in entry["curves"].items()},
              {k: round(v, 3) for k, v in entry["gain_split_vs_k15_s"].items()})
    axes[0].set_ylabel("acceleration trigger − baseline,\ntime to goal (s; below 0 = trigger faster)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(R / "figures/latency_sensitivity.png", dpi=160)
    (R / "latency_sensitivity.json").write_text(json.dumps(
        {"definition": __doc__.strip(), "bootstrap": BOOT, "cohorts": result}, indent=1))


if __name__ == "__main__":
    main()
