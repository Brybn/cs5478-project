"""Result figures from saved rollouts: headline forest, time by force, example episode, calls frontier.

Default: PNGs in results/figures/. With --report: smaller vector PDFs (report layout) in results/figures/report/.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptive_action_chunking.evaluate import _dynamics_summary

R = "results/"
J = lambda p: json.load(open(R + p))
BOOT = {"samples": 5000, "seed": 7}
METRIC = "capped_confirmation_seconds_difference"
REPORT = "--report" in sys.argv
OUT = R + ("figures/report/" if REPORT else "figures/")
os.makedirs(OUT, exist_ok=True)


def size(slides, report):
    return report if REPORT else slides


def pct(rate):
    """Whole percent, halves rounded up (78.5% -> 79%, matching the report tables)."""
    return f"{int(rate * 100 + 0.5 + 1e-9)}%"


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(OUT + name + (".pdf" if REPORT else ".png"), dpi=160)


def pooled(path, ctrl_key, acc, pred):
    d = J(path)
    return {"vs K15": {"acceleration": d["paired_vs_k15"][acc], "prediction error": d["paired_vs_k15"][pred]},
            "vs random": {"acceleration": d[ctrl_key][acc], "prediction error": d[ctrl_key][pred]}}


def fresh(dirs, acc="acceleration_calls15", pred="prediction_error_calls15", ctrl="random_control"):
    eps = [e for d in dirs for e in J(f"{d}/confirmation.json")["episodes"]]
    k15 = _dynamics_summary(eps, BOOT)[1]
    rnd = _dynamics_summary(eps, BOOT, ctrl)[1]
    return {"vs K15": {"acceleration": k15[acc], "prediction error": k15[pred]},
            "vs random": {"acceleration": rnd[acc], "prediction error": rnd[pred]}}


cohorts = {  # main cohorts (two-step approach pulse) in report Table 2 order, then the Table 3 variants
    "PickCube\n(80 groups)": pooled("monitor_confirmation/pooled.json", "paired_vs_random_control",
                                     "acceleration_calls15", "prediction_error_calls15"),
    "PickCube, second policy\n(80 groups)": pooled("monitor_seed2_confirmation/pooled.json", "paired_vs_random_control",
                                                    "acceleration_calls15", "prediction_error_calls15"),
    "StackCube\n(100 groups)": pooled("monitor_stack_confirmation/pooled.json",
                                       "paired_vs_random_control_acceleration_calls14",
                                       "acceleration_calls14", "prediction_error_calls20"),
    "LiftPegUpright\n(100 groups)": pooled("monitor_liftpeg_confirmation/pooled.json", "paired_vs_random_control",
                                           "acceleration_calls25", "prediction_error_calls25"),
    "PickCube, 5-step pulse\n(40 groups)": fresh(["monitor_long_pulse"]),
    "StackCube, 5-step pulse\n(40 groups)": fresh(["monitor_stack_long_pulse"], "acceleration_calls14", "prediction_error_calls20",
                                                  "random_control_acceleration_calls14"),
    "PickCube, grasp pulse\n(80 groups)": fresh(["monitor_grasp_phase", "monitor_grasp_phase_b"]),
    "StackCube, carry pulse\n(40 groups)": fresh(["monitor_stack_carry"], "acceleration_calls14", "prediction_error_calls20",
                                                 "random_control_acceleration_calls14"),
}
STYLE = {"acceleration": ("#2a6fb0", "o", -0.12), "prediction error": ("#c8641e", "s", 0.12)}

fig, axes = plt.subplots(1, 2, figsize=size((10.5, 6.6), (7, 4.5)), sharey=True)
for ax, base in zip(axes, ("vs K15", "vs random")):
    for i, (name, c) in enumerate(cohorts.items()):
        for trig, (color, marker, off) in STYLE.items():
            s = c[base][trig]
            lo, hi = s["ci95"][METRIC]
            ax.plot([lo, hi], [i + off] * 2, color=color, lw=1.6, solid_capstyle="round")
            ax.plot(s[METRIC], i + off, marker, color=color, ms=7, mec="white", mew=1.2,
                    label=trig if i == 0 else None)
    ax.axvline(0, color="#888", lw=1)
    ax.set_title(size(f"Time to goal, {base}", base.replace("K15", "fixed K15")), fontsize=11)
    ax.set_xlabel(size("difference (s); left = trigger faster", "time-to-goal difference (s)"))
    ax.grid(axis="x", alpha=.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_yticks(range(len(cohorts)), list(cohorts)); axes[0].invert_yaxis()
if REPORT:
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=2, fontsize=9)
else:
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
save(fig, "headline_forest", size(None, (0, 0.06, 1, 1)))
for n, c in cohorts.items():
    print(n.replace("\n", " "), {b: {t: round(v[METRIC], 3) for t, v in x.items()} for b, x in c.items()})


# ---- per-force time to goal (approach-phase 2-step cohorts) ----
import numpy as np


def force_means(files, variants):
    eps = [e for f in files for e in (J(f)["episodes"] if isinstance(J(f), dict) else J(f))]
    out = {}
    rng = np.random.default_rng(0)
    for name, v in variants.items():
        for force in sorted({e["force_n"] for e in eps}):
            rows = [e for e in eps if e["variant"] == v and e["force_n"] == force]
            groups = sorted({e["group_id"] for e in rows})
            per = {g: np.mean([e["capped_confirmation_seconds"] for e in rows if e["group_id"] == g]) for g in groups}
            vals = np.array(list(per.values()))
            boots = [rng.choice(vals, len(vals)).mean() for _ in range(2000)]
            out[(name, force)] = (vals.mean(), *np.percentile(boots, [2.5, 97.5]))
    return out


fig, axes = plt.subplots(1, 2, figsize=size((10.5, 4), (7, 2.6)))
COL = {"fixed K15": "#888888", "acceleration": "#2a6fb0", "random": "#c8641e"}
for ax, (task, files, variants) in zip(axes, [
    ("PickCube (80 groups)", ["monitor_level/test.json", "monitor_confirmation/control_on_test.json", "monitor_confirmation/confirmation.json"],
     {"fixed K15": "fixed_k15", "acceleration": "acceleration_calls15", "random": "random_control"}),
    ("StackCube (100 groups)", ["monitor_stack/test.json", "monitor_stack_confirmation/control_on_test.json", "monitor_stack_confirmation/episodes.json"],
     {"fixed K15": "fixed_k15", "acceleration": "acceleration_calls14",
      "random": "random_control_acceleration_calls14"})]):
    m = force_means(files, variants)
    forces = sorted({f for _, f in m})
    for j, name in enumerate(variants):
        x = np.arange(len(forces)) + (j - 1) * 0.26
        mean = np.array([m[name, f][0] for f in forces])
        err = np.array([[m[name, f][0] - m[name, f][1] for f in forces], [m[name, f][2] - m[name, f][0] for f in forces]])
        ax.bar(x, mean, 0.24, color=COL[name], label=name)
        ax.errorbar(x, mean, yerr=err, fmt="none", ecolor="#333", lw=1, capsize=2)
    ax.set_xticks(range(len(forces)), [f"{f:g} N" for f in forces]); ax.set_title(task, fontsize=11)
    ax.set_ylim(3, None); ax.set_xlabel("pulse force"); ax.grid(axis="y", alpha=.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel(size("time to goal (s; axis starts at 3 s)", "time to goal (s)")); axes[0].legend(frameon=False, fontsize=8, loc="upper left")
save(fig, "time_by_force")

# ---- example: same reset and pulse, fixed K15 vs acceleration trigger (PickCube test) ----
t = J("monitor_level/test.json")
cases = {c["episode_id"]: c for c in t["cases"]}
by = {(e["episode_id"], e["variant"]): e for e in t["episodes"]}
best = max((cid for cid in cases if cases[cid]["force_n"] >= 0.4 and by[cid, "acceleration_calls15"]["success_at_end"]
            and by[cid, "fixed_k15"]["success_at_end"]),
           key=lambda cid: by[cid, "fixed_k15"]["capped_confirmation_seconds"] - by[cid, "acceleration_calls15"]["capped_confirmation_seconds"])
c = cases[best]
fig, ax = plt.subplots(figsize=size((8, 3.6), (6, 2.7)))
for v, color, label in (("fixed_k15", "#888888", "fixed K15"), ("acceleration_calls15", "#2a6fb0", "acceleration")):
    e = by[best, v]
    d = np.asarray(e["goal_distances_m"]) * 1000
    ax.plot(np.arange(len(d)) / 20, d, color=color, lw=1.8, label=label)
    for p in e["plans"]:
        if p["reason"] == "monitor":
            ax.plot(p["step"] / 20, d[p["step"]], "v", color=color, ms=7, mec="white")
    ax.axvline(e["capped_confirmation_seconds"], color=color, ls=":", lw=1)
ax.axvspan(c["start_step"] / 20, (c["start_step"] + c["duration_steps"]) / 20, color="#c8641e", alpha=.25, label=f"{c['force_n']:g} N pulse")
ax.set_xlabel("time (s)"); ax.set_ylabel("cube-to-goal distance (mm)"); ax.legend(frameon=False, fontsize=8)
ax.set_title(f"Largest-gap 0.4-0.5 N case ({best}), not typical; triangles = triggered replans, dotted = time to goal", fontsize=8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
save(fig, "example_episode")
print("example", best)

# ---- calls-vs-time frontier: fixed K sweep (same pooled groups) with the triggers and random control ----
FRONTIER = [("PickCube (80 groups)", "monitor_confirmation", "acceleration_calls15", "prediction_error_calls15", "random_control"),
            ("StackCube (100 groups)", "monitor_stack_confirmation", "acceleration_calls14", "prediction_error_calls20",
             "random_control_acceleration_calls14")]
FRONTIER = [f for f in FRONTIER if os.path.exists(R + f[1] + "/fixed_k_sweep.json")]
if FRONTIER:
    fig, axes = plt.subplots(1, len(FRONTIER), figsize=size((5.4 * len(FRONTIER), 4.2), (3.6 * len(FRONTIER), 3.0)), squeeze=False)
    for ax, (task, folder, acc, pred, rnd) in zip(axes[0], FRONTIER):
        summary = {**J(folder + "/fixed_k_sweep.json")["summary"], **J(folder + "/pooled.json")["summary"]}
        point = lambda v: (summary[v]["policy_calls"], summary[v]["capped_confirmation_seconds"],
                           summary[v]["ci95"]["capped_confirmation_seconds"], summary[v]["success_at_end"])
        fixed = sorted((point(v) + (int(v[7:]),) for v in summary if v.startswith("fixed_k")), key=lambda p: p[0])
        x, y = [p[0] for p in fixed], [p[1] for p in fixed]
        ax.plot(x, y, "-", color="#888", lw=1.5, zorder=1)
        for calls, t, (lo, hi), succ, k in fixed:
            ax.errorbar(calls, t, yerr=[[t - lo], [hi - t]], fmt="o", color="#888", ms=6, capsize=2, zorder=2)
            below = task.startswith("StackCube") and k == 10
            ax.annotate(f"K{k}" if REPORT and k != 1 else f"K{k}\n{pct(succ)}",
                        (calls, t) if k == 1 else (calls, lo if below else hi), textcoords="offset points",
                        xytext=(-14, 0) if k == 1 else (0, -3 if below else 3),
                        ha="right" if k == 1 else "center", va="center" if k == 1 else "top" if below else "bottom",
                        fontsize=7, color="#555")
        for v, label, color, marker in ((acc, "acceleration", "#2a6fb0", "D"), (pred, "prediction error", "#7a4fb0", "s"),
                                        (rnd, "random", "#c8641e", "^")):
            calls, t, (lo, hi), succ = point(v)
            ax.errorbar(calls, t, yerr=[[t - lo], [hi - t]], fmt=marker, color=color, ms=7, capsize=2, mec="white",
                        label=f"{label} ({pct(succ)})", zorder=3)
        ax.set_xscale("log"); ax.set_xticks(size([10, 15, 20, 40, 75, 150, 200], [10, 20, 40, 80, 160])); ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("policy calls per episode (log scale)"); ax.set_title(task, fontsize=11); ax.grid(alpha=.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5, loc=size("upper left", "upper center"), title_fontsize=7,
                  title=size("final success in brackets; bars: 95% CI of the mean", None))
    axes[0][0].set_ylabel(size("time to goal (s), mean over all forces", "time to goal (s)"))
    save(fig, "calls_frontier")
    print("frontier", [f[0] for f in FRONTIER])
