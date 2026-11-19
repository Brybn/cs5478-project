"""Trigger checks for the four main cohorts (no new rollouts). Writes results/trigger_comparison.json.

1. Acceleration trigger minus prediction-error trigger, paired time to goal (capped at the horizon), averaged within
   reset group and bootstrapped over whole groups, as in Table 2.
2. Random-control timing: the control's draw is a CRC32 hash of the current state, so check that it fires as often in
   the window from pulse start to three steps after the pulse as in the same steps of the 0 N episode of each group.
   A step is eligible when a draw can fire (2-14 steps after the previous call); calls at 15 steps are scheduled.
3. Acceleration trigger minus random control, separately on the test groups and on the fresh (confirmation) groups.
"""

import json
from pathlib import Path

import numpy as np

R = Path("results")
HZ = 20
BOOT = {"samples": 5000, "seed": 4501}
COHORTS = {
    "PickCube (80 groups)": (["monitor_level/test.json", "monitor_confirmation/confirmation.json"],
                             "acceleration_calls15", "prediction_error_calls15"),
    "PickCube, second policy (80 groups)": (["monitor_seed2/test.json", "monitor_seed2_confirmation/confirmation.json"],
                                            "acceleration_calls15", "prediction_error_calls15"),
    "StackCube (100 groups)": (["monitor_stack/test.json", "monitor_stack_confirmation/episodes.json"],
                               "acceleration_calls14", "prediction_error_calls20"),
    "LiftPegUpright (100 groups)": (["monitor_liftpeg/test.json", "monitor_liftpeg_confirmation/episodes.json"],
                                    "acceleration_calls25", "prediction_error_calls25"),
}
RANDOM = {
    "PickCube (80 groups)": ("monitor_confirmation", "confirmation.json", "random_control"),
    "PickCube, second policy (80 groups)": ("monitor_seed2_confirmation", "confirmation.json", "random_control"),
    "StackCube (100 groups)": ("monitor_stack_confirmation", "episodes.json", "random_control_acceleration_calls14"),
    "LiftPegUpright (100 groups)": ("monitor_liftpeg_confirmation", "episodes.json", "random_control"),
}


def load(files, variants):
    out = {}
    for f in files:
        data = json.loads((R / f).read_text())
        for e in data["episodes"] if isinstance(data, dict) else data:
            if e["variant"] in variants:
                out[e["episode_id"], e["variant"]] = e
    return out


def paired_difference(files, a, b):
    """Time to goal of variant a minus variant b, averaged within reset group, whole-group bootstrap."""
    time = {key: (e["success_confirmation_step"] if e["success_confirmation_step"] is not None else e["steps"]) / HZ
            for key, e in load(files, {a, b}).items()}
    per_group = {}
    for (episode, variant), t in time.items():
        if variant == a and (episode, b) in time:
            per_group.setdefault(episode.rsplit("_force_", 1)[0], []).append(t - time[episode, b])
    groups = sorted(per_group)
    values = np.array([np.mean(per_group[g]) for g in groups])
    indices = np.random.default_rng(BOOT["seed"]).integers(0, len(groups), (BOOT["samples"], len(groups)))
    lo, hi = np.quantile(values[indices].mean(1), [.025, .975])
    return {"minuend": a, "subtrahend": b, "groups": len(groups),
            "difference_s": float(values.mean()), "ci95": [float(lo), float(hi)]}


def random_timing(folder, confirmation, variant):
    """Random-control draws that fired / eligible steps, in the pulse window with and without a pulse."""
    counts = {"pulse": [0, 0], "no_pulse": [0, 0], "all_steps": [0, 0]}
    for e in load([f"{folder}/control_on_test.json", f"{folder}/{confirmation}"], {variant}).values():
        calls, fired, last = set(e["replan_steps"]), set(), 0
        for step in sorted(calls)[1:]:
            if step - last < 15:
                fired.add(step)
            last = step
        window = range(e["start_step"], e["start_step"] + e["duration_steps"] + 3)
        key = "pulse" if e["force_n"] > 0 else "no_pulse"
        last = 0
        for step in range(1, e["steps"]):
            if 2 <= step - last < 15:
                for k in ("all_steps",) + ((key,) if step in window else ()):
                    counts[k][0] += step in fired
                    counts[k][1] += 1
            if step in calls:
                last = step
    return {k: {"fired": f, "eligible": n, "rate": f / n} for k, (f, n) in counts.items()}


def main():
    out = {"bootstrap": BOOT, "acceleration_minus_prediction_error": {}, "random_control_timing": {},
           "acceleration_minus_random_by_split": {}}
    for name, (files, acc, pred) in COHORTS.items():
        r = out["acceleration_minus_prediction_error"][name] = paired_difference(files, acc, pred)
        print(f"{name}: acceleration - prediction error {r['difference_s']:+.2f} s "
              f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]")
    pooled = {"pulse": [0, 0], "no_pulse": [0, 0]}
    for name, args in RANDOM.items():
        r = out["random_control_timing"][name] = random_timing(*args)
        for k in pooled:
            pooled[k][0] += r[k]["fired"]
            pooled[k][1] += r[k]["eligible"]
        print(f"{name}: random fires {r['pulse']['rate']:.1%} of eligible pulse-window steps, "
              f"{r['no_pulse']['rate']:.1%} without a pulse, {r['all_steps']['rate']:.1%} overall")
    out["random_control_timing"]["pooled"] = {k: {"fired": f, "eligible": n, "rate": f / n} for k, (f, n) in pooled.items()}
    print("pooled:", {k: f"{f}/{n} = {f / n:.1%}" for k, (f, n) in pooled.items()})
    for name, (files, acc, _) in COHORTS.items():
        folder, confirmation, control = RANDOM[name]
        splits = {"test": [files[0], f"{folder}/control_on_test.json"], "fresh": [f"{folder}/{confirmation}"]}
        r = out["acceleration_minus_random_by_split"][name] = {k: paired_difference(v, acc, control) for k, v in splits.items()}
        print(f"{name}: acceleration - random, test {r['test']['difference_s']:+.2f} s "
              f"[{r['test']['ci95'][0]:+.2f}, {r['test']['ci95'][1]:+.2f}], fresh {r['fresh']['difference_s']:+.2f} s "
              f"[{r['fresh']['ci95'][0]:+.2f}, {r['fresh']['ci95'][1]:+.2f}]")
    (R / "trigger_comparison.json").write_text(json.dumps({"definition": __doc__.strip(), **out}, indent=1))


if __name__ == "__main__":
    main()
