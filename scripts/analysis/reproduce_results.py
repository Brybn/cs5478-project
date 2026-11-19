"""Re-run the saved rollouts behind the report, policy by policy, and check that they reproduce exactly.

Every saved rollout stores its reset seed, push (start step, duration, angle, force) and policy-noise seed, and
policy noise is drawn per absolute control step, so re-running the same checkpoint on CPU must give identical
success steps, replan steps and policy calls. For each policy the script re-runs the cohorts it was evaluated
on, counts exact matches against the saved files and recomputes the reported numbers from the fresh rollouts.

Policy keys (PickCube unless noted; stages as in the report).
  stage0 ... stage5 seed2_nominal seed2_finetuned
                  lineage benchmark: the 30 groups x 5 forces of results/policy_level_eval, K8/K15, 150 steps,
                  DDIM10 for every checkpoint. Stages 3-5 were saved there (exact check); the others are new numbers.
  stage2 stage3   closing confirmation: 100 groups x 4 forces, K8/K15, 150 steps, DDPM100 as saved.
  stage5          monitor study: test, random control, confirmation, fixed-K sweep; grasp-phase and 5-step pulse cohorts.
  seed2_finetuned monitor study of the second policy: test, random control, confirmation.
  stack           StackCube nominal check and monitor study: test, control, confirmation, sweep,
                  carry/place and 5-step pulse cohorts.
  liftpeg         LiftPegUpright monitor study (corrected policy): test, random control, confirmation.

Usage from the repository root:
  uv run --frozen --no-sync python scripts/analysis/reproduce_results.py               # all policies, 12 workers
  uv run --frozen --no-sync python scripts/analysis/reproduce_results.py stage5 --groups 2  # quick smoke test
Writes <output-dir>/<policy>.json (default results/reproduction/: match counts, mismatches, recomputed vs saved
numbers).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

R, PICK = Path("results"), Path("checkpoints/pickcube")
OUT = Path("results/reproduction")
LINEAGE = {
    "stage0": PICK / "stage0_nominal/best.pt",
    "stage1": PICK / "stage1_pickup/best.pt",
    "stage2": PICK / "stage2_pickup/last.pt",
    "stage3": PICK / "stage3_closing/last.pt",
    "stage4": PICK / "stage4_descent/best.pt",
    "stage5": PICK / "stage5_level/best.pt",
    "seed2_nominal": PICK / "seed2_nominal/best.pt",
    "seed2_finetuned": PICK / "seed2_finetuned/best.pt",
}
MONITORS = {  # headline trigger, its call-matched random control, extra cohorts
    "stage5": dict(task="PickCube", data="data/monitor_level", test="monitor_level/test.json",
                   conf="monitor_confirmation", predictor=PICK / "monitor_level/predictor.pt",
                   trigger="acceleration_calls15", control="random_control", sweep=True,
                   extra=["monitor_grasp_phase", "monitor_grasp_phase_b", "monitor_long_pulse"]),
    "seed2_finetuned": dict(task="PickCube", data="data/monitor_seed2", test="monitor_seed2/test.json",
                            conf="monitor_seed2_confirmation", predictor=PICK / "monitor_seed2/predictor.pt",
                            trigger="acceleration_calls15", control="random_control", sweep=False, extra=[]),
    "stack": dict(task="StackCube", data="data/monitor_stack", test="monitor_stack/test.json",
                  conf="monitor_stack_confirmation", predictor=Path("checkpoints/stackcube/monitor/predictor.pt"),
                  trigger="acceleration_calls14", control="random_control_acceleration_calls14", sweep=True,
                  extra=["monitor_stack_carry", "monitor_stack_long_pulse"]),
    "liftpeg": dict(task="LiftPegUpright", data="data/monitor_liftpeg", test="monitor_liftpeg/test.json",
                    conf="monitor_liftpeg_confirmation", predictor=Path("checkpoints/liftpeg/monitor/predictor.pt"),
                    trigger="acceleration_calls25", control="random_control", sweep=False, extra=[]),
}
MONITOR_POLICIES = {"stack": "checkpoints/stackcube/policy_120k/best_ddim10.pt",
                    "liftpeg": "checkpoints/liftpeg/policy_01_recovery/last.pt"}
KEYS = [*LINEAGE, "stack", "liftpeg"]
COMPARED = ("success_steps", "replan_steps", "policy_calls", "success_at_end", "success_confirmation_step")
TIME = "capped_confirmation_seconds"


def load(path):
    data = json.loads(Path(path).read_text())
    return data["episodes"] if isinstance(data, dict) and "episodes" in data else data


def first_groups(groups, limit):
    return groups if limit is None else groups[:limit]


def cases(groups, forces):
    from adaptive_action_chunking.evaluate import _dynamics_cases
    return _dynamics_cases(groups, forces)


def cohort(name, task, horizon, policies, tasks, saved, predictor=None):
    return dict(name=name, task=task, horizon=horizon, policies=policies, tasks=tasks, saved=saved,
                predictor=str(predictor) if predictor else None)


# ---------- cohort definitions (all inputs read from the saved study files) ----------

def monitor_cohorts(key, limit):
    spec = MONITORS[key]
    meta = json.loads(Path(spec["data"], "metadata.json").read_text())
    policies = {"policy": (meta["policy"], None, None)}  # sampler as stored in the checkpoint (DDIM10)
    test_groups = first_groups([g for g in meta["groups"] if g["partition"] == "test"], limit)
    conf = R / spec["conf"]
    plan = json.loads((conf / "plan.json").read_text())
    calibration = json.loads((conf / "calibration.json").read_text())
    controls = ([{"variant": c["control_variant"], "kind": "random", "threshold": c["threshold"]}
                 for c in calibration["controls"].values()] if "controls" in calibration else
                [{"variant": "random_control", "kind": "random", "threshold": calibration["chosen"]["threshold"]}])
    test_methods = json.loads((R / spec["test"]).read_text())["methods"]
    parts = [("test", test_groups, meta["force_levels_n"], test_methods, R / spec["test"]),
             ("control", test_groups, meta["force_levels_n"], controls, conf / "control_on_test.json"),
             ("confirmation", first_groups(plan["groups"], limit), plan["forces"], plan["methods"],
              conf / "confirmation.json")]
    if spec["sweep"]:
        chunks = json.loads((conf / "fixed_k_sweep.json").read_text())["chunks"]
        sweep = [{"variant": f"fixed_k{c}", "kind": "fixed", "chunk_length": c, "threshold": None} for c in chunks]
        parts.append(("sweep", test_groups + first_groups(plan["groups"], limit), plan["forces"], sweep,
                      conf / "fixed_k_sweep.json"))
    for folder in spec["extra"]:
        extra = json.loads((R / folder / "plan.json").read_text())
        parts.append((folder, first_groups(extra["groups"], limit), extra["forces"], extra["methods"],
                      R / folder / "confirmation.json"))
    result = []
    for name, groups, forces, methods, saved_path in parts:
        tasks = [{**case, **method, "chunk_length": method.get("chunk_length", 15), "policy": "policy"}
                 for case in cases(groups, forces) for method in methods]
        saved = {(e["episode_id"], e["variant"]): e for e in load(saved_path)}
        result.append(cohort(name, spec["task"], meta["horizon"], policies, tasks, saved, spec["predictor"]))
    return result


def lineage_cohort(names, limit):
    plan = json.loads((R / "policy_level_eval/plan.json").read_text())
    groups = list(dict.fromkeys(c["group_id"] for c in plan["cases"]))
    keep = set(first_groups(groups, limit))
    labels = {"retained": "stage3", "start": "stage4", "updated": "stage5"}  # checkpoint labels used in that evaluation
    saved = {(e["episode_id"], f"{labels[e['checkpoint']]}_k{e['chunk_length']}"): e
             for e in load(R / "policy_level_eval/results.json") if labels[e["checkpoint"]] in names}
    tasks = [{**case, "policy": n, "variant": f"{n}_k{k}", "kind": "fixed", "threshold": None, "chunk_length": k}
             for case in plan["cases"] if case["group_id"] in keep for n in names for k in (8, 15)]
    policies = {n: (str(LINEAGE[n]), "ddim", 10) for n in names}
    return cohort("lineage", "PickCube", 150, policies, tasks, saved)


def closing_cohort(names, limit):
    root = R / "policy_closing_confirmation"
    plan = json.loads((root / "comparison/plan.json").read_text())
    groups = list(dict.fromkeys(c["group_id"] for c in plan["cases"]))
    keep = set(first_groups(groups, limit))
    labels = {"stage2": "retained", "stage3": "closing"}
    tasks = [{**case, "policy": n, "variant": f"{labels[n]}_k{k}", "kind": "fixed", "threshold": None,
              "chunk_length": k} for case in plan["cases"] if case["group_id"] in keep for n in names for k in (8, 15)]
    saved = {(e["episode_id"], e["variant"]): e for e in load(root / "horizon150/results.json")}
    policies = {n: (str(LINEAGE[n]), None, None) for n in names}  # stored sampler: DDPM100, as in the original
    return cohort("closing", "PickCube", 150, policies, tasks, saved)


def stack_nominal_cohort(limit):
    root = R / "stackcube_nominal_check"
    plan = json.loads((root / "plan.json").read_text())
    tasks = [{**g, "episode_id": f"{g['group_id']}_f0_k{k}", "force_n": 0.0, "duration_steps": 2,
              "force_xy": [0.0, 0.0], "kind": "fixed", "threshold": None, "chunk_length": k,
              "variant": f"fixed_k{k}", "policy": "policy"}
             for g in first_groups(plan["groups"], limit) for k in plan["chunks"]]
    saved = {(f"{r['group_id']}_f0_k{r['chunk']}", f"fixed_k{r['chunk']}"):
             {"policy_calls": r["calls"], "success_at_end": r["success_at_end"],
              "success_confirmation_step": r["confirmed_step"]} for r in load(root / "results.json")["rows"]}
    return cohort("nominal_check", "StackCube", plan["horizon"], {"policy": (plan["policy"], None, None)}, tasks, saved,
                  MONITORS["stack"]["predictor"])  # unused by fixed chunks; shares the monitor-study worker pool


# ---------- rollouts ----------

WORKER = {}


def _init(task, policies, predictor, horizon):
    os.environ["ROBOTS_TASK"] = task
    import torch
    from adaptive_action_chunking.monitor import DynamicsPredictor
    from adaptive_action_chunking.env import extract_policy_state, make_env
    from adaptive_action_chunking.policy import load_policy

    torch.set_num_threads(1)
    cpu = torch.device("cpu")
    env = make_env(max_episode_steps=horizon, disturbance={"start_step": 0, "duration_steps": 0, "force_xy": [0., 0.]})
    loaded = {name: load_policy(path, device=cpu, sampler=sampler, inference_steps=steps)
              for name, (path, sampler, steps) in policies.items()}
    obs, _ = env.reset(seed=1)
    for policy in loaded.values():  # same warm-up call as the original workers
        policy.sample(np.stack([extract_policy_state(obs)] * 2), torch.Generator().manual_seed(0), cpu)
    WORKER.update(env=env, policies=loaded, horizon=horizon,
                  predictor=DynamicsPredictor.load(predictor) if predictor else None)


def _job(task):
    import torch
    from adaptive_action_chunking.monitor import MotionTrigger
    from adaptive_action_chunking.evaluate import _motion_outcomes, run_episode

    env, horizon = WORKER["env"], WORKER["horizon"]
    env.unwrapped.disturbance = {key: task[key] for key in ("start_step", "duration_steps", "force_xy")}
    trigger = None if task["kind"] == "fixed" else MotionTrigger(task["kind"], task["threshold"], WORKER["predictor"])
    episode = run_episode(env, WORKER["policies"][task["policy"]], task["chunk_length"], task["reset_seed"],
                          torch.device("cpu"), horizon=horizon, policy_seed=task["policy_seed"],
                          motion_trigger=trigger, collect_positions=True)
    expected = list(range(task["start_step"], task["start_step"] + task["duration_steps"])) if any(task["force_xy"]) else []
    keep = ("success_steps", "replan_steps", "policy_calls", "success_at_end", "success_once", "steps",
            "policy_seconds", "monitor_seconds")
    out = {key: episode[key] for key in keep}
    out.update({key: task[key] for key in ("episode_id", "group_id", "force_n", "variant", "policy")})
    out.update(_motion_outcomes(task, episode), inference_seconds=episode["policy_seconds"] + episode["monitor_seconds"],
               schedule_ok=episode["steps"] == horizon and episode["applied_force_steps"] == expected)
    return out


POOL = {}


def run(spec, workers):
    """Run a cohort; consecutive cohorts with the same policies, predictor and horizon share one worker pool."""
    initargs = (spec["task"], spec["policies"], spec["predictor"], spec["horizon"])
    if POOL.get("initargs") != initargs:
        if POOL:
            POOL.pop("pool").shutdown()
        os.environ["ROBOTS_TASK"] = spec["task"]  # spawned workers inherit the task selection
        POOL.update(initargs=initargs, pool=ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                                                initializer=_init, initargs=initargs))
    started, episodes = time.perf_counter(), []
    for index, episode in enumerate(POOL["pool"].map(_job, spec["tasks"], chunksize=1), 1):
        episodes.append(episode)
        if index % 200 == 0:
            print(f"  {spec['name']}: {index}/{len(spec['tasks'])}", flush=True)
    for e in episodes:
        saved = spec["saved"].get((e["episode_id"], e["variant"]))
        fields = [k for k in COMPARED if k in saved] + ([TIME] if TIME in saved else []) if saved else []
        e["identical"] = None if saved is None else e["schedule_ok"] and all(e[k] == saved[k] for k in fields)
    print(f"  {spec['name']}: done in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return episodes


def check(episodes):
    checked = [e for e in episodes if e["identical"] is not None]
    return {"rollouts": len(episodes), "checked_against_saved": len(checked),
            "identical": sum(e["identical"] for e in checked),
            "mismatched": [f"{e['episode_id']} {e['variant']}" for e in checked if not e["identical"]]}


# ---------- recomputed numbers (rerun vs saved) ----------

def diff(episodes, variant, base, seed):
    from adaptive_action_chunking.evaluate import _dynamics_summary
    paired = _dynamics_summary(episodes, {"samples": 5000, "seed": seed}, base)[1][variant]
    return [round(paired[TIME + "_difference"], 3), [round(x, 3) for x in paired["ci95"][TIME + "_difference"]]]


def monitor_numbers(key, runs, limit):
    spec, conf = MONITORS[key], R / MONITORS[key]["conf"]
    trig, ctrl = spec["trigger"], spec["control"]
    numbers = {}
    pooled = [e for name in ("test", "control", "confirmation") for e in runs[name]]
    variants = {e["variant"] for e in runs["confirmation"]}
    pooled = [e for e in pooled if e["variant"] in variants]
    saved = json.loads((conf / "pooled.json").read_text())
    from adaptive_action_chunking.evaluate import _dynamics_summary
    summary = _dynamics_summary(pooled, {"samples": 5000, "seed": 339002})[0]
    numbers["pooled time K15 -> trigger (s)"] = (
        [round(summary["fixed_k15"][TIME], 3), round(summary[trig][TIME], 3)],
        [round(saved["summary"]["fixed_k15"][TIME], 3), round(saved["summary"][trig][TIME], 3)])
    numbers["pooled final success K15 -> trigger"] = (
        [round(summary["fixed_k15"]["success_at_end"], 4), round(summary[trig]["success_at_end"], 4)],
        [round(saved["summary"]["fixed_k15"]["success_at_end"], 4), round(saved["summary"][trig]["success_at_end"], 4)])
    numbers["pooled calls trigger / random"] = (
        [round(summary[trig]["policy_calls"], 2), round(summary[ctrl]["policy_calls"], 2)],
        [round(saved["summary"][trig]["policy_calls"], 2), round(saved["summary"][ctrl]["policy_calls"], 2)])
    for base, label in (("fixed_k15", "vs K15"), (ctrl, "vs random")):
        stored = saved["paired_vs_k15" if base == "fixed_k15" else f"paired_vs_{base}"][trig]
        numbers[f"trigger {label} (s) [95% CI]"] = (diff(pooled, trig, base, 339002), [
            round(stored[TIME + "_difference"], 3), [round(x, 3) for x in stored["ci95"][TIME + "_difference"]]])
    if "sweep" in runs:
        stored = json.loads((conf / "fixed_k_sweep.json").read_text())["summary"]
        rerun = _dynamics_summary(runs["sweep"], {"samples": 5000, "seed": 339002})[0]
        for v in ("fixed_k1", "fixed_k2"):
            numbers[f"sweep {v} time (s) / final success"] = (
                [round(rerun[v][TIME], 3), round(rerun[v]["success_at_end"], 4)],
                [round(stored[v][TIME], 3), round(stored[v]["success_at_end"], 4)])
    for folder in spec["extra"]:
        plan = json.loads((R / folder / "plan.json").read_text())
        stored = json.loads((R / folder / "confirmation.json").read_text())["paired_vs_k15"][trig]
        numbers[f"{folder}: trigger vs K15 (s) [95% CI]"] = (diff(runs[folder], trig, "fixed_k15", plan["seed"] + 1), [
            round(stored[TIME + "_difference"], 3), [round(x, 3) for x in stored["ci95"][TIME + "_difference"]]])
    if limit is not None:
        numbers = {k: (v[0], f"(saved, all groups) {v[1]}") for k, v in numbers.items()}
    return numbers


def success_counts(episodes, variant):
    rows = [e for e in episodes if e["variant"] == variant]
    return f"{sum(e['success_at_end'] for e in rows)}/{len(rows)}"


def lineage_numbers(episodes, names):
    saved = {}
    for e in load(R / "policy_level_eval/results.json"):
        label = {"retained": "stage3", "start": "stage4", "updated": "stage5"}[e["checkpoint"]]
        saved.setdefault(f"{label}_k{e['chunk_length']}", []).append(e)
    numbers = {}
    for n in names:
        for k in (8, 15):
            v = f"{n}_k{k}"
            rows = [e for e in episodes if e["variant"] == v]
            by_force = {f"{f:g}": sum(e["success_at_end"] for e in rows if e["force_n"] == f)
                        for f in sorted({e["force_n"] for e in rows})}
            rerun = {"final success": success_counts(episodes, v), "by force": by_force,
                     "mean time to goal (s)": round(float(np.mean([e[TIME] for e in rows])), 3)}
            stored = saved.get(v)
            numbers[f"{n} K{k}"] = (rerun, f"{sum(e['success_at_end'] for e in stored)}/{len(stored)}" if stored else "not saved")
    return numbers


def closing_numbers(episodes, limit):
    saved = json.loads((R / "policy_closing_confirmation/summary.json").read_text())
    numbers = {f"{v} final success at 150": (success_counts(episodes, v), f"{saved['summaries']['150'][v]['final_success']}/400")
               for v in ("retained_k8", "closing_k8", "retained_k15", "closing_k15")
               if any(e["variant"] == v for e in episodes)}
    if all(any(e["variant"] == v for e in episodes) for v in ("retained_k8", "closing_k8")):
        groups = sorted({e["group_id"] for e in episodes})
        base = {e["episode_id"]: e["success_at_end"] for e in episodes if e["variant"] == "retained_k8"}
        rows = [(e["group_id"], int(e["success_at_end"]) - int(base[e["episode_id"]]))
                for e in episodes if e["variant"] == "closing_k8"]
        grouped = np.array([np.mean([d for g, d in rows if g == group]) for group in groups])
        draws = np.random.default_rng(169002).integers(len(groups), size=(5000, len(groups)))
        primary = saved["primary_k8_150"]
        numbers["closing - retained, K8 at 150 [95% CI]"] = (
            [round(float(grouped.mean()), 4), np.round(np.quantile(grouped[draws].mean(1), [.025, .975]), 4).tolist()],
            [primary["difference"], primary["ci95"]])
    return numbers


def nominal_numbers(episodes):
    rows = load(R / "stackcube_nominal_check/results.json")["rows"]
    return {f"nominal final success {v}": (success_counts(episodes, v), "{}/{}".format(
        sum(r["success_at_end"] for r in rows if f"fixed_k{r['chunk']}" == v),
        sum(f"fixed_k{r['chunk']}" == v for r in rows))) for v in sorted({e["variant"] for e in episodes})}


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policies", nargs="*", metavar="POLICY", help=f"any of {' '.join(KEYS)} (default: all)")
    parser.add_argument("--groups", type=int, help="re-run only the first N reset groups of every cohort")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir
    selected = args.policies or KEYS
    if unknown := set(selected) - set(KEYS):
        parser.error(f"unknown policy {sorted(unknown)}; choose from {' '.join(KEYS)}")
    out.mkdir(parents=True, exist_ok=True)
    reports = {key: {"policy": str(LINEAGE.get(key, MONITOR_POLICIES.get(key))),
                     "groups_limit": args.groups, "cohorts": {}, "numbers": {}} for key in selected}

    names = [n for n in LINEAGE if n in selected]
    if names:
        spec = lineage_cohort(names, args.groups)
        print(f"lineage benchmark: {len(spec['tasks'])} rollouts for {' '.join(names)}", flush=True)
        episodes = run(spec, args.workers)
        numbers = lineage_numbers(episodes, names)
        for n in names:
            reports[n]["cohorts"]["lineage"] = check([e for e in episodes if e["policy"] == n])
            reports[n]["numbers"].update({k: v for k, v in numbers.items() if k.split()[0] == n})
    closing = [n for n in ("stage2", "stage3") if n in selected]
    if closing:
        spec = closing_cohort(closing, args.groups)
        print(f"closing confirmation: {len(spec['tasks'])} rollouts (DDPM100)", flush=True)
        episodes = run(spec, args.workers)
        for n in closing:
            reports[n]["cohorts"]["closing"] = check([e for e in episodes if e["policy"] == n])
            reports[n]["numbers"].update(closing_numbers(episodes, args.groups))
    for key in MONITORS:
        if key not in selected:
            continue
        runs = {}
        specs = monitor_cohorts(key, args.groups) + ([stack_nominal_cohort(args.groups)] if key == "stack" else [])
        for spec in specs:
            print(f"{key} {spec['name']}: {len(spec['tasks'])} rollouts", flush=True)
            runs[spec["name"]] = run(spec, args.workers)
            reports[key]["cohorts"][spec["name"]] = check(runs[spec["name"]])
        reports[key]["numbers"].update(monitor_numbers(key, runs, args.groups))
        if "nominal_check" in runs:
            reports[key]["numbers"].update(nominal_numbers(runs["nominal_check"]))

    if POOL:
        POOL.pop("pool").shutdown()
    print(f"\n{'policy':15} cohort                   rollouts  identical to saved")
    for key, report in reports.items():
        (out / f"{key}.json").write_text(json.dumps(report, indent=1))
        for name, c in report["cohorts"].items():
            same = f"{c['identical']}/{c['checked_against_saved']}" if c["checked_against_saved"] else "no saved rollouts"
            print(f"{key:15} {name:24} {c['rollouts']:8}  {same}")
    print("\nrecomputed numbers: rerun | saved")
    for key, report in reports.items():
        for label, (rerun, saved) in report["numbers"].items():
            print(f"{key:15} {label}: {rerun} | {saved}")
    bad = sum(len(c["mismatched"]) for r in reports.values() for c in r["cohorts"].values())
    print(f"\n{'ALL IDENTICAL' if not bad else f'{bad} MISMATCHED ROLLOUTS'}; details in {out}/")


if __name__ == "__main__":
    main()
