"""Collect expert chunks conditioned on fresh learner approach/retry states."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import torch

from adaptive_action_chunking.artifacts import digest
from adaptive_action_chunking.monitor import _reset_seeds, reset_seed_sources, source_hashes, write_json
from adaptive_action_chunking.env import extract_policy_state, make_env
from adaptive_action_chunking.evaluate import run_episode
from adaptive_action_chunking.corrections.pickup import PickupTeacher, PICKUP_CONTROLLER
from adaptive_action_chunking.policy import load_policy


POLICY = Path('checkpoints/pickcube/stage2_pickup/last.pt')
STRATA = (('approach', 0.), ('approach', .4), ('retry', .3), ('retry', .4))
HORIZON, CHUNK, TARGET_ACTIONS = 150, 8, 15
WORKER = {}


def choose_query(states, actions, kind, force):
    """Query a real K8 conditioning state before a poorly centered closing onset."""
    held = states[:, 18] > .5
    raised = held & (states[:, 31] >= states[0, 31]+.025)
    lifts = np.flatnonzero(np.convolve(raised.astype(int), np.ones(5, dtype=int), 'valid') == 5)
    first_lift = int(lifts[0]+4) if len(lifts) else len(states)
    losses = np.flatnonzero(held[:-1] & ~held[1:])+1
    closing = actions[:, 3] < -.5
    starts = np.flatnonzero(closing & ~np.r_[False, closing[:-1]])
    candidates = starts[:1] if kind == 'approach' else starts[1:]
    for close in candidates:
        query = int(close//CHUNK*CHUNK)
        offset = float(np.linalg.norm(states[close, 19:21]-states[close, 29:31]))
        threshold = .006 if force == 0 else .015
        if not 8 <= query <= 80 or close >= first_lift or offset < threshold:
            continue
        state = states[query]
        if (held[query] or state[7:9].sum() < .04 or state[31] < .015
                or state[31] > states[0, 31]+.015 or np.any(np.abs(state[29:31]) > .3)
                or np.linalg.norm(state[19:21]-state[29:31]) < .008):
            continue
        if kind == 'approach':
            if held[:query+1].any(): continue
            if force and not np.any((losses > close) & (losses <= first_lift)): continue
        elif not np.any(losses <= query):
            continue
        return query, {'source_close_step': int(close), 'source_close_xy_mm': offset*1000,
                       'query_xy_mm': float(np.linalg.norm(state[19:21]-state[29:31])*1000),
                       'prior_grasp_losses': int((losses <= query).sum())}
    return None, {}


def initialize():
    torch.set_num_threads(1)
    WORKER.update(env=make_env(max_episode_steps=HORIZON, disturbance={
        'start_step': 0, 'duration_steps': 0, 'force_xy': [0., 0.]}),
        policy=load_policy(POLICY, device='cpu'))


def collect_case(case):
    env = WORKER['env']
    env.unwrapped.disturbance = {k: case[k] for k in ('start_step', 'duration_steps', 'force_xy')}
    source = run_episode(env, WORKER['policy'], CHUNK, case['reset_seed'], torch.device('cpu'),
                         horizon=HORIZON, policy_seed=case['policy_seed'], collect_positions=True)
    states, actions = np.asarray(source['states'], dtype=np.float32), np.asarray(source['actions'], dtype=np.float32)
    assert states.shape == (151, 42) and actions.shape == (150, 4)
    query, details = choose_query(states, actions, case['kind'], case['force_n'])
    flags = np.zeros(HORIZON, dtype=bool); flags[np.asarray(source['success_steps'], dtype=int)-1] = True
    arrays = {'policy_obs': states, 'policy_actions': actions, 'policy_success': flags,
              'policy_plan_steps': np.asarray(source['replan_steps']),
              'policy_plans': np.asarray([p['actions'] for p in source['plans']], dtype=np.float32)}
    row = {**case, **details, 'query_step': query, 'qualified': False, 'reason': 'no_eligible_query',
           'source_success_at_100': bool(flags[99]), 'source_success_at_150': bool(flags[-1])}
    if query is None: return row, arrays
    assert query in source['replan_steps']
    obs, _ = env.reset(seed=case['reset_seed'])
    state = extract_policy_state(obs)
    np.testing.assert_array_equal(state, states[0])
    for step in range(query):
        obs, _, _, _, _ = env.step(actions[step])
        state = extract_policy_state(obs)
        np.testing.assert_array_equal(state, states[step+1])
    teacher = PickupTeacher(state, reset_cube_z=float(states[0, 31]))
    observations, corrections, success = [state.copy()], [], []
    for step in range(query, HORIZON):
        action = teacher.action(env, state)
        if teacher.failed:
            row['reason'] = 'teacher_attempts_exhausted'
            break
        obs, _, _, truncated, info = env.step(action)
        state = extract_policy_state(obs)
        observations.append(state.copy()); corrections.append(action.copy())
        success.append(bool(info['success'].item()))
        assert bool(truncated.item()) == (step == HORIZON-1)
    observations, corrections = np.asarray(observations), np.asarray(corrections)
    distances = np.linalg.norm(observations[:, 29:32]-observations[:, 26:29], axis=1)
    qualified = (len(corrections) == HORIZON-query and teacher.verified
                 and len(corrections) >= TARGET_ACTIONS and all(success[-10:])
                 and np.all(observations[-10:, 18] > .5) and np.all(distances[-10:] <= .010))
    arrays.update(correction_obs=observations, correction_actions=corrections,
                  correction_success=np.asarray(success, dtype=bool), context_obs=states[query-1:query+1])
    row.update(qualified=bool(qualified), reason='qualified' if qualified else row['reason'] if teacher.failed else 'teacher_not_settled_at_150',
               prefix_replay_exact=True, teacher_verified=teacher.verified, teacher_attempts=teacher.attempts,
               teacher_events=teacher.events, teacher_final_distance_mm=float(distances[-1]*1000),
               correction_steps=len(corrections))
    expected = list(range(case['start_step'], case['start_step']+2)) if case['force_n'] else []
    assert sorted(env.unwrapped.applied_force_steps) == expected
    assert env.unwrapped.applied_force_substeps == len(expected)*5
    return row, arrays


def export_query(handle, trajectory, arrays):
    group = handle.create_group(trajectory)
    group.attrs['conditioning'] = 'learner_takeover'
    group['obs'] = arrays['correction_obs'][:TARGET_ACTIONS+1]
    group['actions'] = arrays['correction_actions'][:TARGET_ACTIONS]
    group['context_obs'] = arrays['context_obs']
    group['action_is_expert'] = np.ones(TARGET_ACTIONS, dtype=bool)


def make_plan(pilot):
    excluded, scanned = set(), {}
    for path in reset_seed_sources():
        records = [json.loads(line) for line in path.read_text().splitlines() if line] if path.suffix == '.jsonl' else [json.loads(path.read_text())]
        seeds = set().union(*(set(_reset_seeds(r)) for r in records))
        excluded.update(seeds); scanned[str(path)] = {'sha256': digest(path), 'seed_count': len(seeds)}
    seed = 90308 if pilot else 90309
    rng, seen, cases, targets = np.random.default_rng(seed), set(excluded), [], {}
    for split, count in ([('pilot', 1)] if pilot else [('train', 8), ('validation', 2)]):
        for kind, force in STRATA:
            targets[split, kind, force] = count
            for index in range(160 if pilot else count*200):
                reset = int(rng.integers(10000, 2**31))
                while reset in seen: reset = int(rng.integers(10000, 2**31))
                seen.add(reset)
                angle = float(rng.uniform(0, 2*np.pi))
                cases.append({'episode_id': f'learner_{split}_{reset}', 'split': split, 'kind': kind,
                    'reset_seed': reset, 'policy_seed': int(rng.integers(0, 2**31)), 'force_n': force,
                    'force_xy': [float(force*np.cos(angle)), float(force*np.sin(angle))],
                    'start_step': int(rng.integers(8, 19)), 'duration_steps': 2, 'candidate_index': index})
    code = {**source_hashes(), str(Path(__file__)): digest(__file__), **{p: digest(p) for p in (
        'src/adaptive_action_chunking/corrections/pickup.py', 'src/adaptive_action_chunking/corrections/collect.py',
        'src/adaptive_action_chunking/finetune.py', 'uv.lock')}}
    plan = {'complete': False, 'mode': 'learner_query_pilot' if pilot else 'fresh_collection',
        'collection_kind': 'learner_conditioned_approach_retry_chunks', 'planned_at_utc': datetime.now(timezone.utc).isoformat(),
        'collection_seed': seed, 'policy': str(POLICY), 'policy_sha256': digest(POLICY),
        'source_horizon': HORIZON, 'chunk_length': CHUNK, 'expert_target_actions': TARGET_ACTIONS,
        'controller': PICKUP_CONTROLLER, 'cases': cases,
        'targets_by_stratum': {f'{s}_{k}_{f:g}': n for (s,k,f),n in targets.items()},
        'selection': 'First approach or first eligible retry closing onset; query its actual K8 plan boundary. Query8..80, unheld, fingers>=40mm, cube near table, current XY>=8mm; close XY>=6mm nominal/15mm disturbed. Disturbed approach requires contact loss before sustained lift; retry requires a prior loss and no prior lift.',
        'acceptance': 'Existing expert reaches verified lift, held official success at150 and ten final successes within10mm. Export only first15 expert actions conditioned on the actual learner history.',
        'sampling': 'One query per fresh reset. Half recovery batch draws reserved for tagged queries, half pooled accumulated recovery; nominal fraction unchanged.',
        'candidate_order': 'Round-robin live strata, up to workers candidates per batch. Export first qualified results in planned order until quotas met; surplus qualified results are recorded but excluded.',
        'excluded_reset_seed_count': len(excluded), 'overlap_sources': scanned, 'code_sha256': code,
        'preserved_sha256': {str(p): digest(p) for folder in ('data','checkpoints','results') for p in Path(folder).rglob('*') if p.is_file()}}
    return plan, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true', help='Collect four disjoint pilot queries, excluded from training')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir or Path('results/debugging/learner_correction_pilot' if args.pilot else 'data/recovery_learner')
    if output.exists() or args.workers < 1: parser.error('Use a fresh output directory and positive workers')
    plan, targets = make_plan(args.pilot)
    output.mkdir(parents=True); (output/'rollouts').mkdir()
    write_json(output/'plan.json', plan)
    write_json(output/'metadata.json', plan)
    queues = {key: [c for c in plan['cases'] if (c['split'],c['kind'],c['force_n']) == key] for key in targets}
    counts, used, attempts, accepted, repeats = Counter(), Counter(), [], [], {}
    files = {split: h5py.File(output/f'{split}.h5','x') for split in sorted({k[0] for k in targets})}
    print(f"Reserved {len(plan['cases'])} fresh groups, excluding {plan['excluded_reset_seed_count']}; target {sum(targets.values())} queries", flush=True)
    try:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
            while True:
                live = [k for k in targets if counts[k] < targets[k] and used[k] < len(queues[k])]
                if not live: break
                batch = []
                while len(batch) < args.workers and live:
                    for key in list(live):
                        batch.append(queues[key][used[key]]); used[key] += 1
                        if used[key] == len(queues[key]): live.remove(key)
                        if len(batch) == args.workers: break
                for (row, arrays) in pool.map(collect_case, batch):
                    key = row['split'], row['kind'], row['force_n']
                    raw = output/'rollouts'/f"{row['episode_id']}.npz"
                    np.savez_compressed(raw, **arrays)
                    exported = row['qualified'] and counts[key] < targets[key]
                    if row['qualified'] and not exported: row['reason'] = 'quota_filled'
                    row.update(accepted=bool(exported), raw_trace=str(raw), raw_trace_sha256=digest(raw))
                    if exported:
                        trajectory = f"traj_{sum(e['split'] == row['split'] for e in accepted)}"
                        export_query(files[row['split']], trajectory, arrays)
                        row.update(trajectory=trajectory, episode_seed=row['reset_seed'], teacher='pickup',
                                   pickup_events=[e for e in row['teacher_events'] if e['step'] <= TARGET_ACTIONS])
                        if counts[key] == 0: repeats[key] = (row, next(c for c in batch if c['episode_id'] == row['episode_id']))
                        counts[key] += 1; accepted.append(row)
                    attempts.append(row)
                write_json(output/'attempts.json', {'attempts': attempts})
                print(f"Sources {len(attempts)}; exported {len(accepted)}/{sum(targets.values())}", flush=True)
            for (previous, case), (row, arrays) in zip(repeats.values(), pool.map(collect_case, [c for _,c in repeats.values()])):
                with np.load(previous['raw_trace']) as original:
                    assert set(original.files) == set(arrays)
                    for k,a in arrays.items(): np.testing.assert_array_equal(a,original[k])
                for k in ('query_step','qualified','teacher_events'): assert row[k] == previous[k]
    finally:
        for handle in files.values(): handle.close()
    complete = all(counts[k] == n for k,n in targets.items())
    for split in files:
        write_json(output/f'{split}.json', {'complete': complete, 'split': split,
            'episodes': [r for r in accepted if r['split'] == split]})
    for p,h in {**plan['preserved_sha256'],**plan['code_sha256']}.items(): assert digest(p) == h,p
    plan.update(complete=complete, completed_at_utc=datetime.now(timezone.utc).isoformat(), attempts=attempts, accepted=accepted,
        accepted_counts={s:sum(r['split']==s for r in accepted) for s in files},
        accepted_by_stratum={f'{s}_{k}_{f:g}':counts[s,k,f] for s,k,f in targets},
        exact_repeats=len(repeats), artifact_sha256={f'{s}.h5':digest(output/f'{s}.h5') for s in files})
    write_json(output/'metadata.json', plan)
    print(json.dumps({'complete':complete,'accepted_counts':plan['accepted_counts'],'sources':len(attempts)},indent=2))
    if not complete: raise RuntimeError('Candidate pool exhausted; partial data remain excluded from training')


if __name__ == '__main__': main()
