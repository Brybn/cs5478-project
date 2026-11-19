"""Collect a small split correction round after the alignment pilot passes."""

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
from adaptive_action_chunking.monitor import _reset_seeds, reset_seed_sources
from adaptive_action_chunking.evaluate import run_episode
from adaptive_action_chunking.corrections.collect import _WORKER, _collect_one, _collection_code, _init_worker, _save_attempt, write_json

POLICY = Path('checkpoints/pickcube/stage2_pickup/last.pt')
FORCES, TARGETS = (0., .4), {'alignment': 2, 'placement': 1, 'successful': 1}
SEED = 90307


def choose_intervention(source, group, force):
    states, actions = np.asarray(source['states']), np.asarray(source['actions'])
    held = states[:, 18] > .5
    xy = np.linalg.norm(states[:, 19:21] - states[:, 29:31], axis=1)
    width = states[:, 7:9].sum(1)
    raised = held & (states[:, 31] >= states[0, 31] + .025)
    lifts = np.flatnonzero(np.convolve(raised.astype(int), np.ones(5, dtype=int), mode='valid') == 5)
    first_lift = int(lifts[0] + 4) if len(lifts) else None
    closing = actions[:, 3] < -.5
    closures = np.flatnonzero(closing & ~np.r_[False, closing[:-1]])
    first_close = int(closures[0]) if len(closures) else None
    losses = np.flatnonzero(held[:-1] & ~held[1:]) + 1
    fragile = bool(len(losses) and (first_lift is None or losses[0] <= first_lift))
    details = {
        'source_success_at_end': bool(source['success_at_end']),
        'first_close_step': first_close,
        'first_close_xy_mm': float(xy[first_close] * 1000) if first_close is not None else None,
        'fragile_first_pickup': fragile,
        'source_sustained_lift_step': first_lift,
    }
    if group == 'successful':
        return (0, 'representative_successful_sequence', details) if source['success_at_end'] else (None, None, details)
    if group == 'alignment':
        threshold = .006 if force == 0 else .015
        details['selection_xy_threshold_mm'] = threshold * 1000
        if first_close is None or (xy[first_close] < threshold and not fragile):
            return None, None, details
        step = first_close - 8
        if (
            step < 1
            or held[:step + 1].any()
            or width[step] < .04
            or states[step, 31] < .015
            or np.any(np.abs(states[step, 29:31]) > .3)
        ):
            return None, None, details
        details['takeover_xy_mm'] = float(xy[step] * 1000)
        return step, 'preclosure_fragile_pickup' if fragile else 'preclosure_alignment_intervention', details
    if group == 'placement':
        for step in range(4, len(states)):
            if (
                held[step - 4:].all()
                and (width[step - 4:step + 1] >= .025).all()
                and (xy[step - 4:step + 1] <= .015).all()
            ):
                if states[step, 31] >= states[0, 31] + .025 and states[step, 28] < states[0, 31] + .025:
                    continue
                return step, 'held_placement_intervention', details
        return None, None, details
    raise ValueError(f'Unknown group: {group}')


def collect_one(task):
    case, method = task['case'], task['method']
    env = _WORKER['env']
    env.unwrapped.disturbance = {k: case[k] for k in ('start_step', 'duration_steps', 'force_xy')}
    source = run_episode(
        env, _WORKER['policy'], 8, case['reset_seed'], torch.device('cpu'),
        horizon=100, policy_seed=case['policy_seed'], collect_positions=True,
    )
    step, kind, details = choose_intervention(source, case['target_group'], case['force_n'])
    case = {**case, 'selection_details': details}
    if step is None:
        return {
            'case': case, 'method': method, 'source': source, 'teacher': 'pickup',
            'takeover_step': None, 'takeover_kind': None, 'accepted': False, 'reason': 'source_not_selected',
        }
    selected = {
        **task, 'case': case, 'expected_source': {**source, 'initial_state': source['states'][0]},
        'teacher': 'pickup',
    }
    if step:
        selected['takeover_step'] = step
    result = _collect_one(selected)
    assert result['takeover_step'] == step
    result['takeover_kind'] = kind
    return result


def training_case(task):
    """Require a failed disturbed source before accepting its alignment label."""
    case = task['case']
    if case['force_n'] == 0 or case['target_group'] != 'alignment': return collect_one(task)
    env = _WORKER['env']
    env.unwrapped.disturbance = {k: case[k] for k in ('start_step', 'duration_steps', 'force_xy')}
    source = run_episode(env, _WORKER['policy'], 8, case['reset_seed'], torch.device('cpu'),
                         horizon=100, policy_seed=case['policy_seed'], collect_positions=True)
    step, kind, details = choose_intervention(source, 'alignment', case['force_n'])
    case = {**case, 'selection_details': details}
    if source['success_at_end'] or step is None:
        return {'case': case, 'method': task['method'], 'source': source, 'teacher': 'pickup',
                'takeover_step': None, 'takeover_kind': None, 'accepted': False, 'reason': 'source_not_selected'}
    result = _collect_one({**task, 'case': case, 'teacher': 'pickup', 'takeover_step': step,
                          'expected_source': {**source, 'initial_state': source['states'][0]}})
    result['takeover_kind'] = kind
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('data/recovery_alignment'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists() or args.workers < 1: parser.error('Use a fresh output directory and positive workers')
    policy_sha256 = digest(POLICY)
    excluded, scanned = set(), {}
    for p in reset_seed_sources():
        records = [json.loads(line) for line in p.read_text().splitlines() if line] if p.suffix == '.jsonl' else [json.loads(p.read_text())]
        seeds = set().union(*(set(_reset_seeds(r)) for r in records))
        excluded.update(seeds)
        scanned[str(p)] = {'sha256': digest(p), 'seed_count': len(seeds)}
    rng = np.random.default_rng(SEED)
    seen, cases, targets = set(excluded), [], {}
    for split, multiplier in [('train', 4), ('validation', 1)]:
        for force in FORCES:
            for group, count in TARGETS.items():
                targets[split, force, group] = count*multiplier
                for index in range(count*multiplier*50):
                    reset = int(rng.integers(10000, 2**31))
                    while reset in seen: reset = int(rng.integers(10000, 2**31))
                    seen.add(reset)
                    angle = float(rng.uniform(0, 2*np.pi))
                    cases.append({'episode_id': f'alignment_{split}_{reset}', 'split': split,
                        'reset_seed': reset, 'policy_seed': int(rng.integers(0, 2**31)), 'force_n': force,
                        'force_xy': [force*np.cos(angle), force*np.sin(angle)], 'start_step': int(rng.integers(8, 19)),
                        'duration_steps': 2, 'chunk_length': 8, 'target_group': group, 'candidate_index': index})
    plan = {'complete': False, 'mode': 'fresh_alignment_collection', 'planned_at_utc': datetime.now(timezone.utc).isoformat(),
        'collection_seed': SEED, 'policy': str(POLICY), 'policy_sha256': policy_sha256, 'teacher': 'pickup',
        'targets': {'train': 32, 'validation': 8}, 'forces_n': list(FORCES),
        'targets_by_stratum': {f'{s}_{f:g}_{g}': n for (s, f, g), n in targets.items()},
        'selection': 'Same pre-closure/placement/successful selectors as the passed pilot; disturbed alignment additionally requires failed source task success at step 100.',
        'source_failure_requirement': 'Every 0.4 N alignment example; nominal alignment remains an intervention if the source succeeds.',
        'acceptance': 'Unchanged verified lift, held task success at absolute step 100, ten settled successes within 10 mm.',
        'cases': cases, 'excluded_reset_seed_count': len(excluded), 'overlap_sources': scanned,
        'code_sha256': {**_collection_code(), __file__: digest(__file__), 'uv.lock': digest('uv.lock')},
        'preserved_sha256': {str(p): digest(p) for folder in ('data', 'checkpoints', 'results') for p in Path(folder).rglob('*') if p.is_file()},
        'attempts': [], 'accepted': []}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'rollouts').mkdir()
    write_json(args.output_dir/'plan.json', plan)
    queues = {key: [c for c in cases if (c['split'], c['force_n'], c['target_group']) == key] for key in targets}
    accepted, used = Counter(), Counter()
    print(f'Frozen {len(cases)} new groups, excluding {len(excluded)} prior seeds; target 32 train + 8 validation', flush=True)
    with h5py.File(args.output_dir/'train.h5', 'x') as train, h5py.File(args.output_dir/'validation.h5', 'x') as validation:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                initializer=_init_worker, initargs=(POLICY, None, False, 'pickup')) as pool:
            while True:
                batch = []
                for key, queue in queues.items():
                    if accepted[key] < targets[key] and used[key] < len(queue):
                        case = queue[used[key]]
                        used[key] += 1
                        batch.append({'case': case, 'teacher': 'pickup',
                            'method': {'variant': 'fixed_k8', 'kind': 'fixed', 'chunk_length': 8, 'threshold': None}})
                if not batch: break
                for result in pool.map(training_case, batch):
                    _save_attempt(result, args.output_dir, {'train': train, 'validation': validation}, plan)
                    if result['accepted']:
                        c = result['case']
                        accepted[c['split'], c['force_n'], c['target_group']] += 1
    for p, h in {**plan['preserved_sha256'], **plan['code_sha256']}.items(): assert digest(p) == h, p
    all_met = all(accepted[k] == n for k, n in targets.items())
    counts = {s: sum(r['split'] == s for r in plan['accepted']) for s in ('train', 'validation')}
    for split in counts:
        write_json(args.output_dir/f'{split}.json', {'complete': all_met, 'split': split,
            'policy': str(POLICY), 'policy_sha256': digest(POLICY), 'episodes': [r for r in plan['accepted'] if r['split'] == split]})
    plan.update(complete=all_met, all_attempts_complete=True, accepted_counts=counts,
        accepted_by_stratum={f'{s}_{f:g}_{g}': accepted[s, f, g] for s, f, g in targets},
        artifact_sha256={str(args.output_dir/f'{s}.h5'): digest(args.output_dir/f'{s}.h5') for s in counts})
    write_json(args.output_dir/'metadata.json', plan)
    print(json.dumps({'complete': all_met, 'accepted_counts': counts, 'source_attempts': len(plan['attempts'])}, indent=2))


if __name__ == '__main__':
    main()
