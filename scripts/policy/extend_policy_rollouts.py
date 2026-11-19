"""Extend every saved comparison rollout without changing its original prefix."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from adaptive_action_chunking.artifacts import digest, resolve_artifact
from adaptive_action_chunking.monitor import source_hashes, write_json
from adaptive_action_chunking.env import make_env
from adaptive_action_chunking.evaluate import run_episode
from adaptive_action_chunking.policy import load_policy


WORKER = {}


class PrefixPolicy:
    """Replay whole saved plans, then resume the actual policy on its schedule."""

    def __init__(self, policy, plans):
        self.policy, self.plans, self.calls = policy, plans, 0
        self.config, self.act_dim = policy.config, policy.act_dim

    def sample(self, observations, generator, device):
        index = self.calls
        self.calls += 1
        if index < len(self.plans):
            return np.asarray(self.plans[index]['actions'], dtype=np.float32)
        return self.policy.sample(observations, generator, device)


def extend_episode(env, policy, case, original, horizon, expected_states=None):
    base, chunk = original['steps'], original['chunk_length']
    assert horizon > base and original['termination'] == 'horizon'
    assert original['replan_steps'] == list(range(0, base, chunk))
    assert [p['step'] for p in original['plans']] == original['replan_steps']
    env.unwrapped.disturbance = {k: case[k] for k in ('start_step', 'duration_steps', 'force_xy')}
    episode = run_episode(env, PrefixPolicy(policy, original['plans']), chunk, case['reset_seed'],
                          torch.device('cpu'), horizon=horizon, policy_seed=case['policy_seed'],
                          collect_positions=True)
    assert episode['steps'] == horizon and episode['termination'] == 'horizon'
    np.testing.assert_array_equal(episode['states'][0], original['initial_state'])
    for key, count in [('actions', base), ('positions', base+1), ('grasped', base+1)]:
        np.testing.assert_array_equal(episode[key][:count], original[key])
    np.testing.assert_allclose(episode['goal_distances_m'][:base+1], original['goal_distances_m'], rtol=1e-6, atol=1e-7)
    assert episode['plans'][:len(original['plans'])] == original['plans']
    assert [s for s in episode['success_steps'] if s <= base] == original['success_steps']
    assert (base in episode['success_steps']) == original['success_at_end']
    assert episode['applied_force_steps'] == original['applied_force_steps']
    assert episode['applied_force_substeps'] == original['applied_force_substeps']
    assert episode['replan_steps'] == list(range(0, horizon, chunk))
    assert episode['policy_calls'] == len(episode['replan_steps'])
    states = np.asarray(episode['states'], dtype=np.float32)
    if expected_states is not None:
        np.testing.assert_array_equal(states[:base+1], expected_states)
    success = ((np.linalg.norm(states[1:, 29:32]-states[1:, 26:29], axis=1) <= .025) &
               (np.abs(states[1:, 9:16]).max(1) <= .2))
    np.testing.assert_array_equal(np.flatnonzero(success)+1, episode['success_steps'])
    return episode


def metrics(episode, base):
    states = np.asarray(episode['states'], dtype=np.float32)
    horizon = len(states)-1
    flags = np.zeros(horizon, dtype=bool)
    flags[np.asarray(episode['success_steps'], dtype=int)-1] = True
    confirmation = np.flatnonzero(np.convolve(flags.astype(int), np.ones(10, dtype=int), 'valid') == 10)
    held = states[:, 18] > .5
    raised = held & (states[:, 31] >= states[0, 31]+.025)
    lift = np.flatnonzero(np.convolve(raised.astype(int), np.ones(5, dtype=int), 'valid') == 5)
    distance = float(np.linalg.norm(states[-1, 29:32]-states[-1, 26:29]))
    old, new = bool(flags[base-1]), bool(flags[-1])
    return {'success_at_original_horizon': old, 'success_at_extended_horizon': new,
            'rescued': not old and new, 'lost': old and not new, 'still_failed': not old and not new,
            'success_once': bool(flags.any()), 'last_ten_success': bool(flags[-10:].all()),
            'confirmation_step': int(confirmation[0]+10) if len(confirmation) else None,
            'first_success_step': episode['first_success_step'],
            'sustained_lift_step': int(lift[0]+4) if len(lift) else None,
            'held_at_horizon': bool(held[-1]), 'final_goal_distance_mm': distance*1000,
            'grasp_loss_transitions': int((held[:-1] & ~held[1:]).sum()),
            'outcome': ('success' if new else 'no_sustained_lift' if not len(lift)
                        else 'within_goal_not_static' if distance <= .025 else 'lifted_outside_goal')}


def initialize(checkpoints, horizon, output):
    torch.set_num_threads(1)
    WORKER.update(env=make_env(max_episode_steps=horizon, disturbance={
        'start_step': 0, 'duration_steps': 0, 'force_xy': [0., 0.]}),
        policies={name: load_policy(path, device='cpu') for name, path in checkpoints.items()},
        horizon=horizon, output=Path(output))


def run(task):
    case, original, trace, repeat = task
    expected = None
    if trace:
        with np.load(trace) as saved:
            expected = saved['states']
    def execute():
        return extend_episode(WORKER['env'], WORKER['policies'][original['checkpoint']],
                              case, original, WORKER['horizon'], expected)
    episode = execute()
    if repeat:
        repeated = execute()
        for key in ('states', 'actions', 'plans', 'success_steps', 'grasped', 'replan_steps'):
            assert episode[key] == repeated[key], key
    path = WORKER['output']/'rollouts'/f"{original['episode_id']}_{original['variant']}.npz"
    states = np.asarray(episode.pop('states'), dtype=np.float32)
    actions = np.asarray(episode.pop('actions'), dtype=np.float32)
    np.savez_compressed(path, states=states, actions=actions)
    episode['metrics'] = metrics({**episode, 'states': states}, original['steps'])
    episode.update({k: original[k] for k in ('group_id', 'episode_id', 'force_n', 'variant', 'checkpoint', 'chunk_length')})
    episode.update(trace=str(path), trace_sha256=digest(path), repeat_exact=repeat,
                   full_state_prefix_checked=expected is not None)
    return episode


def summarize(episodes, bootstrap):
    summary = {}
    for variant in sorted({e['variant'] for e in episodes}):
        rows = [e for e in episodes if e['variant'] == variant]
        groups = sorted({e['group_id'] for e in rows})
        draws = np.random.default_rng(bootstrap['seed']).integers(len(groups), size=(bootstrap['samples'], len(groups)))
        changes = np.array([np.mean([int(e['metrics']['success_at_extended_horizon'])-
            int(e['metrics']['success_at_original_horizon']) for e in rows if e['group_id'] == g]) for g in groups])
        summary[variant] = {'episodes': len(rows),
            **{k: sum(e['metrics'][k] for e in rows) for k in (
                'success_at_original_horizon', 'success_at_extended_horizon', 'rescued', 'lost', 'still_failed', 'last_ten_success')},
            'paired_final_success_change': float(changes.mean()),
            'ci95': np.quantile(changes[draws].mean(1), [.025, .975]).tolist(),
            'outcomes': dict(Counter(e['metrics']['outcome'] for e in rows))}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/policy_training_options_eval/comparison/results.json'))
    parser.add_argument('--horizon', type=int, default=150)
    parser.add_argument('--output-dir', type=Path, default=Path('results/debugging/policy_horizon150'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    if not source['complete'] or args.horizon <= source['plan']['horizon'] or args.workers < 1 or args.output_dir.exists():
        parser.error('Require a completed source, longer horizon, positive workers and fresh output directory')
    checkpoints = {n: str(resolve_artifact(d['path'], d['sha256'])) for n, d in source['plan']['checkpoints'].items()}
    review_path = args.source.parent.parent/'review.json'
    traces = {}
    if review_path.exists():
        review = json.loads(review_path.read_text())
        assert review['complete'] and review['source_sha256'] == digest(args.source)
        for e in review['episodes']:
            assert digest(e['trace']) == e['trace_sha256']
            traces[e['episode_id'], e['variant']] = e['trace']
    preserved = {str(p): digest(p) for folder in ('data', 'checkpoints', 'results')
                 for p in Path(folder).rglob('*') if p.is_file()}
    cases = {c['episode_id']: c for c in source['plan']['cases']}
    first_group = source['plan']['cases'][0]['group_id']
    tasks = [(cases[e['episode_id']], e, traces.get((e['episode_id'], e['variant'])),
              e['group_id'] == first_group and e['force_n'] in (0, .4)) for e in source['episodes']]
    plan = {'planned_at_utc': datetime.now(timezone.utc).isoformat(),
        'role': 'Development horizon extension of an already inspected cohort; original 100-step endpoint remains unchanged.',
        'primary': f"Retained K8 paired final-success change from {source['plan']['horizon']} to {args.horizon} steps, including rescues and lost successes.",
        'source': str(args.source), 'source_sha256': digest(args.source),
        'source_plan': str(args.source.parent/'plan.json'), 'source_plan_sha256': digest(args.source.parent/'plan.json'),
        'original_horizon': source['plan']['horizon'], 'horizon': args.horizon,
        'checkpoints': source['plan']['checkpoints'], 'cases': source['plan']['cases'],
        'selection': 'All source episodes, including original successes; all four policies and both chunk lengths.',
        'continuation': 'Replay complete original plans, including their unexecuted tail; resume policy at its next scheduled absolute call step with original noise seed.',
        'bootstrap': source['plan']['bootstrap'], 'workers': args.workers,
        'code_sha256': {**source_hashes(), str(Path(__file__)): digest(__file__), 'uv.lock': digest('uv.lock')},
        'preserved_sha256': preserved}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'rollouts').mkdir()
    write_json(args.output_dir/'plan.json', plan)
    result = {'complete': False, 'plan': plan, 'episodes': []}
    write_json(args.output_dir/'results.json', result)
    print(f'Extending {len(tasks)} matched rollouts to {args.horizon} steps', flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), initializer=initialize,
                             initargs=(checkpoints, args.horizon, str(args.output_dir))) as pool:
        for episode in pool.map(run, tasks):
            result['episodes'].append(episode)
            if len(result['episodes']) % 40 == 0:
                write_json(args.output_dir/'results.json', result)
                print(f"Extended {len(result['episodes'])}/{len(tasks)}", flush=True)
    result['summary'] = summarize(result['episodes'], plan['bootstrap'])
    result['by_force'] = {f'{f:g}': summarize([e for e in result['episodes'] if e['force_n'] == f], plan['bootstrap'])
                          for f in source['plan']['force_levels_n']}
    for p, h in {**preserved, **plan['code_sha256']}.items(): assert digest(p) == h, p
    result['checks'] = {'original_prefixes_match': len(tasks),
        'full_state_prefixes_match': sum(e['full_state_prefix_checked'] for e in result['episodes']),
        'exact_repeats': sum(e['repeat_exact'] for e in result['episodes']),
        'previous_files_unchanged': len(preserved), 'success_checked_from_physics': True,
        'force_schedules_unchanged': True, 'original_chunk_tails_preserved': True}
    result.update(complete=True, completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output_dir/'results.json', result)
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
