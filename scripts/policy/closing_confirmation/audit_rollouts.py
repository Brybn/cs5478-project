"""Replay the frozen 100/150-step comparison and compute paired policy effects."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from adaptive_action_chunking.artifacts import digest
from adaptive_action_chunking.monitor import _reset_seeds, write_json
from adaptive_action_chunking.env import make_env, extract_policy_state


ROOT = Path('results/policy_closing_confirmation')
ENV = None


def initialize(horizon):
    global ENV
    torch.set_num_threads(1)
    ENV = make_env(max_episode_steps=horizon, disturbance={
        'start_step':0, 'duration_steps':0, 'force_xy':[0.,0.]})


def replay(task):
    case,e,horizon,extended = task
    if extended:
        assert digest(e['trace']) == e['trace_sha256']
        with np.load(e['trace']) as trace: actions,expected=trace['actions'],trace['states']
        base=ROOT/'rollouts'/f"{e['episode_id']}_{e['variant']}.npz"
        with np.load(base) as trace:
            np.testing.assert_array_equal(actions[:100],trace['actions'])
            np.testing.assert_array_equal(expected[:101],trace['states'])
        path=Path(e['trace'])
    else:
        actions=np.asarray(e['actions'],dtype=np.float32);expected=None
        path=ROOT/'rollouts'/f"{e['episode_id']}_{e['variant']}.npz"
        if path.exists():
            with np.load(path) as saved:
                expected=saved['states']
                np.testing.assert_array_equal(actions,saved['actions'])
    assert actions.shape==(horizon,4) and np.isfinite(actions).all() and np.abs(actions).max()<=1.000001
    calls=list(range(0,horizon,e['chunk_length']))
    assert e['replan_steps']==calls and e['policy_calls']==len(calls)
    assert [p['step'] for p in e['plans']]==calls
    for p in e['plans']:
        length=min(e['chunk_length'],horizon-p['step'])
        np.testing.assert_array_equal(actions[p['step']:p['step']+length],np.asarray(p['actions'],dtype=np.float32)[:length])
    ENV.unwrapped.disturbance={k:case[k] for k in ('start_step','duration_steps','force_xy')}
    obs,_=ENV.reset(seed=case['reset_seed'])
    states=[extract_policy_state(obs)];success=[]
    if expected is None:np.testing.assert_array_equal(states[0],e['initial_state'])
    else:np.testing.assert_array_equal(states[0],expected[0])
    for step,action in enumerate(actions):
        obs,_,_,truncated,info=ENV.step(action)
        state=extract_policy_state(obs);states.append(state)
        if expected is not None:np.testing.assert_array_equal(state,expected[step+1])
        assert bool(truncated.item())==(step==horizon-1)
        if info['success'].item():success.append(step+1)
    states=np.asarray(states,dtype=np.float32)
    np.testing.assert_array_equal(np.concatenate([states[:,29:32],states[:,19:22]],axis=1),e['positions'])
    held=states[:,18]>.5
    np.testing.assert_array_equal(held,e['grasped'])
    assert success==e['success_steps'] and (horizon in success)==e['success_at_end']
    distance=np.linalg.norm(states[:,29:32]-states[:,26:29],axis=1)
    np.testing.assert_allclose(distance,e['goal_distances_m'],rtol=1e-6,atol=1e-7)
    physical=(distance[1:]<=.025)&(np.abs(states[1:,9:16]).max(1)<=.2)
    assert (np.flatnonzero(physical)+1).tolist()==success
    forces=list(range(case['start_step'],case['start_step']+case['duration_steps'])) if case['force_n'] else []
    assert sorted(ENV.unwrapped.applied_force_steps)==e['applied_force_steps']==forces
    assert ENV.unwrapped.applied_force_substeps==e['applied_force_substeps']==5*len(forces)
    raised=held&(states[:,31]>=states[0,31]+.025)
    lifts=np.flatnonzero(np.convolve(raised.astype(int),np.ones(5,dtype=int),'valid')==5)
    confirm=next((t for t in range(10,horizon+1) if all(s in success for s in range(t-9,t+1))),None)
    losses=np.flatnonzero(held[:-1]&~held[1:])+1
    closes=np.flatnonzero(actions[:,3]<-.5)
    stop=int(closes[0]) if len(closes) else horizon
    peaks,_=find_peaks(states[:stop+1,21],prominence=.005)
    troughs,_=find_peaks(-states[:stop+1,21],prominence=.005)
    if not extended and not path.exists():np.savez_compressed(path,states=states,actions=actions)
    return {k:e[k] for k in ('episode_id','group_id','variant','chunk_length','force_n','success_at_end')} | {
        'success_at_100':100 in success,'success_at_150':150 in success if extended else None,
        'last_ten_success':all(s in success for s in range(horizon-9,horizon+1)),
        'confirmation_step':confirm,'capped_confirmation_seconds':(confirm or horizon)/20,
        'confirmed_then_failed':confirm is not None and not e['success_at_end'],
        'sustained_lift_step':int(lifts[0]+4) if len(lifts) else None,
        'outcome':'success' if e['success_at_end'] else 'no_sustained_lift' if not len(lifts) else
                  'within_goal_not_static' if distance[-1]<=.025 else 'lifted_outside_goal',
        'preclose_height_peaks_5mm':len(peaks), 'preclose_height_troughs_5mm':len(troughs),
        'preclose_height_extrema_5mm':len(peaks)+len(troughs),
        'no_close_by_horizon':not len(closes),
        'first_close_step':int(closes[0]) if len(closes) else None,
        'first_close_xy_mm':float(np.linalg.norm(states[closes[0],19:21]-states[closes[0],29:31])*1000) if len(closes) else None,
        'grasp_loss_steps':losses.tolist(),'closing_commands_at_grasp_loss':int((actions[losses-1,3]<-.5).sum()),
        'opening_commands_while_grasped':int((held[:-1]&(actions[:,3]>.5)).sum()),
        'held_at_horizon':bool(held[-1]),'final_goal_distance_mm':float(distance[-1]*1000),
        'trace':str(path),'trace_sha256':digest(path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended',action='store_true')
    args=parser.parse_args();extended=args.extended;horizon=150 if extended else 100
    folder=ROOT/'horizon150' if extended else ROOT
    assert not (folder/'review.json').exists()
    baseline=json.loads((ROOT/'comparison/results.json').read_text())
    result=json.loads((folder/'results.json').read_text()) if extended else baseline
    assert result['complete'] and len(result['episodes'])==1600
    plan=baseline['plan'];assert plan['seed']==169001 and plan['groups']==100
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert protocol['planned_at_utc']<plan['planned_at_utc']
    cases={c['episode_id']:c for c in plan['cases']}
    assert len(cases)==400
    groups=sorted({c['group_id'] for c in cases.values()})
    expected={(c,n,k) for c in cases for n in ('retained','closing') for k in (8,15)}
    assert {(e['episode_id'],e['checkpoint'],e['chunk_length']) for e in result['episodes']}==expected
    for name,checkpoint in protocol['checkpoints'].items():
        assert plan['checkpoints'][name]['sha256']==checkpoint['sha256']
        assert digest(checkpoint['path'])==checkpoint['sha256']
    if not extended:
        excluded=set()
        for p,record in plan['overlap_sources'].items():
            assert digest(p)==record['sha256']
            records=[json.loads(line) for line in Path(p).read_text().splitlines() if line] if Path(p).suffix=='.jsonl' else [json.loads(Path(p).read_text())]
            seeds=set().union(*(set(_reset_seeds(r)) for r in records))
            assert len(seeds)==record['seed_count'];excluded.update(seeds)
        assert len(excluded)==plan['excluded_seed_count']
        assert not {c['reset_seed'] for c in cases.values()}&excluded
        (ROOT/'rollouts').mkdir(exist_ok=True)
    rows=[]
    with ProcessPoolExecutor(8,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(horizon,)) as pool:
        for row in pool.map(replay,[(cases[e['episode_id']],e,horizon,extended) for e in result['episodes']]):
            rows.append(row)
            if len(rows)%120==0:print(f'Exact {horizon}-step physics replays: {len(rows)}/1600',flush=True)
    draws=np.random.default_rng(plan['bootstrap']['seed']).integers(100,size=(5000,100))
    contrasts,timing={},{ }
    for treatment,reference in [('closing','retained')]:
        pair=f'{treatment}_vs_{reference}'
        contrasts[pair],timing[pair]={},{}
        for condition,selected in {'overall':rows,**{f'force_{f:g}':[r for r in rows if r['force_n']==f] for f in plan['force_levels_n']}}.items():
            contrasts[pair][condition],timing[pair][condition]={},{}
            for k in (8,15):
                left=[r for r in selected if r['variant']==f'{treatment}_k{k}']
                right={r['episode_id']:r for r in selected if r['variant']==f'{reference}_k{k}'}
                delta={r['episode_id']:int(r['success_at_end'])-int(right[r['episode_id']]['success_at_end']) for r in left}
                grouped=np.asarray([np.mean([delta[r['episode_id']] for r in left if r['group_id']==g]) for g in groups])
                effect={'difference':float(grouped.mean()),'ci95':np.quantile(grouped[draws].mean(1),[.025,.975]).tolist(),
                    'wins':sum(v>0 for v in delta.values()),'losses':sum(v<0 for v in delta.values()),'ties':sum(v==0 for v in delta.values())}
                contrasts[pair][condition][str(k)]=effect
                if not extended and reference=='retained':
                    reported=baseline if condition=='overall' else baseline['by_force'][condition[6:]]
                    expected_effect=reported[f"paired_vs_{plan['baseline']}_same_chunk"][str(k)][f'{treatment}_k{k}']
                    np.testing.assert_allclose(effect['difference'],expected_effect['success_at_end_difference'],atol=1e-14)
                    np.testing.assert_allclose(effect['ci95'],expected_effect['ci95']['success_at_end_difference'],atol=1e-14)
                time_delta={r['episode_id']:r['capped_confirmation_seconds']-right[r['episode_id']]['capped_confirmation_seconds'] for r in left}
                grouped_time=np.asarray([np.mean([time_delta[r['episode_id']] for r in left if r['group_id']==g]) for g in groups])
                timing[pair][condition][str(k)]={'difference_seconds':float(grouped_time.mean()),'ci95':np.quantile(grouped_time[draws].mean(1),[.025,.975]).tolist()}
    behavior={v:{'episodes':sum(r['variant']==v for r in rows),
        'success_at_end':sum(r['success_at_end'] for r in rows if r['variant']==v),
        'outcomes':dict(Counter(r['outcome'] for r in rows if r['variant']==v)),
        'opening_commands_while_grasped':sum(r['opening_commands_while_grasped'] for r in rows if r['variant']==v)}
        for v in sorted({r['variant'] for r in rows})}
    for p,h in {**result['plan']['preserved_sha256'],**result['plan']['code_sha256'],**protocol['code_sha256']}.items():assert digest(p)==h,p
    review={'complete':True,'source_sha256':digest(folder/'results.json' if extended else ROOT/'comparison/results.json'),
        'review_code_sha256':digest(__file__),'horizon':horizon,'checks':{**result['checks'],
            'exact_physics_replays':1600,'physics_actions_replayed':1600*horizon,'independent_paired_success_intervals':True},
        'contrasts':contrasts,'paired_capped_confirmation_time':timing,'behavior':behavior,'episodes':rows}
    write_json(folder/'review.json',review)
    print(json.dumps({'horizon':horizon,'primary_k8':contrasts['closing_vs_retained']['overall']['8'],'behavior':behavior},indent=2))


if __name__=='__main__':main()
