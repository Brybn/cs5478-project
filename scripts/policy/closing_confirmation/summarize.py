"""Summarize verified fixed-deadline success, timing and pre-grasp motion."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from adaptive_action_chunking.artifacts import digest
from adaptive_action_chunking.monitor import write_json


ROOT = Path('results/policy_closing_confirmation')


def main():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    base = json.loads((ROOT/'comparison/results.json').read_text())
    extended = json.loads((ROOT/'horizon150/results.json').read_text())
    reports = {h: json.loads((ROOT/('review.json' if h == 100 else 'horizon150/review.json')).read_text())
               for h in (100, 150)}
    assert base['complete'] and extended['complete'] and all(r['complete'] for r in reports.values())
    for h, result in ((100, base), (150, extended)):
        assert reports[h]['source_sha256'] == digest(ROOT/('comparison/results.json' if h == 100 else 'horizon150/results.json'))
    groups = sorted({r['group_id'] for r in reports[100]['episodes']})
    draws = np.random.default_rng(169002).integers(len(groups), size=(5000, len(groups)))
    summaries, behavior_effects = {}, {}
    for h, report in reports.items():
        rows = report['episodes']
        summaries[str(h)] = {}
        for variant in sorted({r['variant'] for r in rows}):
            selected = [r for r in rows if r['variant'] == variant]
            summaries[str(h)][variant] = {
                'episodes': len(selected), 'final_success': sum(r['success_at_end'] for r in selected),
                'final_success_by_force': {str(f): sum(r['success_at_end'] for r in selected if r['force_n'] == f)
                                           for f in (0., .2, .3, .4)},
                'last_ten_success': sum(r['last_ten_success'] for r in selected),
                'confirmation_then_final_failure': sum(r['confirmed_then_failed'] for r in selected),
                'mean_capped_confirmation_seconds': float(np.mean([r['capped_confirmation_seconds'] for r in selected])),
                'mean_capped_first_close_seconds': float(np.mean([(h if r['first_close_step'] is None else r['first_close_step'])/20 for r in selected])),
                'no_close_by_horizon': sum(r['no_close_by_horizon'] for r in selected),
                'mean_preclose_extrema_5mm': float(np.mean([r['preclose_height_extrema_5mm'] for r in selected])),
                'episodes_with_at_least_two_preclose_peaks': sum(r['preclose_height_peaks_5mm'] >= 2 for r in selected),
                'failure_types': dict(Counter(r['outcome'] for r in selected if not r['success_at_end'])),
                'opening_commands_while_grasped': sum(r['opening_commands_while_grasped'] for r in selected)}
        behavior_effects[str(h)] = {}
        for treatment, reference in (('closing', 'retained'),):
            pair = treatment+'_vs_'+reference
            behavior_effects[str(h)][pair] = {}
            for k in (8, 15):
                left = [r for r in rows if r['variant'] == f'{treatment}_k{k}']
                right = {r['episode_id']: r for r in rows if r['variant'] == f'{reference}_k{k}'}
                effects = {}
                for metric in ('preclose_height_extrema_5mm', 'first_close_step'):
                    def value(row):
                        return (h if row[metric] is None else row[metric])/20 if metric == 'first_close_step' else row[metric]
                    grouped = np.asarray([np.mean([value(r)-value(right[r['episode_id']]) for r in left if r['group_id'] == g]) for g in groups])
                    effects[metric] = {'difference': float(grouped.mean()),
                                       'ci95': np.quantile(grouped[draws].mean(1), [.025, .975]).tolist()}
                behavior_effects[str(h)][pair][str(k)] = effects
    summary = {'complete': True, 'protocol_sha256': digest(ROOT/'protocol.json'),
               'checkpoint_audit_sha256': digest(ROOT/'checkpoint_audit.json'),
               'primary_k8_150': reports[150]['contrasts']['closing_vs_retained']['overall']['8'],
               'summaries': summaries, 'deadline_changes': extended['summary'],
               'success_effects': {str(h): r['contrasts'] for h, r in reports.items()},
               'paired_confirmation_time': {str(h): r['paired_capped_confirmation_time'] for h, r in reports.items()},
               'behavior_effects': behavior_effects,
               'checks': {str(h): r['checks'] for h, r in reports.items()},
               'limitations': protocol['limitations'] + [
                   'The100/150 endpoints share1600 trajectories; they are not3200 independent episodes.',
                   'Preclose extrema describe motion before the first strong close; they do not establish teacher causality or count later retry oscillations.',
                   'A ten-step confirmation can precede a final failure; endpoint success remains separately reported.']}
    write_json(ROOT/'summary.json', summary)
    print(json.dumps({'primary': summary['primary_k8_150'], 'summaries': summaries,
                      'k8_150_time': summary['paired_confirmation_time']['150']['closing_vs_retained']['overall']['8'],
                      'k8_100_behavior': behavior_effects['100']['closing_vs_retained']['8']}, indent=2))


if __name__ == '__main__':
    main()
