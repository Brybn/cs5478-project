"""Freeze the two checkpoints and protocol for a 100-group confirmation."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import torch

from adaptive_action_chunking.artifacts import digest
from adaptive_action_chunking.monitor import source_hashes, write_json


ROOT = Path('results/policy_closing_confirmation')
POLICIES = {'retained': 'stage2_pickup', 'closing': 'stage3_closing'}


def main():
    torch.set_num_threads(1)
    assert not (ROOT/'protocol.json').exists()
    ROOT.mkdir(parents=True, exist_ok=True)
    checkpoints, records, identities, retained = {}, {}, {}, None
    for name, folder in POLICIES.items():
        path = Path('checkpoints/pickcube')/folder/'last.pt'
        expected = digest(path)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['iteration'] == 5000 and all(torch.isfinite(t).all() for t in saved['ema_agent'].values())
        record_path = path.with_name('training.json')
        record = json.loads(record_path.read_text())
        assert record['complete'] and record['config']['total_iters'] == 5000
        records[name] = {'path': str(record_path), 'sha256': digest(record_path),
                         'historical_training_code_sha256': record['code_sha256']}
        identities[str(path)] = expected
        identities[str(record_path)] = digest(record_path)
        checkpoints[name] = {'path': str(path), 'sha256': expected, 'iteration': 5000, 'weights': 'ema_agent'}
        if name == 'retained':
            retained = saved['ema_agent']
        else:
            assert saved['initial_checkpoint_sha256'] == checkpoints['retained']['sha256']
            assert retained.keys() == saved['ema_agent'].keys()
            assert any(not torch.equal(retained[k], saved['ema_agent'][k]) for k in retained)
        del saved
    for split in ('train', 'validation'):
        p = Path('data/split')/(split+'.h5')
        identities[str(p)] = digest(p)
        assert identities[str(p)] == record['demos']['nominal_'+split]['sha256']
    prior = {}
    for folder in ('policy_closing_eval', 'policy_goal_eval'):
        for filename in ('protocol.json', 'summary.json'):
            p = Path('results')/folder/filename
            if p.exists():
                prior[str(p)] = digest(p)
    write_json(ROOT/'checkpoint_audit.json', {'complete': True, 'checkpoints': checkpoints,
        'training_records': records, 'identities_checked': identities, 'checkpoints_distinct': True,
        'note': 'Training-code identities are historical provenance; this confirmation freezes current shared inference code.'})
    files = {Path(p).resolve() for p in source_hashes()} | {Path(p).resolve() for p in (
        'src/adaptive_action_chunking/__init__.py', 'src/adaptive_action_chunking/artifacts.py',
        'scripts/policy/extend_policy_rollouts.py', 'src/adaptive_action_chunking/train.py',
        'pyproject.toml', 'uv.lock')}
    snapshot = {}
    for source in sorted(files):
        relative = source.relative_to(Path.cwd())
        target = ROOT/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        snapshot[str(relative)] = {'sha256': digest(source), 'snapshot': str(target)}
    write_json(ROOT/'source_snapshot.json', snapshot)
    code = {**source_hashes(), **{str(p): digest(p) for p in (
        Path(__file__), Path(__file__).with_name('audit_rollouts.py'), Path(__file__).with_name('plot.py'),
        Path(__file__).with_name('summarize.py'),
        Path('scripts/policy/extend_policy_rollouts.py'))}}
    protocol = {'planned_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': 169001, 'groups': 100,
        'checkpoints': checkpoints,
        'primary': 'Closing minus retained K8 final success at150, equally weighted over forces0/.2/.3/.4 N.',
        'guards': 'Report K8 overall100 and nominal100/150 success with paired intervals, plus capped confirmation time. Do not treat an interval containing zero as proof of nominal equivalence.',
        'secondary': 'K15, individual forces, final-ten-step success, success lost after100, preclose motion and failure categories.',
        'selection': 'Both update5000 EMA checkpoints frozen before confirmation. Fixed100 groups; no outcome-based stopping, tuning, sample extension or pooling with inspected earlier cohorts.',
        'phases': 'Run both policies at K8/K15 through100; independently replay; extend every trajectory to150 preserving exact prefixes and unfinished chunks; independently replay again.',
        'inference': 'CPU, original absolute-step common noise, two-step pulses at8..18,20 Hz.',
        'freshness': 'Exclude all recorded data, collection, selection, evaluation and reserved reset seeds; exact cases depend on prior-history metadata.',
        'statistics': '5000 paired reset-group bootstrap draws, seed169002, retaining all force conditions. Primary95% interval; secondary intervals pointwise.',
        'timing': 'First ten consecutive official-success steps; no confirmation capped at horizon/20 seconds. Include all episodes and separately report later failures. Simulation time excludes inference.',
        'decision': 'Report the primary and nominal/timing tradeoffs before recommending an operating policy. Keep current retained until reviewed; no automatic checkpoint promotion, training or monitor job.',
        'prior_motivation_sha256': prior, 'checkpoint_audit_sha256': digest(ROOT/'checkpoint_audit.json'),
        'code_sha256': code, 'versions': {p: version(p) for p in ('torch', 'numpy', 'mani-skill', 'diffusers', 'sapien', 'gymnasium')},
        'limitations': ['One fixed training seed and one diffusion sampling seed per reset group.',
            'Forces are deliberately equally weighted; the aggregate is not a natural deployment frequency.',
            'The two whole training recipes differ; this comparison cannot isolate closing labels alone.',
            'Earlier inspected cohorts motivated this test and are not pooled into its primary result.']}
    write_json(ROOT/'protocol.json', protocol)
    print(json.dumps({'groups': 100, 'trajectories': 1600, 'checkpoints': checkpoints, 'primary': protocol['primary']}, indent=2))


if __name__ == '__main__':
    main()
