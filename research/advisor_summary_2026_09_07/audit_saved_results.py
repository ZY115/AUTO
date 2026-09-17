"""Read-only audit of saved pilots; does not rerun training or replace results.

Run with tl_sequence_pilot/.venv/bin/python. Only the adjacent audit JSON is written.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'tl_sequence_pilot'
PILOTS = ROOT / 'pilots'
sys.path.insert(0, str(PILOTS / 'pilot_3_0_rl_structure'))
import model
import numpy as np


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    p30 = PILOTS / 'pilot_3_0_rl_structure'
    p31 = PILOTS / 'pilot_3_1_objective_boundary'
    r30, r31 = p30 / 'results/initial_run', p31 / 'results/initial_run'
    result = {'audited_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'Read saved inputs and outputs, recheck finite monitors and final-policy DP. No training rerun.'}
    sources = {}
    for name, directory in [('3.0', r30), ('3.1', r31)]:
        meta = json.loads((directory / 'metadata.json').read_text())
        checks = {path: sha(PILOTS / path) == expected
                  for path, expected in meta['source_sha256'].items()}
        assert all(checks.values()), checks
        sources[name] = checks
    result['saved_source_hashes_match_current'] = sources
    executable = p30 / 'build/learner'
    result['existing_executable_matches_saved_hash_before_tests'] = (
        sha(executable) == json.loads((r30 / 'metadata.json').read_text())['executable_sha256'])
    result['frontend_validation'] = model.validate_frontends(7)
    assert result['frontend_validation']['raw_to_minimized'][0] == 0
    cfg = json.loads((p30 / 'protocol.json').read_text())
    pairs, checkpoints = [], 0
    for distance in cfg['distances']:
        for mode in ('standard', 'counterfactual'):
            for seed in cfg['seeds']:
                left = r30 / 'raw' / f'd{distance}_manual_{mode}_s{seed}'
                right = r30 / 'raw' / f'd{distance}_tl_{mode}_s{seed}'
                inp = left.with_suffix('.txt').read_bytes() == right.with_suffix('.txt').read_bytes()
                q = left.with_suffix('.bin').read_bytes() == right.with_suffix('.bin').read_bytes()
                clean = lambda data: [{k: v for k, v in row.items() if not k.endswith('_seconds')} for row in data]
                a, b = rows(left.with_suffix('.csv')), rows(right.with_suffix('.csv'))
                records = clean(a) == clean(b)
                assert inp and q and records
                assert len(a) == cfg['steps'] // cfg['eval_every'] + 1
                checkpoints += len(a)
                pairs.append({'distance': distance, 'mode': mode, 'seed': seed,
                              'input_bytes_equal': inp, 'final_q_bytes_equal': q,
                              'all_nonclock_records_equal': records})
    dp_errors = []
    for path in sorted((r30 / 'raw').glob('*.bin')):
        distance = int(path.stem.split('_')[0][1:])
        q = np.fromfile(path, dtype=np.float64).reshape(cfg['horizon'], 3, 1 + 3 * distance, 4)
        reference = model.reference_value(distance, cfg['horizon'], cfg['slip'], q)
        saved = float(rows(path.with_suffix('.csv'))[-1]['success_probability'])
        dp_errors.append(abs(reference - saved))
    assert len(dp_errors) == 180 and max(dp_errors) < 1e-12
    result['pilot_3_0'] = {'run_count': len(dp_errors), 'source_pair_count': len(pairs),
                          'paired_checkpoint_count': checkpoints,
                          'final_policy_independent_dp_max_absolute_error': max(dp_errors),
                          'pairs': pairs, 'summary': rows(r30 / 'summary.csv')}
    groups = defaultdict(dict)
    for row in rows(r31 / 'learning_curves.csv'):
        source = row.pop('source')
        groups[(row['gamma'], row['seed'])].setdefault(source, []).append(row)
    assert len(groups) == 36
    assert all(group['tl'] == group['manual'] for group in groups.values())
    assert all(len(group['tl']) == 41 for group in groups.values())
    cases = rows(r31 / 'exact_sweep.csv')
    for row in cases:
        p, pr, length, delay, gamma = [float(row[k]) for k in (
            'fast_probability', 'reliable_probability', 'fast_duration', 'extra_delay', 'gamma')]
        a, b = p * gamma ** (length - 1), pr * gamma ** (length + delay - 1)
        assert abs(a - float(row['fast_expected_return'])) < 1e-14
        assert abs(b - float(row['reliable_expected_return'])) < 1e-14
        assert row['selected_route'] == ('reliable' if b > a else 'fast')
    result['pilot_3_1'] = {'source_pair_count': len(groups), 'paired_checkpoint_count': len(groups) * 41,
                          'all_non_source_records_equal': True, 'exact_cases': len(cases),
                          'objective_mismatches': sum(int(r['objective_mismatch']) for r in cases),
                          'summary': rows(r31 / 'summary.csv')}
    hist = rows(p31 / 'results/history_boundary/compiled_states.csv')
    assert all(int(r['compiled_tl_states']) == int(r['handwritten_distinguishable_bitmasks']) == 2 ** int(r['obligations'])
               and r['bijection_verified'] == 'True' for r in hist)
    result['history_saved_audit'] = {'sizes': [int(r['obligations']) for r in hist],
                                   'total_transitions_checked': sum(int(r['all_transitions_checked']) for r in hist),
                                   'scope': json.loads((p31 / 'results/history_boundary/scope.json').read_text())}
    early = {}
    for path in sorted(PILOTS.glob('pilot_0_*/results/semantics.csv')):
        data = rows(path)
        mismatches = sum(int(v) for row in data for k, v in row.items() if k.endswith('_mismatches'))
        assert mismatches == 0
        early[path.parts[-3]] = {'saved_rows': len(data),
                                'sum_trajectory_evaluations_per_representation': sum(int(r['num_trajectories']) for r in data),
                                'total_mismatches': mismatches,
                                'caution': 'Counts include variants and possibly repeated traces; not unique tasks or RL runs.'}
    result['early_saved_semantic_checks'] = early
    (HERE / 'audit_evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('pilot_3_0',)}, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result['pilot_3_0'].items() if k not in ('pairs', 'summary')}, indent=2))


if __name__ == '__main__':
    audit()
