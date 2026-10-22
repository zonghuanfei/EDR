#!/usr/bin/env python3
"""Rebuild current MedXpertQA/Qwen scores and paired strategy comparisons."""
import argparse
import json
import os
import tempfile
from pathlib import Path

from analyze_systematic_metrics import bootstrap, mcnemar

ROOT = Path(__file__).resolve().parents[1] / 'benchmarks/MedXpertQA/qwen3.8 27b'
CONDITIONS = {
    'direct_model': 'direct_model',
    'ma_rag_int': 'ma_rag_int',
    'ma_rag_int_plus_edr': 'ma_rag_int_plus_edr',
    'edr_gated_random_s42': 'edr_gated_random',
}

def read_traces(root=ROOT):
    traces, gold = {}, {}
    for stem, condition in CONDITIONS.items():
        path = root / 'traces' / f'{stem}_490.jsonl'
        rows = [json.loads(line) for line in path.open(encoding='utf-8-sig') if line.strip()]
        by_id = {r['id']: r for r in rows}
        assert len(rows) == len(by_id) == 490 and set(by_id) == set(range(490)), path
        for row in rows:
            assert row['condition'] == condition, (path, row['id'])
            assert row['correct'] == (row.get('final_option') == row['gold']), (path, row['id'])
            assert gold.setdefault(row['id'], row['gold']) == row['gold']
            if row.get('status') == 'technical_failure':
                assert row.get('final_option') is None and not row['correct']
        traces[stem] = by_id
    return traces

def condition_result(stem, rows):
    ordered = [rows[qid] for qid in sorted(rows)]
    failures = [r for r in ordered if r.get('status') == 'technical_failure']
    correct = sum(r['correct'] for r in ordered)
    result = {
        'dataset': 'MedXpertQA', 'condition': CONDITIONS[stem],
        'metric': 'exact final-option accuracy; missing, tied and technical-failure outcomes count as incorrect',
        'correct': correct, 'total': len(ordered), 'accuracy': correct / len(ordered),
        'rows': [dict({'id': r['id'], 'gold': r['gold'], 'prediction': r.get('final_option'), 'correct': r['correct']},
                     **({'status': r['status']} if 'status' in r else {})) for r in ordered],
    }
    if any('status' in r for r in ordered):
        result.update(completed_traces=len(ordered)-len(failures), technical_failures=len(failures),
                      incomplete_trace_ids=[r['id'] for r in failures],
                      technical_failure_records=[{'id': r['id'], **r.get('technical_failure', {})} for r in failures])
    if stem == 'edr_gated_random_s42':
        result.update(run='random42', allocation_seed=42, generation_seed=None)
    return result

def compare(left_name, right_name, traces, ids=None):
    ids = sorted(ids if ids is not None else traces[left_name])
    left, right = traces[left_name], traces[right_name]
    repairs = [i for i in ids if left[i]['correct'] and not right[i]['correct']]
    harms = [i for i in ids if right[i]['correct'] and not left[i]['correct']]
    lc, rc = sum(left[i]['correct'] for i in ids), sum(right[i]['correct'] for i in ids)
    assert lc-rc == len(repairs)-len(harms)
    return {'left': left_name, 'right': right_name, 'n': len(ids), 'left_correct': lc, 'right_correct': rc,
            'left_accuracy': lc/len(ids), 'right_accuracy': rc/len(ids),
            'repairs': len(repairs), 'harms': len(harms), 'net_correct': lc-rc,
            'gain_percentage_points': (lc-rc)/len(ids)*100,
            'repair_ids': repairs, 'harm_ids': harms,
            'paired_bootstrap_ci95_pp': bootstrap([int(left[i]['correct'])-int(right[i]['correct']) for i in ids]),
            'mcnemar_exact_p': mcnemar(len(repairs), len(harms))}

def allocation_summary(rows):
    post = [r for q in rows.values() for r in q.get('rounds', []) if r['round'] > 1 and r.get('answers')]
    active = [r for r in post if (r.get('routing') or {}).get('independent', 0) > 0]
    independent = sum(r.get('routing', {}).get('independent', 0) for r in post)
    return {'full_correct': sum(r['correct'] for r in rows.values()), 'matched_correct': sum(r['correct'] for r in rows.values()),
            'post_initial_rounds': len(post), 'active_rounds': len(active), 'independent_answers': independent,
            'independent_fraction': independent/(8*len(post)) if post else None,
            'active_round_independent_fraction': independent/(8*len(active)) if active else None}

def comparisons(traces):
    return {
        'dataset': 'MedXpertQA', 'n': 490,
        'comparison_protocol': 'Shared initial responses and ordered first retrieval; each strategy executes its own probes and later trajectory. All 490 endpoints are retained.',
        'statistics': 'Exact two-sided McNemar and 20000 paired bootstrap resamples; statistical seed 20261003.',
        'comparisons': {
            'edr_vs_baseline_490': compare('ma_rag_int_plus_edr', 'ma_rag_int', traces),
            'edr_vs_random42_490': compare('ma_rag_int_plus_edr', 'edr_gated_random_s42', traces),
            'random42_vs_baseline_490': compare('edr_gated_random_s42', 'ma_rag_int', traces),
        },
        'allocation_comparison': {'n': 490, 'excluded_ids': [], 'methods': {
            'adaptive': allocation_summary(traces['ma_rag_int_plus_edr']),
            'random42': allocation_summary(traces['edr_gated_random_s42']),
        }},
    }

def build(root=ROOT):
    traces = read_traces(root)
    outputs = {root/'results'/f'{stem}_490.json': condition_result(stem, rows) for stem, rows in traces.items()}
    outputs[root/'results/comparisons.json'] = comparisons(traces)
    return outputs

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for path, result in build(args.root).items():
        if args.check:
            assert json.loads(path.read_text(encoding='utf-8')) == result, path
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=path.parent, prefix='.stage_', suffix='.json', delete=False) as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
            staged = Path(stream.name)
        os.replace(staged, path)
    print('PASS: '+('checked' if args.check else 'built')+' MedXpertQA scores and paired comparisons')

if __name__ == '__main__':
    main()
