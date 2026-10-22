#!/usr/bin/env python3
"""Verify score reconstruction, paired inputs, generated answers and routing."""
import ast
from collections import Counter
import json
import math
from pathlib import Path
import re

import numpy as np
from build_medxpertqa_results import ROOT, CONDITIONS, build, read_traces

PACKAGE = ROOT.parents[2]

def source_function(relative, name):
    tree = ast.parse((PACKAGE / relative).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'re': re, 'math': math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(relative), 'exec'), namespace)
    return namespace[name]

parse_answer = source_function('src/run_direct_medxpertqa.py', 'parse_answer')
route = source_function('src/edr.py', 'route')

def majority(options):
    votes = Counter(options).most_common()
    return votes[0][0] if len(votes) == 1 or votes[0][1] > votes[1][1] else None

def check_trace(stem, n, correct_expected, gold, root=ROOT, expected_ids=None, predictions=None):
    path = root / 'traces' / f'{stem}_{n}.jsonl'
    rows = [json.loads(line) for line in path.open(encoding='utf-8-sig') if line.strip()]
    ids, total_correct = set(), 0
    stats = Counter()
    for row in rows:
        qid = row['id']
        assert qid not in ids and row['condition'] == CONDITIONS.get(stem, stem), (path, qid)
        ids.add(qid)
        assert row['gold'] == gold[qid]
        assert row['correct'] == (row.get('final_option') == row['gold'])
        if predictions is not None:
            assert row.get('final_option') == predictions[qid]
        total_correct += row['correct']
        if row.get('status') == 'technical_failure':
            assert row.get('final_option') is None and not row['correct']
            stats['technical_failures'] += 1
        assert 1 <= len(row['rounds']) <= 8
        assert [r['round'] for r in row['rounds']] == list(range(1, len(row['rounds'])+1))
        for rd in row['rounds']:
            answers = sorted(rd['answers'], key=lambda a: a['index'])
            assert len(answers) == 8 and [a['index'] for a in answers] == list(range(8))
            assert all(a.get('finish_reason') == 'stop' and a.get('text') for a in answers)
            ps = [parse_answer(a['text'], set('ABCDEFGHIJ')) for a in answers]
            decision = rd['decision']
            assert ps == decision['predictions'] and all(p is not None for p in ps), (path, qid, rd['round'])
            assert majority(ps) == decision['prediction']
            if 'history_order' in rd:
                assert rd['history_order'] == np.argsort(decision['entropies'])[::-1].tolist()
            stats['solver_rounds'] += 1
            stats['parsed_answers'] += 8
            rt = rd.get('routing')
            if rt:
                m = rt['independent']
                assert isinstance(m, int) and 0 <= m <= 6 and rt['discussion'] == 8-m
                stats['routed_rounds'] += 1
                stats['independent_answers'] += m
                if not rt.get('fallback'):
                    p = rt['distributions']
                    reference = route(p['E'], p['H'], p['EH'])
                    for key in ('s', 'tv_e', 'tv_h'):
                        assert math.isclose(rt[key], reference[key], rel_tol=1e-9, abs_tol=1e-10)
                    for key in ('top_e', 'top_eh', 'top_tie'):
                        assert rt[key] == reference[key]
                    if stem == 'edr_gated_random_s42':
                        assert rt['adaptive_independent'] == reference['independent']
                        assert rt['trigger'] == (reference['independent'] > 0)
                        assert 1 <= m <= 6 if rt['trigger'] else m == 0
                        assert rt['allocation_seed'] == 42 and row['allocation_seed'] == 42
                    else:
                        assert m == reference['independent']
            retrieval = rd.get('retrieval') or {}
            for doc in retrieval.get('documents', []):
                assert set(doc) == {'id', 'title', 'score'}
                stats['retrieved_documents'] += 1
                stats['missing_scores'] += doc['score'] is None
        if row.get('status') != 'technical_failure':
            last = row['rounds'][-1]
            assert row['final_option'] == last['decision']['prediction']
            assert last['round'] == 8 or len(set(last['decision']['predictions'])) == 1
    assert ids == (expected_ids if expected_ids is not None else set(range(n)))
    assert total_correct == correct_expected
    return dict(stats)

def check_pairing(traces):
    base = traces['ma_rag_int']
    stats = {}
    for stem in ('ma_rag_int_plus_edr', 'edr_gated_random_s42'):
        equal_retrieval, technical = 0, []
        for qid, row in traces[stem].items():
            b, e = base[qid]['rounds'][0], row['rounds'][0]
            assert b['answers'] == e['answers'] and b['decision'] == e['decision']
            assert base[qid]['first_round_option'] == row['first_round_option']
            if row.get('status') == 'technical_failure':
                technical.append(qid)
                continue
            assert b.get('retrieval') == e.get('retrieval'), (stem, qid, 'first ordered retrieval')
            equal_retrieval += bool(b.get('retrieval'))
        stats[stem] = {'shared_initial_responses': len(base), 'same_ordered_first_retrieval': equal_retrieval,
                       'technical_failure_ids': sorted(technical)}
    return stats

def main():
    for path, expected in build().items():
        assert json.loads(path.read_text(encoding='utf-8')) == expected, path
    traces = read_traces()
    gold = {i:r['gold'] for i,r in traces['ma_rag_int'].items()}
    stats = {}
    for stem in ('ma_rag_int','ma_rag_int_plus_edr','edr_gated_random_s42'):
        results = json.loads((ROOT/'results'/f'{stem}_490.json').read_text(encoding='utf-8'))
        ps = {r['id']:r['prediction'] for r in results['rows']}
        stats[stem] = check_trace(stem,490,results['correct'],gold,predictions=ps)
    for row in traces['direct_model'].values():
        text = row['answer'] if isinstance(row['answer'], str) else row['answer']['text']
        assert parse_answer(text, set('ABCDEFGHIJ')) == row['final_option']
    print(json.dumps({'MedXpertQA':stats,'pairing':check_pairing(traces)},indent=2))
    print('PASS: coverage, answer parsing, final votes, route formula, ordered pairing and current results')

if __name__ == '__main__':
    main()
