"""Recompute allocation-policy outcomes from the released question traces."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from edr import route

FOLDER = ROOT / 'benchmarks/MedXpertQA/qwen3.8 27b'
METHODS = ('ma_rag_int', 'edr_gated_fixed4', 'edr_gated_random_s42', 'ma_rag_int_plus_edr')


def load(name):
    path = FOLDER / 'traces' / (name + '_490.jsonl')
    rows = [json.loads(line) for line in path.open(encoding='utf-8-sig') if line.strip()]
    records = {row['id']: row for row in rows}
    assert len(rows) == len(records) == 490
    assert set(records) == set(range(490))
    return records


def summarize(name, records):
    result = json.loads((FOLDER / 'results' / (name + '_490.json')).read_text(encoding='utf-8'))
    predictions = {row['id']: row['prediction'] for row in result['rows']}
    correct = 0
    counts = Counter()
    failures = []
    for qid, record in records.items():
        assert record['condition'] == name or (name == 'edr_gated_random_s42' and record['condition'] == 'edr_gated_random')
        assert record.get('final_option') == predictions[qid]
        assert record['correct'] == (record.get('final_option') == record['gold'])
        correct += record['correct']
        if record.get('status') == 'technical_failure':
            failures.append(qid)
        for rd in record['rounds']:
            if rd['round'] == 1 or not rd.get('answers'):
                continue
            assert len(rd['answers']) == 8
            routing = rd.get('routing') or {}
            m = routing.get('independent', 0)
            counts[m] += 1
            if name == 'edr_gated_fixed4':
                assert m in (0, 4)
                assert routing['discussion'] == 8 - m
                if not routing.get('fallback'):
                    p = routing['distributions']
                    reference = route(p['E'], p['H'], p['EH'])
                    assert m == (4 if reference['independent'] > 0 else 0)
    assert correct == result['correct']
    rounds = sum(counts.values())
    independent = sum(m * count for m, count in counts.items())
    return {
        'trace': (FOLDER/'traces'/(name+'_490.jsonl')).relative_to(ROOT).as_posix(),
        'n': 490, 'correct': correct, 'accuracy': correct/490,
        'subsequent_rounds': rounds, 'independent_responses': independent,
        'independent_response_share': independent/(8*rounds),
        'allocation_round_counts': {str(m): counts[m] for m in sorted(counts)},
        'technical_failure_ids': sorted(failures),
    }


def pairing(baseline, other):
    initial = document_sets = document_orders = query_orders = comparable = 0
    for qid in baseline:
        b, x = baseline[qid], other[qid]
        assert b['gold'] == x['gold']
        rb, rx = b['rounds'][0], x['rounds'][0]
        same_initial = rb['answers'] == rx['answers'] and rb['decision'] == rx['decision']
        initial += same_initial
        assert same_initial
        eb, ex = rb.get('retrieval'), rx.get('retrieval')
        if eb is None or ex is None:
            continue
        comparable += 1
        db, dx = [d['id'] for d in eb['documents']], [d['id'] for d in ex['documents']]
        document_sets += set(db) == set(dx)
        document_orders += db == dx
        query_orders += eb['queries'] == ex['queries']
    return {'same_initial_responses': initial, 'comparable_first_retrievals': comparable,
            'same_first_retrieval_document_sets': document_sets,
            'same_first_retrieval_document_order': document_orders,
            'same_first_retrieval_query_order': query_orders}


def build():
    data = {name: load(name) for name in METHODS}
    pairings = {name: pairing(data['ma_rag_int'], data[name]) for name in METHODS[1:]}
    for name in METHODS[1:]:
        p = pairings[name]
        assert p['same_first_retrieval_document_order'] == p['comparable_first_retrievals']
        assert p['same_first_retrieval_query_order'] == p['comparable_first_retrievals']
    return {'dataset': 'MedXpertQA', 'model': 'Qwen3.8-27B-NVFP4',
            'methods': {name: summarize(name, records) for name, records in data.items()},
            'first_retrieval_pairing_against_baseline': pairings,
            'ordered_prefix_paired_comparison': ['edr_gated_fixed4', 'edr_gated_random_s42', 'ma_rag_int_plus_edr']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = build()
    output = ROOT / 'analysis/allocation_ablation.json'
    if args.check:
        assert json.loads(output.read_text(encoding='utf-8')) == result
    else:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for name, row in result['methods'].items():
        print(f"{name}: {row['correct']}/{row['n']}; {row['subsequent_rounds']} subsequent rounds; "
              f"{row['independent_responses']} independent responses")
    print('PASS: allocation-policy outcomes and first-retrieval pairing')


if __name__ == '__main__':
    main()
