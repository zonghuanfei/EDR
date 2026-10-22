"""Recompute cross-framework EDR metrics using only the released package."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_PACKAGE = Path(__file__).resolve().parents[1]
GEMMA = "benchmarks/MedXpertQA/Gemma 3 12B IT"
EXPERIMENTS = [
    ("MedXpertQA_Qwen", "MedXpertQA", "Qwen3.8-27B-NVFP4", "MA-RAG-int", "benchmarks/MedXpertQA/qwen3.8 27b", 490, 10),
    ("MedXpertQA_Gemma_initial", "MedXpertQA", "Gemma 3 12B IT", "MA-RAG-int", GEMMA, 490, 10),
    ("MedDDx_Gemma", "MedDDx-Expert", "Gemma 3 12B IT", "MedLA", "benchmarks/MedDDx-Expert", 483, 4),
    ("MMLU_Gemma", "MMLU-Med", "Gemma 3 12B IT", "MedLA", "benchmarks/MMLU-Med", 1089, 4),
    ("MedXpertQA_Gemma_2026091900", "MedXpertQA", "Gemma 3 12B IT", "MA-RAG-int", GEMMA+"/multiseed/seed_2026091900", 490, 10),
    ("MedXpertQA_Gemma_2026091909", "MedXpertQA", "Gemma 3 12B IT", "MA-RAG-int", GEMMA+"/multiseed/seed_2026091909", 490, 10),
]

def load_lines(path):
    with path.open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    result = {r["id"]: r for r in rows}
    assert len(result) == len(rows), path
    return result

def choices(round_record, medla):
    return (round_record.get("decision") or {}).get("options" if medla else "predictions") or []

def ratio(numerator, denominator):
    return numerator / denominator if denominator else None

def mcnemar(repairs, harms):
    n = repairs + harms
    return min(1.0, 2*sum(math.comb(n, k) for k in range(min(repairs, harms)+1))/2**n) if n else 1.0

def bootstrap(differences):
    counts = Counter(differences)
    n = len(differences)
    rng = np.random.default_rng(20261003)
    samples = rng.multinomial(n, [counts[-1]/n, counts[0]/n, counts[1]/n], size=20000)
    estimates = (samples[:, 2]-samples[:, 0])/n*100
    return np.quantile(estimates, [.025, .975]).tolist()

def output_metrics(trace, initial, gold, medla, labels, exclude_failed_terminal=False):
    rounds = trace.get("rounds", [])
    active = any((r.get("routing") or {}).get("independent_count" if medla else "independent", 0) > 0 for r in rounds)
    valid_rounds = [r for r in rounds if (ps := choices(r, medla)) and all(p in labels for p in ps)
                    and (medla or len(ps) == 8)]
    later = valid_rounds if medla else [r for r in valid_rounds if r["round"] > 1]
    discovery = next((r for r in later if gold in choices(r, medla)), None)
    last_round = rounds[-1] if rounds else None
    terminal = last_round if last_round in valid_rounds else None
    if exclude_failed_terminal and trace.get("status") == "technical_failure":
        terminal = None
    initial_terminal = medla and not rounds and bool(initial) and all(p in labels for p in initial)
    pool = choices(terminal, medla) if terminal else initial if initial_terminal else []
    correct = trace.get("final_option") == gold
    if "correct" in trace:
        assert trace["correct"] == correct
    if medla:
        final = trace.get("final_decision") or {}
        consensus = final.get("type") in ("consensus", "initial_consensus")
        limit = 3
        terminal_round = rounds[-1]["round"] if rounds else 0
    else:
        consensus = bool(terminal) and len(set(pool)) == 1 and len(pool) == 8
        limit = 8
        terminal_round = terminal["round"] if terminal else 0
    early_wrong = consensus and trace.get("final_option") in labels and not correct and terminal_round < limit
    stage_count = len(rounds) + 1 if medla else sum(bool(r.get("answers")) for r in rounds)
    retrieval_count = 1 if medla else sum(r.get("retrieval") is not None for r in rounds)
    first_channels = []
    independent_only = discussion_only = 0
    for r in later:
        answers = r.get("answers", [])
        ps = choices(r, medla)
        if medla:
            channels = [a.get("channel", "unspecified") for a in answers]
        else:
            by_index = {a["index"]: a for a in answers}
            channels = [by_index.get(i, {}).get("channel", "unspecified") for i in range(len(ps))]
        correct_channels = {c for p, c in zip(ps, channels) if p == gold}
        if r is discovery:
            first_channels = sorted(correct_channels)
        route = r.get("routing") or {}
        m = route.get("independent_count" if medla else "independent", 0)
        if m > 0:
            independent_only += int("independent" in correct_channels and "discussion" not in correct_channels)
            discussion_only += int("discussion" in correct_channels and "independent" not in correct_channels)
    return {
        "correct": correct, "final_option": trace.get("final_option"), "active": active,
        "discovered_after_initial": discovery is not None,
        "first_discovery_round": discovery["round"] if discovery else None,
        "first_discovery_channels": first_channels,
        "terminal_pool_gold": gold in pool,
        "terminal_pool_available": bool(terminal) or initial_terminal,
        "early_wrong_consensus": early_wrong, "terminal_round": terminal_round,
        "solver_stages": stage_count, "retrieval_stages": retrieval_count,
        "independent_only_correct_rounds": independent_only,
        "discussion_only_correct_rounds": discussion_only,
    }

def group(rows):
    n = len(rows)
    bc = sum(r["baseline"]["correct"] for r in rows)
    ec = sum(r["edr"]["correct"] for r in rows)
    repairs = [r["id"] for r in rows if not r["baseline"]["correct"] and r["edr"]["correct"]]
    harms = [r["id"] for r in rows if r["baseline"]["correct"] and not r["edr"]["correct"]]
    return {"n": n, "baseline_correct": bc, "edr_correct": ec,
            "baseline_accuracy": ratio(bc, n), "edr_accuracy": ratio(ec, n),
            "gain_pp": (ec-bc)/n*100 if n else None,
            "repair_ids": repairs, "harm_ids": harms,
            "repair_rate": ratio(len(repairs), n-bc), "retention_rate": ratio(bc-len(harms), bc)}

def analyze(package, spec):
    key, dataset, model, framework, folder, n, width = spec
    folder = package / folder
    medla = framework == "MedLA"
    stem = "medla_shared_evidence" if medla else "ma_rag_int"
    suffix = "plus_edr"
    paths = [folder / f"traces/{stem}_{n}.jsonl", folder / f"traces/{stem}_{suffix}_{n}.jsonl"]
    base, edr = [load_lines(p) for p in paths]
    shared_path = folder / f"traces/shared_initial_{n}.jsonl"
    shared = load_lines(shared_path) if medla else None
    assert set(base) == set(edr) == set(range(n))
    labels = set("ABCDEFGHIJ"[:width])
    rows = []
    for qid in range(n):
        b, e = base[qid], edr[qid]
        gold = b["gold"]
        assert gold == e["gold"] and gold in labels
        if medla:
            initial = shared[qid].get("initial_answers", [])
            valid = shared[qid]["status"] == "complete" and len(initial) == 17 and all(p in labels for p in initial)
        else:
            initial = choices(b["rounds"][0], False) if b.get("rounds") else []
            other = choices(e["rounds"][0], False) if e.get("rounds") else []
            valid = len(initial) == 8 and initial == other and all(p in labels for p in initial)
        row = {"id": qid, "gold": gold, "valid_shared_initial": valid,
               "initial_disagreement": valid and len(set(initial)) > 1,
               "initial_gold": valid and gold in initial,
               "baseline": output_metrics(b, initial, gold, medla, labels, key == "MedXpertQA_Qwen"),
               "edr": output_metrics(e, initial, gold, medla, labels, key == "MedXpertQA_Qwen")}
        rows.append(row)
    full = group(rows)
    full.update(relative_accuracy_gain=ratio(full["edr_correct"]-full["baseline_correct"], full["baseline_correct"]),
                paired_bootstrap_ci95_pp=bootstrap([int(r["edr"]["correct"])-int(r["baseline"]["correct"]) for r in rows]),
                mcnemar_exact_p=mcnemar(len(full["repair_ids"]), len(full["harm_ids"])))
    disputed = [r for r in rows if r["initial_disagreement"]]
    active = [r for r in rows if r["edr"]["active"]]
    covered = [r for r in disputed if r["initial_gold"] and r["baseline"]["terminal_pool_available"] and r["edr"]["terminal_pool_available"]]
    absent = [r for r in disputed if not r["initial_gold"]]
    mechanism = {"disputed_n": len(disputed), "covered_n": len(covered), "initial_absent_disputed_n": len(absent)}
    for method in ["baseline", "edr"]:
        discovery = [r["id"] for r in absent if r[method]["discovered_after_initial"]]
        mechanism[method] = {
            "early_wrong_consensus_ids": [r["id"] for r in disputed if r[method]["early_wrong_consensus"]],
            "discovery_ids": discovery, "discovery_rate": ratio(len(discovery), len(absent)),
            "discovery_final_correct_ids": [r["id"] for r in absent if r[method]["discovered_after_initial"] and r[method]["correct"]],
            "covered_gold_lost_ids": [r["id"] for r in covered if not r[method]["terminal_pool_gold"]],
            "covered_gold_lost_rate": ratio(sum(not r[method]["terminal_pool_gold"] for r in covered), len(covered)),
            "solver_stages": sum(r[method]["solver_stages"] for r in rows),
            "retrieval_stages": sum(r[method]["retrieval_stages"] for r in rows),
        }
    avoided = [r for r in disputed if r["baseline"]["early_wrong_consensus"] and not r["edr"]["early_wrong_consensus"]]
    mechanism["avoided_early_wrong_ids"] = [r["id"] for r in avoided]
    mechanism["correct_after_avoiding_ids"] = [r["id"] for r in avoided if r["edr"]["correct"]]
    mechanism["edr_first_discovery_channels"] = dict(Counter("+".join(r["edr"]["first_discovery_channels"]) for r in absent if r["edr"]["discovered_after_initial"]))
    all_initial_covered_errors = [r for r in rows if r["valid_shared_initial"] and r["initial_gold"] and not r["baseline"]["correct"]]
    mechanism["initial_covered_error_repair"] = {
        "n": len(all_initial_covered_errors),
        "repaired_ids": [r["id"] for r in all_initial_covered_errors if r["edr"]["correct"]],
    }
    raw_score_file = folder / f"results/final_scores_{n}.json"
    if not raw_score_file.exists():
        raw_score_file = folder / f"results/{stem}_plus_edr_{n}.json"
    score = json.loads(raw_score_file.read_text(encoding="utf-8-sig"))
    if "conditions" in score:
        assert full["baseline_correct"] == score["conditions"][stem]["correct"]
        assert full["edr_correct"] == score["conditions"][stem+"_plus_edr"]["correct"]
    else:
        assert full["edr_correct"] == score["correct"]
        baseline_score = folder / f"results/{stem}_{n}.json"
        assert full["baseline_correct"] == json.loads(baseline_score.read_text(encoding="utf-8-sig"))["correct"]
    return {"key": key, "dataset": dataset, "model": model, "framework": framework,
            "full": full, "active": group(active), "disputed": group(disputed),
            "inactive": group([r for r in rows if not r["edr"]["active"]]),
            "invalid_initial_ids": [r["id"] for r in rows if not r["valid_shared_initial"]],
            "mechanism": mechanism, "question_metrics": rows}

def auxiliary(package):
    neighbors = []
    for folder, n, stem, dataset in [
        (GEMMA+"/madmm_objective", 490, "ma_rag_int", "MedXpertQA"),
        ("benchmarks/MedDDx-Expert/madmm_objective", 483, "medla_shared_evidence", "MedDDx-Expert"),
    ]:
        path = package / folder / f"results/final_scores_{n}.json"
        score = json.loads(path.read_text(encoding="utf-8-sig"))
        counts = {method: score["conditions"][stem+suffix]["correct"]
                  for method,suffix in [("baseline", ""), ("edr", "_plus_edr"), ("mad", "_plus_madmm_objective")]}
        neighbors.append({"dataset": dataset, "framework": "MA-RAG-int" if n == 490 else "MedLA", "n": n,
                          **counts, "edr_vs_mad_gain_pp": (counts["edr"]-counts["mad"])/n*100})
    folder = package / "benchmarks/MedXpertQA/qwen3.8 27b"
    path = folder / "results/comparisons.json"
    comparisons = json.loads(path.read_text(encoding="utf-8-sig"))
    allocation = comparisons["allocation_comparison"]
    ids = set(range(490))
    assert allocation["n"] == len(ids) and allocation["excluded_ids"] == []
    for method, stem in [("adaptive", "ma_rag_int_plus_edr"), ("random42", "edr_gated_random_s42")]:
        path = folder / f"traces/{stem}_490.jsonl"
        traces = load_lines(path)
        post = [r for qid,trace in traces.items() if qid in ids for r in trace["rounds"]
                if r["round"] > 1 and r.get("answers")]
        count = sum((r.get("routing") or {}).get("independent", 0) for r in post)
        activated = [r for r in post if (r.get("routing") or {}).get("independent", 0) > 0]
        actual = {
            "full_correct": sum(t.get("final_option") == t["gold"] for t in traces.values()),
            "matched_correct": sum(traces[i].get("final_option") == traces[i]["gold"] for i in ids),
            "active_rounds": len(activated),
            "post_initial_rounds": len(post), "independent_answers": count,
            "independent_fraction": count/(8*len(post)),
            "active_round_independent_fraction": count/(8*len(activated)),
        }
        assert allocation["methods"][method] == actual
    return {"neighbors": neighbors, "allocation": allocation,
            "allocation_accuracy_comparisons": comparisons["comparisons"]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for spec in EXPERIMENTS:
        result = analyze(args.package, spec)
        results.append(result)
        m = result["mechanism"]
        print(json.dumps({"experiment": result["key"], "full": {k:v for k,v in result["full"].items() if not k.endswith("_ids")},
              "active": {k:v for k,v in result["active"].items() if not k.endswith("_ids")},
              "early_wrong": [len(m[x]["early_wrong_consensus_ids"]) for x in ["baseline", "edr"]],
              "discovery": [len(m["baseline"]["discovery_ids"]),len(m["edr"]["discovery_ids"]),m["initial_absent_disputed_n"]],
              "candidate_loss": [len(m[x]["covered_gold_lost_ids"]) for x in ["baseline", "edr"]]+[m["covered_n"]],
              "channels": m["edr_first_discovery_channels"]}, ensure_ascii=True), flush=True)
    extra = auxiliary(args.package)
    print(json.dumps(extra, ensure_ascii=True), flush=True)
    output = {"schema": "edr-systematic-metrics-v1",
              "cohort_definitions": {
                  "full": "Every released question, with missing final options scored incorrect",
                  "active": "Any actual post-initial allocation assigns at least one independent response; includes later technical failures",
                  "disputed": "Valid shared initial answer choices with at least two distinct legal options",
                  "discovery": "Initially gold-absent disputed questions; a valid later medical-answer round first contains the gold choice",
                  "candidate_loss": "Initially gold-present disputed questions with an available terminal medical-answer pool in both arms; terminal pool excludes gold",
                  "early_wrong": "Disputed questions with a wrong unanimous terminal medical-answer round before the host round limit; excludes initially unanimous questions",
                  "statistics": "Paired-question percentile bootstrap; equivalent multinomial resampling of per-question -1/0/+1 differences; 20000 resamples, statistical seed 20261003; exact two-sided McNemar",
              }, "experiments": results, "additional_comparisons": extra}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

if __name__ == "__main__":
    main()
