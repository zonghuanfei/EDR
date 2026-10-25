"""Summarize paired final accuracy on questions with shared initial disagreement.

Usage:
    python -B tools/summarize_initial_disagreement.py
    python -B tools/summarize_initial_disagreement.py --check
"""

import argparse
import json
from pathlib import Path

from analyze_systematic_metrics import EXPERIMENTS, bootstrap, mcnemar


PACKAGE = Path(__file__).resolve().parents[1]


def initial_group(options, expected_count, labels, complete=True):
    if not complete or len(options) != expected_count or any(x not in labels for x in options):
        return "invalid_initial"
    return "initial_disagreement" if len(set(options)) > 1 else "initial_unanimous"


def project_record(record, shared=False):
    """Keep only fields needed for the comparison; do not copy generated text."""
    if shared:
        return {
            "gold": record["gold"],
            "initial": record.get("initial_answers") or [],
            "complete": record.get("status") == "complete",
        }
    rounds = record.get("rounds") or []
    first = next((r for r in rounds if r.get("round") == 1), {})
    correct = record.get("final_option") == record["gold"]
    if "correct" in record and record["correct"] != correct:
        raise ValueError(f"Incorrect stored score for question {record['id']}")
    return {
        "gold": record["gold"],
        "initial": (first.get("decision") or {}).get("predictions") or [],
        "shared_initial_ref": record.get("shared_initial_ref"),
        "correct": correct,
    }


def load_records(path, shared=False):
    records = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["id"]
            if qid in records:
                raise ValueError(f"Duplicate question ID {qid} in {path.name}")
            records[qid] = project_record(row, shared)
    return records


def select_cohort(methods, initial, expected_count, labels, shared_file=None, trace_files=None):
    groups = {k: [] for k in ("initial_disagreement", "initial_unanimous", "invalid_initial")}
    for qid, source in sorted(initial.items()):
        if source["gold"] not in labels:
            raise ValueError(f"Invalid gold label for question {qid}")
        category = initial_group(source["initial"], expected_count, labels, source.get("complete", True))
        for name, records in methods.items():
            row = records[qid]
            if row["gold"] != source["gold"]:
                raise ValueError(f"Gold mismatch for {name}, question {qid}")
            if shared_file is not None:
                ref = row.get("shared_initial_ref") or {}
                if ref.get("id") != qid or "file" not in ref:
                    raise ValueError(f"Missing shared initial reference for {name}, question {qid}")
                if (trace_files[name].parent / ref["file"]).resolve() != shared_file.resolve():
                    raise ValueError(f"Different shared initial reference for {name}, question {qid}")
            elif row["initial"] != source["initial"]:
                raise ValueError(f"Initial option mismatch for {name}, question {qid}")
        groups[category].append(qid)
    return groups


def paired_summary(reference, method, ids):
    n = len(ids)
    reference_correct = sum(reference[i]["correct"] for i in ids)
    method_correct = sum(method[i]["correct"] for i in ids)
    corrections = [i for i in ids if not reference[i]["correct"] and method[i]["correct"]]
    regressions = [i for i in ids if reference[i]["correct"] and not method[i]["correct"]]
    reference_errors = n - reference_correct
    differences = [int(method[i]["correct"]) - int(reference[i]["correct"]) for i in ids]
    return {
        "n": n,
        "reference_correct": reference_correct,
        "method_correct": method_correct,
        "reference_accuracy_percent": 100 * reference_correct / n if n else None,
        "method_accuracy_percent": 100 * method_correct / n if n else None,
        "gain_pp": 100 * (method_correct - reference_correct) / n if n else None,
        "corrections": len(corrections),
        "regressions": len(regressions),
        "reference_errors": reference_errors,
        "correction_rate_percent": 100 * len(corrections) / reference_errors if reference_errors else None,
        "both_correct": reference_correct - len(regressions),
        "both_incorrect": reference_errors - len(corrections),
        "paired_bootstrap_ci95_pp": bootstrap(differences) if n else None,
        "mcnemar_exact_p": mcnemar(len(corrections), len(regressions)) if n else None,
        "correction_ids": corrections,
        "regression_ids": regressions,
    }


def trace_specs(folder, key, n, medla):
    stem = "medla_shared_evidence" if medla else "ma_rag_int"
    traces = {"baseline": folder / f"traces/{stem}_{n}.jsonl"}
    if key == "MedXpertQA_Qwen":
        traces["gated_fixed4"] = folder / f"traces/edr_gated_fixed4_{n}.jsonl"
        traces["gated_random_s42"] = folder / f"traces/edr_gated_random_s42_{n}.jsonl"
    if key in ("MedXpertQA_Gemma_initial", "MedDDx_Gemma"):
        traces["madmm_objective"] = folder / f"madmm_objective/traces/{stem}_plus_madmm_objective_{n}.jsonl"
    traces["edr"] = folder / f"traces/{stem}_plus_edr_{n}.jsonl"
    return traces


def check_full_scores(folder, trace_files, methods, stem, n):
    common_scores = folder / f"results/final_scores_{n}.json"
    common = json.loads(common_scores.read_text(encoding="utf-8-sig")) if common_scores.exists() else None
    for name, path in trace_files.items():
        if name == "madmm_objective":
            score_file = folder / f"madmm_objective/results/final_scores_{n}.json"
            expected = json.loads(score_file.read_text(encoding="utf-8-sig"))["conditions"][stem + "_plus_madmm_objective"]["correct"]
        elif common is not None:
            expected = common["conditions"][stem + ("_plus_edr" if name == "edr" else "")]["correct"]
        else:
            score_file = folder / "results" / (path.stem + ".json")
            expected = json.loads(score_file.read_text(encoding="utf-8-sig"))["correct"]
        if sum(r["correct"] for r in methods[name].values()) != expected:
            raise ValueError(f"Full score mismatch for {name}")


def analyze_experiment(package, spec):
    key, dataset, model, framework, relative_folder, n, option_count = spec
    folder = package / relative_folder
    medla = framework == "MedLA"
    labels = set("ABCDEFGHIJ"[:option_count])
    files = trace_specs(folder, key, n, medla)
    methods = {name: load_records(path) for name, path in files.items()}
    for name, records in methods.items():
        if set(records) != set(range(n)):
            raise ValueError(f"Unexpected question coverage for {key}/{name}")
    shared_file = folder / f"traces/shared_initial_{n}.jsonl" if medla else None
    initial = load_records(shared_file, shared=True) if medla else methods["baseline"]
    if set(initial) != set(range(n)):
        raise ValueError(f"Unexpected shared initial coverage for {key}")
    groups = select_cohort(methods, initial, 17 if medla else 8, labels, shared_file, files)
    ids = groups["initial_disagreement"]
    stem = "medla_shared_evidence" if medla else "ma_rag_int"
    check_full_scores(folder, files, methods, stem, n)
    summaries = {}
    for name, records in methods.items():
        correct = sum(records[i]["correct"] for i in ids)
        summaries[name] = {
            "trace": files[name].relative_to(package).as_posix(),
            "full_correct": sum(r["correct"] for r in records.values()),
            "correct": correct,
            "accuracy_percent": 100 * correct / len(ids) if ids else None,
        }
    pairs = [("baseline", name) for name in methods if name != "baseline"]
    pairs += [(name, "edr") for name in methods if name not in ("baseline", "edr")]
    comparisons = []
    for reference, method in pairs:
        comparisons.append({
            "reference": reference,
            "method": method,
            **paired_summary(methods[reference], methods[method], ids),
        })
    return {
        "key": key,
        "dataset": dataset,
        "model": model,
        "framework": framework,
        "total_questions": n,
        "initial_source": (shared_file or files["baseline"]).relative_to(package).as_posix(),
        "initial_counts": {name: len(qids) for name, qids in groups.items()},
        "question_ids": ids,
        "methods": summaries,
        "comparisons": comparisons,
    }


def check_existing_metrics(package, experiments):
    path = package / "analysis/systematic_edr_metrics.json"
    existing = {e["key"]: e["disputed"] for e in json.loads(path.read_text(encoding="utf-8"))["experiments"]}
    for e in experiments:
        old = existing[e["key"]]
        new = next(c for c in e["comparisons"] if c["reference"] == "baseline" and c["method"] == "edr")
        pairs = [("n", "n"), ("reference_correct", "baseline_correct"), ("method_correct", "edr_correct"),
                 ("correction_ids", "repair_ids"), ("regression_ids", "harm_ids")]
        if any(new[a] != old[b] for a, b in pairs):
            raise ValueError(f"Initial-disagreement reconstruction mismatch for {e['key']}")


def build(package=PACKAGE):
    experiments = [analyze_experiment(package, spec) for spec in EXPERIMENTS]
    check_existing_metrics(package, experiments)
    return {
        "definitions": {
            "cohort": "Shared valid initial answer choices with at least two distinct legal options; eight choices in MA-RAG-int and 17 in MedLA.",
            "accuracy": "Final correct answers divided by all questions in the initial-disagreement cohort; missing final options count as incorrect.",
            "gain_pp": "Method accuracy minus reference accuracy, in percentage points.",
            "correction": "Reference incorrect and comparison method correct on the same question.",
            "regression": "Reference correct and comparison method incorrect on the same question.",
            "correction_rate_percent": "Corrections divided by reference errors in the same cohort, multiplied by 100.",
            "uncertainty": "95% paired-question percentile bootstrap intervals, 20000 resamples, statistical seed 20261003; two-sided exact McNemar p-values for each comparison.",
        },
        "experiments": experiments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output or args.package / "analysis/initial_disagreement_metrics.json"
    result = build(args.package)
    if args.check:
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError("Stored initial-disagreement metrics differ from reconstructed results")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for e in result["experiments"]:
        print(e["key"] + f" (n={len(e['question_ids'])})")
        for c in e["comparisons"]:
            print(f"  {c['method']} vs {c['reference']}: "
                  f"{c['reference_correct']} -> {c['method_correct']}; "
                  f"gain {c['gain_pp']:+.2f} pp; "
                  f"corrections/regressions {c['corrections']}/{c['regressions']}")
    print("PASS: paired initial states, complete scores and initial-disagreement results")


if __name__ == "__main__":
    main()
