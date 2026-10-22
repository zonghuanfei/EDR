"""Summarize observed full-benchmark and routing-active paired outcomes."""

import argparse
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    {
        "name": "MedXpertQA / Qwen3.8-27B-NVFP4",
        "baseline": PACKAGE / "benchmarks/MedXpertQA/qwen3.8 27b/traces/ma_rag_int_490.jsonl",
        "edr": PACKAGE / "benchmarks/MedXpertQA/qwen3.8 27b/traces/ma_rag_int_plus_edr_490.jsonl",
        "output": PACKAGE / "benchmarks/MedXpertQA/qwen3.8 27b/results/routing_subset_analysis.json",
    },
    {
        "name": "MedXpertQA / Gemma 3 12B IT / initial run",
        "baseline": PACKAGE / "benchmarks/MedXpertQA/Gemma 3 12B IT/traces/ma_rag_int_490.jsonl",
        "edr": PACKAGE / "benchmarks/MedXpertQA/Gemma 3 12B IT/traces/ma_rag_int_plus_edr_490.jsonl",
        "output": PACKAGE / "benchmarks/MedXpertQA/Gemma 3 12B IT/results/routing_subset_analysis.json",
    },
    {
        "name": "MedXpertQA / Gemma 3 12B IT / seed 2026091900",
        "baseline": PACKAGE / "benchmarks/MedXpertQA/Gemma 3 12B IT/multiseed/seed_2026091900/traces/ma_rag_int_490.jsonl",
        "edr": PACKAGE / "benchmarks/MedXpertQA/Gemma 3 12B IT/multiseed/seed_2026091900/traces/ma_rag_int_plus_edr_490.jsonl",
        "output": PACKAGE / "benchmarks/MedXpertQA/Gemma 3 12B IT/multiseed/seed_2026091900/results/routing_subset_analysis.json",
    },
    {
        "name": "MedXpertQA / Gemma 3 12B IT / seed 2026091909",
        "baseline": PACKAGE / "benchmarks/MedXpertQA/Gemma 3 12B IT/multiseed/seed_2026091909/traces/ma_rag_int_490.jsonl",
        "edr": PACKAGE / "benchmarks/MedXpertQA/Gemma 3 12B IT/multiseed/seed_2026091909/traces/ma_rag_int_plus_edr_490.jsonl",
        "output": PACKAGE / "benchmarks/MedXpertQA/Gemma 3 12B IT/multiseed/seed_2026091909/results/routing_subset_analysis.json",
    },
]


def read_jsonl(path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def summarize(rows):
    repairs = [row["id"] for row in rows if row["edr_correct"] and not row["baseline_correct"]]
    harms = [row["id"] for row in rows if row["baseline_correct"] and not row["edr_correct"]]
    baseline_correct = sum(row["baseline_correct"] for row in rows)
    edr_correct = sum(row["edr_correct"] for row in rows)
    count = len(rows)
    return {
        "questions": count,
        "baseline_correct": baseline_correct,
        "edr_correct": edr_correct,
        "baseline_accuracy": baseline_correct / count if count else None,
        "edr_accuracy": edr_correct / count if count else None,
        "gain_percentage_points": 100 * (edr_correct - baseline_correct) / count if count else None,
        "repairs": len(repairs),
        "harms": len(harms),
        "net_correct": len(repairs) - len(harms),
        "changed_outcomes": len(repairs) + len(harms),
        "repair_ids": repairs,
        "harm_ids": harms,
    }


def build(experiment, write=True):
    baseline = read_jsonl(experiment["baseline"])
    edr = read_jsonl(experiment["edr"])
    if set(baseline) != set(edr):
        raise ValueError(f"Unpaired trace IDs: {experiment['name']}")

    rows = []
    for question_id in sorted(baseline):
        base_row = baseline[question_id]
        edr_row = edr[question_id]
        active = any(
            (round_row.get("routing") or {}).get("independent", 0) > 0
            for round_row in edr_row.get("rounds", [])
        )
        first_predictions = []
        if edr_row.get("rounds"):
            first_predictions = (edr_row["rounds"][0].get("decision") or {}).get("predictions") or []
        rows.append({
            "id": question_id,
            "routing_active": active,
            "first_round_disagreement": len(set(first_predictions)) > 1 if first_predictions else None,
            "baseline_correct": bool(base_row.get("correct")),
            "edr_correct": bool(edr_row.get("correct")),
        })

    active_rows = [row for row in rows if row["routing_active"]]
    result = {
        "experiment": experiment["name"],
        "activation_rule": "At least one recorded EDR round has independent > 0.",
        "full_observed_run": summarize(rows),
        "routing_active_subset": summarize(active_rows),
    }
    if write:
        experiment["output"].parent.mkdir(parents=True, exist_ok=True)
        experiment["output"].write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    names = ["medxpertqa_qwen", "gemma_initial", "gemma_2026091900", "gemma_2026091909"]
    parser.add_argument("--experiment", choices=["all"]+names, default="all")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    selected = EXPERIMENTS if args.experiment == "all" else [EXPERIMENTS[names.index(args.experiment)]]
    for item in selected:
        if item["baseline"].exists() and item["edr"].exists():
            result = build(item, write=not args.check)
            if args.check:
                assert json.loads(item["output"].read_text(encoding="utf-8")) == result, item["output"]
            summary = result["full_observed_run"]
            active = result["routing_active_subset"]
            print(
                f"{item['name']}: full {summary['baseline_correct']} -> "
                f"{summary['edr_correct']} ({summary['gain_percentage_points']:.2f} pp); "
                f"active {active['baseline_correct']} -> {active['edr_correct']} "
                f"({active['gain_percentage_points']:.2f} pp)"
            )
