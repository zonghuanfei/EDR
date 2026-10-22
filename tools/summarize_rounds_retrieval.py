"""Summarize logical solver and retrieval counts from the retained trajectories."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def summarize(path):
    ids = set()
    solver_rounds = retrieval_rounds = retrieval_queries = partial_rounds = 0
    conditions = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["id"] in ids:
                raise ValueError(f"Duplicate question ID in {path.name}: {item['id']}")
            ids.add(item["id"])
            conditions.add(item["condition"])
            if item["condition"] == "direct_model":
                solver_rounds += int(bool(item.get("answer")))
                continue
            for round_item in item["rounds"]:
                if round_item.get("answers"):
                    solver_rounds += 1
                    partial_rounds += int(not round_item.get("decision"))
                retrieval = round_item.get("retrieval")
                if retrieval is not None:
                    queries = retrieval.get("queries", [])
                    if not queries:
                        raise ValueError(f"Empty retrieval queries: {path.name}, {item['id']}")
                    retrieval_rounds += 1
                    retrieval_queries += len(queries)
    if not ids or len(conditions) != 1:
        raise ValueError(f"Expected a nonempty, single-condition trace: {path}")
    totals = {
        "solver_rounds": solver_rounds,
        "retrieval_rounds": retrieval_rounds,
        "retrieval_queries": retrieval_queries,
    }
    return {
        "condition": next(iter(conditions)),
        "questions": len(ids),
        **totals,
        "partial_solver_rounds_included": partial_rounds,
        "mean_per_question": {name: value / len(ids) for name, value in totals.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    paths = sorted((args.root / "benchmarks").glob("**/traces/*.jsonl"))
    if not paths:
        parser.error("No benchmark trajectory files found")
    results = [{"trace": path.relative_to(args.root).as_posix(), **summarize(path)}
               for path in paths]
    print(json.dumps({
        "counting_unit": "logical stages and queries in retained trajectories; not transport attempts",
        "shared_prefix_policy": "count reused stages once in each condition",
        "partial_round_policy": "include a retained solver round with answers but no valid decision",
        "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
