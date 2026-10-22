"""Recompute post-initial candidate discovery from the released trajectories."""

import argparse
import json
from collections import Counter
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
GEMMA = Path("benchmarks/MedXpertQA/Gemma 3 12B IT")
EXPERIMENTS = [
    ("MedXpertQA / Qwen", Path("benchmarks/MedXpertQA/qwen3.8 27b"), 490, 10),
    ("MedXpertQA / Gemma / initial", GEMMA, 490, 10),
    ("MedXpertQA / Gemma / seed_2026091900", GEMMA / "multiseed/seed_2026091900", 490, 10),
    ("MedXpertQA / Gemma / seed_2026091909", GEMMA / "multiseed/seed_2026091909", 490, 10),
]


def read_rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate question IDs")
    return result


def predictions(round_record):
    return (round_record.get("decision") or {}).get("predictions") or []


def active(trace):
    return any((r.get("routing") or {}).get("independent", 0) > 0
               for r in trace.get("rounds", []))


def outcome(trace, exclude_failed_terminal=False):
    gold = trace["gold"]
    rounds = trace.get("rounds", [])
    first = None
    channels = []
    activated = False
    for r in rounds:
        activated |= (r.get("routing") or {}).get("independent", 0) > 0
        ps = predictions(r)
        if gold in ps and first is None:
            first = r["round"]
            answers = {a["index"]: a for a in r.get("answers", [])}
            channels = sorted({answers.get(i, {}).get("channel", "unspecified")
                               for i, p in enumerate(ps) if p == gold})
            active_by_discovery = activated
    return {
        "first_correct_candidate_round": first,
        "first_correct_candidate_channels": channels,
        "independent_active_by_discovery": active_by_discovery if first is not None else None,
        "historical_correct_candidate": first is not None,
        "final_pool_has_correct_candidate": gold in predictions(rounds[-1]) if rounds and not (exclude_failed_terminal and trace.get("status") == "technical_failure") else False,
        "final_correct": trace.get("final_option") == gold,
    }


def summarize(rows, cohort, method):
    selected = [r for r in rows if r[cohort]]
    discovered = [r for r in selected if r[method]["first_correct_candidate_round"] is not None
                  and r[method]["first_correct_candidate_round"] > 1]
    n = len(selected)
    channels = Counter("+".join(r[method]["first_correct_candidate_channels"])
                       for r in discovered)
    final = [r["id"] for r in discovered if r[method]["final_correct"]]
    return {
        "questions": n,
        "discoveries": len(discovered),
        "discovery_rate": len(discovered) / n if n else None,
        "discovery_ids": [r["id"] for r in discovered],
        "first_discovery_channels": dict(sorted(channels.items())),
        "discovered_with_independent_already_active": sum(
            r[method]["independent_active_by_discovery"] is True for r in discovered),
        "final_correct_after_discovery": len(final),
        "final_correct_ids": final,
        "final_conversion_rate": len(final) / len(discovered) if discovered else None,
        "cumulative_discoveries": {
            str(t): sum(r[method]["first_correct_candidate_round"] <= t for r in discovered)
            for t in range(1, 9)
        },
    }


def analyze(name, base, edr, options, random=None):
    if set(base) != set(edr) or (random is not None and set(random) != set(base)):
        raise ValueError("Unmatched question IDs")
    methods = {"baseline": base, "edr": edr}
    if random is not None:
        methods["gated_random_s42"] = random
    rows = []
    for qid in sorted(base):
        b, e = base[qid], edr[qid]
        bp, ep = predictions(b["rounds"][0]), predictions(e["rounds"][0])
        valid = len(bp) == len(ep) == 8 and all(x in options for x in bp + ep)
        shared = valid and bp == ep
        absent = shared and b["gold"] not in bp
        if b["gold"] != e["gold"]:
            raise ValueError("Gold mismatch")
        row = {
            "id": qid, "gold": b["gold"], "valid_matched_initial_choices": shared,
            "initial_correct_candidate": b["gold"] in bp if shared else None,
            "eligible_no_initial_gold": absent,
            "eligible_disputed_no_initial_gold": absent and len(set(bp)) > 1,
            "eligible_active_no_initial_gold": absent and active(e),
            "edr_routing_active": active(e),
        }
        for method, records in methods.items():
            if records[qid]["gold"] != b["gold"]:
                raise ValueError("Gold mismatch")
            row[method] = outcome(records[qid], exclude_failed_terminal=random is not None)
        rows.append(row)
    cohorts = ["eligible_disputed_no_initial_gold", "eligible_no_initial_gold",
               "eligible_active_no_initial_gold"]
    return {
        "experiment": name,
        "primary_cohort": cohorts[0],
        "definition": "Eight valid matched first-round choices, disagreement, and no gold choice; discovery is the first later round containing the gold choice.",
        "questions": len(rows),
        "excluded_initial_ids": [r["id"] for r in rows if not r["valid_matched_initial_choices"]],
        "cohorts": {c: {m: summarize(rows, c, m) for m in methods} for c in cohorts},
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ["medxpertqa_qwen", "gemma_initial", "gemma_2026091900", "gemma_2026091909"]
    parser.add_argument("--experiment", choices=["all"]+names, default="all")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    selected = EXPERIMENTS if args.experiment == "all" else [EXPERIMENTS[names.index(args.experiment)]]
    for name, folder, count, width in selected:
        folder = PACKAGE / folder
        base = read_rows(folder / f"traces/ma_rag_int_{count}.jsonl")
        edr = read_rows(folder / f"traces/ma_rag_int_plus_edr_{count}.jsonl")
        random_path = folder / f"traces/edr_gated_random_s42_{count}.jsonl"
        random = read_rows(random_path) if random_path.exists() else None
        if len(base) != count:
            raise ValueError("Incomplete benchmark")
        result = analyze(name, base, edr, set("ABCDEFGHIJ"[:width]), random)
        target = folder / "results/candidate_discovery.json"
        if args.check:
            assert json.loads(target.read_text(encoding="utf-8")) == result, target
        else:
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        primary = result["cohorts"][result["primary_cohort"]]
        print(name, {k: (v["discoveries"], v["questions"]) for k, v in primary.items()})


if __name__ == "__main__":
    main()
