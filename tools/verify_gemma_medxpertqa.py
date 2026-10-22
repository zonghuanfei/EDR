#!/usr/bin/env python3
"""Verify the compact Gemma MedXpertQA result and trajectory export."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "MedXpertQA" / "Gemma 3 12B IT"
CONDITIONS = ("ma_rag_int", "ma_rag_int_plus_edr")


def main():
    result = json.loads((ROOT / "results" / "final_scores_490.json").read_text(encoding="utf-8"))
    rows = result["rows"]
    assert len(rows) == 490 and [row["id"] for row in rows] == list(range(490))
    by_id = {row["id"]: row for row in rows}
    traces = {}
    for condition in CONDITIONS:
        path = ROOT / "traces" / f"{condition}_490.jsonl"
        with path.open(encoding="utf-8") as stream:
            items = [json.loads(line) for line in stream if line.strip()]
        assert len(items) == 490 and [item["id"] for item in items] == list(range(490))
        failures = set(result["conditions"][condition]["technical_failure_ids"])
        missing_option = set(result["conditions"][condition]["completed_without_final_option_ids"])
        assert not failures & missing_option
        assert result["conditions"][condition]["completed"] == 490 - len(failures)
        correct = 0
        for item in items:
            row = by_id[item["id"]]
            assert item["condition"] == condition and item["gold"] == row["gold"]
            assert item["final_option"] == row[f"{condition}_prediction"]
            assert item["correct"] == row[f"{condition}_correct"]
            assert item["correct"] == (item["final_option"] == item["gold"])
            assert item["status"] == ("technical_failure" if item["id"] in failures else "complete")
            if item["id"] in missing_option:
                assert item["final_option"] is None and item["status"] == "complete"
            correct += item["correct"]
            for round_item in item["rounds"]:
                answers = round_item.get("answers", [])
                if item["status"] == "complete":
                    assert len(answers) == 8
                for answer in answers:
                    assert set(answer) <= {"index", "text", "finish_reason", "channel"}
                if "routing" in round_item:
                    routing = round_item["routing"]
                    assert routing["independent"] + routing["discussion"] == 8
                for document in round_item.get("retrieval", {}).get("documents", []):
                    assert set(document) == {"id", "title", "score"}
        assert correct == result["conditions"][condition]["correct"]
        traces[condition] = items

    repairs = [qid for qid in range(490) if not traces[CONDITIONS[0]][qid]["correct"]
               and traces[CONDITIONS[1]][qid]["correct"]]
    harms = [qid for qid in range(490) if traces[CONDITIONS[0]][qid]["correct"]
             and not traces[CONDITIONS[1]][qid]["correct"]]
    change = result["edr_vs_ma_rag_int"]
    assert repairs == change["repair_ids"] and harms == change["harm_ids"]
    assert len(repairs) - len(harms) == change["net_correct"] == 19
    print(json.dumps({"status": "PASS", "questions": 490,
                      "baseline_correct": result["conditions"][CONDITIONS[0]]["correct"],
                      "edr_correct": result["conditions"][CONDITIONS[1]]["correct"],
                      "repairs": len(repairs), "harms": len(harms)}))


if __name__ == "__main__":
    main()
