"""Tests for initial-state selection and paired final-outcome scoring."""

import json
from pathlib import Path
import tempfile
import unittest

import summarize_initial_disagreement as metrics


class InitialDisagreementTests(unittest.TestCase):
    def test_initial_consensus_and_invalid_materials_are_distinct(self):
        labels = set("ABCD")
        self.assertEqual(metrics.initial_group(["A"] * 8, 8, labels), "initial_unanimous")
        self.assertEqual(metrics.initial_group(["A"] * 7 + ["B"], 8, labels), "initial_disagreement")
        self.assertEqual(metrics.initial_group(["A", "B"], 8, labels), "invalid_initial")
        self.assertEqual(metrics.initial_group(["A"] * 7 + [None], 8, labels), "invalid_initial")
        self.assertEqual(metrics.initial_group(["A"] * 16 + ["B"], 17, labels, False), "invalid_initial")

    def test_selection_uses_initial_state_not_later_route_or_final_outcome(self):
        initial = {0: {"gold": "B", "initial": ["A"] * 8},
                   1: {"gold": "B", "initial": ["A"] * 7 + ["B"]}}
        methods = {
            "baseline": {i: {**r, "correct": False} for i, r in initial.items()},
            "edr": {i: {**r, "correct": True, "activated": True} for i, r in initial.items()},
        }
        groups = metrics.select_cohort(methods, initial, 8, set("ABCD"))
        self.assertEqual(groups["initial_disagreement"], [1])
        self.assertEqual(groups["initial_unanimous"], [0])
        methods["edr"][1].update(correct=False, activated=False)
        self.assertEqual(metrics.select_cohort(methods, initial, 8, set("ABCD")), groups)

    def test_different_initial_choices_raise_instead_of_changing_the_cohort(self):
        source = {0: {"gold": "A", "initial": ["A"] * 7 + ["B"]}}
        other = {0: {"gold": "A", "initial": ["B"] * 8}}
        with self.assertRaisesRegex(ValueError, "Initial option mismatch"):
            metrics.select_cohort({"baseline": source, "edr": other}, source, 8, set("ABCD"))

    def test_post_initial_failure_stays_in_denominator(self):
        failed = metrics.project_record({
            "id": 0, "gold": "B", "final_option": None, "correct": False,
            "status": "technical_failure",
            "rounds": [{"round": 1, "decision": {"predictions": ["A"] * 7 + ["B"]}}],
        })
        self.assertEqual(metrics.initial_group(failed["initial"], 8, set("ABCD")), "initial_disagreement")
        reference = {0: {"correct": True}, 1: {"correct": False}, 2: {"correct": False}, 3: {"correct": False}}
        method = {0: failed, 1: {"correct": True}, 2: {"correct": True}, 3: {"correct": False}}
        result = metrics.paired_summary(reference, method, [0, 1, 2, 3])
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["gain_pp"], 25)
        self.assertEqual(result["correction_ids"], [1, 2])
        self.assertEqual(result["regression_ids"], [0])
        self.assertAlmostEqual(result["correction_rate_percent"], 200 / 3)

    def test_duplicate_ids_and_inconsistent_stored_scores_raise(self):
        record = {"id": 0, "gold": "A", "final_option": "A", "correct": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            path.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate question ID"):
                metrics.load_records(path)
        with self.assertRaisesRegex(ValueError, "Incorrect stored score"):
            metrics.project_record({**record, "correct": False})

    def test_medla_reference_must_identify_the_shared_initial_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / "shared.jsonl"
            trace_files = {"baseline": root / "baseline.jsonl"}
            initial = {0: {"gold": "A", "initial": ["A"] * 16 + ["B"], "complete": True}}
            methods = {"baseline": {0: {"gold": "A", "shared_initial_ref": {"file": "shared.jsonl", "id": 0}}}}
            groups = metrics.select_cohort(methods, initial, 17, set("ABCD"), shared, trace_files)
            self.assertEqual(groups["initial_disagreement"], [0])
            methods["baseline"][0]["shared_initial_ref"]["id"] = 1
            with self.assertRaisesRegex(ValueError, "Missing shared initial reference"):
                metrics.select_cohort(methods, initial, 17, set("ABCD"), shared, trace_files)


if __name__ == "__main__":
    unittest.main()
