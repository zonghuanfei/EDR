"""Offline checks for routed generation, restart safety and paired initial calls."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from edr_random_allocation import AllocationIntegrityError, Plugin, random_metadata
from test_edr_core import response


class Harness:
    def __init__(self, fail_probe=False):
        self.calls = []
        self.fail_probe = fail_probe

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def save_receipt(path, value):
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(value, stream)

    def api_call(self, path, url, payload, source_path=None):
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                saved = json.load(stream)
            if saved["request"] != payload:
                raise RuntimeError("Cached request differs")
            return saved["response"]
        if source_path is not None:
            with gzip.open(source_path, "rt", encoding="utf-8") as stream:
                saved = json.load(stream)
            self.save_receipt(path, saved)
            return saved["response"]
        self.calls.append((path.name, copy.deepcopy(payload)))
        if "_probe_" in path.name:
            if self.fail_probe:
                raise RuntimeError("Unavailable probe")
            view = path.name.removesuffix(".json.gz").rsplit("_", 1)[1]
            probs = {"E": {"A": .9, "B": .1}, "H": {"A": .1, "B": .9},
                     "EH": {"A": .2, "B": .8}}[view]
            raw = response(probs)
        else:
            raw = {"choices": [{"index": i, "finish_reason": "stop",
                                "message": {"content": "<answer>A</answer>"}}
                               for i in range(payload["n"])], "usage": {}}
        self.save_receipt(path, {"request": payload, "response": raw})
        return raw


class RandomAllocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.folder = self.root / "question_1"
        self.folder.mkdir()
        self.path = self.folder / "round_2_solver.json.gz"
        self.inputs = {"question": "Q", "options": "A or B", "documents": "EVIDENCE",
                       "answers": "SHARED_HISTORY"}
        self.payload = {"model": "test-model", "n": 8, "temperature": .7,
                        "messages": [{"role": "system", "content": "Solve"},
                                     {"role": "user", "content": "EVIDENCE SHARED_HISTORY"}]}

    def tearDown(self):
        assert self.folder.resolve().is_relative_to(self.root)
        self.temp.cleanup()

    def plugin(self, harness, seed=42):
        plugin = Plugin(harness, {"A": 1, "B": 2}, seed)
        plugin.context.set({"id": 1, "labels": ["A", "B"], "inputs": self.inputs})
        return plugin

    def info(self):
        return json.loads((self.folder / "round_2_edr.json").read_text(encoding="utf-8"))

    def test_budget_visibility_and_restart(self):
        harness = Harness()
        plugin = self.plugin(harness)
        raw = plugin.routed_call(self.path, "mock", self.payload, 2)
        info = self.info()
        self.assertTrue(info["trigger"])
        self.assertEqual(info["independent"] + info["discussion"], 8)
        self.assertEqual([c["edr_channel"] for c in raw["choices"]],
                         ["independent"] * info["independent"] + ["discussion"] * info["discussion"])
        calls = dict(harness.calls)
        independent = calls["round_2_solver_independent.json.gz"]
        discussion = calls["round_2_solver_discussion.json.gz"]
        self.assertIn("EVIDENCE", independent["messages"][1]["content"])
        self.assertNotIn("SHARED_HISTORY", independent["messages"][1]["content"])
        self.assertEqual(discussion["messages"], self.payload["messages"])
        self.assertTrue(all("seed" not in request for _, request in harness.calls))
        before = len(harness.calls)
        plugin.validate_question(self.folder, 1)
        self.assertEqual(plugin.routed_call(self.path, "mock", self.payload, 2), raw)
        self.assertEqual(len(harness.calls), before)

    def test_changed_seed_rejects_cached_run(self):
        harness = Harness()
        self.plugin(harness).routed_call(self.path, "mock", self.payload, 2)
        before = len(harness.calls)
        with self.assertRaises(AllocationIntegrityError):
            self.plugin(harness, 43).routed_call(self.path, "mock", self.payload, 2)
        self.assertEqual(len(harness.calls), before)

    def test_changed_branch_request_rejected_before_generation(self):
        harness = Harness()
        plugin = self.plugin(harness)
        plugin.routed_call(self.path, "mock", self.payload, 2)
        path = self.folder / "round_2_solver_independent.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            saved = json.load(stream)
        saved["request"]["temperature"] = .1
        harness.save_receipt(path, saved)
        before = len(harness.calls)
        with self.assertRaises(AllocationIntegrityError):
            plugin.routed_call(self.path, "mock", self.payload, 2)
        self.assertEqual(len(harness.calls), before)

    def test_probe_failure_uses_full_discussion_budget(self):
        harness = Harness(fail_probe=True)
        raw = self.plugin(harness).routed_call(self.path, "mock", self.payload, 2)
        info = self.info()
        self.assertEqual((info["independent"], info["discussion"]), (0, 8))
        self.assertTrue(info["fallback"])
        self.assertFalse(info["trigger"])
        self.assertEqual(info["rng"]["draw_count"], 0)
        self.assertEqual(len(raw["choices"]), 8)
        self.assertFalse(any("_solver_independent" in name for name, _ in harness.calls))

    def test_paired_initial_call_uses_supplied_source(self):
        harness = Harness()
        plugin = self.plugin(harness)
        source = self.root / "baseline"
        source.mkdir()
        path = self.folder / "round_1_solver.json.gz"
        expected = {"choices": [{"message": {"content": "frozen initial answer"}}]}
        harness.save_receipt(source / path.name, {"request": self.payload, "response": expected})
        plugin.source_map = {self.folder.name: source}
        self.assertEqual(plugin.api_call(path, "mock", self.payload), expected)
        self.assertEqual(harness.calls, [])

    def test_published_allocations_match_executable_draws(self):
        trace = Path(__file__).resolve().parents[1] / "benchmarks" / "MedXpertQA" / "qwen3.8 27b" / "traces" / "edr_gated_random_s42_490.jsonl"
        checked = triggered = 0
        with trace.open(encoding="utf-8-sig") as stream:
            for line in stream:
                row = json.loads(line)
                for rd in row["rounds"]:
                    info = rd.get("routing")
                    if info is None:
                        continue
                    draw = random_metadata(row["allocation_seed"], "MedXpertQA", row["id"], rd["round"], info["trigger"])
                    self.assertEqual(info["independent"], draw["selected_independent"] or 0,
                                     (row["id"], rd["round"]))
                    self.assertEqual(info["discussion"], 8 - info["independent"])
                    checked += 1
                    triggered += info["trigger"]
        self.assertGreater(checked, 0)
        self.assertGreater(triggered, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
