import copy
import random
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_medla as study


class NativeTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        sample = {"question": "Which is correct?", "options": dict(A="one", B="two", C="three", D="four")}
        self.state = {"id": 0, "sample": sample, "root": Path(self.temp.name), "base": "http://unused/v1",
                      "model": "dummy", "arm": "baseline", "phase": "debate", "round": 1,
                      "independent_names": set(), "candidate_labels": list("ABCD"), "calls": [],
                      "routes": [], "selection_rng": random.Random(42), "errors": [],
                      "diverged": False, "evidence": "Independent source evidence.", "lock": threading.Lock(),
                      "label_ids": dict(A=1, B=2, C=3, D=4)}
        self.token = study.CASE.set(self.state)
        self.args = SimpleNamespace(model="vllm", llm_name="dummy", url="http://unused/v1", temp=.7, debug=False)
        self.initial = {}
        for i, label in enumerate("ABC"):
            agent = study.MedicalAgent(args=self.args, instruction=f"you are [{i}]agent{i}.",
                                       agent_name=f"[{i}]agent{i}", logger=[])
            agent.ans, agent.reason, agent.logic_report = label, f"PRIOR_REASON_{i}", f"CRITIQUE_{i}"
            agent.ans_list, agent.reason_list = [label], [agent.reason]
            agent.question = [sample["question"], sample["options"]]
            self.initial[str(i)] = agent

    def tearDown(self):
        study.CASE.reset(self.token)
        self.temp.cleanup()

    def network(self, url, payload):
        if payload.get("logprobs"):
            raw = {"choices": [{"message": {"content": "A"}, "finish_reason": "length",
                    "logprobs": {"content": [{"top_logprobs": [
                        {"token": k, "logprob": v} for k, v in zip("ABCD", (-.3, -2, -3, -4))]}]}}]}
        else:
            raw = {"choices": [{"message": {"content": "<Answer>Answer: A</Answer>\nA is supported."},
                                "finish_reason": "stop"}]}
        return raw, []

    def execute(self, arm, policy):
        self.state.update(arm=arm, calls=[], routes=[], independent_names=set(), diverged=False,
                          selection_rng=random.Random(42))
        agents = copy.deepcopy(self.initial)
        with patch.object(study, "request", self.network), patch.object(study.edr, "route", return_value=policy):
            study.choose_route(agents)
            return study.NATIVE_TALK(agents, 0, retry_count=[0])

    def test_original_selector_and_zero_route_replay_exact_requests(self):
        baseline = self.execute("baseline", {"independent": 0})
        routed = self.execute("edr", {"independent": 0})
        self.assertEqual(list(baseline), list(routed))
        self.assertEqual([a.ans for a in baseline.values()], [a.ans for a in routed.values()])
        self.assertTrue(all(c["replayed"] for c in self.state["calls"]))
        self.assertFalse(self.state["diverged"])

    def test_independent_mask_keeps_evidence_and_drops_both_histories(self):
        self.execute("baseline", {"independent": 0})
        self.execute("edr", {"independent": 4})
        self.assertEqual(self.state["routes"][0]["independent_count"], 2)
        independent_calls = [c for c in self.state["calls"] if c["channel"] == "independent"]
        self.assertEqual(len(independent_calls), 2)
        for call in independent_calls:
            saved = study.read_receipt(self.state["root"] / "edr/receipts" / (call["tag"] + ".json.gz"))
            text = str(saved["request"]["messages"])
            self.assertIn(self.state["evidence"], text)
            self.assertNotIn("PRIOR_REASON_", text)
            self.assertNotIn("CRITIQUE_", text)
            self.assertNotIn("Your opinion is:", text)
            self.assertNotIn("The opinion of", text)

    def test_ambiguous_native_tags_rejected(self):
        with self.assertRaises(ValueError):
            study.strict_parse("<Answer>Answer: A</Answer><Answer>Answer: B</Answer>")
        self.assertEqual(study.strict_parse("<Eliminate>Answer: C</Eliminate>", "Eliminate"), "C")
        self.assertEqual(study.strict_parse("<Answer>A</Answer>"), "A")
        self.assertEqual(study.strict_parse("<Eliminate>C</Eliminate>", "Eliminate"), "C")
        with self.assertRaises(ValueError):
            study.strict_parse("<Answer>A/B</Answer><Answer>Answer: C</Answer>")

    def test_technical_retry_refreshes_truncated_receipt_but_keeps_valid_cache(self):
        agent = copy.deepcopy(self.initial["0"])
        self.state.update(phase="initial", round=0, arm="shared")
        message = "Question: Which is correct?"
        with patch.object(study, "request", self.network):
            agent.chat_without_history(message)
        tag = self.state["calls"][0]["tag"]
        path = self.state["root"] / "shared/receipts" / (tag + ".json.gz")
        receipt = study.read_receipt(path)
        receipt["response"]["choices"][0]["finish_reason"] = "length"
        study.save_receipt(path, receipt)
        agent._call_seq = 0
        self.state.update(technical_retry=True, calls=[])
        with patch.object(study, "request", side_effect=self.network) as request:
            agent.chat_without_history(message)
            self.assertEqual(request.call_count, 1)
        archived = path.with_name(tag + ".failed_technical_attempt_1.json.gz")
        self.assertEqual(study.read_receipt(archived)["response"]["choices"][0]["finish_reason"], "length")
        agent._call_seq = 0
        with patch.object(study, "request", side_effect=AssertionError("Valid cache must not resample")) as request:
            agent.chat_without_history(message)
            self.assertEqual(request.call_count, 0)


class FinalScoringTests(unittest.TestCase):
    def test_one_arm_failure_preserves_other_arm_correct_answer(self):
        rows = [{"id": 0, "gold": "A", "status": "technical_failure",
                 "baseline": {"prediction": "A", "correct": True}},
                {"id": 1, "gold": "B", "status": "completed",
                 "baseline": {"prediction": "A", "correct": False},
                 "edr": {"prediction": "B", "correct": True}}]
        scores = study.final_scores(rows, 2)
        self.assertEqual(scores["baseline_correct"], 1)
        self.assertEqual(scores["edr_correct"], 1)
        self.assertEqual(scores["matched_completed_correct"], {"baseline": 0, "edr": 1})
        self.assertEqual(scores["technical_missing_prediction_ids"], {"baseline": [], "edr": [0]})
        self.assertEqual(scores["full_delta_pp"], 0)

    def test_shared_failure_counts_wrong_for_both(self):
        scores = study.final_scores([{"id": 0, "gold": "C", "status": "technical_failure"}], 1)
        self.assertEqual(scores["baseline_accuracy"], 0)
        self.assertEqual(scores["edr_accuracy"], 0)

    def test_incomplete_coverage_rejected(self):
        with self.assertRaises(ValueError):
            study.final_scores([], 483)


if __name__ == "__main__":
    unittest.main()
