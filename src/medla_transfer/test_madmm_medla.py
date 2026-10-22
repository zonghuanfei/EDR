"""Offline checks for the MAD-M2 medical-history adapter."""
import copy
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_madmm_medla as adapter


class MaskTests(unittest.TestCase):
    def test_original_strict_median(self):
        for n, kept in [(17, 8), (4, 2), (3, 1), (2, 1)]:
            scores = [(i+1)/(n+1) for i in range(n)]
            mask, threshold = adapter.objective_mask(scores)
            self.assertEqual(sum(mask), kept)
            self.assertEqual(mask, [v > threshold for v in scores])
        self.assertEqual(adapter.objective_mask([.8]*17)[0], [False]*17)

    def test_top1_not_sampled_token(self):
        choice = {"logprobs": {"content": [
            {"logprob": -9, "top_logprobs": [{"logprob": -.1}, {"logprob": -4}]},
            {"logprob": -8, "top_logprobs": [{"logprob": -.3}]}]}}
        result = adapter.generation_score(choice)
        self.assertAlmostEqual(result["score"], math.exp(-.2))
        self.assertEqual(result["valid_token_count"], 2)

    def test_weighted_segments(self):
        parts = [{"score": math.exp(-.1), "valid_token_count": 1},
                 {"score": math.exp(-.3), "valid_token_count": 3}]
        self.assertAlmostEqual(adapter.combine_scores(parts), math.exp(-.25))
        with self.assertRaises(ValueError):
            adapter.combine_scores([])

    def test_teacher_forcing_positions(self):
        saved = {"request": {"prompt": [10, 11, 12, 13]},
                 "response": {"prompt_logprobs": [None, {},
                     {"12": {"logprob": -4}, "20": {"logprob": -.1}},
                     {"13": {"logprob": -.3}}]}}
        result = adapter.forced_score(saved, 2)
        self.assertAlmostEqual(result["score"], math.exp(-.2))
        self.assertEqual(result["valid_token_count"], 2)
        broken = copy.deepcopy(saved)
        broken["response"]["prompt_logprobs"][2].pop("12")
        with self.assertRaises(ValueError):
            adapter.forced_score(broken, 2)


class HistoryTests(unittest.TestCase):
    def state(self, visible):
        return {"candidate_labels": ["A", "C"], "visible_keys": set(visible),
                "snapshot": {"0": {"name": "own", "ans": "A", "reason": "OWN + LOGIC"},
                             "1": {"name": "peer", "ans": "C", "reason": "PEER + LOGIC"}}}

    def test_own_and_peer_mask_not_candidate_mask(self):
        agent = SimpleNamespace(agent_name="own", question=["QUESTION", {"A": "alpha", "C": "gamma"}],
                                reason="OWN")
        kept, own_visible = adapter.render_history(agent, self.state(["0"]))
        self.assertTrue(own_visible)
        self.assertIn("Your opinion is: A. Your reasoning is: OWN", kept)
        self.assertNotIn("The opinion of 1", kept)
        self.assertIn("A/C", kept)
        masked, own_visible = adapter.render_history(agent, self.state(["1"]))
        self.assertFalse(own_visible)
        self.assertNotIn("Your opinion is:", masked)
        self.assertNotIn("OWN", masked)
        self.assertIn("The opinion of 1 is C", masked)
        self.assertIn("A/C", masked)
        self.assertIn("QUESTION", masked)

    def test_all_ties_hide_histories_not_question(self):
        agent = SimpleNamespace(agent_name="own", question=["QUESTION", {"A": "alpha", "C": "gamma"}], reason="OWN")
        rendered, visible = adapter.render_history(agent, self.state([]))
        self.assertFalse(visible)
        self.assertNotIn("OWN", rendered)
        self.assertNotIn("PEER", rendered)
        self.assertIn("QUESTION", rendered)
        self.assertIn("A/C", rendered)

    def test_masked_real_request_preserves_evidence_and_seed(self):
        state = self.state(["1"])
        state.update(phase="debate", round=1, id=0, root=Path("/unused"), arm="madmm",
                     base="http://example/v1", evidence="EVIDENCE", errors=[], calls=[], latest_receipt={})
        agent = object.__new__(adapter.core.MedicalAgent)
        agent.args = SimpleNamespace(debug=False)
        agent.agent_name, agent.question, agent.reason, agent._call_seq = "own", "QUESTION", "OWN", 0
        response = {"choices": [{"finish_reason": "stop", "message": {"content": "<Answer>Answer: A</Answer>"},
                    "logprobs": {"content": [{"top_logprobs": [{"logprob": -.2}]}]}}]}
        token = adapter.core.CASE.set(state)
        try:
            with patch.object(adapter.core, "request", return_value=(response, [])) as request, \
                 patch.object(adapter.core, "save_receipt"):
                text = adapter.invoke(agent, "gemma-3-12b-it", [{"role": "user", "content": "Question: ORIGINAL"}], .7)
            payload = request.call_args.args[1]
            self.assertTrue(payload["logprobs"])
            self.assertEqual(payload["top_logprobs"], 1)
            self.assertEqual(payload["temperature"], .7)
            user = payload["messages"][-1]["content"]
            self.assertIn("EVIDENCE", user)
            self.assertNotIn("Your opinion is:", user)
            self.assertNotIn("OWN", user)
            self.assertIn("The opinion of 1", user)
            self.assertEqual(text, "<Answer>Answer: A</Answer>")
        finally:
            adapter.core.CASE.reset(token)


class ComparisonTests(unittest.TestCase):
    def test_same_completed_cohort(self):
        rows = [{"id": 0, "gold": "A", "status": "completed", "prediction": "A"},
                {"id": 1, "gold": "B", "status": "inherited_initial_failure"},
                {"id": 2, "gold": "C", "status": "technical_failure"}]
        refs = {0: {"baseline": {"prediction": "B"}, "edr": {"prediction": "A"}}}
        result = adapter.aggregate(rows, refs, 3, 0)
        self.assertEqual(result["completed_pairs"], 1)
        self.assertEqual(result["matched_completed_correct"], {"baseline": 0, "edr": 1, "madmm": 1})
        self.assertEqual(result["technical_failure_ids"], [2])
        self.assertEqual(result["inherited_initial_failure_ids"], [1])


if __name__ == "__main__":
    unittest.main()
