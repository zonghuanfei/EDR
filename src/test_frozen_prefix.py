"""Offline checks for frozen source failures and restart-stable query order."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import frozen_prefix as frozen
import run_marag_queryguard_w8 as harness
from technical_retry import TechnicalRetry


class FrozenPrefixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source/question_0"
        self.folder = self.root / "output/question_0"
        self.source.mkdir(parents=True)
        self.folder.mkdir(parents=True)
        self.http = patch.object(harness.requests, "post", side_effect=AssertionError("Unexpected HTTP"))
        self.post = self.http.start()
        self.addCleanup(self.http.stop)

    def saved(self, folder, name, request, response):
        path = folder / name
        harness.save_receipt(path, {"request": request, "response": response})
        return path

    def test_missing_source_never_generates(self):
        with self.assertRaisesRegex(RuntimeError, "missing"):
            harness.api_call(self.folder / "round_1_solver.json.gz", "unused", {"n": 8},
                             self.source / "round_1_solver.json.gz")
        self.post.assert_not_called()

    def test_incompatible_source_never_generates(self):
        source = self.saved(self.source, "round_1_solver.json.gz", {"n": 8}, {"choices": []})
        with self.assertRaisesRegex(RuntimeError, "different parameters"):
            harness.api_call(self.folder / source.name, "unused", {"n": 4}, source)
        self.post.assert_not_called()

    def test_existing_output_must_equal_source_response(self):
        request = {"n": 8}
        source = self.saved(self.source, "round_1_solver.json.gz", request, {"choices": ["source"]})
        local = self.saved(self.folder, source.name, request, {"choices": ["different sample"]})
        with self.assertRaisesRegex(RuntimeError, "Cached response differs"):
            harness.api_call(local, "unused", request, source)
        self.post.assert_not_called()

    def test_failed_frozen_query_is_retained_without_retry(self):
        request = {"n": 1}
        raw = {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}
        source = self.saved(self.source, "round_1_query.json.gz", request, raw)
        wrapper = TechnicalRetry(harness)
        result = wrapper(self.folder / source.name, "unused", request, source)
        self.assertEqual(result, raw)
        self.assertEqual(frozen.receipt(self.folder / source.name)["response"], raw)
        with self.assertRaisesRegex(RuntimeError, "Incomplete query"):
            harness.parse_queries(result["choices"][0])
        self.assertFalse(list(self.folder.glob("*retry*")))
        self.post.assert_not_called()


    def query_source(self, order):
        payload = {"n": 1}
        raw = {"choices": [{"finish_reason": "stop", "message": {"content": "[Query 1] a\n[Query 2] b"}}]}
        self.saved(self.source, "round_1_query.json.gz", payload, raw)
        harness.write_json(self.source / "round_1_transition.json", {"queries": order})
        return payload, raw

    def test_source_query_order_is_restored(self):
        payload, raw = self.query_source(["b", "a"])
        order = frozen.source_query_order(self.source, payload, raw, ["a", "b"], harness.reuse_reason)
        self.assertEqual(order, ["b", "a"])

    def test_missing_or_duplicate_source_query_order_is_rejected(self):
        payload, raw = self.query_source(["a", "a"])
        with self.assertRaisesRegex(RuntimeError, "query order"):
            frozen.source_query_order(self.source, payload, raw, ["a", "b"], harness.reuse_reason)
        (self.source / "round_1_transition.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            frozen.source_query_order(self.source, payload, raw, ["a", "b"], harness.reuse_reason)

    def transition(self):
        result = {"queries": ["b", "a"], "documents": [{"id": "b", "content": "B"}],
                  "history_order": list(range(8)), "query_usage": {"total_tokens": 1}}
        harness.write_json(self.source / "round_1_transition.json", result)
        return result

    def test_document_content_and_history_are_frozen(self):
        saved = self.transition()
        actual = frozen.initial_transition(self.source, saved["queries"], saved["documents"],
                                           [.5] * 8, saved["query_usage"])
        self.assertEqual(actual["history_order"], list(range(8)))
        changed = copy.deepcopy(saved["documents"])
        changed[0]["content"] = "same ID but different evidence"
        with self.assertRaisesRegex(RuntimeError, "content or order"):
            frozen.initial_transition(self.source, saved["queries"], changed, [.5] * 8,
                                      saved["query_usage"])
        with self.assertRaisesRegex(RuntimeError, "descending"):
            frozen.initial_transition(self.source, saved["queries"], saved["documents"],
                                      list(range(8)), saved["query_usage"])

    def test_query_journal_survives_changed_set_iteration(self):
        request, response = {"prompt": "q"}, {"output": "a,b"}
        first = frozen.local_query_order(self.folder, 2, ["b", "a"], request, response, harness.write_json)
        resumed = frozen.local_query_order(self.folder, 2, ["a", "b"], request, response, harness.write_json)
        self.assertEqual(first, resumed)
        self.assertEqual(resumed, ["b", "a"])
        with self.assertRaisesRegex(RuntimeError, "different request or response"):
            frozen.local_query_order(self.folder, 2, ["a", "b"], {"prompt": "changed"}, response,
                                     harness.write_json)

    def test_missing_journal_does_not_reinterpret_existing_retrieval(self):
        self.saved(self.folder, "round_2_retrieval_0.json.gz", {}, {})
        with self.assertRaisesRegex(RuntimeError, "no query-order journal"):
            frozen.local_query_order(self.folder, 2, ["a", "b"], {}, {}, harness.write_json)

    def test_done_fast_path_rejects_changed_initial_response(self):
        first = {"round": 1, "predictions": ["A"] * 8, "prediction": "A",
                 "entropies": [.5] * 8, "usage": {}}
        for folder in (self.source, self.folder):
            harness.write_json(folder / "round_1.json", first)
        request = {"n": 8}
        self.saved(self.source, "round_1_solver.json.gz", request, {"choices": ["source"]})
        self.saved(self.folder, "round_1_solver.json.gz", request, {"choices": ["different"]})
        harness.write_json(self.folder / "done.json", {"prediction": "A"})
        args = SimpleNamespace(query_order_root=self.source.parent)
        with self.assertRaisesRegex(RuntimeError, "different frozen initial responses"):
            harness.question_run({"id": 0}, args, {}, self.folder.parent)
        self.post.assert_not_called()

    @staticmethod
    def solver_raw(predictions):
        return {"choices": [{"index": i, "finish_reason": "stop",
                             "message": {"content": f"<answer>{p}</answer>"},
                             "logprobs": {"content": [{"top_logprobs": [
                                 {"token": "A", "logprob": -.2}, {"token": "B", "logprob": -1.8}]}]}}
                            for i, p in enumerate(predictions)], "usage": {"total_tokens": 8}}

    def test_full_question_uses_source_documents_and_tied_history_order(self):
        item = {"id": 0, "question": "question", "option": {"A": "a", "B": "b"}}
        seen = []
        prompts = {"system_prompt": "solver", "system_prompt_query": "query-system",
                   "user_prompt": SimpleNamespace(render=lambda **values: "first"),
                   "user_prompt_query": SimpleNamespace(render=lambda **values: "query"),
                   "user_prompt_round": SimpleNamespace(render=lambda **values: seen.append(values) or "later")}
        args = SimpleNamespace(query_order_root=self.source.parent, reuse_root=None,
                               model="dummy", max_tokens=32, base_url="unused", retriever_url="unused")
        initial = self.solver_raw(["A", "B"] * 4)
        solver_payload = {"model": "dummy", "messages": [{"role": "system", "content": "solver"},
                          {"role": "user", "content": "first"}], "temperature": .7,
                          "max_completion_tokens": 32, "n": 8, "logprobs": True, "top_logprobs": 20,
                          "chat_template_kwargs": {"enable_thinking": False}}
        self.saved(self.source, "round_1_solver.json.gz", solver_payload, initial)
        entropy = float(harness.np.mean(harness.token_entropies(initial["choices"][0]["logprobs"]["content"])))
        record = {"round": 1, "predictions": ["A", "B"] * 4, "prediction": None,
                  "entropies": [entropy] * 8, "usage": initial["usage"]}
        harness.write_json(self.source / "round_1.json", record)
        query_payload = {"model": "dummy", "messages": [{"role": "system", "content": "query-system"},
                         {"role": "user", "content": "query"}], "temperature": 0.,
                         "max_completion_tokens": 8192, "n": 1,
                         "chat_template_kwargs": {"enable_thinking": True}, "stop": harness.STOP_STRINGS}
        query_raw = {"choices": [{"finish_reason": "stop", "message": {"content": "[Query 1] alpha\n[Query 2] beta"}}],
                     "usage": {"total_tokens": 1}}
        self.saved(self.source, "round_1_query.json.gz", query_payload, query_raw)
        docs = []
        for i, query in enumerate(["beta", "alpha"]):
            doc = {"id": query, "title": query, "content": query.upper()}
            docs.append(doc)
            payload = {"query": query, "total_k": 32, "top_k": 2,
                       "combine_docs": False, "use_reranker": True}
            self.saved(self.source, f"round_1_retrieval_{i}.json.gz", payload,
                       {"documents": [doc], "scores": [1.]})
        transition = {"queries": ["beta", "alpha"], "documents": docs,
                      "history_order": list(range(8)), "query_usage": query_raw["usage"]}
        harness.write_json(self.source / "round_1_transition.json", transition)
        original = harness.api_call
        def transport(path, url, payload, source_path=None):
            source = self.source / path.name if path.name.startswith("round_1_") else None
            return original(path, url, payload, source_path=source)
        self.post.side_effect = None
        self.post.return_value = SimpleNamespace(raise_for_status=lambda: None,
                                                 json=lambda: self.solver_raw(["A"] * 8))
        with patch.object(harness, "api_call", transport):
            result = harness.question_run(item, args, prompts, self.folder.parent)
        self.assertEqual(result["prediction"], "A")
        self.assertEqual(result["rounds"], 2)
        self.post.assert_called_once()
        self.assertEqual(seen[0]["documents"], "BETA\nALPHA")
        self.assertIn("1. The assistant's previous answer is", seen[0]["answers"])
        self.assertIn("1. The assistant's previous answer is (Entropy", seen[0]["answers"])
        self.assertEqual(json.loads((self.folder / "round_1_transition.json").read_text()), transition)
        self.assertEqual(json.loads((self.folder / "round_1.json").read_text()), record)
        self.post.reset_mock()
        cached = harness.question_run(item, args, prompts, self.folder.parent)
        self.assertEqual(cached, result)
        self.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
