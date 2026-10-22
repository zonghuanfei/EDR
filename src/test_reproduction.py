"""Offline smoke tests for request replay and actual framework hooks."""
import copy
import gzip
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from madmm_plugin import Plugin as MADPlugin
from query_output_guard import reuse_reason


class Harness:
    def __init__(self):
        self.calls = []

    reuse_reason = staticmethod(reuse_reason)

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def save_receipt(path, value):
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(value, stream)

    def api_call(self, path, url, payload, source_path=None):
        self.calls.append((path.name, copy.deepcopy(payload), source_path))
        if path.exists() or source_path:
            with gzip.open(path if path.exists() else source_path, "rt", encoding="utf-8") as stream:
                saved = json.load(stream)
            if source_path and not path.exists():
                self.save_receipt(path, saved)
            return saved["response"]
        raw = {"choices": [{"index": i, "message": {"content": "<answer>A</answer>"},
                            "finish_reason": "stop"} for i in range(payload["n"])], "usage": {}}
        self.save_receipt(path, {"request": payload, "response": raw})
        return raw






class MaskHookTests(unittest.TestCase):
    def test_render_masks_history_without_changing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            scores = list(range(1, 9))
            choices = [{"message": {"content": f"ANSWER_{i}"}} for i in range(8)]
            order = list(reversed(range(8)))
            uncertainties = [.1] * 8
            previous = "\n\n".join(f"{i}. The assistant's previous answer is (Entropy 0.10):\nANSWER_{j}"
                                     for i, j in enumerate(order, 1))
            Harness.write_json(folder / "round_1.json", {"entropies": uncertainties})
            Harness.write_json(folder / "round_1_transition.json", {"history_order": order})
            base = SimpleNamespace(api_call=lambda *a, **k: None, parse_queries=lambda x: None,
                                   complete_choice=lambda c: c["message"]["content"], write_json=Harness.write_json)
            plugin = MADPlugin(base, {})
            context = {"last_round": 1, "folder": folder, "last_choices": choices,
                       "last_scores": [{"score": score/10} for score in scores]}
            original = SimpleNamespace(render=lambda **v: v["documents"] + "\n" + v["answers"])
            text = plugin.render_masked(original, context,
                {"question": "q", "options": "A. a", "documents": "EVIDENCE", "answers": previous})
            self.assertIn("EVIDENCE", text)
            self.assertNotIn("ANSWER_0", text)
            self.assertIn("ANSWER_7", text)
            info = json.loads((folder / "round_2_madmm.json").read_text())
            self.assertEqual(info["retained_count"], 4)


class EntryPointTests(unittest.TestCase):
    def test_all_conditions_run_against_mocked_transport(self):
        import run_marag as runner

        class SolverHarness(Harness):
            def api_call(self, path, url, payload, source_path=None):
                raw = super().api_call(path, url, payload, source_path)
                if source_path is None and not raw["choices"][0].get("logprobs"):
                    for choice in raw["choices"]:
                        choice["logprobs"] = {"content": [{"top_logprobs": [
                            {"token": "A", "logprob": -.1}, {"token": "B", "logprob": -2}]}]}
                    self.save_receipt(path, {"request": payload, "response": raw})
                return raw

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream = root / "upstream"
            upstream.mkdir()
            dataset_path = upstream / "dataset.json"
            items = [{"id": i, "question": "q", "option": {"A": "a", "B": "b"},
                      "answer": "AB"[i]} for i in range(2)]
            dataset_path.write_text(json.dumps(items), encoding="utf-8")
            (upstream / "ma_rag_entropy.py").write_text("# mocked prompt source\n", encoding="utf-8")
            prompts = {"system_prompt": "system", "user_prompt": SimpleNamespace(render=lambda **k: "question"),
                       "user_prompt_round": SimpleNamespace(render=lambda **k: "round")}
            outputs = {condition: root / condition for condition in
                       ("baseline", "edr", "gated_random", "gated_fixed4", "madmm_objective")}
            for condition, output in outputs.items():
                argv = ["run_marag.py", "--root", str(upstream), "--condition", condition,
                        "--model", "dummy", "--output", str(output), "--workers", "1"]
                if condition in ("edr", "gated_random", "gated_fixed4"):
                    argv.extend(["--tokenizer", str(root)])
                if condition in ("edr", "gated_random", "gated_fixed4", "madmm_objective"):
                    argv.extend(["--source-baseline", str(outputs["baseline"])])
                if condition == "gated_random":
                    argv.extend(["--allocation-seed", "42"])
                transport = SolverHarness()
                with patch("sys.argv", argv), patch.object(runner, "load_dataset", return_value=(items, dataset_path, "AB")), \
                     patch.object(runner.harness, "official_prompts", return_value=prompts), \
                     patch.object(runner.harness, "api_call", transport.api_call), \
                     patch.object(runner.harness, "parse_queries", runner.harness.parse_queries), \
                     patch.object(runner, "label_token_ids", return_value={"A": 1, "B": 2}), \
                     patch.object(runner, "check_endpoint") as preflight, redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(), 0)
                    self.assertEqual(preflight.call_count, int(condition in ("edr", "gated_random", "gated_fixed4")))
                score = json.loads((output / "scores.json").read_text())
                self.assertEqual((score["total"], score["correct"]), (2, 1))
                self.assertEqual([row["id"] for row in score["rows"]], [0, 1])


if __name__ == "__main__":
    unittest.main()
