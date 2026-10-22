import math
import random
import unittest

from edr import INDEPENDENT_TEMPLATE, distribution, probe_payload, route


def response(probs):
    return {
        "choices": [{
            "index": 0,
            "finish_reason": "length",
            "message": {"content": max(probs, key=probs.get)},
            "logprobs": {"content": [{
                "top_logprobs": [
                    {"token": key, "logprob": math.log(value)}
                    for key, value in probs.items()
                ]
            }]},
        }]
    }


class EDRCoreTests(unittest.TestCase):
    def test_routing_example(self):
        result = route(
            dict(A=.9, B=.05, C=.05),
            dict(A=.05, B=.9, C=.05),
            dict(A=.1, B=.85, C=.05),
        )
        self.assertEqual(result["independent"], 5)
        self.assertEqual(result["discussion"], 3)

    def test_same_top_and_ties_do_not_route(self):
        self.assertEqual(route(dict(A=.6, B=.4), dict(A=.9, B=.1), dict(A=.8, B=.2))["independent"], 0)
        self.assertEqual(route(dict(A=.5, B=.5), dict(A=.1, B=.9), dict(A=.1, B=.9))["independent"], 0)

    def test_route_bounds(self):
        rng = random.Random(1)
        for count in (2, 3, 10):
            for _ in range(50):
                distributions = []
                for _ in range(3):
                    values = [rng.random() for _ in range(count)]
                    distributions.append({str(i): value / sum(values) for i, value in enumerate(values)})
                result = route(*distributions)
                self.assertTrue(0 <= result["independent"] <= 6)
                self.assertEqual(result["independent"] + result["discussion"], 8)

    def test_one_token_distribution(self):
        result = distribution(response(dict(A=.1, B=.9)), ["A", "B"])
        self.assertAlmostEqual(result["B"], .9)
        with self.assertRaises(ValueError):
            distribution(response(dict(A=1.0)), ["A", "B"])

    def test_view_visibility(self):
        inputs = dict(question="Q", options="OPTIONS", documents="DOC", answers="PEERS")
        for view in ("E", "H", "EH"):
            payload = probe_payload("model", inputs, view, dict(A=32, B=33))
            text = payload["messages"][1]["content"]
            self.assertEqual("DOC" in text, view != "H")
            self.assertEqual("PEERS" in text, view != "E")
            self.assertEqual(payload["max_completion_tokens"], 1)

    def test_independent_channel_hides_history(self):
        text = INDEPENDENT_TEMPLATE.render(question="Q", options="OPTIONS", documents="DOC", answers="PEERS")
        self.assertIn("DOC", text)
        self.assertNotIn("PEERS", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
