"""Focused tests for the cross-framework metric definitions."""
import unittest

import analyze_systematic_metrics as metrics

class MetricTests(unittest.TestCase):
    def round(self, index, predictions, independent=0):
        return {"round": index, "decision": {"predictions": predictions},
                "routing": {"independent": independent},
                "answers": [{"index": i, "channel": "independent" if i < independent else "discussion"}
                            for i in range(len(predictions))]}

    def test_discovery_and_channel(self):
        initial = ["A"]*4 + ["B"]*4
        trace = {"final_option": "C", "correct": True,
                 "rounds": [self.round(1, initial), self.round(2, ["C"]+["A"]*7, 1)]}
        result = metrics.output_metrics(trace, initial, "C", False, set("ABCD"))
        self.assertTrue(result["active"])
        self.assertEqual(result["first_discovery_round"], 2)
        self.assertEqual(result["first_discovery_channels"], ["independent"])
        self.assertFalse(result["early_wrong_consensus"])

    def test_technical_final_pool_is_not_available(self):
        initial = ["A"]*4 + ["B"]*4
        trace = {"final_option": None, "correct": False,
                 "rounds": [self.round(1, initial), {"round":2, "answers": [{"index":0}]}]}
        result = metrics.output_metrics(trace, initial, "B", False, set("ABCD"))
        self.assertFalse(result["terminal_pool_available"])
        self.assertFalse(result["early_wrong_consensus"])

    def test_wrong_consensus_before_cap(self):
        initial = ["A"]*4 + ["B"]*4
        trace = {"final_option":"A", "correct":False,
                 "rounds":[self.round(1,initial), self.round(2,["A"]*8)]}
        self.assertTrue(metrics.output_metrics(trace, initial, "B", False, set("ABCD"))["early_wrong_consensus"])
        trace["rounds"][-1]["round"] = 8
        self.assertFalse(metrics.output_metrics(trace, initial, "B", False, set("ABCD"))["early_wrong_consensus"])

    def test_query_failure_does_not_make_initial_pool_a_terminal_pool(self):
        initial = ["A"]*4 + ["B"]*4
        trace = {"final_option": None, "correct": False, "status": "technical_failure",
                 "rounds": [self.round(1, initial)]}
        result = metrics.output_metrics(trace, initial, "B", False, set("ABCD"), exclude_failed_terminal=True)
        self.assertFalse(result["terminal_pool_available"])
        self.assertFalse(result["terminal_pool_gold"])
        self.assertFalse(result["correct"])

    def test_missing_channels_stay_unspecified(self):
        initial = ["A"]*4 + ["B"]*4
        later = self.round(2, ["C"]+["A"]*7, 1)
        for answer in later["answers"]:
            answer.pop("channel")
        trace = {"final_option": "C", "correct": True,
                 "rounds": [self.round(1, initial), later]}
        result = metrics.output_metrics(trace, initial, "C", False, set("ABCD"))
        self.assertEqual(result["first_discovery_channels"], ["unspecified"])
        self.assertEqual(result["independent_only_correct_rounds"], 0)

    def test_medla_initial_consensus(self):
        trace = {"final_option":"A", "correct":True, "rounds":[],
                 "final_decision":{"type":"initial_consensus", "option":"A"}}
        result = metrics.output_metrics(trace,["A"]*17,"A",True,set("ABCD"))
        self.assertTrue(result["terminal_pool_gold"])
        self.assertEqual(result["solver_stages"],1)
        self.assertFalse(result["discovered_after_initial"])

    def test_medla_incomplete_terminal_does_not_reuse_initial(self):
        trace = {"final_option": None, "correct": False,
                 "rounds": [{"round": 1, "answers": [{"channel": "independent"}]}]}
        result = metrics.output_metrics(trace, ["A", "B"], "B", True, set("ABCD"))
        self.assertFalse(result["terminal_pool_available"])
        self.assertFalse(result["terminal_pool_gold"])

    def test_pair_counts_and_repair_denominator(self):
        rows = [{"id":1,"baseline":{"correct":False},"edr":{"correct":True}},
                {"id":2,"baseline":{"correct":False},"edr":{"correct":False}},
                {"id":3,"baseline":{"correct":True},"edr":{"correct":False}}]
        group = metrics.group(rows)
        self.assertEqual(group["repair_rate"], .5)
        self.assertEqual(group["gain_pp"], 0)
        self.assertEqual(group["harm_ids"],[3])

    def test_mcnemar_and_reproducible_resampling(self):
        self.assertEqual(metrics.mcnemar(6,1), .125)
        self.assertEqual(metrics.bootstrap([1,0,0,-1]), metrics.bootstrap([1,0,0,-1]))

if __name__ == "__main__":
    unittest.main()
