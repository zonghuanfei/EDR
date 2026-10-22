# Allocation ablation on MedXpertQA

All policies use Qwen3.8-27B-NVFP4, eight responses per round and all 490
questions. Gated fixed 4+4 and gated random allocation use EDR's positive-count
gate on their own trajectories. Fixed allocation assigns four independent
responses when the gate activates; random allocation draws uniformly from
one to six with allocation seed 42. Other rounds use eight discussion responses.

| Policy | Correct / N | Accuracy (%) | Independent responses | Independent share (%) | Subsequent rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gated fixed 4+4 | 214 / 490 | 43.67 | 1,816 | 23.07 | 984 |
| Gated random, seed 42 | 216 / 490 | 44.08 | 1,701 | 21.94 | 969 |
| Adaptive EDR | 236 / 490 | 48.16 | 1,286 | 18.67 | 861 |

Independent share is calculated over all responses after round 1. Retained
partial response stages count in the round and allocation totals; missing
final answers count as incorrect in the accuracy denominator.

All three policies share initial responses, ordered first retrieval and history
order. For adaptive EDR versus gated random allocation, the gain is 4.08
percentage points, with a paired 95% confidence interval of [1.84, 6.33] and
exact two-sided McNemar p = 0.00053505.

## Files

- [Fixed 4+4 trajectory](../benchmarks/MedXpertQA/qwen3.8%2027b/traces/edr_gated_fixed4_490.jsonl)
- [Fixed 4+4 score](../benchmarks/MedXpertQA/qwen3.8%2027b/results/edr_gated_fixed4_490.json)
- [Fixed-allocation implementation](../src/edr_gated_fixed4.py)
- [Allocation summary](../analysis/allocation_ablation.json)
- [Paired random-allocation comparison](../benchmarks/MedXpertQA/qwen3.8%2027b/results/comparisons.json)

From the package root, recompute or check the allocation summary with:

```bash
python -B tools/analyze_allocation_ablation.py
python -B tools/analyze_allocation_ablation.py --check
```

The [integration guide](INTEGRATION.md) gives commands for new evaluations.
