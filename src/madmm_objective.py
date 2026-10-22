"""MAD-M2 objective masking, adapted to OpenAI-format generation logprobs."""
import math


def generation_score(choice):
    """Match upstream exp(mean(rank-1 logprob)), not sampled-token perplexity."""
    tokens = (choice.get("logprobs") or {}).get("content") or []
    if not tokens:
        raise ValueError("Missing generated-token probabilities")
    values = []
    for token in tokens:
        candidates = token.get("top_logprobs") or []
        if not candidates:
            raise ValueError("Missing top-1 probability at a generated position")
        probabilities = [entry["logprob"] for entry in candidates]
        if any(p is None or math.isnan(p) or p > 1e-6 for p in probabilities):
            raise ValueError("Invalid generation log probability")
        top1 = max(probabilities)
        if top1 != float("-inf"):
            values.append(top1)
    # Do not interpret absent probability records as confident answers.
    if not values:
        raise ValueError("No finite rank-1 log probabilities")
    return {"score": math.exp(sum(values) / len(values)),
            "valid_token_count": len(values), "generated_position_count": len(tokens)}


def objective_mask(scores):
    """Exact upstream strict-above-descending-middle selector, including ties."""
    if not scores or any(not math.isfinite(s) or s < 0 or s > 1.000001 for s in scores):
        raise ValueError("Invalid objective score list")
    threshold = sorted(scores, reverse=True)[len(scores) // 2]
    return [score > threshold for score in scores], threshold
