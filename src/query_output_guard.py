"""Bound the released MA-RAG query interface without changing its prompts."""
import re

POLICY = "first_numbered_block_max4_v1"
MAX_QUERIES = 4
STOP_STRINGS = ["\n[Query 5]"]
QUERY_LINE = re.compile(r"^[ \t]*\[Query ([0-9]+)\][ \t]*([^\r\n]*)", re.MULTILINE)


def parse_queries(choice):
    if choice.get("finish_reason") != "stop":
        raise RuntimeError(f"Incomplete query generation: finish_reason={choice.get('finish_reason')}")
    text = choice.get("message", {}).get("content") or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    matches = list(QUERY_LINE.finditer(text))
    if not matches:
        raise RuntimeError("Query generator returned no numbered queries")
    selected = []
    for expected, match in enumerate(matches[:MAX_QUERIES], 1):
        if int(match.group(1)) != expected:
            raise RuntimeError("Queries must start at 1 with consecutive numbering")
        query = match.group(2).strip()
        if not query:
            raise RuntimeError("Query generator returned an empty query")
        selected.append(query)
    # Keep the released set-based ordering.
    queries = list(set(selected))
    return queries, {
        "policy": POLICY,
        "numbered_lines": len(matches),
        "accepted_numbered_lines": len(selected),
        "unique_queries": len(queries),
        "ignored_extra_lines": max(0, len(matches) - MAX_QUERIES),
        "finish_reason": choice["finish_reason"],
        "stop_reason": choice.get("stop_reason"),
    }


def reuse_reason(receipt, requested):
    """Check cache reuse against the original stored request."""
    original = receipt.get("request", {})
    if original == requested:
        return "identical_request"
    if requested.get("stop") != STOP_STRINGS or "stop" in original:
        return None
    without_stop = {key: value for key, value in requested.items() if key != "stop"}
    if original != without_stop:
        return None
    choices = receipt.get("response", {}).get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        return None
    message = choices[0].get("message", {})
    text = message.get("content") or ""
    all_generated = text + (message.get("reasoning") or "") + (message.get("reasoning_content") or "")
    if any(stop in all_generated for stop in STOP_STRINGS):
        return None
    try:
        queries, _ = parse_queries(choices[0])
    except RuntimeError:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    legacy = set(query.strip() for query in re.findall(r"\[Query .*?\](.*?)$", text, re.MULTILINE))
    if set(queries) != legacy:
        return None
    return "added_stop_absent_from_output_and_identical_query_set"
