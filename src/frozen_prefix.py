"""Keep paired initial materials and this run's query order immutable."""
import copy
import gzip
import hashlib
import json
import math


class PrefixIntegrityError(RuntimeError):
    """A paired prefix or cached request is inconsistent."""


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def receipt(path):
    if not path.is_file():
        raise RuntimeError(f"Missing frozen receipt: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def ordered_queries(ordered, parsed):
    if (not isinstance(ordered, list) or not all(isinstance(q, str) and q for q in ordered)
            or len(ordered) != len(parsed) or len(set(ordered)) != len(ordered)
            or set(ordered) != set(parsed)):
        raise RuntimeError("Saved query order does not match the parsed query set")
    return list(ordered)


def source_query_order(folder, payload, raw, parsed, compatible):
    saved = receipt(folder / "round_1_query.json.gz")
    transition = folder / "round_1_transition.json"
    if not transition.is_file() or not compatible(saved, payload) or saved["response"] != raw:
        raise RuntimeError("Frozen first query or transition is incompatible")
    return ordered_queries(read_json(transition).get("queries"), parsed)


def initial_record(folder, current):
    path = folder / "round_1.json"
    if not path.is_file():
        raise RuntimeError("Frozen first-round decision is missing")
    saved = read_json(path)
    keys = ("round", "predictions", "prediction", "usage")
    if any(current.get(k) != saved.get(k) for k in keys):
        raise RuntimeError("First-round decision differs from the frozen source")
    a, b = current.get("entropies", []), saved.get("entropies", [])
    if len(a) != 8 or len(b) != 8 or any(not math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-10)
                                        for x, y in zip(a, b)):
        raise RuntimeError("First-round entropy differs from the frozen source")
    return copy.deepcopy(saved)


def initial_transition(folder, queries, documents, entropies, query_usage):
    path = folder / "round_1_transition.json"
    if not path.is_file():
        raise RuntimeError("Frozen first-retrieval transition is missing")
    saved = read_json(path)
    if saved.get("queries") != queries or saved.get("documents") != documents:
        raise RuntimeError("Frozen first retrieved documents, content or order differ")
    if saved.get("query_usage") != query_usage:
        raise RuntimeError("Frozen first query usage differs")
    order = saved.get("history_order")
    if (not isinstance(order, list) or len(order) != 8 or
            any(type(i) is not int for i in order) or sorted(order) != list(range(8))):
        raise RuntimeError("Frozen history order is not a permutation")
    if any(entropies[order[i]] < entropies[order[i + 1]] for i in range(7)):
        raise RuntimeError("Frozen history order is not descending in entropy")
    return copy.deepcopy(saved)


def local_query_order(folder, round_id, parsed, payload, raw, write_json):
    path = folder / f"round_{round_id}_query_order.json"
    identity = {"request_sha256": digest(payload), "response_sha256": digest(raw)}
    if path.is_file():
        saved = read_json(path)
        if any(saved.get(k) != v for k, v in identity.items()):
            raise RuntimeError("Query-order journal belongs to a different request or response")
        return ordered_queries(saved.get("queries"), parsed)
    if ((folder / f"round_{round_id}_transition.json").exists() or
            any(folder.glob(f"round_{round_id}_retrieval_*.json.gz"))):
        raise RuntimeError("Existing retrieval has no query-order journal; refusing to infer its order")
    order = ordered_queries(list(parsed), parsed)
    write_json(path, dict(identity, queries=order))
    return order


def completed_prefix(folder, source, parse_queries, compatible):
    """Validate the retained initial prefix before returning an existing endpoint."""
    first = source / "round_1.json"
    if not first.is_file() or read_json(folder / "round_1.json") != read_json(first):
        raise RuntimeError("Completed output has a different frozen initial decision")
    source_solver = receipt(source / "round_1_solver.json.gz")
    local_solver = receipt(folder / "round_1_solver.json.gz")
    if (not compatible(local_solver, source_solver["request"]) or
            local_solver["response"] != source_solver["response"]):
        raise RuntimeError("Completed output has different frozen initial responses")
    first_predictions = read_json(first).get("predictions", [])
    if len(first_predictions) != 8:
        raise RuntimeError("Frozen initial prediction budget differs")
    if len(set(first_predictions)) == 1:
        return
    source_query = receipt(source / "round_1_query.json.gz")
    local_query = receipt(folder / "round_1_query.json.gz")
    if (not compatible(local_query, source_query["request"]) or
            local_query["response"] != source_query["response"]):
        raise RuntimeError("Completed output has a different frozen first query")
    choices = source_query["response"].get("choices", [])
    if len(choices) != 1:
        raise RuntimeError("Frozen first query response count differs")
    parsed, _ = parse_queries(choices[0])
    transition = read_json(source / "round_1_transition.json")
    ordered = ordered_queries(transition.get("queries"), parsed)
    local_transition = read_json(folder / "round_1_transition.json")
    if local_transition != transition:
        raise RuntimeError("Completed output has a different frozen first transition")
    documents, seen = [], set()
    for i, query in enumerate(ordered):
        name = f"round_1_retrieval_{i}.json.gz"
        original, cached = receipt(source / name), receipt(folder / name)
        if original.get("request", {}).get("query") != query:
            raise RuntimeError("Frozen retrieval index belongs to a different query")
        if (not compatible(cached, original["request"]) or
                cached["response"] != original["response"]):
            raise RuntimeError("Completed output has different frozen retrieval content")
        result = original["response"]
        added = [doc for doc in result["documents"] if doc["id"] not in seen]
        documents.extend(added)
        seen.update(doc["id"] for doc in added)
    initial_transition(source, ordered, documents, read_json(first)["entropies"],
                       source_query["response"].get("usage"))
