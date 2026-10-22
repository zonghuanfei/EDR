"""MA-RAG-int with bounded queries and validated cache reuse."""
import ast
import gzip
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from liquid import Template
from scipy.stats import entropy

from run_direct_medxpertqa import parse_answer
from query_output_guard import MAX_QUERIES, STOP_STRINGS, parse_queries, reuse_reason
from frozen_prefix import (completed_prefix, initial_record, initial_transition,
                           local_query_order, source_query_order)


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def official_prompts(path):
    wanted = {"system_prompt", "user_prompt", "user_prompt_round", "system_prompt_query", "user_prompt_query"}
    result = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in wanted:
            value = node.value
            result[node.targets[0].id] = Template(ast.literal_eval(value.args[0])) if isinstance(value, ast.Call) else ast.literal_eval(value)
    assert result.keys() == wanted
    return result


def majority(predictions):
    votes = Counter(predictions).most_common()
    return votes[0][0] if len(votes) == 1 or votes[0][1] > votes[1][1] else None


def token_entropies(tokens):
    values = [[value["logprob"] for value in token["top_logprobs"]] for token in tokens]
    if len({len(row) for row in values}) == 1:
        return entropy(np.exp(np.asarray(values, dtype=np.float64)), axis=1)
    return np.asarray([entropy(np.exp(row)) for row in values])


def save_receipt(path, receipt):
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as output:
        json.dump(receipt, output, ensure_ascii=False)
    temporary.replace(path)


def api_call(path, url, payload, source_path=None):
    frozen = None
    if source_path is not None:
        if not source_path.is_file():
            raise RuntimeError(f"Frozen source request is missing: {source_path}")
        with gzip.open(source_path, "rt", encoding="utf-8") as source:
            frozen = json.load(source)
        if not reuse_reason(frozen, payload):
            raise RuntimeError(f"Frozen source request has different parameters: {source_path}")
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as source:
            receipt = json.load(source)
        assert reuse_reason(receipt, payload), "Cached call belongs to different parameters"
        if frozen is not None and receipt["response"] != frozen["response"]:
            raise RuntimeError("Cached response differs from the frozen source")
        return receipt["response"]
    if frozen is not None:
        saved = dict(frozen, effective_request=payload, reused_from=str(source_path),
                     reuse_validation=reuse_reason(frozen, payload))
        save_receipt(path, saved)
        return saved["response"]
    started = time.perf_counter()
    response = requests.post(url, json=payload, headers={"Authorization": "Bearer local"}, timeout=(30, 1800))
    response.raise_for_status()
    raw = response.json()
    save_receipt(path, {"request": payload, "response": raw, "seconds": time.perf_counter()-started})
    return raw


def complete_choice(choice):
    if choice.get("finish_reason") != "stop":
        raise RuntimeError(f"Incomplete generation: finish_reason={choice.get('finish_reason')}")
    return choice["message"].get("content") or ""




def question_run(item, args, prompts, output):
    path = output / f"question_{item['id']}"
    path.mkdir(exist_ok=True)
    order_root = getattr(args, "query_order_root", None)
    frozen_folder = order_root / path.name if order_root is not None else None
    if (path / "done.json").exists():
        if frozen_folder is not None:
            completed_prefix(path, frozen_folder, parse_queries, reuse_reason)
        return json.loads((path / "done.json").read_text())
    source_root = getattr(args, "reuse_root", None)

    def call(name, url, payload):
        source_path = source_root / path.name / name if source_root is not None else None
        return api_call(path / name, url, payload, source_path=source_path)
    options = "\n".join(f"{k}. {v}" for k, v in item["option"].items() if v.strip())
    previous, documents = [], ""
    history = []
    for round_id in range(1, 9):
        if round_id == 1:
            prompt = prompts["user_prompt"].render(question=item["question"], options=options)
        else:
            prompt = prompts["user_prompt_round"].render(question=item["question"], options=options, answers="\n\n".join(previous), documents=documents)
        payload = {
            "model": args.model, "messages": [{"role": "system", "content": prompts["system_prompt"]}, {"role": "user", "content": prompt}],
            "temperature": 0.7, "max_completion_tokens": args.max_tokens, "n": 8,
            "logprobs": True, "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False},
        }
        raw = call(f"round_{round_id}_solver.json.gz", args.base_url + "/chat/completions", payload)
        assert len(raw["choices"]) == 8
        answers, predictions, uncertainties = [], [], []
        for choice in sorted(raw["choices"], key=lambda c: c["index"]):
            text = complete_choice(choice)
            if choice["message"].get("reasoning") or choice["message"].get("reasoning_content"):
                raise RuntimeError("Unexpected hidden reasoning in Solver")
            prediction = parse_answer(text, set(item["option"]))
            if prediction is None:
                raise RuntimeError("Solver did not produce an explicit final answer")
            tokens = choice["logprobs"]["content"]
            if not tokens or not all(t.get("top_logprobs") for t in tokens):
                raise RuntimeError("Missing generation log probabilities")
            values = token_entropies(tokens)
            mean_entropy = float(np.mean(values))
            if not np.isfinite(mean_entropy):
                raise RuntimeError("Nonfinite entropy")
            answers.append(text)
            predictions.append(prediction)
            uncertainties.append(mean_entropy)
        record = {"round": round_id, "predictions": predictions, "entropies": uncertainties, "prediction": majority(predictions), "usage": raw["usage"]}
        if round_id == 1 and frozen_folder is not None:
            record = initial_record(frozen_folder, record)
            uncertainties = record["entropies"]
        history.append(record)
        write_json(path / f"round_{round_id}.json", record)
        if len(set(predictions)) == 1 or round_id == 8:
            done = {"id": item["id"], "prediction": record["prediction"], "first_round_prediction": history[0]["prediction"], "rounds": round_id, "history": history, "state": "complete"}
            write_json(path / "done.json", done)
            return done
        answers_for_query = [f"{i}. The assistant's previous answer is:\n{answer}" for i, answer in enumerate(answers, 1)]
        query_prompt = prompts["user_prompt_query"].render(question=item["question"], options=options, answers="\n\n".join(answers_for_query))
        query_payload = {
            "model": args.model, "messages": [{"role": "system", "content": prompts["system_prompt_query"]}, {"role": "user", "content": query_prompt}],
            "temperature": 0.0, "max_completion_tokens": 8192, "n": 1,
            "chat_template_kwargs": {"enable_thinking": True},
            "stop": STOP_STRINGS,
        }
        query_raw = call(f"round_{round_id}_query.json.gz", args.base_url + "/chat/completions", query_payload)
        if len(query_raw.get("choices", [])) != 1:
            raise RuntimeError("Query generator must return exactly one response")
        queries, query_guard = parse_queries(query_raw["choices"][0])
        if round_id == 1 and frozen_folder is not None:
            queries = source_query_order(frozen_folder, query_payload, query_raw, queries, reuse_reason)
        else:
            queries = local_query_order(path, round_id, queries, query_payload, query_raw, write_json)
        assert 1 <= len(queries) <= MAX_QUERIES
        retrieved, seen = [], set()
        def retrieve(pair):
            i, query = pair
            body = {"query": query, "total_k": 32, "top_k": 2, "combine_docs": False, "use_reranker": True}
            return call(f"round_{round_id}_retrieval_{i}.json.gz", args.retriever_url + "/retriever", body)
        if queries:
            with ThreadPoolExecutor(max_workers=len(queries)) as pool:
                results = list(pool.map(retrieve, enumerate(queries)))
            for result in results:
                if "documents" not in result or "scores" not in result:
                    raise RuntimeError("Malformed retrieval response")
                new_docs = [doc for doc in result["documents"] if doc["id"] not in seen]
                retrieved.extend(new_docs)
                seen.update(doc["id"] for doc in new_docs)
            documents = "\n".join(doc["content"] for doc in retrieved)
        else:
            documents = "null"
        # Preserve the released source's descending-entropy order exactly.
        frozen_transition = (initial_transition(frozen_folder, queries, retrieved, uncertainties,
                                               query_raw["usage"])
                             if round_id == 1 and frozen_folder is not None else None)
        order = (np.asarray(frozen_transition["history_order"], dtype=int)
                 if frozen_transition is not None else np.argsort(uncertainties)[::-1])
        previous = [f"{i}. The assistant's previous answer is (Entropy {uncertainties[j]:.2f}):\n{answers[j]}" for i, j in enumerate(order, 1)]
        transition = frozen_transition if frozen_transition is not None else {"queries": queries, "documents": retrieved, "history_order": order.tolist(), "query_usage": query_raw["usage"], "query_output_guard": query_guard}
        write_json(path / f"round_{round_id}_transition.json", transition)
