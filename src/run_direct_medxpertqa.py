#!/usr/bin/env python3
"""Single-response evaluation on MedXpertQA."""

import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a medical assistant. Select the single best answer to the "
    "multiple-choice question. Return only the capital option enclosed in "
    "the tag <answer>X</answer>."
)


def build_messages(item):
    options = "\n".join(f"{k}. {v}" for k, v in item["option"].items() if v.strip())
    prompt = (
        "Below is a multiple-choice medical question.\n\n"
        f"### Question\n{item['question']}\n\n"
        f"### Options\n{options}\n\n"
        "Select the single best answer."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def parse_answer(text, options):
    tags = re.findall(r"<answer>\s*([A-Z])\s*</answer>", text, re.IGNORECASE)
    if tags:
        labels = {tag.upper() for tag in tags}
        terminal = re.search(r"<answer>\s*([A-Z])\s*</answer>\s*$", text, re.IGNORECASE)
        if terminal:
            label = terminal.group(1).upper()
            return label if label in options else None
        return next(iter(labels)) if len(labels) == 1 and labels.issubset(options) else None
    # An isolated whole response is safe; letters embedded in prose are not answers.
    bare = re.fullmatch(r"\s*([A-Z])\s*[.]?\s*", text)
    return bare.group(1) if bare and bare.group(1) in options else None


def classify_response(raw, options):
    choice = raw["choices"][0]
    if choice.get("finish_reason") != "stop":
        return "incomplete", None
    message = choice["message"]
    if message.get("reasoning") or message.get("reasoning_content"):
        return "unexpected_thinking", None
    prediction = parse_answer(message.get("content") or "", options)
    return ("valid", prediction) if prediction is not None else ("unparsed", None)


BENCHMARKS = {"MedXpertQA": (490, "ABCDEFGHIJ")}


def validate_dataset(items, benchmark="MedXpertQA"):
    expected_count, alphabet = BENCHMARKS[benchmark]
    assert len(items) == expected_count, "Expected the complete official repository subset"
    assert len({item["id"] for item in items}) == expected_count, "Duplicate IDs"
    assert {item["id"] for item in items} == set(range(expected_count)), "Unexpected IDs"
    for item in items:
        assert isinstance(item["question"], str) and item["question"].strip()
        choices = {k for k, v in item["option"].items() if v.strip()}
        assert choices and choices.issubset(set(alphabet)), f"Unexpected options for {item['id']}"
        assert choices == set(alphabet), f"Expected ten options for {item['id']}"
        assert item["answer"] in choices, f"Invalid gold for {item['id']}"


def evaluate_one(item, config):
    from openai import OpenAI

    request = {
        "model": config["model"],
        "messages": build_messages(item),
        "temperature": 0.0,
        "max_completion_tokens": config["max_tokens"],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    started = time.perf_counter()
    record = {"id": item["id"], "request": request}
    try:
        with OpenAI(api_key="local", base_url=config["base_url"], timeout=600, max_retries=0) as client:
            response = client.chat.completions.create(**request)
        raw = response.model_dump(mode="json")
        status, prediction = classify_response(raw, {k for k, v in item["option"].items() if v.strip()})
        record.update(raw_response=raw, status=status, prediction=prediction)
    except Exception as error:
        record.update(status="request_error", prediction=None, error=f"{type(error).__name__}: {error}")
    record["latency_seconds"] = time.perf_counter() - started
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", default="qwen3.8-27b-nvfp4")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    data_bytes = args.dataset.read_bytes()
    items = json.loads(data_bytes)
    benchmark = args.dataset.stem
    validate_dataset(items, benchmark)
    config = {
        "model": args.model, "base_url": args.base_url, "workers": args.workers,
        "max_tokens": args.max_tokens, "temperature": 0.0, "thinking": False,
        "dataset_name": benchmark, "expected_questions": len(items),
        "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "system_prompt": SYSTEM_PROMPT,
        "top_p_top_k": "Inherited unchanged from the existing model service",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Fresh directories prevent incompatible settings or earlier retries being merged.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, args.output_dir / Path(__file__).name)
    results = []
    started = time.perf_counter()
    inputs = [{"id": i["id"], "question": i["question"], "option": i["option"]} for i in items]
    with (args.output_dir / "requests.jsonl").open("x", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_one, item, config) for item in inputs]
            for future in as_completed(futures):
                record = future.result()
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                results.append(record)
                if len(results) % 25 == 0 or len(results) == len(items):
                    print(json.dumps({"completed": len(results), "status": dict(Counter(r["status"] for r in results))}), flush=True)
    wall_seconds = time.perf_counter() - started
    gold = {i["id"]: i["answer"] for i in items}
    for record in results:
        record["gold"] = gold[record["id"]]
        record["correct"] = record["prediction"] == record["gold"] if record["status"] == "valid" else None
    results.sort(key=lambda r: r["id"])
    with (args.output_dir / "scored.jsonl").open("x", encoding="utf-8") as output:
        for record in results:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = Counter(r["status"] for r in results)
    all_valid = counts["valid"] == len(items)
    summary = {
        "status": "complete" if all_valid else "requires_review",
        "questions": len(items), "completed_requests": len(results), "response_status": dict(counts),
        "correct": sum(r["correct"] is True for r in results),
        "accuracy": sum(r["correct"] is True for r in results) / len(items) if all_valid else None,
        "wall_seconds": wall_seconds,
        "usage": {key: sum((r.get("raw_response", {}).get("usage") or {}).get(key, 0) or 0 for r in results)
                  for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "finish_reasons": dict(Counter(r["raw_response"]["choices"][0].get("finish_reason") for r in results if "raw_response" in r)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not all_valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
