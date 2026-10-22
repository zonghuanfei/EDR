"""Run MA-RAG, EDR, allocation ablations or the MAD-M2 adapter."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path

import run_marag_queryguard_w8 as harness
from endpoint_check import check_endpoint, label_token_ids
from frozen_prefix import PrefixIntegrityError
from technical_retry import TechnicalRetry

PACKAGE = Path(__file__).resolve().parents[1]
DATASETS = {"MedXpertQA": (490, "ABCDEFGHIJ")}


def load_dataset(upstream, name):
    count, labels = DATASETS[name]
    path = upstream / "datasets" / (name + ".json")
    items = json.loads(path.read_text(encoding="utf-8-sig"))
    if len(items) != count or {item["id"] for item in items} != set(range(count)):
        raise ValueError("The pinned, complete dataset with unique zero-based IDs is required")
    for item in items:
        present = {k for k, v in item["option"].items() if v.strip()}
        if present != set(labels) or item["answer"] not in present or not item["question"].strip():
            raise ValueError(f"Invalid question/options/gold: {item['id']}")
    return items, path, labels


def frozen_initial_valid(source, item):
    path = source / f"question_{item['id']}" / "round_1_solver.json.gz"
    decision = path.parent / "round_1.json"
    if not path.is_file() or not decision.is_file():
        return False
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        choices = json.load(stream)["response"].get("choices", [])
    return len(choices) == 8 and all(
        choice.get("finish_reason") == "stop" and
        harness.parse_answer(choice.get("message", {}).get("content") or "", set(item["option"]))
        is not None for choice in choices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Pinned upstream MA-RAG checkout")
    parser.add_argument("--dataset", choices=DATASETS, default="MedXpertQA")
    parser.add_argument("--condition", choices=("baseline", "edr", "gated_random", "gated_fixed4", "madmm_objective"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="A separate directory for this new run")
    parser.add_argument("--model", required=True, help="The served model ID")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--retriever-url", default="http://127.0.0.1:18990")
    parser.add_argument("--tokenizer", type=Path, help="Local tokenizer matching the served weights")
    parser.add_argument("--source-baseline", type=Path, help="Raw baseline run for frozen initial materials")
    parser.add_argument("--allocation-seed", type=int,
                        help="Local allocation RNG seed for gated_random")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--validate-only", action="store_true", help="Check local inputs without any API requests")
    args = parser.parse_args()
    if args.workers < 1 or args.max_tokens < 1:
        parser.error("Worker count and token limit must be positive")
    if args.condition == "gated_random":
        if args.source_baseline is None or args.allocation_seed is None or args.allocation_seed < 0:
            parser.error("gated_random requires --source-baseline and --allocation-seed")
    elif args.allocation_seed is not None:
        parser.error("--allocation-seed is only used by gated_random")
    if args.condition in ("madmm_objective", "gated_fixed4") and args.source_baseline is None:
        parser.error("This condition requires --source-baseline")
    if args.condition in ("edr", "gated_random", "gated_fixed4") and args.tokenizer is None:
        parser.error("EDR conditions require --tokenizer")
    if args.condition == "baseline" and args.source_baseline is not None:
        parser.error("A fresh baseline does not import another condition's initial materials")
    for source in (args.source_baseline,):
        if source is not None and (not source.is_dir() or source.resolve() == args.output.resolve()):
            parser.error("A source must be an existing directory different from --output")
    if args.output.resolve().is_relative_to(PACKAGE):
        parser.error("Write new runs outside the release package")
    items, dataset_path, labels = load_dataset(args.root, args.dataset)
    prompts = harness.official_prompts(args.root / "ma_rag_entropy.py")
    ids = label_token_ids(args.tokenizer, labels) if args.tokenizer is not None else {}
    config = {"dataset": args.dataset, "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
              "upstream_prompt_sha256": hashlib.sha256((args.root / "ma_rag_entropy.py").read_bytes()).hexdigest(),
              "condition": args.condition, "model": args.model, "base_url": args.base_url.rstrip("/"),
              "retriever_url": args.retriever_url.rstrip("/"), "max_tokens": args.max_tokens,
              "solver_temperature": .7, "query_temperature": 0.0, "probe_temperature": 1.0,
              "source_baseline": str(args.source_baseline.resolve()) if args.source_baseline else None,
              "label_token_ids": ids}
    if args.condition == "gated_random":
        from edr_random_allocation import allocation_config
        config["allocation_seed"] = args.allocation_seed
        config["allocation"] = allocation_config(args.allocation_seed, args.dataset)
    source = args.source_baseline
    if source is not None:
        source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        keys = ("dataset", "dataset_sha256", "upstream_prompt_sha256", "model", "max_tokens",
                "solver_temperature", "query_temperature")
        if source_config.get("condition") != "baseline" or any(
            source_config.get(key) != config[key] for key in keys
        ):
            raise ValueError("Frozen source has a different condition, dataset, prompts or generation settings")
    if args.validate_only:
        print(json.dumps({"status": "PASS", "questions": len(items), "model_calls": 0, **config}))
        return 0
    args.base_url, args.retriever_url = config["base_url"], config["retriever_url"]
    args.reuse_root = None
    args.query_order_root = source
    if ids:
        check_endpoint(args.base_url, args.model, ids)
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing output belongs to a different experiment")
    harness.write_json(config_path, config)
    harness.api_call = TechnicalRetry(harness)
    sources = {f"question_{item['id']}": source / f"question_{item['id']}" for item in items} if source else {}
    if args.condition == "edr":
        from plugin_install import install_edr
        plugin = install_edr(harness, ids, sources)
    elif args.condition == "gated_fixed4":
        from edr_gated_fixed4 import Plugin
        plugin = Plugin(harness, ids, sources, mode="gated_fixed4")
        plugin.install()
    elif args.condition == "gated_random":
        from edr_random_allocation import Plugin
        plugin = Plugin(harness, ids, args.allocation_seed, args.dataset)
        plugin.source_map = sources
        plugin.install()
    elif args.condition == "madmm_objective":
        from madmm_plugin import Plugin
        plugin = Plugin(harness, sources)
        plugin.install()
    else:
        plugin = None
    gold = {item["id"]: item["answer"] for item in items}

    def solve(item):
        qid = item["id"]
        folder = args.output / f"question_{qid}"
        folder.mkdir(exist_ok=True)
        try:
            if source is not None and not frozen_initial_valid(source, item):
                raise RuntimeError("Frozen initial material is missing or technically invalid; not regenerated")
            solver_input = {k: item[k] for k in ("id", "question", "option")}
            run = plugin.question_run if plugin else harness.question_run
            result = run(solver_input, args, prompts, args.output)
            return {"id": qid, "gold": gold[qid], "prediction": result["prediction"],
                    "correct": result["prediction"] == gold[qid], "status": "complete",
                    "rounds": result["rounds"]}
        except PrefixIntegrityError:
            raise
        except Exception as error:
            failure = {"id": qid, "gold": gold[qid], "prediction": None, "correct": False,
                       "status": "technical_failure", "error": f"{type(error).__name__}: {error}"}
            harness.write_json(folder / "error.json", failure)
            return failure

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(solve, item) for item in items]):
            rows.append(future.result())
            progress = {"processed": len(rows), "total": len(items),
                        "correct": sum(row["correct"] for row in rows),
                        "technical_failures": sum(row["status"] == "technical_failure" for row in rows)}
            harness.write_json(args.output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    rows.sort(key=lambda row: row["id"])
    correct = sum(row["correct"] for row in rows)
    summary = {"condition": args.condition, "total": len(items), "correct": correct,
               "accuracy": correct / len(items), "rows": rows}
    harness.write_json(args.output / "scores.json", summary)
    return 2 if any(row["status"] == "technical_failure" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
