"""MAD-M2 objective history masking on frozen MedLA initial materials."""
import argparse
import copy
import hashlib
import json
import math
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_medla as core
from madmm_objective import generation_score, objective_mask

SOURCE = HERE / "run"
FORMAT_GUARD = (
    "\n\nOutput-format requirement: the <{tag}> tag must contain "
    "exactly ONE capital option letter after 'Answer:'. Choose one allowed option. "
    "Do not copy a slash-separated list of options. Do not give multiple conflicting tags. "
    "Put all explanations outside the tag."
)
ORIGINAL_REBUTTAL = core.MedicalAgent.rebuttal
TOKENIZER = None


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted_initial_calls(source, agent):
    prefix = "initial_0_" + agent.agent_name + "_"
    paths = []
    for path in (source / "shared/receipts").glob("*.json.gz"):
        if path.name.startswith(prefix):
            suffix = path.name[len(prefix):-len(".json.gz")]
            if suffix.isdigit():
                paths.append((int(suffix), path))
    remaining, accepted = agent.reason, []
    for _, path in sorted(paths):
        saved = core.read_receipt(path)
        text = saved["response"]["choices"][0]["message"]["content"]
        part = text + "\n--------------\n"
        if remaining.startswith(part):
            accepted.append(path)
            remaining = remaining[len(part):]
    if remaining or len(accepted) != 3:
        raise ValueError("Could not map accepted medical reasoning to frozen calls")
    return accepted


def forced_score(record, prefix_length):
    payload, raw = record["request"], record["response"]
    target = payload["prompt"][prefix_length:]
    positions = raw.get("prompt_logprobs")
    if positions is None:
        positions = raw["choices"][0].get("prompt_logprobs")
    if not target or not isinstance(positions, list) or len(positions) != len(payload["prompt"]):
        raise ValueError("Incomplete teacher-forced probability response")
    top1 = []
    for token_id, row in zip(target, positions[prefix_length:]):
        if not row or str(token_id) not in row:
            raise ValueError("Missing probability position in canonical answer replay")
        value = max(float(v["logprob"]) for v in row.values())
        if not math.isfinite(value) or value > 1e-6:
            raise ValueError("Invalid rank-1 log probability")
        top1.append(value)
    return {"score": math.exp(sum(top1) / len(top1)), "valid_token_count": len(top1),
            "generated_position_count": len(target)}


def combine_scores(parts):
    count = sum(p["valid_token_count"] for p in parts)
    if not count or any(p["score"] <= 0 for p in parts):
        raise ValueError("Empty or numerically invalid quality score")
    return math.exp(sum(math.log(p["score"]) * p["valid_token_count"] for p in parts) / count)


def score_record(saved, state):
    choice = saved["response"]["choices"][0]
    if (choice.get("logprobs") or {}).get("content"):
        return generation_score(choice)
    text = choice["message"]["content"]
    prefix = TOKENIZER.apply_chat_template(
        saved["request"]["messages"], tokenize=True, add_generation_prompt=True)
    body = TOKENIZER.encode(text, add_special_tokens=False)
    # Canonical text replay includes the terminating turn token, not a new answer.
    end_turn = TOKENIZER.convert_tokens_to_ids("<end_of_turn>")
    if end_turn is None or end_turn == TOKENIZER.unk_token_id:
        raise ValueError("Gemma end-of-turn token unavailable")
    target = body + [end_turn]
    payload = {"model": state["model"], "prompt": prefix + target,
               "temperature": 0.0, "max_tokens": 1, "prompt_logprobs": 1}
    path = state["root"] / "rescoring" / (core.digest(payload) + ".json.gz")
    if path.exists():
        record = core.read_receipt(path)
        assert record["request"] == payload
        assert record["source_request_sha256"] == saved["request_sha256"]
        assert record["source_text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    else:
        raw, attempts = core.request(state["base"].removesuffix("/v1") + "/v1/completions", payload)
        record = {"request": payload, "response": raw, "attempts": attempts,
                  "source_request_sha256": saved["request_sha256"],
                  "source_text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        core.save_receipt(path, record)
    return {**forced_score(record, len(prefix)), "source": "canonical_text_teacher_forcing",
            "original_completion_tokens": saved["response"].get("usage", {}).get("completion_tokens"),
            "retokenized_text_tokens": len(body), "termination_tokens": 1,
            "rescoring_receipt": str(path.relative_to(state["root"]))}


def quality_score(agent, state):
    parts = []
    for path in state["quality"][agent.agent_name]:
        part = score_record(core.read_receipt(path), state)
        if part.get("source") != "canonical_text_teacher_forcing":
            part["generation_receipt"] = str(path.relative_to(state["root"]))
        parts.append(part)
    count = sum(p["valid_token_count"] for p in parts)
    score = combine_scores(parts)
    return {"score": score, "valid_token_count": count, "segments": parts,
            "scored_content": "accepted medical-agent generations; not the separate logic report"}


def choose_mask(agents):
    state = core.CASE.get()
    quality = [quality_score(a, state) for a in agents.values()]
    mask, threshold = objective_mask([v["score"] for v in quality])
    snapshot = {name: {"name": a.agent_name, "ans": a.ans,
                "medical_reason": a.reason,
                "reason": a.reason + "logic_report" + a.logic_report}
                for name, a in agents.items()}
    selected = core.next_route_selected(agents, state["selection_rng"])
    visible = [name for name, flag in zip(snapshot, mask) if flag]
    entry = {"round": state["round"], "candidate_labels": sorted({a.ans for a in agents.values()}),
             "selected_names": [agents[n].agent_name for n in selected], "peer_pool_size": len(agents),
             "updated_agents": len(selected), "mask": mask, "visible_keys": visible,
             "agent_scores": quality, "threshold": threshold,
             "history_order": list(snapshot), "snapshot": snapshot,
             "independent_count": 0, "fallback": False}
    state.update(snapshot=snapshot, visible_keys=set(visible), independent_names=set(),
                 candidate_labels=entry["candidate_labels"], diverged=True)
    state["routes"].append(entry)
    return entry


def render_history(agent, state):
    snapshot, visible = state["snapshot"], state["visible_keys"]
    own_key = next(k for k, v in snapshot.items() if v["name"] == agent.agent_name)
    opinions = "---\n"
    for key, value in snapshot.items():
        if key in visible and key != own_key:
            opinions += (f"The opinion of {key} is {value['ans']}, and the reasoning is: "
                         f"{value['reason']}.\n ---\n")
    template = core.NATIVE["eliminate_rebuttal_template"].original
    if own_key not in visible:
        marker = "Your opinion is: {ans}. Your reasoning is: {reason}\n"
        if template.template.count(marker) != 1:
            raise ValueError("Native own-history template changed")
        template = core.PromptTemplate.from_template(template.template.replace(marker, "", 1))
    values = {"question": agent.question, "ans": snapshot[own_key]["ans"],
              "reason": agent.reason, "opinions": opinions,
              "option_str": "/".join(state["candidate_labels"])}
    return template.format(**values), own_key in visible


def invoke(self, llm_name, messages, temperature=None):
    state = core.CASE.get()
    tag = f"{state['phase']}_{state['round']}_{self.agent_name}_{self._call_seq}"
    self._call_seq += 1
    inputs = copy.deepcopy(messages)
    masked = state["phase"] == "debate" and isinstance(self, core.MedicalAgent)
    own_visible = None
    if masked:
        rendered, own_visible = render_history(self, state)
        if "Question:" not in inputs[-1]["content"]:
            rendered += "\n\n" + inputs[-1]["content"]
        inputs[-1]["content"] = rendered
    inputs[-1]["content"] += "\n\n[Shared retrieved medical evidence]\n" + state["evidence"]
    if isinstance(self, (core.MedicalAgent, core.DecisionAgent)):
        inputs[-1]["content"] += FORMAT_GUARD.format(
            tag="Eliminate" if state["phase"] == "initial" else "Answer")
    payload = {"model": llm_name, "messages": inputs, "temperature": float(temperature),
               "max_tokens": 8000,
               "seed": int(core.digest([state["id"], tag, "generation"])[:8], 16) % 2147483647}
    if state["arm"] == "shared":
        path = state["source"] / "shared/receipts" / (tag + ".json.gz")
        record = core.read_receipt(path)
        if record["request_sha256"] != core.digest(payload):
            raise ValueError("Frozen initial request differs: " + tag)
        replayed = True
    else:
        if masked:
            payload.update(logprobs=True, top_logprobs=1)
        path = state["root"] / "madmm/receipts" / (tag + ".json.gz")
        record = core.read_receipt(path) if path.exists() else None
        if record:
            assert record["request_sha256"] == core.digest(payload)
            c = record["response"]["choices"][0]
            invalid = c.get("finish_reason") != "stop" or not c["message"].get("content")
            if masked and not (c.get("logprobs") or {}).get("content"):
                invalid = True
            if invalid and state.get("technical_retry"):
                backup = path.with_name(path.name + ".failed_first_attempt")
                if not backup.exists():
                    core.save_receipt(backup, record)
                record = None
        if record is None:
            raw, attempts = core.request(state["base"] + "/chat/completions", payload)
            record = {"request": payload, "request_sha256": core.digest(payload), "response": raw,
                      "attempts": attempts, "phase": state["phase"], "round": state["round"],
                      "actor": self.agent_name, "masked_history": masked,
                      "own_history_visible": own_visible,
                      "visible_history_keys": sorted(state.get("visible_keys", set()), key=int)}
            core.save_receipt(path, record)
        replayed = False
    choice = record["response"]["choices"][0]
    if choice.get("finish_reason") != "stop" or not choice["message"].get("content"):
        state["errors"].append({"tag": tag, "finish_reason": choice.get("finish_reason")})
        raise ValueError("Incomplete generation: " + tag)
    if masked:
        generation_score(choice)
    state["calls"].append({"tag": tag, "replayed": replayed,
                           "channel": "filtered_history" if masked else "initial" if replayed else "judge"})
    state["latest_receipt"][self.agent_name] = path
    return choice["message"]["content"]


def rebuttal(self, records, **kwargs):
    result = ORIGINAL_REBUTTAL(self, records, **kwargs)
    state = core.CASE.get()
    state["quality"][self.agent_name] = [state["latest_receipt"][self.agent_name]]
    return result


def install():
    core.NATIVE["AgentHelpler"].invoke = invoke
    core.MedicalAgent.rebuttal = rebuttal
    core.choose_route = choose_mask


def execute(index, sample, examples, config, reference, technical_retry=False):
    root = Path(config.root) / "cases" / str(index)
    result = {"id": index, "gold": sample["answer_idx"], "status": "completed"}
    if reference["status"] != "completed":
        result.update(status="inherited_initial_failure", error=reference["error"])
        if not config.preflight:
            core.write_json(root / "result.json", result)
        return result
    source = SOURCE / "cases" / str(index)
    state = {"id": index, "root": root, "source": source, "sample": sample,
             "base": config.base, "model": config.model, "arm": "shared", "phase": "initial",
             "round": 0, "independent_names": set(), "candidate_labels": list("ABCD"),
             "selection_rng": random.Random(20261001 + index), "errors": [], "calls": [],
             "routes": [], "quality": {}, "latest_receipt": {}, "diverged": False,
             "lock": threading.Lock(), "technical_retry": technical_retry}
    args = SimpleNamespace(model="vllm", llm_name=config.model, url=config.base, temp=.7,
                           few_shots=1, knowledges=0, baseline_bool=0, debug=False, dataset="medddx_expert")
    token = core.CASE.set(state)
    try:
        retrieval = core.read_receipt(source / "shared/retrieval.json.gz")
        assert retrieval["request"]["query"] == sample["question"]
        documents = retrieval["response"]["documents"]
        state["evidence"] = "\n\n".join(f"[D{i+1}] {d.get('title', '')}\n{d['content'][:1200]}"
                                              for i, d in enumerate(documents))
        # Native initial calls are replayed locally; no new initial generations occur.
        agents = {}
        for i in range(17):
            agent, _ = core.initial_agent(i, sample, examples[i], args, state)
            agents[str(i)] = agent
            state["quality"][agent.agent_name] = accepted_initial_calls(source, agent)
        initial = [{"name": a.agent_name, "answer": a.ans, "reason": a.reason,
                    "logic_report": a.logic_report, "eliminated_labels": a._eliminated_labels}
                   for a in agents.values()]
        assert initial == json.loads((source / "shared/initial_materials.json").read_text())
        if not config.preflight:
            core.write_json(root / "initial_materials.json", initial)
        result["initial_answers"] = [a.ans for a in agents.values()]
        result["initial_replayed_calls"] = len(state["calls"])
        state.update(arm="madmm", calls=[], phase="debate", round=0)
        if config.preflight:
            result["preflight_only"] = True
        else:
            outcome = core.solve(agents, args, state)
            result.update(prediction=outcome["prediction"], correct=outcome["prediction"] == result["gold"],
                          outcome=outcome, evidence_sha256=hashlib.sha256(state["evidence"].encode()).hexdigest())
    except Exception:
        result.update(status="technical_failure", error=traceback.format_exc())
    finally:
        core.CASE.reset(token)
    if not config.preflight:
        core.write_json(root / "result.json", result)
    return result


def aggregate(rows, references, total, started):
    completed = [r for r in rows if r["status"] == "completed"]
    n = len(completed)
    correct = {"baseline": sum(references[r["id"]]["baseline"]["prediction"] == r["gold"] for r in completed),
               "edr": sum(references[r["id"]]["edr"]["prediction"] == r["gold"] for r in completed),
               "madmm": sum(r["prediction"] == r["gold"] for r in completed)}
    repairs = [r["id"] for r in completed if r["prediction"] == r["gold"] and
               references[r["id"]]["edr"]["prediction"] != r["gold"]]
    harms = [r["id"] for r in completed if r["prediction"] != r["gold"] and
             references[r["id"]]["edr"]["prediction"] == r["gold"]]
    return {"dataset": "MedDDx-Expert", "total": total, "finished": len(rows), "completed_pairs": n,
            "matched_completed_correct": correct, "matched_completed_accuracy":
            {k: v/n if n else None for k, v in correct.items()},
            "madmm_vs_edr_delta_pp": 100*(correct["madmm"]-correct["edr"])/n if n else None,
            "madmm_repairs_edr_ids": repairs, "madmm_harms_edr_ids": harms,
            "inherited_initial_failure_ids": [r["id"] for r in rows if r["status"] == "inherited_initial_failure"],
            "technical_failure_ids": [r["id"] for r in rows if r["status"] == "technical_failure"],
            "elapsed_seconds": time.time()-started, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def audit(rows, samples, config):
    assert {r["id"] for r in rows} == set(range(len(samples)))
    masked = rescored = 0
    for row in rows:
        assert row["gold"] == samples[row["id"]]["answer_idx"]
        if row["status"] != "completed":
            continue
        case = Path(config.root) / "cases" / str(row["id"])
        assert json.loads((case / "initial_materials.json").read_text()) == json.loads(
            (SOURCE / "cases" / str(row["id"]) / "shared/initial_materials.json").read_text())
        assert row["correct"] == (row["prediction"] == row["gold"])
        retrieval = core.read_receipt(SOURCE / "cases" / str(row["id"]) / "shared/retrieval.json.gz")
        evidence = "\n\n".join(f"[D{i+1}] {d.get('title', '')}\n{d['content'][:1200]}"
                               for i, d in enumerate(retrieval["response"]["documents"]))
        assert row["evidence_sha256"] == hashlib.sha256(evidence.encode()).hexdigest()
        for route in row["outcome"]["routes"]:
            scores = [q["score"] for q in route["agent_scores"]]
            mask, threshold = objective_mask(scores)
            assert mask == route["mask"] and threshold == route["threshold"]
            assert route["candidate_labels"] == sorted({x["ans"] for x in route["snapshot"].values()})
            for quality in route["agent_scores"]:
                parts = quality["segments"]
                expected = combine_scores(parts)
                assert abs(expected-quality["score"]) < 1e-12
                for part in parts:
                    if part.get("rescoring_receipt"):
                        receipt = core.read_receipt(case / part["rescoring_receipt"])
                        prefix_length = len(receipt["request"]["prompt"]) - part["generated_position_count"]
                        recomputed = forced_score(receipt, prefix_length)
                        assert recomputed["valid_token_count"] == part["valid_token_count"]
                        assert abs(recomputed["score"]-part["score"]) < 1e-12
                    if part.get("generation_receipt"):
                        receipt = core.read_receipt(case / part["generation_receipt"])
                        recomputed = generation_score(receipt["response"]["choices"][0])
                        assert recomputed["valid_token_count"] == part["valid_token_count"]
                        assert abs(recomputed["score"]-part["score"]) < 1e-12
        for call in row["outcome"]["calls"]:
            saved = core.read_receipt(case / "madmm/receipts" / (call["tag"] + ".json.gz"))
            assert core.digest(saved["request"]) == saved["request_sha256"]
            if call["channel"] == "filtered_history":
                masked += 1
                route = next(r for r in row["outcome"]["routes"] if r["round"] == saved["round"])
                assert set(saved["visible_history_keys"]) == set(route["visible_keys"])
                expected = {k for k, v in route["snapshot"].items() if v["name"] == saved["actor"]}
                assert len(expected) == 1
                assert saved["own_history_visible"] == bool(expected & set(route["visible_keys"]))
                user = saved["request"]["messages"][-1]["content"]
                assert "[Shared retrieved medical evidence]" in user
                assert user.endswith("\n\n[Shared retrieved medical evidence]\n" + evidence +
                                     FORMAT_GUARD.format(tag="Answer"))
                own = route["snapshot"][next(iter(expected))]
                agent = SimpleNamespace(agent_name=saved["actor"], reason=own["medical_reason"],
                                        question=[samples[row["id"]]["question"], samples[row["id"]]["options"]])
                rendered, _ = render_history(agent, {"snapshot": route["snapshot"],
                    "visible_keys": set(route["visible_keys"]), "candidate_labels": route["candidate_labels"]})
                assert user.startswith(rendered)
                for k, packet in route["snapshot"].items():
                    if packet["name"] != saved["actor"]:
                        assert (f"The opinion of {k} is {packet['ans']}, and the reasoning is: {packet['reason']}" in user) == (k in route["visible_keys"])
                generation_score(saved["response"]["choices"][0])
        rescored += len(list((case / "rescoring").glob("*.json.gz")))
    return {"verified": True, "coverage": len(rows), "masked_medical_calls": masked,
            "teacher_forced_scoring_calls": rescored, "initial_materials_unchanged": True,
            "candidate_set_unchanged_by_mask": True, "representative_selector": "native, unchanged",
            "final_judge": "native, unmasked", "source_paired_results_sha256": file_sha(SOURCE / "paired_results.json")}


def main():
    global TOKENIZER, SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:18089/v1")
    parser.add_argument("--model", default="gemma-3-12b-it")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-limit", type=int, default=0)
    config = parser.parse_args()
    SOURCE = config.source.resolve()
    output = Path(config.root).resolve()
    if output.is_relative_to(HERE.parents[1].resolve()) or output == SOURCE:
        parser.error("Write comparator results outside the release and outside the source run")
    source_protocol = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    if source_protocol.get("model") != config.model or source_protocol.get("temperature") != .7:
        parser.error("Comparator model and generation settings must match the frozen source")
    samples = [json.loads(s) for s in (core.UPSTREAM / "data/medddx/test_hard.jsonl").read_text().splitlines() if s.strip()]
    reference_rows = json.loads((SOURCE / "paired_results.json").read_text())
    references = {r["id"]: r for r in reference_rows}
    assert len(samples) == len(references) == 483
    assert all(references[i]["gold"] == s["answer_idx"] for i, s in enumerate(samples))
    examples = core.fewshots(len(samples))
    install()
    if config.preflight:
        indices = range(config.preflight_limit or len(samples))
        rows = [execute(i, samples[i], examples[i], config, references[i]) for i in indices]
        assert all(r["status"] in ("completed", "inherited_initial_failure") for r in rows), [r for r in rows if r["status"] == "technical_failure"]
        print(json.dumps({"preflight_verified": True, "questions": len(rows), "model_calls": 0}))
        return 0
    from transformers import AutoTokenizer
    if config.tokenizer is None:
        parser.error("--tokenizer is required for a model-backed run")
    TOKENIZER = AutoTokenizer.from_pretrained(str(config.tokenizer), local_files_only=True)
    output = Path(config.root)
    protocol = {"dataset": "MedDDx-Expert", "n": 483, "model": config.model, "temperature": .7,
                "max_tokens": 8000, "initial_agents": 17, "max_discussion_rounds": 3,
                "question_concurrency": config.concurrency, "request_concurrency": 32,
                "source_dataset_sha256": file_sha(core.UPSTREAM / "data/medddx/test_hard.jsonl"),
                "source_paired_results_sha256": file_sha(SOURCE / "paired_results.json"),
                "module": "MAD-M2 objective medical-history masking transfer",
                "score": "exp(token-weighted mean of per-position rank-1 log probability)",
                "scored_history": "accepted medical generations; logic reports attached to kept peers but not included in scores",
                "frozen_initial_probabilities": "teacher force canonical decoded text and end-of-turn token under original messages; not original saved sampling-token arrays",
                "new_generation_probabilities": "logprobs=true, top_logprobs=1",
                "selector": "score > sorted(scores, reverse=True)[n//2]; no tie fill",
                "mask_scope": "medical rebuttal own/peer histories only; retain evidence and all current candidate labels",
                "unchanged_host": "native elimination, logic checks, representatives, judge, stopping, seeds",
                "retry": "native finite format retries; one additional technical-only case attempt with valid receipts reused",
                "initial_failures_inherited": [r["id"] for r in reference_rows if r["status"] != "completed"],
                "code_sha256": {p.name: file_sha(p) for p in [Path(__file__), HERE.parent/'madmm_objective.py', HERE/'run_medla.py']}}
    previous = output / "protocol.json"
    if previous.exists():
        assert json.loads(previous.read_text()) == protocol
    else:
        core.write_json(previous, protocol)
    started_path = output / "started_at.json"
    started = json.loads(started_path.read_text())["epoch"] if started_path.exists() else time.time()
    if not started_path.exists():
        core.write_json(started_path, {"epoch": started})
    rows, pending = [], []
    for i, sample in enumerate(samples):
        path = output / "cases" / str(i) / "result.json"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved["id"] == i and saved["gold"] == sample["answer_idx"]
            rows.append(saved)
        else:
            pending.append(i)
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        futures = [pool.submit(execute, i, samples[i], examples[i], config, references[i]) for i in pending]
        for future in as_completed(futures):
            rows.append(future.result())
            status = aggregate(rows, references, len(samples), started)
            core.write_json(output / "progress.json", status)
            print("MADMM_PROGRESS " + json.dumps(status), flush=True)
    for failed in [r for r in rows if r["status"] == "technical_failure"]:
        case = output / "cases" / str(failed["id"])
        marker = case / "technical_retry.json"
        if marker.exists():
            continue
        core.write_json(marker, {"original_error": failed["error"], "attempt": 2})
        i = failed["id"]
        replacement = execute(i, samples[i], examples[i], config, references[i], technical_retry=True)
        rows = [replacement if r["id"] == i else r for r in rows]
        core.write_json(output / "progress.json", aggregate(rows, references, len(samples), started))
    rows.sort(key=lambda r: r["id"])
    core.write_json(output / "results.json", rows)
    audit_result = audit(rows, samples, config)
    core.write_json(output / "execution_audit.json", audit_result)
    summary = aggregate(rows, references, len(samples), started)
    full_correct = {arm: sum(r.get(arm, {}).get("prediction") == r["gold"] for r in reference_rows)
                    for arm in ["baseline", "edr"]}
    full_correct["madmm"] = sum(r.get("prediction") == r["gold"] for r in rows)
    summary.update(state="full_completed", full_correct=full_correct,
                   full_accuracy={k: v/483 for k, v in full_correct.items()},
                   full_madmm_vs_edr_delta_pp=100*(full_correct['madmm']-full_correct['edr'])/483)
    core.write_json(output / "summary.json", summary)
    print("MADMM_FINAL " + json.dumps(summary), flush=True)
    return 2 if summary["technical_failure_ids"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
