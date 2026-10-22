"""Full paired MedLA evidence-interface/EDR study using fixed upstream functions."""
import argparse
import ast
import copy
import gzip
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import urllib.error
import urllib.request
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from langchain_core.prompts import PromptTemplate
import edr

HERE = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ.get("MEDLA_UPSTREAM", str(HERE / "upstream"))).resolve()
RETRIEVER_URL = os.environ.get("MEDLA_RETRIEVER_URL", "http://127.0.0.1:18990/retriever")
CASE = ContextVar("medla_case")
ACTOR_RNG = ContextVar("actor_rng")
NETWORK = threading.BoundedSemaphore(32)
LOG = logging.getLogger("medla_native")
LOG.setLevel(logging.ERROR)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_receipt(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False)
    temporary.replace(path)


def read_receipt(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def request(url, payload):
    attempts = []
    for attempt in range(3):
        start = time.time()
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with NETWORK, urllib.request.urlopen(req, timeout=1200) as response:
                raw = json.load(response)
            attempts.append({"attempt": attempt + 1, "seconds": time.time() - start, "status": 200})
            return raw, attempts
        except Exception as error:
            attempts.append({"attempt": attempt + 1, "seconds": time.time() - start, "error": str(error)})
            if isinstance(error, urllib.error.HTTPError) and error.code not in (429, 500, 502, 503, 504):
                break
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Transport failure: {attempts}")


def strict_parse(content, tag="Answer"):
    blocks = re.findall(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", content, re.I | re.S)
    labels = []
    for block in blocks:
        match = re.fullmatch(r"\s*(?:(?:Answer|Option)\s*:?\s*)?([A-E])\s*[.，,]?\s*", block, re.I)
        if not match:
            raise ValueError(f"Non-single-option {tag} tag")
        labels.append(match.group(1).upper())
    if len(set(labels)) != 1 or not labels:
        raise ValueError(f"Missing or ambiguous {tag} tag")
    return labels[-1]


def load_native():
    scope = {"PromptTemplate": PromptTemplate, "logger": LOG, "pd": pd, "io": io,
             "json": json, "Path": Path, "datetime": datetime, "os": os, "sleep": time.sleep,
             "random": SimpleNamespace(shuffle=lambda values: ACTOR_RNG.get().shuffle(values))}
    for filename in ("util/prompt.py", "util/agent.py"):
        tree = ast.parse((UPSTREAM / filename).read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), scope)
    scope["parse_content"] = strict_parse
    source = ast.parse((UPSTREAM / "main.py").read_text(encoding="utf-8"))
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "agents_talk")
    scope["randint"] = lambda low, high: CASE.get()["selection_rng"].randint(low, high)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), scope)
    return scope


NATIVE = load_native()
BaseAgent = NATIVE["Agent"]
MedicalAgent = NATIVE["MedAgent_Eliminate"]
LogicAgent = NATIVE["LogicAgent"]
DecisionAgent = NATIVE["DecisionMakersAgent"]


def helper_init(self, engine, agent_name, url=""):
    state = CASE.get()
    self.default_temp = None
    self.agent_name = agent_name + digest([state["id"], agent_name])[:10]
    self._call_seq = 0
    self._rng = random.Random(int(digest([state["id"], agent_name, "shuffle"])[:8], 16))


def invoke(self, llm_name, messages, temperature=None):
    state = CASE.get()
    phase, round_id = state["phase"], state["round"]
    counter = self._call_seq
    self._call_seq += 1
    tag = f"{phase}_{round_id}_{self.agent_name}_{counter}"
    inputs = copy.deepcopy(messages)
    independent = phase == "debate" and self.agent_name in state["independent_names"]
    if independent:
        allowed = "/".join(state["candidate_labels"])
        inputs = [{"role": "system", "content": self.system_instruction +
                   "\nAnalyze the medical question using the supplied evidence. Return a correct option, not an eliminated option."},
                  {"role": "user", "content": f"Question: {state['sample']['question']}\n"
                   f"Options: {json.dumps(state['sample']['options'], ensure_ascii=False)}\n\n"
                   f"[Shared retrieved medical evidence]\n{state['evidence']}\n\n"
                   f"Choose from the current candidates: {allowed}. Explain your medical reasoning.\n"
                   f"End with <Answer>Answer: one of {allowed}</Answer>."}]
    else:
        if phase == "debate" and isinstance(self, MedicalAgent):
            if "Question:" in inputs[-1]["content"]:
                self._native_debate_messages = copy.deepcopy(inputs)
            elif hasattr(self, "_native_debate_messages"):
                repair = inputs[-1]["content"]
                inputs = copy.deepcopy(self._native_debate_messages)
                inputs[-1]["content"] += "\n\n" + repair
        inputs[-1]["content"] += "\n\n[Shared retrieved medical evidence]\n" + state["evidence"]
    if isinstance(self, MedicalAgent) or isinstance(self, DecisionAgent):
        tag_name = "Eliminate" if phase == "initial" else "Answer"
        inputs[-1]["content"] += (f"\n\nOutput-format requirement: the <{tag_name}> tag must contain "
            f"exactly ONE capital option letter after 'Answer:'. Choose one allowed option. "
            "Do not copy a slash-separated list of options. Do not give multiple conflicting tags. "
            "Put all explanations outside the tag.")
    payload = {"model": llm_name, "messages": inputs, "temperature": float(temperature),
               "max_tokens": 8000, "seed": int(digest([state["id"], tag, "generation"])[:8], 16) % 2147483647}
    receipt_path = state["root"] / state["arm"] / "receipts" / (tag + ".json.gz")
    source_path = state["root"] / "baseline" / "receipts" / (tag + ".json.gz")
    replay = state["arm"] == "edr" and not state["diverged"]
    if replay:
        saved = read_receipt(source_path)
        assert saved["request_sha256"] == digest(payload), f"Shared-prefix mismatch: {tag}"
        record = dict(saved, replayed=True)
    elif receipt_path.exists():
        record = read_receipt(receipt_path)
        assert record["request_sha256"] == digest(payload), f"Resume request mismatch: {tag}"
        choice = record["response"].get("choices", [{}])[0]
        invalid_output = (choice.get("finish_reason") != "stop"
                          or not (choice.get("message", {}).get("content") or "").strip())
        if state.get("technical_retry") and invalid_output:
            archived = receipt_path.with_name(tag + ".failed_technical_attempt_1.json.gz")
            if not archived.exists():
                save_receipt(archived, record)
            raw, attempts = request(state["base"] + "/chat/completions", payload)
            record = dict(record, response=raw, attempts=attempts, technical_retry=True)
    else:
        raw, attempts = request(state["base"] + "/chat/completions", payload)
        record = {"request": payload, "request_sha256": digest(payload), "response": raw,
                  "attempts": attempts, "replayed": False, "phase": phase, "round": round_id,
                  "actor": self.agent_name, "channel": "independent" if independent else
                  ("initial" if phase == "initial" else "judge" if phase == "judge" else "discussion")}
    save_receipt(receipt_path, record)
    raw = record["response"]
    choice = raw.get("choices", [{}])[0]
    if choice.get("finish_reason") != "stop":
        state["errors"].append({"tag": tag, "finish_reason": choice.get("finish_reason")})
        raise RuntimeError(f"Generation did not stop normally: {tag}")
    text = choice.get("message", {}).get("content") or ""
    if not text.strip():
        raise RuntimeError(f"Empty model output: {tag}")
    with state["lock"]:
        state["calls"].append({"tag": tag, "replayed": replay, "channel": record["channel"]})
    return text


NATIVE["AgentHelpler"].__init__ = helper_init
NATIVE["AgentHelpler"].invoke = invoke
ORIGINAL_ELIMINATE = MedicalAgent.Eliminate_one_answer


def eliminate(self, *args, **kwargs):
    token = ACTOR_RNG.set(self._rng)
    try:
        result = ORIGINAL_ELIMINATE(self, *args, **kwargs)
        option = args[1] if len(args) > 1 else kwargs["option"]
        try:
            label = strict_parse(result[1], "Eliminate")
            if label in option:
                if not hasattr(self, "_eliminated_labels"):
                    self._eliminated_labels = []
                self._eliminated_labels.append(label)
        except ValueError:
            pass
        return result
    finally:
        ACTOR_RNG.reset(token)


MedicalAgent.Eliminate_one_answer = eliminate


class SortedTemplate:
    def __init__(self, original):
        self.original = original

    def format(self, **kwargs):
        kwargs["option_str"] = "/".join(sorted(set(kwargs["option_str"].split("/"))))
        return self.original.format(**kwargs)


NATIVE["eliminate_rebuttal_template"] = SortedTemplate(NATIVE["eliminate_rebuttal_template"])
ORIGINAL_REBUTTAL = MedicalAgent.rebuttal


def rebuttal(self, records, **kwargs):
    state = CASE.get()
    if self.agent_name in state["independent_names"]:
        state["diverged"] = True
    answer, reasoning = ORIGINAL_REBUTTAL(self, records, **kwargs)
    if answer not in state["sample"]["options"] or state["errors"]:
        raise RuntimeError(f"Invalid candidate or technical failure in rebuttal: {self.agent_name}")
    return answer, reasoning


MedicalAgent.rebuttal = rebuttal
NATIVE_TALK = NATIVE["agents_talk"]


def choose_route(agents):
    state = CASE.get()
    snapshot = [{"name": a.agent_name, "answer": a.ans, "reason": a.reason,
                 "logic_report": a.logic_report} for a in agents.values()]
    candidate_labels = sorted({a.ans for a in agents.values()})
    rng_copy = random.Random()
    rng_copy.setstate(state["selection_rng"].getstate())
    groups = {}
    for name, agent in agents.items():
        groups.setdefault(agent.ans, []).append(name)
    selected = [names[rng_copy.randint(0, len(names) - 1)] if len(names) > 1 else names[0]
                for names in groups.values()]
    info = {"round": state["round"], "candidate_labels": candidate_labels,
            "selected_names": [agents[name].agent_name for name in selected],
            "peer_pool_size": len(agents), "updated_agents": len(selected), "fallback": False}
    m = 0
    if state["arm"] == "edr":
        inp = {"question": state["sample"]["question"],
               "options": "\n".join(f"{k}. {v}" for k, v in state["sample"]["options"].items()),
               "documents": state["evidence"], "answers": json.dumps(snapshot, ensure_ascii=False)}
        distributions = {}
        try:
            for view in ("E", "H", "EH"):
                payload = edr.probe_payload(state["model"], inp, view, state["label_ids"])
                path = state["root"] / "edr" / "receipts" / f"probe_{state['round']}_{view}.json.gz"
                if path.exists():
                    record = read_receipt(path)
                    assert digest(record["request"]) == digest(payload)
                else:
                    raw, attempts = request(state["base"] + "/chat/completions", payload)
                    record = {"request": payload, "response": raw, "attempts": attempts}
                    save_receipt(path, record)
                distributions[view] = edr.distribution(record["response"], list("ABCD"))
            info.update(edr.route(distributions["E"], distributions["H"], distributions["EH"]))
            m = min(len(selected) - 1, math.floor(len(selected) * info["independent"] / 8 + .5))
        except Exception as error:
            info.update(fallback=True, error=str(error))
        info["distributions"] = distributions
    info["independent_count"] = m
    independent = {agents[name].agent_name for name in selected[:m]}
    info["independent_names"] = sorted(independent)
    state.update(independent_names=independent, candidate_labels=candidate_labels)
    state["routes"].append(info)
    return info


def next_route_selected(agents, rng):
    shadow = random.Random()
    shadow.setstate(rng.getstate())
    groups = {}
    for name, agent in agents.items():
        groups.setdefault(agent.ans, []).append(name)
    return [names[shadow.randint(0, len(names)-1)] if len(names) > 1 else names[0]
            for names in groups.values()]


def fewshots(rows_count):
    old = Path.cwd()
    rng = random.Random(999)
    scope = {"logger": LOG, "np": SimpleNamespace(arange=np.arange,
              random=np.random.RandomState(20261001)), "random": rng, "os": os}
    tree = ast.parse((UPSTREAM / "util/rag.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QADB")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "rag.py", "exec"), scope)
    try:
        os.chdir(UPSTREAM)
        db = scope["QADB"](db_size=100)
        return [db.get_few_shot(1, 17) for _ in range(rows_count)]
    finally:
        os.chdir(old)


def initial_agent(index, sample, examples, args, state):
    token = CASE.set(state)
    logs = []
    try:
        agent = MedicalAgent(args=args, instruction=f"you are [{index}]agent{index}. ",
                             agent_name=f"[{index}]agent{index}", logger=logs)
        agent.answer_question([sample["question"], sample["options"]], examples, retry_count=[0])
        if len(getattr(agent, "_eliminated_labels", [])) != 3 or state["errors"]:
            raise RuntimeError(f"Incomplete initial elimination for {agent.agent_name}")
        critic = LogicAgent(args=args, instruction="you are a logic agent.",
                            agent_name=f"LogicAgent_{index}", logger=logs)
        agent.logic_report = critic.check_logic(agent.reason, retry_count=[0])
        if not agent.logic_report or state["errors"]:
            raise RuntimeError(f"Initial logic-check failure: {index}")
        return agent, logs
    finally:
        CASE.reset(token)


def solve(agents, args, state):
    for round_index in range(3):
        if len({a.ans for a in agents.values()}) == 1:
            break
        state.update(phase="debate", round=round_index + 1)
        choose_route(agents)
        expected = [agents[name].agent_name for name in
                    next_route_selected(agents, state["selection_rng"])]
        agents = NATIVE_TALK(agents, state["id"], retry_count=[0])
        assert [a.agent_name for a in agents.values()] == expected
    answers = [a.ans for a in agents.values()]
    if len(set(answers)) == 1:
        prediction, used_judge = answers[0], False
    else:
        state.update(phase="judge", round=3, independent_names=set())
        judge = DecisionAgent(args=args, logger=[])
        prediction = judge.make_decision(agents, [state["sample"]["question"], state["sample"]["options"]],
                                         num_rounds=3, retry_count=[0])
        used_judge = True
    if prediction not in state["sample"]["options"] or state["errors"]:
        raise RuntimeError("No valid final prediction or prior technical failure")
    return {"prediction": prediction, "used_judge": used_judge,
            "final_agents": [{"name": a.agent_name, "answers": a.ans_list, "reasonings": a.reason_list,
                              "initial_logic_report": a.logic_report} for a in agents.values()],
            "routes": copy.deepcopy(state["routes"]), "calls": copy.deepcopy(state["calls"])}


def paired_case(index, sample, examples, config, label_ids, technical_retry=False):
    root = Path(config.root) / "cases" / str(index)
    state = {"id": index, "sample": sample, "root": root, "base": config.base, "model": config.model,
             "arm": "shared", "phase": "initial", "round": 0, "independent_names": set(),
             "candidate_labels": list("ABCD"), "calls": [], "routes": [], "diverged": False,
             "selection_rng": random.Random(20261001 + index), "errors": [], "lock": threading.Lock(),
             "label_ids": label_ids, "technical_retry": technical_retry}
    token = CASE.set(state)
    args = SimpleNamespace(model="vllm", llm_name=config.model, url=config.base, temp=.7,
                           few_shots=1, knowledges=0, baseline_bool=0, debug=False, dataset="medddx_expert")
    result = {"id": index, "gold": sample["answer_idx"], "status": "completed"}
    try:
        evidence_path = root / "shared" / "retrieval.json.gz"
        if evidence_path.exists():
            retrieval = read_receipt(evidence_path)
        else:
            payload = {"query": sample["question"], "total_k": 32, "top_k": 6,
                       "combine_docs": False, "use_reranker": True}
            raw, attempts = request(RETRIEVER_URL, payload)
            retrieval = {"request": payload, "response": raw, "attempts": attempts}
            save_receipt(evidence_path, retrieval)
        documents = retrieval["response"]["documents"]
        assert documents, "Empty retrieval"
        state["evidence"] = "\n\n".join(f"[D{i+1}] {d.get('title', '')}\n{d['content'][:1200]}"
                                            for i, d in enumerate(documents))
        with ThreadPoolExecutor(max_workers=8) as pool:
            initial = list(pool.map(lambda i: initial_agent(i, sample, examples[i], args, state), range(17)))
        agents = {str(i): value[0] for i, value in enumerate(initial)}
        initial_records = [{"name": a.agent_name, "answer": a.ans, "reason": a.reason,
                            "logic_report": a.logic_report, "eliminated_labels": a._eliminated_labels}
                           for a in agents.values()]
        write_json(root / "shared" / "initial_materials.json", initial_records)
        result["initial_answers"] = [a.ans for a in agents.values()]
        result["initial_calls"] = len(state["calls"])
        for arm in ("baseline", "edr"):
            state.update(arm=arm, phase="debate", round=0, independent_names=set(), calls=[], routes=[],
                         selection_rng=random.Random(20261001 + index), diverged=False)
            result[arm] = solve(copy.deepcopy(agents), args, state)
            result[arm]["correct"] = result[arm]["prediction"] == sample["answer_idx"]
        activated = any(r["independent_count"] > 0 for r in result["edr"]["routes"])
        result["activated"] = activated
        if not activated:
            assert result["baseline"]["prediction"] == result["edr"]["prediction"]
        write_json(root / "paired_result.json", result)
    except Exception:
        result.update(status="technical_failure", error=traceback.format_exc())
        write_json(root / "paired_result.json", result)
    finally:
        CASE.reset(token)
    return result


def aggregate(records, total, start):
    matched = [r for r in records if r["status"] == "completed"]
    b = sum(r["baseline"]["correct"] for r in matched)
    e = sum(r["edr"]["correct"] for r in matched)
    repair = [r["id"] for r in matched if not r["baseline"]["correct"] and r["edr"]["correct"]]
    harm = [r["id"] for r in matched if r["baseline"]["correct"] and not r["edr"]["correct"]]
    return {"total": total, "finished": len(records), "completed_pairs": len(matched),
            "technical_failure_ids": [r["id"] for r in records if r["status"] != "completed"],
            "baseline_correct": b, "edr_correct": e, "repair_ids": repair, "harm_ids": harm,
            "matched_delta_pp": 100*(e-b)/len(matched) if matched else None,
            "initial_disagreement_questions": sum(len(set(r["initial_answers"])) > 1 for r in matched),
            "activated_questions": sum(r["activated"] for r in matched),
            "independent_generations": sum(r["independent_count"] for x in matched for r in x["edr"]["routes"]),
            "probe_failures": sum(r["fallback"] for x in matched for r in x["edr"]["routes"]),
            "elapsed_seconds": time.time()-start, "updated_at": datetime.now().isoformat()}


def final_scores(records, total):
    if len(records) != total or len({r["id"] for r in records}) != total:
        raise ValueError("Final scoring requires complete, unique question coverage")
    scores = {"total": total, "technical_missing_prediction_ids": {},
              "matched_completed_correct": {},
              "scoring_rule": "Score each arm separately; missing technical outputs count as incorrect"}
    for arm in ("baseline", "edr"):
        valid = [r for r in records if isinstance(r.get(arm, {}).get("prediction"), str)
                 and r[arm]["prediction"] in {"A", "B", "C", "D"}]
        for row in valid:
            assert row[arm]["correct"] == (row[arm]["prediction"] == row["gold"])
        correct = sum(row[arm]["prediction"] == row["gold"] for row in valid)
        scores[arm + "_correct"] = correct
        scores[arm + "_accuracy"] = correct / total
        scores[arm + "_valid_predictions"] = len(valid)
        valid_ids = {row["id"] for row in valid}
        scores["technical_missing_prediction_ids"][arm] = [r["id"] for r in records
                                                         if r["id"] not in valid_ids]
        scores["matched_completed_correct"][arm] = sum(r[arm]["correct"] for r in records
                                                     if r["status"] == "completed")
    scores["full_delta_pp"] = 100 * (scores["edr_correct"] - scores["baseline_correct"]) / total
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="New run directory outside the release")
    parser.add_argument("--base", default="http://127.0.0.1:18089/v1")
    parser.add_argument("--model", default="gemma-3-12b-it")
    parser.add_argument("--tokenizer", required=True, help="Local Gemma tokenizer directory")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dataset-jsonl", type=Path,
                        help="Optional normalized four-option dataset, e.g. prepared MMLU-Med")
    parser.add_argument("--expected-count", type=int, default=483)
    parser.add_argument("--dataset-name", default="MedDDx-Expert")
    args = parser.parse_args()
    root = Path(args.root)
    package = HERE.parents[1]
    if root.resolve().is_relative_to(package.resolve()):
        parser.error("Write new runs outside the release package")
    dataset_path = args.dataset_jsonl or UPSTREAM / "data/medddx/test_hard.jsonl"
    samples = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(samples) == args.expected_count and len({r["idx"] for r in samples}) == args.expected_count
    assert all(set(r["options"]) == set("ABCD") and r["answer_idx"] in r["options"] for r in samples)
    examples = fewshots(len(samples))
    if args.limit:
        samples = samples[:args.limit]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    label_ids = {k: tokenizer.encode(k, add_special_tokens=False) for k in "ABCD"}
    assert all(len(ids) == 1 for ids in label_ids.values())
    label_ids = {k: ids[0] for k, ids in label_ids.items()}
    protocol = {"harness": "MedLA with shared MedCorp evidence interface",
               "upstream_commit": "5c12cfc8d67170b1f4b131b9c120a54a573c634d",
               "dataset": args.dataset_name, "n": len(samples), "source_n": args.expected_count,
               "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
               "model": args.model, "num_initial_agents": 17, "num_rounds": 3,
               "temperature": .7, "max_tokens": 8000, "few_shots": 1, "db_size": 100,
               "question_concurrency": args.concurrency, "network_concurrency": 32,
               "allocation": "m=min(N-1,floor(N*m8/8+0.5)); original EDR route; native representative order",
               "source_execution": "Original elimination, logic checker, representative selector, rebuttal and judge via AST",
               "common_adapters": ["Local Gemma HTTP backend", "Deterministic names/seeds/shuffle",
                 "Shared MedCorp evidence in both conditions", "Strict native tag parser", "Sorted candidate string",
                 "Single-letter format guard", "Preserve native question/history on format-only retries"],
               "edr_change": "At representative rebuttal only, remove own/peer history for selected turns",
               "prefix": "Shared initial materials; request-validated baseline replay until first history mask",
               "source_hashes": {p: hashlib.sha256((UPSTREAM/p).read_bytes()).hexdigest()
                                  for p in ("main.py", "util/agent.py", "util/prompt.py", "util/rag.py")}}
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        core_keys = set(protocol) - {"question_concurrency"}
        if any(previous.get(key) != protocol[key] for key in core_keys):
            raise ValueError("Existing run belongs to different data, prompts or model settings")
    else:
        write_json(protocol_path, protocol)
    from endpoint_check import check_endpoint
    check_endpoint(args.base, args.model, label_ids)
    started = time.time()
    results = []
    pending = []
    for index, sample in enumerate(samples):
        path = root / "cases" / str(index) / "paired_result.json"
        saved = json.loads(path.read_text()) if path.exists() else None
        if saved and saved["status"] == "completed":
            results.append(saved)
        else:
            pending.append((index, sample))
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(paired_case, index, sample, examples[index], args, label_ids) for index, sample in pending]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = aggregate(results, len(samples), started)
            write_json(root / "progress.json", status)
            print("PAIRED_PROGRESS " + json.dumps(status), flush=True)
    # Only technical failures receive one further same-protocol attempt.
    for failed in [r for r in results if r["status"] != "completed"]:
        index = failed["id"]
        save_receipt(root / "cases" / str(index) / "technical_attempt_1.json.gz", failed)
        replacement = paired_case(index, samples[index], examples[index], args, label_ids, technical_retry=True)
        results = [replacement if r["id"] == index else r for r in results]
        write_json(root / "progress.json", aggregate(results, len(samples), started))
    results.sort(key=lambda r: r["id"])
    write_json(root / "paired_results.json", results)
    status = aggregate(results, len(samples), started)
    status.update(final_scores(results, len(samples)))
    write_json(root / "summary.json", status)
    return 0 if not status["technical_failure_ids"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
