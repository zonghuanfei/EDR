"""Gate-matched random allocation of independent and discussion responses."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
from concurrent.futures import ThreadPoolExecutor

import edr
from frozen_prefix import PrefixIntegrityError

SCHEMA = "edr_gated_random_allocation_v1"
KEY_SPEC = "UTF-8 JSON sorted keys, ensure_ascii=True, separators=(',',':'); SHA-256; unsigned big-endian integer seed"


class AllocationIntegrityError(PrefixIntegrityError):
    """Abort on changed allocation provenance; never convert to probe fallback."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def allocation_config(seed, dataset="MedXpertQA"):
    if type(seed) is not int or seed < 0 or dataset != "MedXpertQA":
        raise ValueError("Random allocation requires MedXpertQA and a nonnegative integer seed")
    return {"schema": SCHEMA, "condition": "gated_random", "dataset": dataset,
            "allocation_seed": seed,
            "trigger": "edr.route(E,H,EH)['independent'] > 0",
            "distribution": "discrete_uniform", "support": [1, 2, 3, 4, 5, 6],
            "candidate_budget": 8, "fallback_allocation": [0, 8],
            "key_spec": KEY_SPEC, "rng": "python.random.Random(int(SHA256(key),16)).randrange(1,7)",
            "key_fields": ["protocol", "allocation_seed", "dataset", "question_id", "round"],
            "model_request_seed": "not_set", "draws_per_trigger": 1}


def random_metadata(seed, dataset, qid, round_id, trigger):
    allocation_config(seed, dataset)
    if type(qid) is not int or qid < 0 or type(round_id) is not int or not 2 <= round_id <= 8:
        raise AllocationIntegrityError("Invalid allocation question/round identity")
    key = {"protocol": SCHEMA, "allocation_seed": seed, "dataset": dataset,
           "question_id": qid, "round": round_id}
    key_text = canonical(key)
    key_sha256 = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
    selected = random.Random(int(key_sha256, 16)).randrange(1, 7) if trigger else None
    return {"key": key, "key_json": key_text, "key_sha256": key_sha256,
            "key_spec": KEY_SPEC, "generator": "python.random.Random",
            "operation": "randrange(1,7)", "draw_count": int(trigger),
            "selected_independent": selected}


def read_json(path):
    try:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                value = json.load(stream)
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Receipt root is not an object")
        return value
    except (OSError, ValueError, TypeError) as error:
        raise AllocationIntegrityError(f"Unreadable allocation receipt: {path.name}") from error


def branch_payloads(payload, inputs, independent_count):
    if independent_count == 0:
        return []
    independent = copy.deepcopy(payload)
    independent["n"] = independent_count
    independent["messages"][1]["content"] = edr.INDEPENDENT_TEMPLATE.render(**inputs)
    discussion = copy.deepcopy(payload)
    discussion["n"] = 8 - independent_count
    return [("independent", independent), ("discussion", discussion)]


def merge_responses(responses):
    merged = {"choices": [], "usage": {k: 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")}}
    for name, response in responses:
        for choice in sorted(response["choices"], key=lambda c: c["index"]):
            new_choice = copy.deepcopy(choice)
            new_choice["index"] = len(merged["choices"])
            new_choice["edr_channel"] = name
            merged["choices"].append(new_choice)
        for key in merged["usage"]:
            merged["usage"][key] += response.get("usage", {}).get(key, 0) or 0
    if len(merged["choices"]) != 8:
        raise RuntimeError("Merged candidate count is not eight")
    return merged


class Plugin(edr.Plugin):
    def __init__(self, baseline, label_ids, allocation_seed, dataset="MedXpertQA"):
        self.allocation = allocation_config(allocation_seed, dataset)
        self.allocation_seed, self.dataset = allocation_seed, dataset
        super().__init__(baseline, label_ids, source_map=None)
        self.mode = "gated_random"

    def question_run(self, item, args, prompts, output):
        # The baseline has a done.json shortcut. Check our records before that
        # shortcut, including incomplete branch attempts from interrupted work.
        self.validate_question(Path(output) / f"question_{item['id']}", item["id"])
        return super().question_run(item, args, prompts, output)

    def validate_question(self, folder, qid):
        rounds = set()
        for path in folder.glob("round_*"):
            match = re.match(r"round_(\d+)(?:_|\.)", path.name)
            if match and int(match[1]) > 1:
                rounds.add(int(match[1]))
        for round_id in sorted(rounds):
            meta = folder / f"round_{round_id}_edr.json"
            solver = folder / f"round_{round_id}_solver.json.gz"
            if not meta.exists():
                # A crash during probing can leave only probe receipts. They are
                # checked against reconstructed prompts when routed_call resumes.
                rest = [p for p in folder.glob(f"round_{round_id}_*")
                        if "_probe_" not in p.name and not p.name.endswith(".tmp")]
                if rest or (folder / f"round_{round_id}.json").exists():
                    raise AllocationIntegrityError("Solver/decision exists without allocation metadata")
                continue
            info = read_json(meta)
            self.validate_info(info, folder, qid, round_id)
            if solver.exists():
                self.validate_aggregate(read_json(solver), info, folder)
        done = folder / "done.json"
        if done.exists():
            last = read_json(done).get("rounds")
            if type(last) is not int or not 1 <= last <= 8:
                raise AllocationIntegrityError("Invalid completed-round count")
            for round_id in range(2, last + 1):
                if not (folder / f"round_{round_id}_edr.json").is_file() or not (folder / f"round_{round_id}_solver.json.gz").is_file():
                    raise AllocationIntegrityError("Completed question lacks allocation/solver receipts")

    def validate_info(self, info, folder, qid, round_id, payload=None, inputs=None):
        identity = {"schema": SCHEMA, "round": round_id, "mode": self.mode,
                    "dataset": self.dataset, "question_id": qid,
                    "allocation_seed": self.allocation_seed,
                    "allocation_protocol_sha256": digest(self.allocation)}
        if any(info.get(k) != v for k, v in identity.items()):
            raise AllocationIntegrityError("Allocation identity/protocol differs")
        if type(info.get("fallback")) is not bool or type(info.get("trigger")) is not bool:
            raise AllocationIntegrityError("Allocation fallback/trigger must be booleans")
        labels = info.get("labels")
        distributions = info.get("distributions")
        if not isinstance(labels, list) or labels != sorted(self.label_ids) or not isinstance(distributions, dict):
            raise AllocationIntegrityError("Allocation labels/distributions differ")
        if info["fallback"]:
            expected = {"adaptive_independent": None, "trigger": False, "independent": 0, "discussion": 8}
            if not isinstance(info.get("error"), str) or not info["error"] or info.get("failed_probe_view") not in ("E", "H", "EH"):
                raise AllocationIntegrityError("Fallback lacks its probe failure record")
        else:
            if set(distributions) != {"E", "H", "EH"}:
                raise AllocationIntegrityError("Allocation lacks all three probe distributions")
            try:
                routed = edr.route(distributions["E"], distributions["H"], distributions["EH"])
            except (ValueError, TypeError, KeyError) as error:
                raise AllocationIntegrityError("Invalid saved probe distributions") from error
            expected = {**routed, "adaptive_independent": routed["independent"], "trigger": routed["independent"] > 0}
            rng = random_metadata(self.allocation_seed, self.dataset, qid, round_id, expected["trigger"])
            expected.update(independent=rng["selected_independent"] or 0,
                            discussion=8 - (rng["selected_independent"] or 0))
        if any(info.get(k) != v for k, v in expected.items()):
            raise AllocationIntegrityError("Saved allocation does not match the original gate/keyed draw")
        rng = random_metadata(self.allocation_seed, self.dataset, qid, round_id, expected["trigger"])
        if info.get("rng") != rng:
            raise AllocationIntegrityError("Saved random key/draw metadata differs")
        if payload is not None and info.get("solver_request_sha256") != digest(payload):
            raise AllocationIntegrityError("Saved allocation belongs to a different solver request")
        if inputs is not None and info.get("context_sha256") != digest(inputs):
            raise AllocationIntegrityError("Saved allocation belongs to different evidence/history")
        branch_hashes = info.get("branch_request_sha256")
        expected_branches = {"independent", "discussion"} if info["independent"] else set()
        if not isinstance(branch_hashes, dict) or set(branch_hashes) != expected_branches:
            raise AllocationIntegrityError("Saved branch request inventory differs")
        if payload is not None and inputs is not None:
            expected_hashes = {name: digest(request) for name, request in branch_payloads(payload, inputs, info["independent"])}
            if branch_hashes != expected_hashes:
                raise AllocationIntegrityError("Saved branch payloads differ from the original templates")
        for name in ("independent", "discussion"):
            for cache in folder.glob(f"round_{round_id}_solver_{name}*.json.gz"):
                receipt = read_json(cache)
                if name not in branch_hashes or digest(receipt.get("request")) != branch_hashes[name] or receipt.get("reused_from"):
                    raise AllocationIntegrityError("Saved branch cache changed/imported")
        hashes = info.get("probe_receipt_sha256")
        if not isinstance(hashes, dict) or set(hashes) - {"E", "H", "EH"}:
            raise AllocationIntegrityError("Invalid probe receipt inventory")
        for view in ("E", "H", "EH"):
            probe_path = folder / f"round_{round_id}_probe_{view}.json.gz"
            if probe_path.exists() != (view in hashes):
                raise AllocationIntegrityError("Probe receipt inventory changed after allocation")
            if view in hashes:
                receipt = read_json(probe_path)
                if digest(receipt) != hashes[view] or receipt.get("reused_from"):
                    raise AllocationIntegrityError("Saved probe receipt changed/imported")
                if view in distributions:
                    try:
                        observed = edr.distribution(receipt["response"], labels)
                    except Exception as error:
                        raise AllocationIntegrityError("Saved probe receipt does not support allocation") from error
                    if observed != distributions[view]:
                        raise AllocationIntegrityError("Saved probe distribution differs from receipt")
        if set(distributions) - set(hashes):
            raise AllocationIntegrityError("Distribution has no supporting local receipt")

    def validate_aggregate(self, receipt, info, folder):
        if digest(receipt.get("request")) != info["solver_request_sha256"] or receipt.get("reused_from"):
            raise AllocationIntegrityError("Aggregate request changed/imported")
        if info["independent"] > 0:
            if receipt.get("edr") != info or not receipt.get("derived"):
                raise AllocationIntegrityError("Merged receipt allocation metadata differs")
            choices = receipt.get("response", {}).get("choices", [])
            expected = ["independent"] * info["independent"] + ["discussion"] * info["discussion"]
            if len(choices) != 8 or [c.get("edr_channel") for c in choices] != expected:
                raise AllocationIntegrityError("Merged receipt candidate allocation differs")
            source_files = [f"round_{info['round']}_solver_{name}.json.gz" for name in ("independent", "discussion")]
            if receipt.get("source_files") != source_files:
                raise AllocationIntegrityError("Merged source inventory differs")
            responses = []
            for name, filename in zip(("independent", "discussion"), source_files):
                branch = read_json(folder / filename)
                if len(branch.get("response", {}).get("choices", [])) != info[name]:
                    raise AllocationIntegrityError("Saved branch candidate count differs")
                responses.append((name, branch["response"]))
            try:
                expected_response = merge_responses(responses)
            except (ValueError, TypeError, KeyError, RuntimeError) as error:
                raise AllocationIntegrityError("Saved branch responses cannot reproduce aggregate") from error
            if receipt.get("response") != expected_response:
                raise AllocationIntegrityError("Saved aggregate differs from its two local branches")

    def checked_call(self, path, url, payload):
        # TechnicalRetry can read these sibling files instead of the main cache.
        # Validate them before calling it so integrity errors cannot become retry
        # or fallback events. These are only files under the new output folder.
        stem = path.name.removesuffix(".json.gz")
        for name in (path.name, stem + ".failed_attempt0.json.gz", stem + ".retry1.json.gz"):
            cache = path.parent / name
            if cache.exists():
                receipt = read_json(cache)
                if receipt.get("request") != payload or receipt.get("reused_from"):
                    raise AllocationIntegrityError("Cached request differs or imports a different run")
        return self.original(path, url, payload)

    def routed_call(self, path, url, payload, round_id):
        context = self.context.get()
        if context is None or "inputs" not in context or payload.get("n") != 8:
            raise AllocationIntegrityError("Missing round context or incorrect candidate budget")
        if "seed" in payload:
            raise AllocationIntegrityError("Allocation seed must not become a model generation seed")
        inputs, labels, qid = context["inputs"], context["labels"], context["id"]
        metadata_path = path.parent / f"round_{round_id}_edr.json"
        if metadata_path.exists():
            info = read_json(metadata_path)
            self.validate_info(info, path.parent, qid, round_id, payload, inputs)
        else:
            if path.exists() or any(path.parent.glob(f"round_{round_id}_solver_*.json.gz")):
                raise AllocationIntegrityError("Prior solver attempt lacks allocation metadata")
            info = {"schema": SCHEMA, "round": round_id, "mode": self.mode,
                    "dataset": self.dataset, "question_id": qid,
                    "allocation_seed": self.allocation_seed,
                    "allocation_protocol_sha256": digest(self.allocation),
                    "labels": labels, "fallback": False,
                    "solver_request_sha256": digest(payload), "context_sha256": digest(inputs)}
            distributions = {}
            try:
                for view in ("E", "H", "EH"):
                    probe = edr.probe_payload(payload["model"], inputs, view, {k: self.label_ids[k] for k in labels})
                    raw = self.checked_call(path.parent / f"round_{round_id}_probe_{view}.json.gz", url, probe)
                    distributions[view] = edr.distribution(raw, labels)
                routed = edr.route(distributions["E"], distributions["H"], distributions["EH"])
                info.update(routed, adaptive_independent=routed["independent"], trigger=routed["independent"] > 0)
            except PrefixIntegrityError:
                raise
            except Exception as error:
                info.update(adaptive_independent=None, trigger=False, fallback=True,
                            failed_probe_view=view, error=f"{type(error).__name__}: {error}")
            rng = random_metadata(self.allocation_seed, self.dataset, qid, round_id, info["trigger"])
            info.update(rng=rng, independent=rng["selected_independent"] or 0,
                        discussion=8 - (rng["selected_independent"] or 0), distributions=distributions,
                        probe_receipt_sha256={view: digest(read_json(path.parent / f"round_{round_id}_probe_{view}.json.gz"))
                                              for view in ("E", "H", "EH")
                                              if (path.parent / f"round_{round_id}_probe_{view}.json.gz").exists()})
            info["branch_request_sha256"] = {name: digest(request) for name, request in branch_payloads(payload, inputs, info["independent"])}
            self.validate_info(info, path.parent, qid, round_id, payload, inputs)
            # Commit the allocation before launching either solver branch.
            self.base.write_json(metadata_path, info)
        if path.exists():
            receipt = read_json(path)
            self.validate_aggregate(receipt, info, path.parent)
            return self.checked_call(path, url, payload)
        m = info["independent"]
        if m == 0:
            return self.checked_call(path, url, payload)
        groups = branch_payloads(payload, inputs, m)

        def generate(group):
            name, request = group
            response = self.checked_call(path.parent / f"round_{round_id}_solver_{name}.json.gz", url, request)
            if len(response.get("choices", [])) != request["n"]:
                raise RuntimeError("Channel did not return requested candidate count")
            return name, response

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(generate, groups))
        merged = merge_responses(responses)
        self.base.save_receipt(path, {"request": payload, "response": merged, "derived": True,
                                     "edr": info, "source_files": [f"round_{round_id}_solver_{k}.json.gz" for k, _ in groups]})
        return merged
