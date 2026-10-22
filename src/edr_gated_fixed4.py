"""Evidence-discussion routing; no changes to the baseline source files."""
import copy
import math
import re
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar

from liquid import Template


INDEPENDENT_TEMPLATE = Template('''Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

### Documents
{{documents}}

---

The above are the question and some relevant documents. Please analyze and then re-answer. Please give your response towards **lower entropy**. Give your analysis and final answer.''')

PROBE_SYSTEM = "Answer the multiple-choice question using the supplied information. Output exactly one capital option letter, with no explanation."


def distribution(raw, labels):
    choices = raw.get("choices", [])
    if len(choices) != 1:
        raise ValueError("Probe must have one choice")
    choice = choices[0]
    if choice.get("finish_reason") not in ("stop", "length"):
        raise ValueError("Invalid probe finish reason")
    message = choice.get("message", {})
    if message.get("reasoning") or message.get("reasoning_content"):
        raise ValueError("Probe unexpectedly generated reasoning")
    positions = choice.get("logprobs", {}).get("content") or []
    if len(positions) != 1:
        raise ValueError("Probe must score exactly one generated position")
    scores = {}
    for entry in positions[0].get("top_logprobs", []):
        token = entry["token"]
        if token in labels:
            if token in scores or not math.isfinite(entry["logprob"]):
                raise ValueError("Duplicate or nonfinite label score")
            scores[token] = entry["logprob"]
    if set(scores) != set(labels):
        raise ValueError(f"Missing exact label scores: {set(labels) - set(scores)}")
    if (message.get("content") or "").strip() not in labels:
        raise ValueError("Generated token is not a legal label")
    peak = max(scores.values())
    weights = {k: math.exp(scores[k] - peak) for k in labels}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def route(e, h, eh):
    labels = sorted(e)
    if len(labels) < 2 or set(h) != set(e) or set(eh) != set(e):
        raise ValueError("Mismatched label distributions")
    for p in (e, h, eh):
        if any(not math.isfinite(v) or v < 0 for v in p.values()) or not math.isclose(sum(p.values()), 1, abs_tol=1e-6):
            raise ValueError("Invalid probability distribution")
    tv_e = sum(abs(eh[k] - e[k]) for k in labels) / 2
    tv_h = sum(abs(eh[k] - h[k]) for k in labels) / 2
    s = max(0.0, tv_e - tv_h)
    top_e = sorted(labels, key=lambda k: (-e[k], k))
    top_eh = sorted(labels, key=lambda k: (-eh[k], k))
    tied = e[top_e[0]] == e[top_e[1]] or eh[top_eh[0]] == eh[top_eh[1]]
    m = 0 if tied or top_e[0] == top_eh[0] else min(6, math.floor(6 * s + 0.5))
    return {"independent": m, "discussion": 8 - m, "s": s, "tv_e": tv_e, "tv_h": tv_h,
            "top_e": top_e[0], "top_eh": top_eh[0], "top_tie": tied}


def probe_payload(model, inputs, view, label_ids):
    blocks = ["### Question\n" + inputs["question"], "### Options\n" + inputs["options"]]
    if view in ("E", "EH"):
        blocks.append("### Documents\n" + inputs["documents"])
    if view in ("H", "EH"):
        blocks.append("### Previous answers\n" + inputs["answers"])
    blocks.append("Answer with one option letter:")
    return {"model": model, "messages": [{"role": "system", "content": PROBE_SYSTEM},
            {"role": "user", "content": "\n\n".join(blocks)}], "temperature": 1.0,
            "max_completion_tokens": 1, "n": 1, "logprobs": True, "top_logprobs": 20,
            "allowed_token_ids": list(label_ids.values()), "logprob_token_ids": list(label_ids.values()),
            "chat_template_kwargs": {"enable_thinking": False}}


class CapturePrompt:
    def __init__(self, original, context):
        self.original, self.context = original, context

    def render(self, **kwargs):
        self.context["inputs"] = dict(kwargs)
        return self.original.render(**kwargs)


class Plugin:
    def __init__(self, baseline, label_ids, source_map=None, mode="edr"):
        if mode not in ("edr", "off", "fixed4", "gated_fixed4"):
            raise ValueError(mode)
        self.base, self.label_ids, self.mode = baseline, label_ids, mode
        self.original = baseline.api_call
        self.source_map = source_map or {}
        self.context = ContextVar("edr_question", default=None)

    def install(self):
        self.base.api_call = self.api_call

    def question_run(self, item, args, prompts, output):
        context = {"id": item["id"], "labels": sorted(k for k, v in item["option"].items() if v.strip())}
        token = self.context.set(context)
        local_prompts = dict(prompts)
        local_prompts["user_prompt_round"] = CapturePrompt(prompts["user_prompt_round"], context)
        try:
            return self.base.question_run(item, args, local_prompts, output)
        finally:
            self.context.reset(token)

    def api_call(self, path, url, payload, source_path=None):
        match = re.fullmatch(r"round_(\d+)_solver.json.gz", path.name)
        if match and int(match[1]) > 1 and self.mode != "off":
            return self.routed_call(path, url, payload, int(match[1]))
        # Share only the unmodified first-round prefix, never import completed answers.
        source = None
        if path.name.startswith("round_1_"):
            source_root = self.source_map.get(path.parent.name)
            if source_root is not None:
                source = source_root / path.name
        return self.original(path, url, payload, source_path=source)

    def routed_call(self, path, url, payload, round_id):
        context = self.context.get()
        if context is None or "inputs" not in context or payload["n"] != 8:
            raise RuntimeError("Missing round context or incorrect solver budget")
        if path.exists():
            return self.original(path, url, payload)
        inputs = context["inputs"]
        labels = context["labels"]
        info = {"round": round_id, "mode": self.mode, "labels": labels, "fallback": False}
        if self.mode == "fixed4":
            info.update(independent=4, discussion=4)
        else:
            distributions = {}
            try:
                for view in ("E", "H", "EH"):
                    probe = probe_payload(payload["model"], inputs, view, {k: self.label_ids[k] for k in labels})
                    raw = self.original(path.parent / f"round_{round_id}_probe_{view}.json.gz", url, probe)
                    distributions[view] = distribution(raw, labels)
                info.update(route(distributions["E"], distributions["H"], distributions["EH"]))
                if self.mode == "gated_fixed4":
                    adaptive = info["independent"]
                    info.update(adaptive_independent=adaptive, trigger=adaptive > 0,
                                independent=4 if adaptive > 0 else 0,
                                discussion=4 if adaptive > 0 else 8)
            except Exception as error:
                info.update(independent=0, discussion=8, fallback=True, error=f"{type(error).__name__}: {error}")
            info["distributions"] = distributions
        self.base.write_json(path.parent / f"round_{round_id}_edr.json", info)
        m = info["independent"]
        if m == 0:
            return self.original(path, url, payload)
        independent = copy.deepcopy(payload)
        independent["n"] = m
        independent["messages"][1]["content"] = INDEPENDENT_TEMPLATE.render(**inputs)
        discussion = copy.deepcopy(payload)
        discussion["n"] = 8 - m
        groups = [("independent", independent), ("discussion", discussion)]

        def generate(group):
            name, request = group
            response = self.original(path.parent / f"round_{round_id}_solver_{name}.json.gz", url, request)
            if len(response.get("choices", [])) != request["n"]:
                raise RuntimeError("Channel did not return requested candidate count")
            return name, response

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(generate, groups))
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
        self.base.save_receipt(path, {"request": payload, "response": merged, "derived": True,
                                     "edr": info, "source_files": [f"round_{round_id}_solver_{k}.json.gz" for k, _ in groups]})
        return merged
