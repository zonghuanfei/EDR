"""Mask only solver history; preserve host queries, evidence, votes and stopping."""
import gzip
import hashlib
import json
import re
import threading
from contextvars import ContextVar

from madmm_objective import generation_score, objective_mask


EVIDENCE_ONLY = '''Below is a multiple-choice question.
### Question
{question}

### Options
{options}

### Documents
{documents}

---

The above are the question and some relevant documents. Please analyze and then re-answer. Please give your response towards **lower entropy**. Give your analysis and final answer.'''


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def receipt(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


class MaskPrompt:
    def __init__(self, plugin, original, context):
        self.plugin, self.original, self.context = plugin, original, context

    def render(self, **values):
        return self.plugin.render_masked(self.original, self.context, values)


class Plugin:
    def __init__(self, base, sources, request_limit=24):
        self.base, self.sources = base, sources
        self.original = base.api_call
        self.original_parser = base.parse_queries
        self.context = ContextVar("madmm_objective_question", default=None)
        self.capacity = threading.BoundedSemaphore(request_limit)

    def install(self):
        self.base.api_call = self.api_call
        self.base.parse_queries = self.parse_queries

    def question_run(self, item, args, prompts, output):
        context = {"id": item["id"], "folder": output / f"question_{item['id']}"}
        token = self.context.set(context)
        local = dict(prompts)
        local["user_prompt_round"] = MaskPrompt(self, prompts["user_prompt_round"], context)
        try:
            return self.base.question_run(item, args, local, output)
        finally:
            self.context.reset(token)

    def api_call(self, path, url, payload, source_path=None):
        context = self.context.get()
        source = None
        if path.name.startswith("round_1_"):
            source = self.sources[path.parent.name] / path.name
            if not source.exists() or not self.base.reuse_reason(receipt(source), payload):
                raise RuntimeError(f"Frozen first-round request is incompatible: {source}")
        with self.capacity:
            raw = self.original(path, url, payload, source_path=source)
        match = re.fullmatch(r"round_(\d+)_solver.json.gz", path.name)
        if match:
            choices = sorted(raw["choices"], key=lambda choice: choice["index"])
            if [c["index"] for c in choices] != list(range(8)) or payload["n"] != 8:
                raise RuntimeError("Solver must return exactly eight indexed responses")
            scores = [generation_score(choice) for choice in choices]
            context.update(last_round=int(match[1]), last_choices=choices, last_scores=scores)
            self.base.write_json(path.parent / f"round_{match[1]}_madmm_scores.json",
                                 {"round": int(match[1]), "agent_scores": scores})
        if path.name == "round_1_query.json.gz":
            context["frozen_query_root"] = self.sources[path.parent.name]
        return raw

    def parse_queries(self, choice):
        queries, guard = self.original_parser(choice)
        context = self.context.get()
        source = context.pop("frozen_query_root", None) if context else None
        if source is None:
            return queries, guard
        transition = read(source / "round_1_transition.json")
        original_choice = receipt(source / "round_1_query.json.gz")["response"]["choices"][0]
        if choice != original_choice or set(queries) != set(transition["queries"]):
            raise RuntimeError("Frozen first-query material differs")
        return list(transition["queries"]), guard

    def render_masked(self, original, context, values):
        previous_round = context["last_round"]
        folder = context["folder"]
        prior_record = read(folder / f"round_{previous_round}.json")
        transition = read(folder / f"round_{previous_round}_transition.json")
        order = transition["history_order"]
        if sorted(order) != list(range(8)):
            raise RuntimeError("Invalid host history order")
        answers = [self.base.complete_choice(choice) for choice in context["last_choices"]]
        chunks = [f"{i}. The assistant's previous answer is (Entropy {prior_record['entropies'][j]:.2f}):\n{answers[j]}"
                  for i, j in enumerate(order, 1)]
        if "\n\n".join(chunks) != values["answers"]:
            raise RuntimeError("Host history does not match immutable previous answers")
        scores = [record["score"] for record in context["last_scores"]]
        mask, threshold = objective_mask(scores)
        visible = [j for j in order if mask[j]]
        kept = "\n\n".join(chunk for chunk, j in zip(chunks, order) if mask[j])
        info = {"round": previous_round + 1, "source_round": previous_round,
                "method": "MAD-M2 objective official-code selector",
                "score": scores, "agent_scores": context["last_scores"], "threshold": threshold,
                "mask": mask, "kept_indices": [i for i, flag in enumerate(mask) if flag],
                "removed_indices": [i for i, flag in enumerate(mask) if not flag],
                "original_history_order": order, "visible_indices": visible,
                "retained_count": len(visible), "all_eight_agents_share_mask": True,
                "branch": "filtered_history" if visible else "evidence_only",
                "original_history_sha256": hashlib.sha256(values["answers"].encode()).hexdigest(),
                "visible_history_sha256": hashlib.sha256(kept.encode()).hexdigest(),
                "evidence_sha256": hashlib.sha256(values["documents"].encode()).hexdigest()}
        if not visible:
            prompt = EVIDENCE_ONLY.format(**values)
        else:
            prompt = original.render(**dict(values, answers=kept))
        info["rendered_solver_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        self.base.write_json(folder / f"round_{previous_round + 1}_madmm.json", info)
        return prompt
