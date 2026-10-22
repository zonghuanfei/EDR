"""Fetch pinned MIRAGE tests; only normalize the public dataset fields."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

COMMIT = "392943af99cd94cafd50a0de2e7fca24bbf65494"
URL = f"https://raw.githubusercontent.com/gzxiong/MIRAGE/{COMMIT}/benchmark.json"
SPECS = {"mmlu": (1089, "ABCD")}


def normalize(name, source):
    count, labels = SPECS[name]
    if len(source) != count:
        raise ValueError(f"Unexpected source size for {name}: {len(source)}")
    rows = []
    for source_id, item in source.items():
        assert set(item["options"]) == set(labels)
        assert item["answer"] in labels and item["question"].strip()
        rows.append({"idx": source_id, "question": item["question"],
                     "options": {k: item["options"][k] for k in labels},
                     "answer_idx": item["answer"]})
    assert len({r["idx"] for r in rows}) == count
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output
    target.mkdir(parents=True, exist_ok=True)
    source_path = target / "mirage_benchmark.json"
    if source_path.exists():
        raw = source_path.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=120) as response:
            raw = response.read()
        source_path.write_bytes(raw)
    source = json.loads(raw)
    manifest = {"repository": "https://github.com/gzxiong/MIRAGE", "commit": COMMIT,
                "download_url": URL, "source_sha256": hashlib.sha256(raw).hexdigest(),
                "selection": "Complete official MMLU-Med test; original order; no filtering",
                "normalization": "question/options/answer to question/options/answer_idx; retain original source id",
                "retrieval": "Question only; no gold answers, PMIDs or reference snippets in model inputs",
                "datasets": {}}
    for name in SPECS:
        rows = normalize(name, source[name])
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        path = target / (name + ".jsonl")
        if path.exists() and path.read_bytes() != content:
            raise RuntimeError(f"Refusing to replace differing existing dataset: {path}")
        path.write_bytes(content)
        manifest["datasets"][name] = {"file": path.name, "n": len(rows),
            "labels": list(SPECS[name][1]), "sha256": hashlib.sha256(content).hexdigest()}
    path = target / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("Existing dataset manifest does not match pinned source")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
