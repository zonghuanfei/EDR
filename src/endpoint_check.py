"""Small option-probability preflight; no benchmark questions are sent."""
import argparse
import json

from edr import distribution, probe_payload


def label_token_ids(tokenizer_path, labels):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    encoded = {label: tokenizer.encode(label, add_special_tokens=False) for label in labels}
    if any(len(tokens) != 1 for tokens in encoded.values()):
        raise ValueError("Every option letter must be one token in the served model's tokenizer")
    if len({tokens[0] for tokens in encoded.values()}) != len(encoded):
        raise ValueError("Option letters must have distinct token IDs")
    return {label: tokens[0] for label, tokens in encoded.items()}


def check_endpoint(base_url, model, label_ids):
    import requests

    inputs = {"question": "Which organ filters blood?", "options": "\n".join(
        f"{label}. {['Kidney', 'Skin', 'Hair', 'Nail'][i] if i < 4 else 'Other'}"
        for i, label in enumerate(label_ids)),
        "documents": "The kidneys filter blood.", "answers": "A prior answer selected B."}
    results = {}
    for view in ("E", "H", "EH"):
        payload = probe_payload(model, inputs, view, label_ids)
        response = requests.post(base_url.rstrip("/") + "/chat/completions", json=payload,
                                 headers={"Authorization": "Bearer local"}, timeout=(30, 120))
        response.raise_for_status()
        results[view] = distribution(response.json(), list(label_ids))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--labels", default="ABCDEFGHIJ")
    args = parser.parse_args()
    if len(set(args.labels)) != len(args.labels) or len(args.labels) < 2:
        parser.error("--labels must contain distinct capital option letters")
    ids = label_token_ids(args.tokenizer, args.labels)
    result = check_endpoint(args.base_url, args.model, ids)
    print(json.dumps({"status": "PASS", "label_token_ids": ids, "distributions": result}))


if __name__ == "__main__":
    main()
