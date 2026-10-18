"""Normalize and deduplicate supervision without exposing targets to inference."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from .protocol import parse_out


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def question(prompt):
    if "### Problem\n" not in prompt:
        raise ValueError("Missing problem section")
    return prompt.split("### Problem\n", 1)[1].split("\n### Solver result", 1)[0].strip()


def question_key(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.]", " ", text.casefold())).strip()


def masked_key(text):
    return re.sub(r"\d+(?:\.\d+)?", "NUMBER", question_key(text))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def normalize(row):
    prompt = row["prompt"]
    model = prompt.split("### Model", 1)[-1].split("### Task", 1)[0]
    ids = list(dict.fromkeys(re.findall(r"\[([SPVOC]\d+)\]", model)))
    parsed = parse_out(row["target"], ids)
    if parsed is None:
        raise ValueError("Invalid diagnostic supervision")
    q = question(prompt)
    core = {
        "errors": [{"id": i, "type": t} for i, t in parsed["errors"]],
        "missing": parsed["missing"],
    }
    canonical = json.dumps(
        {"errors": sorted(parsed["errors"]), "missing": parsed["missing"]},
        sort_keys=True,
    )
    source = row.get("source", "")
    if source not in {"optmath", "resocratic", "model_generated"} and not source.startswith(
        "procedural/"
    ):
        sid = row.get("sid", "")
        source = (
            "optmath"
            if sid.startswith("optmath")
            else "resocratic"
            if sid.startswith("reso")
            else "model_generated"
        )
    return {
        "id": digest(prompt + "\n" + canonical),
        "question_id": digest(question_key(q)),
        "source": source,
        "family": row.get("family", "natural_or_mutated"),
        "question": q,
        "prompt": prompt,
        "comp_ids": ids,
        "target": row["target"],
        "errors": core["errors"],
        "missing": core["missing"],
    }


def consolidate(paths):
    unique = {}
    labels = {}
    conflicts = set()
    total = 0
    for path in paths:
        for row in read_jsonl(path):
            item = normalize(row)
            total += 1
            key = digest(item["prompt"])
            if key in labels and labels[key] != item["id"]:
                conflicts.add(key)
            labels[key] = item["id"]
            unique.setdefault(item["id"], item)
    kept = [v for v in unique.values() if digest(v["prompt"]) not in conflicts]
    return sorted(kept, key=lambda r: r["id"]), {
        "input_rows": total,
        "unique_rows": len(unique),
        "conflicting_prompts_removed": len(conflicts),
        "output_rows": len(kept),
    }


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--holdout", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    seen = set()
    report = {}
    for split, paths in [
        ("train", args.train),
        ("validation", args.validation),
        ("holdout", args.holdout),
    ]:
        rows, stats = consolidate(paths)
        clean = [r for r in rows if masked_key(r["question"]) not in seen]
        stats["cross_split_rows_removed"] = len(rows) - len(clean)
        seen.update(masked_key(r["question"]) for r in clean)
        stats["output_rows"] = len(clean)
        stats["sources"] = dict(Counter(r["source"] for r in clean))
        write_jsonl(args.out / f"{split}.jsonl", clean)
        report[split] = stats
    (args.out / "statistics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
