"""Validate exported supervision and lexical split separation without a model."""

import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq
from or_semantic_detector.data import read_jsonl, masked_key, question
from or_semantic_detector.protocol import parse_out


def verify(root):
    splits = {}
    counts = {}
    for name in ("train", "validation", "holdout"):
        rows = list(read_jsonl(root / "data" / f"sft_{name}.jsonl"))
        assert len({r["id"] for r in rows}) == len(rows), f"Duplicate IDs: {name}"
        assert len({r["prompt"] for r in rows}) == len(rows), f"Duplicate prompts: {name}"
        for row in rows:
            pred = parse_out(row["target"], row["comp_ids"])
            assert pred is not None, row["id"]
            assert sorted(pred["errors"]) == sorted((e["id"], e["type"]) for e in row["errors"])
            assert pred["missing"] == row["missing"]
            assert question(row["prompt"]) == row["question"]
        splits["sft_" + name] = {masked_key(r["question"]) for r in rows}
        counts["sft_" + name] = len(rows)
    for name in ("train", "validation"):
        rows = pq.read_table(root / "data" / f"grpo_{name}.parquet").to_pylist()
        keys = set()
        for row in rows:
            gold = json.loads(row["extra_info"]["gold"])
            assert len(gold["comp_ids"]) == len(gold["comp_labels"]) == len(gold["comp_types"])
            keys.add(masked_key(question(row["prompt"][0]["content"])))
        splits["grpo_" + name] = keys
        counts["grpo_" + name] = len(rows)
    training = splits["sft_train"] | splits["grpo_train"]
    for name in ("sft_validation", "sft_holdout", "grpo_validation"):
        assert not training & splits[name], f"Training overlap: {name}"
    assert not splits["sft_validation"] & splits["sft_holdout"], "SFT evaluation overlap"
    return {
        "rows": counts,
        "checks": "passed",
        "scope": "Label/schema integrity and number-masked lexical separation; not semantic independence or benchmark decontamination.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
