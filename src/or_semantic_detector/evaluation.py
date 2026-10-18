"""Offline objective scoring; reference answers never enter inference."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from .data import read_jsonl


def correct(result, gold, tolerance=1e-4):
    if result.get("status") != "ok" or result.get("objective") is None or gold is None:
        return False
    try:
        a, b = float(result["objective"]), float(gold)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance * max(1.0, abs(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    refs = list(read_jsonl(args.references))
    preds = list(read_jsonl(args.predictions))
    if len({r["id"] for r in refs}) != len(refs) or len({r["id"] for r in preds}) != len(preds):
        raise ValueError("Duplicate IDs")
    p = {r["id"]: r for r in preds}
    if set(p) != {r["id"] for r in refs}:
        raise ValueError("Prediction/reference coverage mismatch")
    stats = defaultdict(lambda: dict(n=0, initial_correct=0, final_correct=0, fixed=0, broken=0))
    for row in refs:
        item = p[row["id"]]
        a = correct(item.get("initial", {}), row["objective"])
        b = correct(item.get("final", {}), row["objective"])
        for group in [row.get("dataset", "unspecified"), "overall"]:
            d = stats[group]
            d["n"] += 1
            d["initial_correct"] += a
            d["final_correct"] += b
            d["fixed"] += not a and b
            d["broken"] += a and not b
    for d in stats.values():
        d["accuracy"] = d["final_correct"] / d["n"]
    args.output.write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
