"""Generate independent semantic-contract examples with solver checks."""

import argparse
import json
import random
from pathlib import Path
from or_semantic_detector import simulation as sim
from or_semantic_detector.data import normalize, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups-per-family", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "val", "holdout"], default="train")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = []
    if args.out.exists():
        raise FileExistsError(args.out)
    for name in [
        "full_partial",
        "role_binding",
        "bounds_scope",
        "ratio_contract",
        "prerequisites",
        "domain_group",
        "scope_group",
    ]:
        generator = getattr(sim, name)
        for i in range(args.groups_per_family):
            for row in generator(
                rng, f"simulation_{args.split}_{args.seed}_{name}_{i}", args.split, i
            ):
                rows.append(normalize(row))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    print(json.dumps({"rows": len(rows), "seed": args.seed, "split": args.split}))


if __name__ == "__main__":
    main()
