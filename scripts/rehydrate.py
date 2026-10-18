#!/usr/bin/env python
"""Restore upstream problem text into the FaithOR-Supervision files.

The published files replace every problem statement that derives from OptMATH-Train or
ReSocratic-29k with a placeholder of the form ``[[UPSTREAM:<dataset>:<sha256>]]`` and
record, per row, which upstream record it came from.  Download the two upstream datasets
yourself (their terms apply), then run::

    python scripts/rehydrate.py --optmath /path/to/OptMATH-Train.jsonl \
        --resocratic /path/to/resocratic-29k.json --data-dir data --out-dir data_rehydrated

``--optmath`` accepts the JSONL export of ``Aurora-Gem/OptMATH-Train`` (fields
``instruction``, ``input``, ``output``, ``answer``) or a directory produced by
``datasets.load_dataset(...).save_to_disk``; ``--resocratic`` accepts the ReSocratic-29k
JSON list (field ``question``).  The script verifies every restored file against
``REHYDRATED_SHA256SUMS`` so you know the result is byte-identical to the files the
detector was trained on.  Simulated (``procedural/*``) rows carry their own text and are
copied unchanged.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PLACEHOLDER = re.compile(r"\[\[UPSTREAM:(OptMATH-Train|ReSocratic-29k):([0-9a-f]{64})\]\]")
NOTE_MARKERS = ["\n# Note", "\n## Note", "\n```python", "\nNote:"]


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def optmath_question(raw_input):
    """Problem statement as used in FaithOR: drop the '# Question:' header and the trailing code note."""
    text = raw_input
    if text.startswith("# Question:"):
        text = text[len("# Question:") :]
    for marker in NOTE_MARKERS:
        i = text.find(marker)
        if i > 0:
            text = text[:i]
    return text.strip()


def load_optmath(path):
    path = Path(path)
    rows = []
    if path.is_dir():
        from datasets import load_from_disk

        rows = [r["input"] for r in load_from_disk(str(path))]
    else:
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line)["input"] for line in f if line.strip()]
    table = {}
    for raw in rows:
        table.setdefault(sha(raw), optmath_question(raw))
    return table


def load_resocratic(path):
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    table = {}
    for r in data:
        raw = r["question"]
        table.setdefault(sha(raw), raw.strip())
    return table


def fill(text, tables, missing):
    def repl(m):
        ds, digest = m.group(1), m.group(2)
        q = tables[ds].get(digest)
        if q is None:
            missing.add((ds, digest))
            return m.group(0)
        return q

    return PLACEHOLDER.sub(repl, text)


def rehydrate_jsonl(src, dst, tables, missing):
    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            if r.get("upstream"):
                r["question"] = fill(r["question"], tables, missing)
                r["prompt"] = fill(r["prompt"], tables, missing)
                r.pop("upstream")
            g.write(json.dumps(r, ensure_ascii=False) + "\n")


def rehydrate_parquet(src, dst, tables, missing):
    import pandas as pd

    df = pd.read_parquet(src)
    prompts = []
    for p in df["prompt"]:
        p = [dict(m) for m in p]
        for m in p:
            m["content"] = fill(m["content"], tables, missing)
        prompts.append(p)
    df["prompt"] = prompts
    if "upstream" in df.columns:
        df = df.drop(columns=["upstream"])
    df.to_parquet(dst, index=False)


def content_digest(path):
    """Byte digest for JSONL; for Parquet a digest of the canonical row content, so that the check
    does not depend on the Parquet writer version."""
    path = Path(path)
    if path.suffix != ".parquet":
        return hashlib.sha256(path.read_bytes()).hexdigest()
    import pandas as pd

    df = pd.read_parquet(path)
    h = hashlib.sha256()
    for rec in df.to_dict("records"):
        h.update(json.dumps(rec, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--optmath", required=True)
    ap.add_argument("--resocratic", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="data_rehydrated")
    a = ap.parse_args()
    tables = {"OptMATH-Train": load_optmath(a.optmath), "ReSocratic-29k": load_resocratic(a.resocratic)}
    print({k: len(v) for k, v in tables.items()}, "upstream rows indexed", flush=True)
    src_dir, out_dir = Path(a.data_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = set()
    expected = {}
    sums = src_dir / "REHYDRATED_SHA256SUMS"
    if sums.exists():
        for line in sums.read_text().splitlines():
            h, name = line.split("  ", 1)
            expected[name] = h
    ok = True
    for src in sorted(src_dir.iterdir()):
        if src.name in ("REHYDRATED_SHA256SUMS",):
            continue
        dst = out_dir / src.name
        if src.suffix == ".jsonl":
            rehydrate_jsonl(src, dst, tables, missing)
        elif src.suffix == ".parquet":
            rehydrate_parquet(src, dst, tables, missing)
        else:
            dst.write_bytes(src.read_bytes())
            continue
        digest = content_digest(dst)
        status = "unchecked"
        if src.name in expected:
            status = "OK" if digest == expected[src.name] else "MISMATCH"
            ok &= status == "OK"
        print(f"{src.name}: {status}", flush=True)
    if missing:
        ok = False
        print(f"{len(missing)} upstream records not found; is the upstream file complete and unmodified?", file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
