"""Standard and CoT code-generation baselines with solver status checks."""

import argparse
import json
from pathlib import Path
from or_semantic_detector.client import Client
from or_semantic_detector.data import read_jsonl
from or_semantic_detector.prompts import STANDARD, COT, extract_code
from .semopt.common import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["standard", "cot"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default="http://localhost:8001/v1")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    client = Client(args.url, args.model, args.out / "usage.jsonl", max_tokens=12288)
    with (args.out / "predictions.jsonl").open("w") as stream:
        for i, row in enumerate(read_jsonl(args.input)):
            template = STANDARD if args.method == "standard" else COT
            prompt = template.format(problem=row["question"], problem_description=row["question"])
            try:
                raw = client.call(prompt, args.method)
                result = execute(extract_code(raw), args.out / str(i))
                status = "ok" if result["ok"] else str(result["status"])
                final = {"status": status, "objective": result["objective"]}
                record = {"id": row["id"], "initial": final, "final": final, "raw": raw}
            except Exception as e:
                record = {"id": row["id"], "error": type(e).__name__}
            stream.write(json.dumps(record) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
