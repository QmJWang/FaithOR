"""Initial mathematical draft and execution debugging, without a detector."""

import argparse
import json
from pathlib import Path
from or_semantic_detector.client import Client
from or_semantic_detector.data import read_jsonl
from or_semantic_detector.modeling import form_prompt_v51, extract_v5
from or_semantic_detector.pipeline import generate_and_solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--writer-url", default="http://localhost:8001/v1")
    parser.add_argument("--writer-model", required=True)
    parser.add_argument("--coder-url", default="http://localhost:8001/v1")
    parser.add_argument("--coder-model", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    writer = Client(args.writer_url, args.writer_model, args.out / "usage.jsonl", max_tokens=12288)
    coder = Client(args.coder_url, args.coder_model, args.out / "usage.jsonl", max_tokens=12288)
    with (args.out / "predictions.jsonl").open("w") as stream:
        for i, row in enumerate(read_jsonl(args.input)):
            try:
                model = extract_v5(writer.call(form_prompt_v51(row["question"]), "initial_math"))
                result = generate_and_solve(coder, model, args.out / str(i))
                record = {
                    "id": row["id"],
                    "initial": result,
                    "final": result,
                    "initial_model": model,
                }
            except Exception as error:
                record = {"id": row["id"], "error": type(error).__name__}
            stream.write(json.dumps(record) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
