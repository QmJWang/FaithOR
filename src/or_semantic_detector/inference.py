"""Run the frozen diagnostic protocol against a model-serving endpoint."""

import argparse
import json
from pathlib import Path
from .client import Client
from .data import read_jsonl
from .protocol import gen_prompt, parse_out


RETRY_NOTE = (
    "\nYour previous answer was not a valid diagnosis. End with exactly one JSON object line."
)


def diagnose(client, text, comp_ids):
    """Two bounded attempts; any transport failure or invalid output is an abstention.

    The second attempt appends a short format reminder, because a greedy decoder
    would otherwise reproduce the first invalid answer verbatim.
    """
    prompt = gen_prompt(text)
    attempts = []
    error = None
    for attempt in range(2):
        current = prompt if attempt == 0 else prompt + RETRY_NOTE
        try:
            raw = client.call(current, "detector", max_tokens=512)
        except Exception as failure:  # truncation, transport or protocol failure
            error = type(failure).__name__ + ": " + str(failure)
            attempts.append(None)
            continue
        attempts.append(raw)
        parsed = parse_out(raw, comp_ids)
        if parsed is not None:
            return {
                "valid": True,
                "diagnosis": parsed,
                "raw": raw,
                "attempts": attempts,
            }
    return {
        "valid": False,
        "diagnosis": None,
        "raw": attempts[-1],
        "attempts": attempts,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = Client(args.url, args.model, args.output.with_suffix(".usage.jsonl"))
    with args.output.open("w") as stream:
        for row in read_jsonl(args.input):
            # Only these inference fields are read; supervision is never forwarded.
            result = diagnose(client, row["text"], row["comp_ids"])
            stream.write(json.dumps({"id": row.get("id", row.get("sid")), **result}) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
