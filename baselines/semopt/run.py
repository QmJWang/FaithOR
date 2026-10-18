"""Run reward-selected search without loading reference answers."""

import argparse
import json
from pathlib import Path
from or_semantic_detector.client import Client
from or_semantic_detector.data import read_jsonl
from .search import Search
from .reward import reward_text


class Reward:
    def __init__(self, checkpoint, device="cuda", max_length=8192):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
        )

    def __call__(self, problem, code):
        x = self.tokenizer(reward_text(problem, code), return_tensors="pt", truncation=False)
        if x.input_ids.shape[-1] > self.max_length:
            raise ValueError("Reward input exceeds context budget")
        with self.torch.inference_mode():
            score = self.model(**x.to(self.device)).logits.flatten()[0].sigmoid()
        if not self.torch.isfinite(score):
            raise ValueError("Non-finite reward")
        return score.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reward", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--url", default="http://localhost:8001/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--breadth", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--reward-max-length", type=int, default=8192)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    client = Client(
        args.url,
        args.model,
        args.out / "usage.jsonl",
        temperature=0.9,
        max_tokens=args.max_tokens,
    )
    reward = Reward(args.reward, args.device, args.reward_max_length)
    with (args.out / "predictions.jsonl").open("w") as f:
        for i, row in enumerate(read_jsonl(args.input)):
            try:
                result = Search(
                    client, reward, args.out / str(i), args.simulations, args.breadth
                ).run(row["question"])
            except Exception as e:
                result = {"error": type(e).__name__, "message": str(e)}
            f.write(json.dumps({"id": row["id"], **result}) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
