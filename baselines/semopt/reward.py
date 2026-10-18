"""Scalar semantic reward and pairwise Bradley-Terry training."""

import argparse
import json
from pathlib import Path


def reward_text(problem, code):
    return "Problem:\n" + problem + "\n\nSolver code:\n" + code + "\n\nSemantic faithfulness:"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    torch.manual_seed(42)
    args.out.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=1, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.config.pad_token_id = tok.pad_token_id
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rows = [json.loads(line) for line in args.pairs.open() if line.strip()]
    if not rows:
        raise ValueError("Empty preference dataset")
    model.train()
    for epoch in range(args.epochs):
        for row in rows:
            text = [reward_text(row["problem"], row[k]) for k in ["chosen", "rejected"]]
            encoded = tok(text, padding=True, return_tensors="pt", truncation=False)
            if encoded.input_ids.shape[-1] > args.max_length:
                raise ValueError("Reward input exceeds context budget")
            scores = model(**encoded.to("cuda")).logits.flatten().float()
            loss = -torch.nn.functional.logsigmoid(scores[0] - scores[1])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(json.dumps({"epoch": epoch + 1, "last_loss": loss.item()}), flush=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
