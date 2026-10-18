"""Response-only LoRA SFT with optional decision-token weighting."""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path


def read(p):
    return [json.loads(line) for line in p.open()]


def sample_weight(source, table):
    """Exact key first, then longest matching 'prefix/*' key, else 1.0."""
    if source in table:
        return float(table[source])
    best = None
    for key, val in table.items():
        if key.endswith("*") and source.startswith(key[:-1]):
            if best is None or len(key) > len(best[0]):
                best = (key, float(val))
    return best[1] if best else 1.0


def accum_multipliers(weights, accum):
    """Per-sample loss multipliers so that each gradient-accumulation window applies
    sum_i s_i * loss_i / sum_i s_i over the samples in that window.  With all weights 1
    this equals 1 / window_size, identical to the unweighted implementation."""
    out = []
    for start in range(0, len(weights), accum):
        window = weights[start : start + accum]
        total = float(sum(window))
        assert total > 0
        out.extend(w / total for w in window)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=2400)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter", default="adapter")
    ap.add_argument("--merge-name", default="merged")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--memory-fraction", type=float, default=0.9)
    ap.add_argument("--choice-weight", type=float, default=1.0)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument(
        "--sample-weights",
        default="",
        help='JSON {source_prefix: weight}; "procedural/*" matches sources starting with "procedural/"; default 1',
    )
    ap.add_argument(
        "--sources",
        default="",
        help='Comma-separated source prefixes to keep (e.g. "optmath,resocratic" or "procedural/"); default keeps every row',
    )
    ap.add_argument(
        "--include-replay",
        action="store_true",
        help="Also keep rows with replay=true (writer-generated formulations replayed in the contract stage)",
    )
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    import torch

    tok = AutoTokenizer.from_pretrained(a.base)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.merge_only:
        assert (a.out / a.adapter / "adapter_config.json").exists()
        m = AutoModelForCausalLM.from_pretrained(
            a.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cpu"
        )
        m = PeftModel.from_pretrained(m, str(a.out / a.adapter)).merge_and_unload()
        m.save_pretrained(a.out / a.merge_name, safe_serialization=True)
        tok.save_pretrained(a.out / a.merge_name)
        print("MERGE_COMPLETE", flush=True)
        return
    from or_semantic_detector.loss import weights_for_response, weighted_response_loss

    def enc(r):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        pi = tok(prompt, add_special_tokens=False)["input_ids"]
        encoded = tok(
            r["target"] + "<|im_end|>\n",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ti = encoded["input_ids"]
        w = weights_for_response(
            r["target"], encoded["offset_mapping"], a.choice_weight, a.value_weight
        )
        assert len(ti) == len(w)
        return pi, ti, w

    mixed = read(a.data)
    if a.sources:
        prefixes = [x.strip() for x in a.sources.split(",") if x.strip()]
        mixed = [
            r
            for r in mixed
            if any(r.get("source", "unspecified").startswith(x) for x in prefixes)
            or (a.include_replay and r.get("replay"))
        ]
        assert mixed, "No rows left after --sources filtering"
    weight_table = json.loads(a.sample_weights) if a.sample_weights else {}
    assert all(float(v) > 0 for v in weight_table.values())
    sample_weights = [sample_weight(r.get("source", "unspecified"), weight_table) for r in mixed]
    multipliers = accum_multipliers(sample_weights, a.accum)
    assert a.choice_weight >= 1 and a.value_weight >= 1
    assert all(len(enc(r)[0]) + len(enc(r)[1]) <= a.max_len for r in mixed), (
        "Prepared training data exceeds context cap"
    )
    eligible = mixed
    plan = dict(
        base=a.base,
        epochs=1,
        lr=a.lr,
        rank=a.rank,
        accum=a.accum,
        max_len=a.max_len,
        seed=20261013,
        samples=len(mixed),
        sources=dict(Counter(r.get("source", "unspecified") for r in mixed)),
        replay_eligible=len(eligible),
        reasoning_samples=sum(bool(r.get("reasoning_mode")) for r in mixed),
        choice_weight=a.choice_weight,
        value_weight=a.value_weight,
        response_only=True,
        trainable="attention and MLP LoRA; bf16 frozen base",
        sources_filter=a.sources,
        include_replay=a.include_replay,
        replay_rows=sum(bool(r.get("replay")) for r in mixed),
        sample_weights=weight_table,
        sample_weight_sum=float(sum(sample_weights)),
        sample_weight_by_source={
            k: sample_weight(k, weight_table)
            for k in sorted(set(r.get("source", "unspecified") for r in mixed))
        },
    )
    (a.out / "training_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2), flush=True)
    if a.prepare_only:
        return
    assert not (a.out / a.adapter / "adapter_config.json").exists(), (
        "Refusing to overwrite completed adapter."
    )
    torch.manual_seed(20261013)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=a.rank,
            lora_alpha=2 * a.rank,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(mixed) / a.accum)
    warm = max(1, int(0.05 * steps))
    half = max(1, steps // 2)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))),
    )
    optim.zero_grad(set_to_none=True)
    started = time.time()
    step = 0
    losses = []
    prompt_tokens = 0
    target_tokens = 0
    for i, r in enumerate(mixed):
        pi, ti, w = enc(r)
        prompt_tokens += len(pi)
        target_tokens += len(ti)
        ids = torch.tensor([pi + ti], device="cuda")
        # Only response tokens contribute to SFT loss. Keep exactly their
        # predictive logits (plus the final unused position), avoiding a
        # sequence_length x 151936 vocabulary tensor for the masked prompt.
        labels = torch.tensor(ti, device="cuda")
        out = model(input_ids=ids, logits_to_keep=len(ti) + 1)
        response_logits = out.logits[:, :-1, :].reshape(-1, out.logits.shape[-1])
        if a.choice_weight == a.value_weight == 1:
            loss = torch.nn.functional.cross_entropy(response_logits.float(), labels)
        else:
            loss = weighted_response_loss(response_logits, labels, w)
        # Sample-level source weights: within one accumulation window the applied loss is
        # sum_i s_i * loss_i / sum_i s_i (all s_i == 1 reproduces loss / window_size exactly).
        (loss * multipliers[i]).backward()
        losses.append(loss.item())
        del out, ids, labels, loss, response_logits
        if (i + 1) % a.accum == 0 or i + 1 == len(mixed):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            schedule.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            state = dict(
                stage="training",
                step=step,
                total_steps=steps,
                samples_seen=i + 1,
                loss=sum(losses[-a.accum :]) / len(losses[-a.accum :]),
                elapsed_seconds=time.time() - started,
                peak_allocated_mib=torch.cuda.max_memory_allocated() // 2**20,
                prompt_tokens=prompt_tokens,
                target_tokens=target_tokens,
                api_cost_usd=0,
            )
            if step % 10 == 0 or step == 1:
                print(json.dumps(state), flush=True)
                (a.out / "progress.json").write_text(json.dumps(state, indent=2) + "\n")
            if step % 50 == 0:
                model.save_pretrained(a.out / f"checkpoint_{step}")
            if step == half:
                model.save_pretrained(a.out / "adapter_half")
                tok.save_pretrained(a.out / "adapter_half")
    model.save_pretrained(a.out / "adapter")
    tok.save_pretrained(a.out / "adapter")
    state.update(stage="trained", elapsed_seconds=time.time() - started)
    (a.out / "progress.json").write_text(json.dumps(state, indent=2) + "\n")
    print("TRAIN_COMPLETE", json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
