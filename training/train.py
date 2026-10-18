"""Unified entry point for the consolidated SFT and GRPO recipes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["sft1", "sft2", "sft", "grpo"], required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    here = Path(__file__).resolve().parent
    if args.stage in ("sft1", "sft2", "sft"):
        # sft1: general supervision from the base model; sft2: contract stage continued from the
        # merged sft1 model (pass it as --base); sft: legacy single pass over everything.
        key = {"sft1": "sft_stage1", "sft2": "sft_stage2", "sft": "sft"}[args.stage]
        options = dict(cfg[key])
        flags = ["--include-replay"] if options.pop("include_replay", False) else []
        options.pop("note", None)
        data_file = options.pop("data_file", "data/sft_train.jsonl")
        cmd = [
            sys.executable,
            str(here / "sft.py"),
            "--base",
            args.base,
            "--data",
            str(args.data_root / data_file),
            "--out",
            str(args.out),
        ]
        for key, val in options.items():
            cmd.extend(["--" + key.replace("_", "-"), str(val)])
        cmd.extend(flags)
    else:
        options = cfg["grpo"]
        cmd = [
            sys.executable,
            "-m",
            "verl.trainer.main_ppo",
            "algorithm.adv_estimator=grpo",
            "algorithm.norm_adv_by_std_in_grpo=True",
            "algorithm.use_kl_in_reward=False",
            "data.train_files=" + str((args.data_root / "data/grpo_train.parquet").resolve()),
            "data.val_files=" + str((args.data_root / "data/grpo_validation.parquet").resolve()),
            "data.prompt_key=prompt",
            "data.return_raw_chat=True",
            "data.filter_overlong_prompts=True",
            "data.truncation=error",
            "+data.apply_chat_template_kwargs.enable_thinking=False",
            "actor_rollout_ref.model.path=" + args.base,
            "actor_rollout_ref.model.use_remove_padding=False",
            "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            "actor_rollout_ref.model.lora_rank=0",
            "actor_rollout_ref.actor.use_kl_loss=True",
            "actor_rollout_ref.actor.kl_loss_coef=0.001",
            "actor_rollout_ref.actor.entropy_coeff=0",
            "actor_rollout_ref.actor.fsdp_config.param_offload=True",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            "actor_rollout_ref.ref.fsdp_config.param_offload=True",
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.mode=async",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
            "actor_rollout_ref.rollout.val_kwargs.temperature=0",
            "custom_reward_function.path=" + str(here / "grpo_reward.py"),
            "custom_reward_function.name=compute_score",
            "reward.custom_reward_function.path=" + str(here / "grpo_reward.py"),
            "reward.custom_reward_function.name=compute_score",
            "trainer.default_local_dir=" + str(args.out.resolve()),
            "trainer.project_name=or_semantic_detector",
            "trainer.experiment_name=consolidated_grpo",
            "trainer.logger=[console]",
        ]
        cmd.extend(k + "=" + str(v) for k, v in options.items())
    print(json.dumps({"command": cmd}, indent=2), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
