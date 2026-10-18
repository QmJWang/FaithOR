# FaithOR: one semantic diagnosis makes the math model faithful to the problem

LLMs can turn a natural-language optimization problem into a mathematical model that **solves to optimality and is still wrong**: a countable quantity declared continuous, an "exactly k times" ratio written as `>=`, a capacity that should bind two machines applied to all of them, a net-profit objective missing a cost term. Solver status cannot see any of this.

FaithOR trains a compact 4B **semantic detector** that reads the problem, the numbered candidate model and its solver feedback, and returns the wrong components, their error types and a missing-constraint flag. A frozen writer applies **one** targeted repair. On seven public benchmarks (1,511 problems) this single round lifts Qwen3.6-35B from 87.4% to 92.6%, DeepSeek-v4.1-flash to 93.8% and Gemini-3-flash to 94.7%, captures 98% of what three rounds would give, and costs about one tenth of the API tokens of the strongest prior pipeline.

- **Paper**: *arXiv, coming soon*
- **Training data**: [FaithOR-Supervision](https://huggingface.co/QMJWang/FaithOR-Supervision) — 24.6K component-level diagnoses from writer formulations and semantic-contract simulation, plus 3.7K GRPO prompts
- **Detector**: [FaithOR-Detector-4B](https://huggingface.co/QMJWang/FaithOR-Detector-4B) — Qwen3-4B, two-stage SFT on the data above

This repository holds the data construction, training, inference, one-round repair pipeline, scoring and baselines.

## Main results

One semantic detection and one targeted repair per problem; no reference answer is used to choose when to stop. Accuracy in %, weighted over 1,511 problems (NL4Opt 214, IndustryOR 42, EasyLP 545, ComplexLP 111, NLP4LP 178, ReSocratic 403, ComplexOR 18). Five-run entries are mean ± sample SD; single runs are reported as is.

| Group | Method | Backbone | Runs | NL4Opt | IndustryOR | EasyLP | ComplexLP | NLP4LP | ReSocratic | ComplexOR | Overall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | Standard \* | Qwen3.6-35B | 5 | 22.62 ± 4.83 | 23.33 ± 4.26 | 13.06 ± 3.86 | 12.79 ± 0.75 | 25.51 ± 4.40 | 23.23 ± 3.04 | 7.78 ± 4.97 | 18.80 ± 2.60 |
| Baseline | CoT \* | Qwen3.6-35B | 5 | 50.19 ± 2.68 | 49.52 ± 4.26 | 49.91 ± 2.22 | 17.30 ± 5.07 | 56.85 ± 4.87 | 47.59 ± 1.21 | 32.22 ± 7.24 | 47.53 ± 0.93 |
| Baseline | One-pass | Qwen3.6-35B | fixed draft | 78.04 | 64.29 | 93.94 | 54.05 | 87.64 | 81.39 | 50.00 | 83.32 |
| Baseline | One-pass | GPT-4o | 1 | 79.91 | 61.90 | 97.43 | 45.95 | 91.57 | 84.12 | 50.00 | 85.37 |
| Baseline | One-pass | GPT-4.1-nano | 1 | 83.18 | 40.48 | 82.02 | 41.44 | 87.08 | 78.91 | 38.89 | 77.30 |
| Baseline | One-pass | DeepSeek-v4.1-flash | 1 | 79.44 | 71.43 | 97.61 | 64.86 | 94.38 | 84.12 | 50.00 | 87.36 |
| Baseline | One-pass | Gemini-3-flash | 1 | 77.10 | 78.57 | 97.80 | 63.96 | 92.13 | 86.10 | 66.67 | 87.69 |
| Prior work | Reflexion | Qwen3.6-35B | 5 | 75.51 ± 3.18 | 67.62 ± 3.61 | 85.32 ± 2.10 | 61.80 ± 1.87 | 89.33 ± 3.46 | 82.83 ± 2.43 | 67.78 ± 4.65 | 81.31 ± 1.75 |
| Prior work | OptiMUS-0.2 | Qwen3.6-35B | 5 | 49.72 ± 4.89 | 25.71 ± 9.28 | 42.72 ± 1.55 | 24.68 ± 4.30 | 57.75 ± 2.56 | 49.63 ± 2.29 | 21.11 ± 6.09 | 45.27 ± 1.98 |
| Prior work | OptiMUS-0.3 | Qwen3.6-35B | 5 | 74.49 ± 1.80 | 57.62 ± 3.91 | 79.63 ± 2.27 | 52.79 ± 2.89 | 88.88 ± 2.68 | 79.26 ± 0.57 | 48.89 ± 4.65 | 76.94 ± 1.34 |
| Prior work | SAC-Opt | Qwen3.6-35B | 5 | 75.23 ± 2.82 | 56.19 ± 3.98 | 86.39 ± 1.49 | 48.83 ± 4.05 | 88.31 ± 3.09 | 80.74 ± 1.05 | 40.00 ± 7.24 | 79.38 ± 1.69 |
| Prior work | SemOPT | Qwen3.6-35B | 5 | 87.48 ± 0.90 | 72.38 ± 1.30 | 96.88 ± 0.61 | 74.95 ± 2.66 | 97.98 ± 0.64 | 92.75 ± 0.87 | 67.78 ± 2.48 | 91.94 ± 0.37 |
| Prior work | SAC-Opt (1,429 covered) | GPT-4o | 1 | 76.59 | 30.30 | 77.61 | 68.75 | 89.41 | 80.76 | 50.00 | 77.96 |
| Prior work | SemOPT | GPT-4.1-nano | 1 | 75.23 | 52.38 | 97.06 | 54.05 | 88.20 | 84.62 | 61.11 | 84.78 |
| Prior work | SemOPT | DeepSeek-v4.1-flash | 1 | 83.64 | 69.05 | 96.88 | 66.67 | 96.63 | 89.08 | 77.78 | 89.68 |
| Prior work | SemOPT | Gemini-3-flash | 1 | 85.05 | 83.33 | 97.43 | 69.37 | 98.31 | 91.81 | 66.67 | 91.46 |
| **Ours, one round** | detector + targeted repair | Qwen3.6-35B | 5 | 88.97 ± 0.26 | 63.33 ± 1.30 | 97.14 ± 0.10 | 94.41 ± 1.48 | 93.71 ± 0.25 | 91.86 ± 0.21 | 56.67 ± 2.48 | **92.55 ± 0.10** |
| **Ours, one round** | detector + targeted repair | GPT-4o | 1 | 90.19 | 64.29 | 97.61 | 80.18 | 96.63 | 92.31 | 55.56 | **92.32** |
| **Ours, one round** | detector + targeted repair | GPT-4.1-nano | 1 | 86.45 | 45.24 | 88.99 | 62.16 | 92.70 | 83.62 | 44.44 | **83.92** |
| **Ours, one round** | detector + targeted repair | DeepSeek-v4.1-flash | 5 | 88.13 ± 0.85 | 68.10 ± 4.94 | 98.24 ± 0.10 | 97.12 ± 1.61 | 97.08 ± 0.47 | 92.70 ± 0.38 | 58.89 ± 3.04 | **93.81 ± 0.21** |
| **Ours, one round** | detector + targeted repair | Gemini-3-flash | 1 | 88.32 | 76.19 | 98.17 | 99.10 | 98.31 | 93.80 | 66.67 | **94.71** |

Notes. All "Ours" rows are one round; baselines keep their own full procedures. SemOPT is our re-implementation of the published procedure with a locally trained reward model, not the authors' weights. On GPT-4.1-nano the one-round result trails SemOPT by 13 problems; we do not claim a win on every backbone. \* Standard and CoT are scored with the strict optimality certification; under the historical numeric-only convention they score 33.53% and 51.09%.

### Cost of one round

Everything is accumulated up to the end of the first round: initial draft, code generation and debugging, one detection, one math edit and its re-generation. Tokens are provider-reported.

| Backbone | Problems | API calls | Input tokens | Output tokens | Total tokens | Est. cost (CNY) | Request time / problem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-v4.1-flash | 1,511 | 3,559 | 3,223,080 | 1,233,279 | 4,456,359 | 13.05 | 8.8 s |
| Gemini-3-flash | 1,511 | 3,530 | 3,299,444 | 1,479,015 | 4,778,459 | 24.35 | 9.2 s |

Against the complete SemOPT pipeline on the same backbones (per problem, over the problems each method has billing records for):

| Backbone | Tokens / problem: ours vs. SemOPT | Ratio | Cost / problem (CNY): ours vs. SemOPT | Ratio | Request time / problem: ours vs. SemOPT |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-v4.1-flash | 2,949 vs. 27,631 | 10.7% | 0.0086 vs. 0.0829 | 10.4% | 8.8 s vs. 181.1 s |
| Gemini-3-flash | 3,162 vs. 31,501 | 10.0% | 0.0161 vs. 0.1696 | 9.5% | 9.2 s vs. 94.2 s |

Unified prices: DeepSeek 0.0016 / 0.0064 CNY per 1K input / output tokens, Gemini 0.002 / 0.012. Times are accumulated client-side request latencies and exclude local detector batching and SemOPT's reward computation, so they are not end-to-end speed-ups. Local GPU time for the detector is not free; API cost is not the whole compute cost.

### One round captures almost all of the gain

Rounds two and three are historical ablations: the same trajectories re-scored after each round.

| Backbone | Runs | Draft + debug | Round 1 (ours) | Round 2 | Round 3 | Net problems gained by rounds 2–3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B | 5 | 87.36 | **92.55 ± 0.10** | 92.63 ± 0.22 | 92.65 ± 0.13 | +1.6 |
| GPT-4o | 1 | 87.69 | **92.32** | 92.32 | 92.72 | +6.0 |
| GPT-4.1-nano | 1 | 80.41 | **83.92** | 83.32 | 84.18 | +4.0 |
| DeepSeek-v4.1-flash | 5 | 88.14 | **93.81 ± 0.21** | 93.51 ± 0.22 | 93.79 ± 0.14 | −0.2 |
| Gemini-3-flash | 1 | 89.34 | **94.71** | 94.97 | 94.90 | +3.0 |

For Qwen3.6-35B the first round delivers 98% of the three-round net improvement.

### The detector matters, not just the repair loop

Same frozen drafts, same Qwen3.6-35B writer and coder, same prompts; only the detector is swapped. "Repaired" = initially wrong → correct; "broken" = initially correct → wrong.

| Detector | Runs | Repaired | Broken | Repair rate | Break rate | Final accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **FaithOR-Detector-4B** | 5 | 86.6 | 8.2 | 45.6% | 0.62% | **92.61%** |
| GPT-5.4-mini as detector | 1 | 40 | 28 | 21.1% | 2.12% | 88.21% |
| DeepSeek-v4.1-flash as detector | 1 | 39 | 5 | 20.5% | 0.38% | 89.67% |

On ComplexLP alone the 4B detector repairs 39 of 45 initially wrong problems (86.7%) where the two API detectors repair 4 and 5.

## How it works

```
problem ──► writer drafts numbered model ──► coder writes Pyomo, solver runs (≤3 debug rewrites)
                                                           │
                              FaithOR-Detector-4B reads problem + model + solver result
                                                           │
                         {"errors": [...], "missing": ...}  ─┬─ empty ──► keep the draft
                                                            └─ non-empty ──► writer edits only the flagged
                                                                              components ──► re-code, re-solve
```

The detector sees the problem text, the solver status and objective of the current model, and the model with every component numbered (`S` sets, `P` parameters, `V` variables, `O1` objective, `C` constraints). It answers with one check line and a JSON object:

```json
{"errors": [{"id": "V1", "type": "var_to_int"}], "missing": false}
```

The ten error types are `var_to_int`, `var_to_bin`, `var_to_cont`, `var_bound`, `con_sense`, `num_value`, `con_structure`, `con_extra`, `obj_sense`, `obj_expr`; `missing` flags an absent constraint. Each type maps to a concrete edit instruction for the writer, which may decline an instruction that contradicts the problem. Reference answers are used by offline scoring only; they never reach the detector, the writer or the coder.

## Quick start

Python 3.10+.

```bash
pip install -e '.[data,solve,test]'
```

Serve the detector and run detection on numbered inputs (`id`, `text`, `comp_ids`):

```bash
vllm serve QMJWang/FaithOR-Detector-4B --served-model-name faithor --port 8000 --max-model-len 16384
or-detect --input examples/detector_input.jsonl --output outputs/detection.jsonl --model faithor
```

Run the full one-round repair with any OpenAI-compatible writer/coder (they may share a model), then score offline:

```bash
or-repair --input examples/problems.jsonl --out outputs/ours \
  --writer-model writer --writer-url http://localhost:8001/v1 \
  --coder-model writer --coder-url http://localhost:8001/v1 \
  --detector-model faithor --detector-url http://localhost:8000/v1
or-score --predictions outputs/ours/predictions.jsonl --references examples/references.jsonl --output outputs/scores.json
```

Generated Python is executed locally; use an isolated environment. Nonlinear problems need SCIP on `PATH`; Gurobi baselines need `pip install -e '.[baselines]'` and a licence.

## Training the detector

The published dataset ships the upstream-derived problem statements as placeholders (see the dataset card); restore them once before training:

```bash
python scripts/rehydrate.py --optmath /path/to/OptMATH-Train.jsonl --resocratic /path/to/resocratic-29k.json \
  --data-dir ../FaithOR-Supervision/data --out-dir ../FaithOR-Supervision/data   # verified against REHYDRATED_SHA256SUMS
```

Two supervised stages on [FaithOR-Supervision](https://huggingface.co/QMJWang/FaithOR-Supervision) from `Qwen/Qwen3-4B` (`configs/training.json` holds every hyper-parameter):

```bash
# stage 1: data/sft_stage1.jsonl (optmath + resocratic rows), LoRA r64, lr 1e-4
python training/train.py --config configs/training.json --stage sft1 --base Qwen/Qwen3-4B \
  --data-root ../FaithOR-Supervision --out checkpoints/stage1
python training/sft.py --base Qwen/Qwen3-4B --data ../FaithOR-Supervision/data/sft_train.jsonl --out checkpoints/stage1 --merge-only

# stage 2: data/sft_stage2.jsonl (simulated families, then replayed writer rows), LoRA r16, lr 6e-6, from the merged stage-1 model
python training/train.py --config configs/training.json --stage sft2 --base checkpoints/stage1/merged \
  --data-root ../FaithOR-Supervision --out checkpoints/stage2
python training/sft.py --base checkpoints/stage1/merged --data ../FaithOR-Supervision/data/sft_train.jsonl --out checkpoints/stage2 --merge-only
```

Each SFT stage runs on one 48 GB GPU (stage 1 about 3.6 h, stage 2 about 1 h). A short GRPO stage follows (two GPUs, about 3 h for 50 steps; needs a [verl](https://github.com/verl-project/verl) 0.9.1 environment):

```bash
python training/train.py --config configs/training.json --stage grpo --base checkpoints/stage2/merged \
  --data-root ../FaithOR-Supervision --out checkpoints/grpo
python -m verl.model_merger merge --backend fsdp --local_dir checkpoints/grpo/global_step_50/actor --target_dir checkpoints/grpo/hf_step_50
```
The loss is response-only with token weights 10 / 3 / 1 on the first `errors` decision, the IDs/types/`missing` values and everything else. The GRPO reward is rule-based (`training/grpo_reward.py`): format, false alarms on correct models, component-localization F1 and error-type bonus. `--stage sft` is the single-pass variant. Add `--dry-run` to print the launch command. `--sources`, `--include-replay` and `--sample-weights` on `training/sft.py` select and weight rows without editing the data file.

Run from the public release, this recipe gives 1,393–1,394 / 1,511 (92.2%) after the two SFT stages and 1,391–1,394 / 1,511 after GRPO (two runs each) in the one-round harness with Qwen3.6-35B as writer and coder (initial drafts 1,319; writer self-check 1,355; a single SFT pass over all rows 1,362). The released [FaithOR-Detector-4B](https://huggingface.co/QMJWang/FaithOR-Detector-4B) weights are the historical checkpoint behind the tables above, which scores 1,398 in a single run of the same harness; its model card describes how they were produced.

## Data construction

```bash
python scripts/simulate.py --out outputs/simulated.jsonl --groups-per-family 20 --seed 42
```

The simulator samples a semantic contract (divisible vs. indivisible decisions, parameter-role binding, bounds and scope, exact vs. inequality ratios, prerequisites, implicit domains, resource costs), renders the problem and the correct model from it, then applies tracked mutations; correct and equivalent variants are kept. Real-formulation rows are built from writer drafts on OptMATH-Train and ReSocratic-29k with execution-checked perturbation and counterfactual localization. The dataset card documents the composition and the overlap audit against the benchmarks (zero exact or number-masked duplicates, no shared 13-word span).

## Repository layout

```text
src/or_semantic_detector/  Protocol, losses, simulation, inference, repair, scoring
training/                 Two-stage SFT / GRPO entry point, LoRA trainer, reward adapter
configs/                  Training configuration (stage 1, stage 2, GRPO, single-pass)
baselines/                Initial, Standard, CoT, SemOPT, pinned official sources
examples/                 Small interface examples (not benchmark data)
tests/                    Parser, reward, loss weights, data, simulation, repair, scoring, solver
scripts/                  Simulation, upstream-text rehydration, release verification
```

```bash
pytest -q && ruff check . && ruff format --check .
```

## Citation

Jiaxin Wang et al., *FaithOR* (paper coming soon; a BibTeX entry will be added here).

## Attribution

See [NOTICE.md](NOTICE.md) and the dataset card. Third-party baselines are fetched at pinned commits rather than copied; upstream data licences remain applicable.
