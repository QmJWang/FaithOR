# Baselines

## Implemented runners

- `python -m baselines.initial`: one mathematical draft and code execution/debugging, without semantic diagnosis.
- `python -m baselines.simple --method standard|cot`: a single code-generation request, with explicit handling of response truncation and a Gurobi status check. No semantic repair is performed.
- `python -m baselines.semopt.run`: our paper-based semantic-reward search reconstruction. Requires a scalar reward checkpoint and Gurobi. It records candidates and reports the reward-selected candidate, not an oracle-selected answer.
- `python -m baselines.semopt.reward`: pairwise scalar reward training. Input JSONL fields are `problem`, `chosen`, and `rejected`.

All inference runners consume question-only records. Keep objective references in a separate file for offline evaluation. SemOPT output uses `selected`/`candidates`; convert the selected solver status to the scorer's `final` schema before comparison. Gurobi status 2 denotes optimality. Do not treat any-candidate correctness as the reward-selected final accuracy.

The public SemOPT reconstruction preserves the search structure and prompts while refactoring transport and execution. Context overflow is explicit and model responses are not silently accepted when truncated. It is not an official implementation and the public refactor has not been benchmarked end-to-end. The old provider-specific retry scripts, CPU/GPU experiments, and recovery orchestration are intentionally absent.

## Official sources

```bash
python -m baselines.fetch_official --out external
```

This downloads pinned SAC-Opt and OptiMUS revisions. Run their documented entry points with your own model endpoint and solver configuration. The repository does not claim that a generic wrapper reproduces each official native workflow. Keep any local patches, budgets, model choices, and scoring changes in your experiment record.

SAC-Opt and OptiMUS are not reimplemented or vendored here. This avoids silently changing their native algorithms and preserves their licensing requirements. Network access occurs only when explicitly invoking the fetch command.
