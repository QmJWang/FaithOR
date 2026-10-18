# Attribution

- Qwen3 provides the detector base model. Consult the upstream Qwen3-4B model card and license.
- The five-element writer prompt follows the LLMOPT prompt structure, with component-level formatting and modeling rules used by this project.
- Standard and CoT prompts follow the Chain-of-Experts baseline style; code execution and scoring are provided by this project.
- SAC-Opt and OptiMUS are third-party baselines. Their pinned sources are listed in `baselines/official.json` and retain their original licenses.
- The SemOPT directory is our implementation from the paper, not official author code or author weights. Its refactored runner must be evaluated before assigning it historical results.
- Training supervision includes OptMATH and ReSocratic-derived material alongside procedural examples. Source terms remain applicable.

This release does not grant rights beyond those held by the contributors. No third-party source repository, API credential, solver license, benchmark answer archive, or internal experiment log is embedded in this code package.
