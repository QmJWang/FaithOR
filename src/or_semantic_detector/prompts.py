import re

"""Prompts for mathematical repair and code translation."""

V5 = True
GEN_PROMPT = """Translate the given mathematical model into a Pyomo script, faithfully and literally. Do NOT change the model in any way: do not add, remove, or modify any set, parameter value, variable, domain, bound, objective term, or constraint, even if you think the model is wrong or incomplete. Use only the data written in the model.

Requirements for the script:
- `import pyomo.environ as pyo` and build a `pyo.ConcreteModel()` assigned to a top-level variable named `model` (not inside a function or `if __name__ == "__main__":`).
- Exactly one active `pyo.Objective` with the correct sense.
- Do NOT call any solver and do not print results; an external harness will solve `model` and read the objective value.
- Output a single ```python code block and nothing else.

Mathematical model:
{formulation}"""


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    m = re.search(r"```(?:python|py)?\s*\n(.*)", text, re.S)  # unterminated block
    return m.group(1) if m else text


GEN_PLUS_RULE = """

Additional requirement: Pyomo variable bounds may only be a tuple of constants, e.g. bounds=(0, 10), or a rule function that returns a (lb, ub) tuple, e.g. def x_bounds(m, i): return (0, m.U[i]) and then bounds=x_bounds. Never put a lambda or a function inside the bounds tuple (no bounds=(0, lambda ...), no bounds=(f, g)). Write any complicated bound as a pyo.Constraint instead."""

GEN_PROMPT_PLUS = GEN_PROMPT + GEN_PLUS_RULE
TYPE_SUGG = {
    "var_to_int": "the variable(s) declared in [{i}] should be INTEGER: change their domain to integer",
    "var_to_bin": "the variable(s) declared in [{i}] should be BINARY (0-1): change their domain to binary",
    "var_to_cont": "the variable(s) in [{i}] should be CONTINUOUS: change their domain to continuous (remove the integrality requirement)",
    "var_bound": "the bounds of the variable(s) in [{i}] are probably wrong: correct the lower/upper bounds according to the problem statement",
    "con_sense": "the direction of constraint [{i}] is probably reversed (<= vs >= vs =): correct the relation",
    "num_value": "a number in [{i}] (coefficient, right-hand side or parameter value) is probably wrong: check every number against the problem statement and correct it",
    "con_structure": "constraint [{i}] is written in the wrong form or structure (terms, variables involved, index ranges): rewrite it to match the problem statement",
    "con_extra": "constraint [{i}] is probably superfluous (the problem does not require it): delete it",
    "obj_sense": "the objective sense in [{i}] is probably reversed (minimize vs maximize): correct it",
    "obj_expr": "the objective expression [{i}] is probably wrong: correct its terms and coefficients according to the problem statement",
}

REPAIR_T = (
    """You are an expert in operations research. Below are an optimization problem and a mathematical model written for it. Every component of the model is labelled with an ID ("""
    + ("[S..] sets, [P..] parameters" if V5 else "[P..] sets/parameters")
    + """, [V..] decision variables with domains and bounds, [O1] objective, [C..] constraints).

A checker produced the following concrete fix suggestions:
{flags}

Apply the suggested changes, unless a suggestion clearly contradicts the problem statement. Keep the model's notation and style and do not change any other component.

Output a JSON object in a ```json code block:
{{"replace": {{"<ID>": "<full corrected text of that component>", ...}}, "delete": ["<ID>", ...], "add_constraints": ["<full text of a missing constraint, in the style of the model's constraint lines>", ...]}}
Use empty objects/lists when nothing needs to change.

### Problem
{question}

### Model
{model}"""
)
STANDARD = """You are a Python programmer in the field of operations research and optimization. Your proficiency in utilizing third-party libraries such as Gurobi is essential. In addition to your expertise in Gurobi, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy, or PuLP.
You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem.
Now the origin problem is as follow:\n{problem}\nGive your Python code directly."""

COT = """You are a Python programmer in the field of operations research and optimization. Your proficiency in utilizing third-party libraries such as Gurobi is essential. In addition to your expertise in Gurobi, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy, or PuLP.
You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem.
Now the origin problem is as follow:
{problem_description}
Let's analyse the problem step by step, and then give your Python code."""
