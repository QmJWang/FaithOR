"""Five-element model segmentation and component edit application."""

import re

V5 = True
FORM_PROMPT_V5 = r"""In mathematics, any optimization problem can be modeled as the following expression $\min_{\boldsymbol{x} \in \mathcal{X}} f(\boldsymbol{x}), {\rm s.t.} g(\boldsymbol{x}) \leq b$, where $\boldsymbol{x} = (x_1, x_2, \ldots, x_d)^\top$ is the $d$-dimensional decision variable, $\mathcal{X} \subset \mathbb{R}^d$ is the feasible domain, $f: \mathcal{X} \rightarrow \mathbb{R}$ is the objective function and the goal is to find the minima of $f$, $g(\boldsymbol{x}) \leq b$ is the constraint of $\boldsymbol{x}$. 
The above definition can be mapped to a five-element consisting of ``Variables, Objective, Constraints, Sets, Parameters''. Variables indicates what $\boldsymbol{x}$ is, Objective describes the form of the objective function $f(\boldsymbol{x})$, and Constraints indicates the constraints $g(\boldsymbol{x})$ and $\mathcal{X}$. These three can abstract the optimization problem. Sets and Parameters are their specific explanations: Sets describes and explains the subscripts of the vectors or matrices in them, and Parameters supplements their specific values. 
You need to write the corresponding five-element model based on the problem description and information provided. 
The problem description is as follows: 
```
{question}
```
Please write the corresponding five-element model. Please use LaTeX and ``` plain text environment to complete the following template to model the above optimization problem into five elements: 

```
## Sets: 
[You need to fill in]

## Parameters: 
[You need to fill in]

## Variables: 
[You need to fill in]

## Objective: 
[You need to fill in]

## Constraints: 
[You need to fill in]
```

Rules for filling in the template:
- Sets: one index set per line (write "None" if there are no index sets).
- Parameters: one parameter (or one indexed parameter family) per line, with ALL of its numeric values taken from the problem (write tables out explicitly), its unit, and its type (integer or real).
- Variables: one decision variable (or one indexed variable family) per line, with its meaning, its type (continuous / integer / binary), its non-negativity, and its lower/upper bounds.
- IMPORTANT: the type, non-negativity and simple lower/upper bounds of the variables must be stated ONLY in the Variables section. Do NOT repeat them in the Constraints section (no constraints such as x >= 0, x in Z, x in {0,1}, or x <= u for a single variable).
- Objective: exactly one line: minimize or maximize, followed by the full expression. The objective value must be exactly the quantity the question asks for.
- Constraints: one constraint (or one indexed family) per line, numbered C1, C2, ...: a short description, then the full algebraic expression (with "for all" ranges where indexed).
- Do NOT write any code."""


def extract_v5(text):
    """the five-element model: content of the last fenced block that contains '## Variables', else the whole text"""
    blocks = re.findall(r"```(?:[a-zA-Z]*)\s*\n(.*?)```", text, re.S)
    blocks = [b for b in blocks if re.search(r"#+\s*Variables", b, re.I)]
    t = blocks[-1] if blocks else text
    return t.strip() + "\n\nEND OF MODEL"


SEC_RE = re.compile(
    r"^\s{0,3}#{1,4}\s*\**\s*(sets\s*(?:and|&)\s*parameters|sets|parameters|data|decision\s+variables|variables|objective(?:\s+function)?|constraints)\b[^\n]*$",
    re.I | re.M,
)

BULLET_RE = re.compile(r"^(\s{0,3})([*\-•+]|\d+[.)])\s+")

CON_RE = re.compile(r"^\s*(?:[*\-]\s+)?(?:\*\*)?\s*\(?C\s*\d+\b")

SUBHEAD_RE = re.compile(
    r"^\s*(\*\*[^*\n]{1,60}\*\*:?\s*|#{3,}\s*.*|[A-Z][A-Za-z0-9 /&,()\-]{0,50}:\s*)$"
)


def _sec_kind(name):
    n = name.lower()
    if V5 and n.strip() == "sets":
        return "S"
    if "variable" in n:
        return "V"
    if "objective" in n:
        return "O"
    if "constraint" in n:
        return "C"
    return "P"


def _split_blocks(lines, kind):
    """lines of one section body -> list of [is_component, [lines]]"""
    blocks, cur, in_table = [], None, False

    def new_block(is_c, ln):
        b = [is_c, [ln]]
        blocks.append(b)
        return b

    for ln in lines:
        s = ln.strip()
        if kind == "C":
            if CON_RE.match(ln):
                cur = new_block(True, ln)
            elif cur is None:
                cur = new_block(False, ln)
            else:
                cur[1].append(ln)
            continue
        if not s:  # blank line: belongs to the current block
            in_table = False
            if cur is None:
                cur = new_block(False, ln)
            else:
                cur[1].append(ln)
            continue
        if s.startswith("|"):
            if in_table:
                cur[1].append(ln)
            else:
                cur = new_block(True, ln)
                in_table = True
            continue
        in_table = False
        m = BULLET_RE.match(ln)
        if m and len(m.group(1)) <= 1:
            cur = new_block(True, ln)
            continue
        if SUBHEAD_RE.match(ln) and not re.search(r"[=\\$]", s):
            cur = new_block(False, ln)
            continue
        if V5:  # five-element format: one item per line, only indented lines continue an item
            cont = ln[:1] in (" ", "\t") or re.match(r"^\s*(where\b|[=<>+])", ln) is not None
        else:
            cont = (
                ln[:1] in (" ", "\t")
                or re.match(r"^\s*([a-z(]|where\b|\\[a-z]|[=<>+\-])", ln) is not None
            )
        if cur is not None and cur[0] and cont:
            cur[1].append(ln)
            continue
        cur = new_block(True, ln)
    return blocks


def segment(text):
    """flg formulation text -> list of segments dict(id or None, kind, text). "".join(text) reproduces the model up to
    END OF MODEL (assemble() appends it back). Returns None if no section headings are found."""
    t = text
    e = t.find("END OF MODEL")
    if e >= 0:
        t = t[:e]
    ms = list(SEC_RE.finditer(t))
    if not ms:
        return None
    segs = [dict(id=None, kind="H", text=t[: ms[0].start()])] if ms[0].start() > 0 else []
    cnt = {"S": 0, "P": 0, "V": 0, "O": 0, "C": 0}
    for i, m in enumerate(ms):
        kind = _sec_kind(m.group(1))
        body = t[m.end() : ms[i + 1].start() if i + 1 < len(ms) else len(t)]
        segs.append(dict(id=None, kind="H", text=t[m.start() : m.end()]))
        if kind == "O":
            if body.strip():
                cnt["O"] += 1
                segs.append(dict(id=f"O{cnt['O']}", kind="O", text=body))
            else:
                segs.append(dict(id=None, kind="X", text=body))
            continue
        lines = body.split("\n")
        if kind == "C" and not any(CON_RE.match(line) for line in lines):
            kind_split = (
                "P"  # constraints not labelled C1, C2, ...: fall back to bullet/paragraph splitting
            )
        else:
            kind_split = kind
        for j, (is_c, bl) in enumerate(_split_blocks(lines, kind_split)):
            if j > 0:
                segs.append(dict(id=None, kind="X", text="\n"))
            txt = "\n".join(bl)
            if is_c and txt.strip():
                cnt[kind] += 1
                segs.append(dict(id=f"{kind}{cnt[kind]}", kind=kind, text=txt))
            else:
                segs.append(dict(id=None, kind="X", text=txt))
    assert "".join(x["text"] for x in segs) == t
    return segs


def assemble(segs):
    out = "".join(s["text"] for s in segs if not s.get("deleted"))
    return out.rstrip() + "\n\nEND OF MODEL"


def comp_ids(segs):
    return [s["id"] for s in segs if s["id"]]


def render_numbered(segs):
    """text shown to the detector / aligner: every component prefixed with [ID]"""
    out = []
    for s in segs:
        if s.get("deleted"):
            continue
        if s["id"]:
            body = s["text"].strip("\n")
            out.append(f"[{s['id']}] " + body.lstrip())
        else:
            out.append(s["text"].strip("\n") if s["kind"] == "H" else s["text"].strip("\n"))
    txt = "\n".join(x for x in out if x.strip())
    return re.sub(r"\n{3,}", "\n\n", txt)


MARK, GMARK = "<|box_end|>", "<|quad_end|>"


def render_marked(segs):
    """numbered model with a marker token after every component (detector input); returns (text, comp_ids)"""
    out, ids = [], []
    for s in segs:
        if s.get("deleted"):
            continue
        if s["id"]:
            out.append(f"[{s['id']}] " + s["text"].strip("\n").lstrip() + " " + MARK)
            ids.append(s["id"])
        elif s["text"].strip():
            out.append(s["text"].strip("\n"))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)), ids


def solver_line(status, obj):
    if status == "ok" and obj is not None:
        return f"solved to optimality, objective value = {obj:.10g}"
    if status in ("infeasible", "unbounded", "infeasible_or_unbounded"):
        return f"the solver reports the model is {status.replace('_', ' ')} (no objective value)"
    if status == "timeout":
        return "the solver hit the time limit (no objective value)"
    return (
        f"the model could not be solved ({status}; the generated code failed even after debugging)"
    )


DET_TEMPLATE = (
    """Check the mathematical optimization model below against the problem statement. Every component of the model is labelled with an ID ("""
    + ("[S..] sets, [P..] parameters" if V5 else "[P..] sets/parameters")
    + """, [V..] decision variables with domains and bounds, [O1] objective, [C..] constraints). Decide which components are wrong and whether a required constraint is missing.

### Problem
{question}

### Solver result of this model
{solver}

### Model
{model}

### Verdict: is the model wrong or incomplete?"""
    + GMARK
)


def detector_text(question, segs, status, obj):
    model, ids = render_marked(segs)
    return DET_TEMPLATE.format(
        question=question.strip(), solver=solver_line(status, obj), model=model
    ), ids


def apply_edits(segs, edits):
    """edits: list of dict(op=replace|delete|insert, id, text). Returns new segment list (deep copy)."""
    new = [dict(s) for s in segs]
    byid = {s["id"]: s for s in new if s["id"]}
    ins = []
    for e in edits:
        if e["op"] == "replace" and e["id"] in byid:
            o = byid[e["id"]]["text"]
            lead, trail = re.match(r"^\s*", o).group(0), re.search(r"\s*$", o).group(0)
            byid[e["id"]]["text"] = lead + e["text"].strip() + (trail if "\n" in trail else "\n")
        elif e["op"] == "delete" and e["id"] in byid:
            byid[e["id"]]["deleted"] = True
        elif e["op"] == "insert":
            ins.append(e)
    if ins:
        # insert at end of the constraints section (or end of model if none)
        cidx = [i for i, s in enumerate(new) if s["kind"] == "C"]
        pos = (cidx[-1] + 1) if cidx else len(new)
        n0 = sum(1 for s in new if s["kind"] == "C")
        add = []
        for j, e in enumerate(ins):
            add.append(dict(id=None, kind="X", text="\n\n"))
            add.append(
                dict(
                    id=f"C{n0 + j + 1}",
                    kind="C",
                    text=e["text"].strip() + "\n",
                    inserted=True,
                )
            )
        new = new[:pos] + add + new[pos:]
    return new


FORM_PROMPT_V51 = (
    FORM_PROMPT_V5[: FORM_PROMPT_V5.index("Rules for filling in the template:")]
    + r"""Rules for filling in the template:
- Sets: one index set per line (write "None" if there are no index sets).
- Parameters: one parameter (or one indexed parameter family) per line, with its unit, its type (integer or real) and ALL of its numeric values. Every number in the problem must appear in Parameters with its value, including constraint coefficients ("twice", "3 times", "0.5 of"), right-hand sides, and every entry of every table (write tables out explicitly). Never declare a parameter without listing its values. Keep numbers exactly as stated in the problem (do not rescale to thousands/millions, do not convert units; keep the unit the question asks for). If a table only lists some index pairs, either restrict the index set to the listed pairs or state that all other entries are 0.
- Variables: one decision variable (or one indexed variable family) per line: its meaning, its index set, its type (continuous / integer / binary) and whether it is non-negative. Nothing else.
- IMPORTANT: the type and non-negativity of the variables must be stated ONLY in the Variables section. Do NOT repeat them in the Constraints section (no constraints such as x >= 0, x in Z, x in {0,1}).
- Bounds: do NOT put any other bound in the Variables section. Every lower bound other than 0 and every upper bound stated in the problem must be written ONCE in the Constraints section as its own line, e.g. "C5: Upper bounds: x_i <= U_i for all i in I". Only write bounds that the problem states; never derive implied bounds (no min(...), no B - sum of other lower bounds, no ceil(...)).
- Objective: exactly one line: minimize or maximize, followed by the full expression. The objective value must be exactly the quantity the question asks for. If the asked quantity is a net amount (net flow out of the source, profit), write it as the net expression or define it with a variable plus constraints.
- Constraints: one constraint (or one indexed family) per line, numbered C1, C2, ...: a short description, then the full algebraic expression (with "for all" ranges where indexed). Use exactly the coefficients from Parameters. Write every constraint the model needs, including balance/conservation constraints at every node type (source, sink, intermediate) and the definition of every auxiliary quantity used in the objective; do not rely on the objective or on "implicit" assumptions. Use "=" only when the problem requires equality; for supply use <= and for demand use >= unless stated otherwise. Do not add constraints the problem does not state.
- When a constraint depends on a derived structure (e.g. which shifts cover a period, which nodes are adjacent), state the derivation in its short description before the expression.
- Prefer compact formulations with finitely indexed constraints (e.g. MTZ for subtours) over constraints quantified "for all subsets".
- Write only the final model: no notes, no revisions, no alternative formulations.
- Do NOT write any code."""
)


def form_prompt_v51(question):
    return FORM_PROMPT_V51.replace("{question}", question)
