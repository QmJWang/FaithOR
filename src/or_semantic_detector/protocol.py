"""Frozen diagnostic labels, prompt construction, parser, and GRPO reward."""

import json
import re

TC = [
    "var_to_int",
    "var_to_bin",
    "var_to_cont",
    "var_bound",
    "con_sense",
    "num_value",
    "con_structure",
    "con_extra",
    "obj_sense",
    "obj_expr",
]
FALLBACK_T = {
    "V": "var_to_int",
    "C": "con_structure",
    "O": "obj_expr",
    "P": "num_value",
    "S": "num_value",
}
INSTR = """### Task
Check every component against the problem statement. First write a short check (a few sentences at most). Then output, as the last line, only a JSON object:
{"errors": [{"id": "<component ID>", "type": "<error type>"}, ...], "missing": <true|false>}
- "errors": the wrong components, most likely first; [] if no component is wrong.
- "missing": true if a constraint required by the problem is absent from the model.
- error types: var_to_int / var_to_bin / var_to_cont (the variable should be integer / binary / continuous), var_bound (wrong variable bound), con_sense (wrong inequality direction), num_value (wrong number), con_structure (wrong constraint expression), con_extra (the constraint should be removed), obj_sense (wrong min/max), obj_expr (wrong objective expression)."""
MAX_REASON_CHARS = 1000  # ~256 tokens; a longer check part is a format error


def gen_prompt(text):
    t = text.replace(" <|box_end|>", "").replace("<|box_end|>", "").replace("<|quad_end|>", "")
    i = t.rfind("### Verdict")
    t = (t[:i] if i >= 0 else t).rstrip()
    return t + "\n\n" + INSTR


def gold(d):
    """-> (ordered list of (id, type) labelled wrong, missing flag 1/0/-1, set of masked ids)"""
    errs = []
    for c, y, t in zip(
        d["comp_ids"],
        d["comp_labels"],
        d.get("comp_types") or [-1] * len(d["comp_ids"]),
    ):
        if y == 1:
            errs.append(
                (
                    c,
                    TC[t] if t is not None and t >= 0 else FALLBACK_T.get(c[0], "num_value"),
                )
            )
    masked = {c for c, y in zip(d["comp_ids"], d["comp_labels"]) if y == -1}
    return errs, d.get("missing", 0), masked


def gen_target(d):
    errs, miss, _ = gold(d)
    return f"Checked {len(d['comp_ids'])} components against the problem.\n" + json.dumps(
        {"errors": [{"id": c, "type": t} for c, t in errs], "missing": miss == 1}
    )


def strip_think(s):
    if "</think>" in s:
        s = s.split("</think>")[-1]
    return s


def parse_out(s, comp_ids):
    """-> dict(errors=[(id,type)...], missing=bool, reason_chars=int) or None (format error)"""
    s = strip_think(s).strip()
    dec = json.JSONDecoder()
    obj, start = None, None
    for m in reversed(list(re.finditer(r"\{", s))):
        try:
            o, end = dec.raw_decode(s[m.start() :])
        except Exception:
            continue
        if isinstance(o, dict) and "errors" in o:
            if s[m.start() + end :].strip().strip("`").strip():
                continue  # trailing text after the JSON
            obj, start = o, m.start()
            break
    if obj is None:
        return None
    reason = s[:start].strip()
    if len(reason) > MAX_REASON_CHARS:
        return None
    E, miss = obj.get("errors"), obj.get("missing", False)
    if not isinstance(E, list) or not isinstance(miss, bool):
        return None
    ids = set(comp_ids)
    out, seen = [], set()
    for e in E:
        if not isinstance(e, dict) or e.get("id") not in ids or e.get("type") not in TC:
            return None
        if e["id"] in seen:
            continue
        seen.add(e["id"])
        out.append((e["id"], e["type"]))
    return dict(errors=out, missing=miss, reason_chars=len(reason))


def reward(pred, d, w_type=0.2, r_miss_pos=-0.5, r_neg_ok=1.0, r_neg_bad=-1.0, r_fmt=-1.0):
    """Solver-label reward.  V lines are neutral (neither rewarded nor penalised, not in F1); masked (-1) components are neutral;
    a missing-constraint label (missing==1) counts as one extra gold item '__missing__' (pred missing=true hits it).
    Returns (score, info)."""
    info = dict(fmt_ok=0.0, f1=0.0, said_none=0.0, type_bonus=0.0, n_pred=0.0, v_flag=0.0)
    if pred is None:
        return r_fmt, info
    info["fmt_ok"] = 1.0
    errs, miss, masked = gold(d)
    gpos = {c: t for c, t in errs if c[0] != "V"}
    P = [(c, t) for c, t in pred["errors"] if c[0] != "V" and c not in masked]
    info["n_pred"] = float(len(pred["errors"]))
    info["v_flag"] = float(any(c[0] == "V" for c, _ in pred["errors"]))
    use_miss = miss in (0, 1)
    if d["any_error"] == 0:
        bad = bool(P) or (use_miss and pred["missing"])
        info["said_none"] = float(not bad)
        return (r_neg_bad if bad else r_neg_ok), info
    if not pred["errors"] and not pred["missing"]:
        info["said_none"] = 1.0
        return r_miss_pos, info
    G = set(gpos) | ({"__missing__"} if miss == 1 else set())
    S = {c for c, _ in P} | ({"__missing__"} if (use_miss and pred["missing"]) else set())
    if not G:
        return 0.0, info  # only V / unknown labels (not used for RL)
    tp = len(G & S)
    f1 = 2 * tp / (len(G) + len(S)) if S else 0.0
    hits = [(c, t) for c, t in P if c in gpos]
    bonus = w_type * sum(t == gpos[c] for c, t in hits) / len(hits) if hits else 0.0
    info["f1"] = f1
    info["type_bonus"] = bonus
    return f1 + bonus, info


def rl_eligible(d):
    """negatives, or positives with >=1 non-V labelled component or a missing label"""
    if d["kind"] == "cf_unloc":
        return False
    if d["any_error"] == 0:
        return True
    errs, miss, _ = gold(d)
    return any(c[0] != "V" for c, _ in errs) or miss == 1
