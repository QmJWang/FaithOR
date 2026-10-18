"""Conservative structural expansion of detector-selected components."""

import re
import difflib

ISO_THR = 0.9
ISO_MAX = 6
_LABEL = re.compile(r"^\s*(?:[*\-]\s+)?(?:\*\*)?\s*\(?\s*[SPVOC]\s*\d+\s*\)?\s*(?:\*\*)?\s*[:.)]?")

_MATHW = {
    "geq",
    "leq",
    "ge",
    "le",
    "sum",
    "in",
    "for",
    "all",
    "forall",
    "cdot",
    "times",
    "quad",
    "qquad",
    "left",
    "right",
    "frac",
    "text",
    "mathrm",
    "mathbf",
    "mathbb",
    "ldots",
    "dots",
    "cdots",
    "neq",
    "eq",
    "and",
    "lt",
    "gt",
    "max",
    "min",
    "begin",
    "end",
    "array",
    "displaystyle",
    "limits",
    "big",
    "bigg",
    "mid",
}

_GREEK = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "varphi",
    "chi",
    "psi",
    "omega",
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Phi",
    "Psi",
    "Omega",
}


def iso_tokens(text):
    """normalised token list: label removed, sub/superscripts dropped, numbers -> N, single letters / greek -> V,
    runs of operands (V, N) joined by + - , * cdot -> E (so 'x1+x5+x6' == 'x1+x2+x5+x6')"""
    t = _LABEL.sub("", text, count=1).replace("$", " ")
    t = re.sub(
        r"\\(?:text|mathrm|mathbf|operatorname|textbf|textit)\s*\{([^{}]*)\}",
        r" \1 ",
        t,
    )
    for _ in range(3):
        t = re.sub(r"[_^]\s*\{[^{}]*\}", " ", t)
    t = re.sub(r"[_^]\s*\\?[A-Za-z0-9]+", " ", t)
    t = re.sub(r"\d+(?:[.,:]\d+)*", " N ", t)
    raw = re.findall(r"\\?[A-Za-z]+|[^\sA-Za-z{}]", t)
    out = []
    for w in raw:
        bs = w.startswith("\\")
        w = w.lstrip("\\")
        if (len(w) == 1 and w.isalpha()) or w in _GREEK:
            out.append("V")
        elif w == "N":
            out.append("N")
        elif bs and w not in _MATHW:
            out.append(w.lower())
        else:
            out.append(w.lower())
    # collapse operand runs
    res, i = [], 0
    OPD, OPS = {"V", "N"}, {"+", "-", ",", "*", "cdot", "times"}
    while i < len(out):
        if out[i] in OPD:
            j = i + 1
            while j < len(out):
                if out[j] in OPD:
                    j += 1
                elif out[j] in OPS and j + 1 < len(out) and out[j + 1] in OPD:
                    j += 2
                else:
                    break
            res.append("E")
            i = j
        else:
            res.append(out[i])
            i += 1
    return res


def iso_sim(a, b):
    ta, tb = iso_tokens(a), iso_tokens(b)
    wa, wb = _words(ta), _words(tb)
    full = difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()
    if not wa and not wb:
        return 1.0 if ta == tb else 0.0  # pure-math lines: same skeleton required
    if not wa or not wb:
        return 0.0
    words = difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio()
    return min(full, words)


def iso_expand(segs, flagged, thr=None, max_add=None, exclude=()):
    """flagged: detector ids (ordered).  -> dict(src_id -> [added ids]) with <= max_add added ids in total"""
    thr = ISO_THR if thr is None else thr
    max_add = ISO_MAX if max_add is None else max_add
    comps = [(s["id"], s["text"]) for s in segs if s.get("id") and not s.get("deleted")]
    txt = dict(comps)
    order = [c for c, _ in comps]
    taken = set(flagged) | set(exclude)
    out = {}
    n = 0
    cand = []
    for f in flagged:
        if f not in txt:
            continue
        for c in order:
            if c in taken or c[0] != f[0]:
                continue
            sim = iso_sim(txt[f], txt[c])
            if sim >= thr:
                cand.append((-sim, abs(order.index(c) - order.index(f)), f, c))
    for _, _, f, c in sorted(cand):
        if n >= max_add:
            break
        if c in taken:
            continue
        taken.add(c)
        out.setdefault(f, []).append(c)
        n += 1
    return out


def _words(toks):
    return [w for w in toks if len(w) >= 2 and w.isalpha() and w not in _MATHW]
