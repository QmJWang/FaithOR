"""One diagnostic round, targeted mathematical edits, and execution debugging."""

import argparse
import json
from pathlib import Path
from .client import Client
from .data import read_jsonl
from .execution import execute
from .inference import diagnose
from .modeling import (
    segment,
    assemble,
    render_numbered,
    detector_text,
    apply_edits,
    form_prompt_v51,
    extract_v5,
)
from .prompts import GEN_PROMPT_PLUS, TYPE_SUGG, REPAIR_T, extract_code
from .structure import iso_expand


def json_object(text):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                pass
    raise ValueError("No JSON object found")


def validate_edits(edits, allowed, missing):
    replace = edits.get("replace", {})
    delete = edits.get("delete", [])
    add = edits.get("add_constraints", [])
    if not isinstance(replace, dict) or not isinstance(delete, list) or not isinstance(add, list):
        raise ValueError("Invalid edit schema")
    if not all(isinstance(x, str) for x in delete):
        raise ValueError("Invalid deletion ID")
    if not set(replace).union(delete) <= set(allowed):
        raise ValueError("Unauthorized component edit")
    if set(replace) & set(delete):
        raise ValueError("Conflicting edits")
    if not all(isinstance(x, str) and x.strip() for x in list(replace.values()) + add):
        raise ValueError("Empty edit")
    if len(add) > (3 if missing else 0):
        raise ValueError("Unauthorized constraint insertion")
    return {"replace": replace, "delete": delete, "add_constraints": add}


def generate_and_solve(client, model, work, max_debug=3):
    code = extract_code(client.call(GEN_PROMPT_PLUS.format(formulation=model), "code"))
    for step in range(max_debug + 1):
        result = execute(code, Path(work) / str(step))
        if (
            result["status"] in {"ok", "infeasible", "unbounded", "infeasible_or_unbounded"}
            or step == max_debug
        ):
            return dict(result, code=code, debug_calls=step)
        prompt = (
            "Fix execution errors in this Pyomo script without changing its mathematical model. "
            "Return one Python code block.\nMathematical model:\n"
            + model
            + "\nCode:\n"
            + code
            + "\nFeedback:\n"
            + result["error"]
        )
        code = extract_code(client.call(prompt, "code_debug"))


def run_case(question, writer, coder, detector, work):
    # Reference answers are intentionally absent from this interface.
    raw = writer.call(form_prompt_v51(question), "initial_math")
    formulation = extract_v5(raw)
    segments = segment(formulation)
    if not segments:
        raise ValueError("Initial mathematical model has no valid sections")
    initial = generate_and_solve(coder, formulation, Path(work) / "initial")
    record = {
        "initial": initial,
        "initial_model": formulation,
        "detector": None,
        "final": initial,
        "edited": False,
    }
    # From here on, any failure keeps the initial answer: a detector or writer
    # problem is an abstention from repair, never a lost case.
    try:
        text, ids = detector_text(question, segments, initial["status"], initial["objective"])
        diagnosis = diagnose(detector, text, ids)
        record["detector"] = diagnosis
        if not diagnosis["valid"]:
            return record
        return repair_case(question, writer, coder, segments, diagnosis, record, work)
    except Exception as error:
        record["repair_error"] = type(error).__name__ + ": " + str(error)
        return record


def repair_case(question, writer, coder, segments, diagnosis, record, work):
    pred = diagnosis["diagnosis"]
    flagged = pred["errors"][:3]
    if not flagged and not pred["missing"]:
        return record
    hints = {i: TYPE_SUGG[t].format(i=i) for i, t in flagged}
    for source, extras in iso_expand(segments, list(hints)).items():
        for cid in extras:
            hints[cid] = "Check together with [" + source + "]: " + hints[source]
    flags = "\n".join("- [" + i + "]: " + hint for i, hint in hints.items())
    if pred["missing"]:
        flags += "\n- Required constraints may be missing. Add at most three."
    prompt = REPAIR_T.format(flags=flags, question=question, model=render_numbered(segments))
    repair = writer.call(prompt, "math_repair", max_tokens=12288)
    record["repair_raw"] = repair
    try:
        edits = validate_edits(json_object(repair), hints, pred["missing"])
    except ValueError as error:
        record["edit_error"] = str(error)
        return record
    operations = [{"op": "replace", "id": i, "text": t} for i, t in edits["replace"].items()]
    operations += [{"op": "delete", "id": i} for i in edits["delete"]]
    operations += [{"op": "insert", "text": t} for t in edits["add_constraints"]]
    updated = apply_edits(segments, operations)
    repaired = assemble(updated)
    record.update(
        edited=True,
        edits=edits,
        final_model=repaired,
        final=generate_and_solve(coder, repaired, Path(work) / "repaired"),
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    for role, port in [("writer", 8001), ("coder", 8001), ("detector", 8000)]:
        parser.add_argument("--" + role + "-url", default=f"http://localhost:{port}/v1")
        parser.add_argument("--" + role + "-model", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    clients = {
        role: Client(
            getattr(args, role + "_url"),
            getattr(args, role + "_model"),
            args.out / "usage.jsonl",
            max_tokens=12288,
        )
        for role in ["writer", "coder", "detector"]
    }
    with (args.out / "predictions.jsonl").open("w") as stream:
        for i, row in enumerate(read_jsonl(args.input)):
            try:
                result = run_case(
                    row["question"],
                    clients["writer"],
                    clients["coder"],
                    clients["detector"],
                    args.out / str(i),
                )
            except Exception as error:
                result = {"error": type(error).__name__, "message": str(error)}
            stream.write(json.dumps({"id": row["id"], **result}) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
