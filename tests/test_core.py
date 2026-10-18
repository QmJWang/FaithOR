import random
import pytest
from or_semantic_detector.protocol import parse_out, reward
from or_semantic_detector.loss import weights_for_response
from or_semantic_detector.data import normalize, consolidate, masked_key, write_jsonl
from or_semantic_detector.modeling import segment, assemble, apply_edits
from or_semantic_detector.pipeline import validate_edits
from or_semantic_detector.evaluation import correct
from or_semantic_detector.inference import diagnose
from or_semantic_detector.simulation import prerequisites, full_partial


def test_parser_rejects_unknown_labels_and_components():
    assert parse_out('{"errors":[{"id":"V9","type":"var_to_int"}],"missing":false}', ["V1"]) is None
    assert parse_out('{"errors":[{"id":"V1","type":"new_type"}],"missing":false}', ["V1"]) is None
    assert parse_out('{"errors":[],"missing":"false"}', ["V1"]) is None
    assert parse_out('{"errors":[],"missing":false} trailing', ["V1"]) is None


def test_reward_respects_variable_and_masked_neutrality():
    gold = {
        "comp_ids": ["V1", "C1", "C2"],
        "comp_labels": [1, 1, -1],
        "comp_types": [0, 4, -1],
        "missing": 0,
        "any_error": 1,
    }
    a = {"errors": [("C1", "con_sense")], "missing": False}
    b = {
        "errors": [("C1", "con_sense"), ("V1", "var_to_int"), ("C2", "con_extra")],
        "missing": False,
    }
    assert reward(a, gold)[0] == pytest.approx(1.2)
    assert reward(a, gold)[0] == reward(b, gold)[0]
    assert reward(None, gold)[0] == -1


def test_weight_offsets_cover_decision_values_not_prompt():
    target = 'Checked components.\n{"errors":[{"id":"V1","type":"var_to_int"}],"missing":false}'
    offsets = [(i, i + 1) for i in range(len(target))]
    weights = weights_for_response(target, offsets)
    assert weights[target.index("[{") + 1] == 10
    assert all(
        weights[i] == 3 for i in range(target.index("var_to_int"), target.index("var_to_int") + 10)
    )
    assert weights[0] == 1
    clean = '{"errors":[],"missing":false}'
    assert (
        weights_for_response(clean, [(i, i + 1) for i in range(len(clean))])[clean.index("]")] == 10
    )


def test_consolidation_removes_replay_and_conflicting_labels(tmp_path):
    row = prerequisites(random.Random(3), "test", "train", 0)[0]
    p = tmp_path / "data.jsonl"
    write_jsonl(p, [row, row])
    rows, stats = consolidate([p])
    assert len(rows) == 1 and stats["input_rows"] == 2
    other = dict(row, target='{"errors":[{"id":"V1","type":"var_to_int"}],"missing":false}')
    write_jsonl(p, [row, other])
    rows, stats = consolidate([p])
    assert not rows
    assert masked_key("Buy 12 cars") == masked_key("Buy 99 cars")


def test_semantic_contract_labels_and_solver_proof():
    rows = full_partial(random.Random(7), "test", "train", 0)
    normalized = [normalize(r) for r in rows]
    assert len(normalized) == 4
    assert sum(bool(r["errors"]) for r in normalized) == 2
    assert {e["type"] for r in normalized for e in r["errors"]} == {"var_to_int", "var_to_cont"}
    assert all("validator" in r["proof"] for r in rows)


def test_edit_bounds_and_application():
    with pytest.raises(ValueError):
        validate_edits({"replace": {"C9": "x <= 3"}}, ["C1"], False)
    with pytest.raises(ValueError):
        validate_edits({"add_constraints": ["x <= 3"]}, [], False)
    model = "## Variables:\nx: continuous, nonnegative\n## Objective:\nmaximize x\n## Constraints:\nC1: x <= 3\n"
    seg = segment(model)
    new = apply_edits(seg, [{"op": "replace", "id": "V1", "text": "x: integer, nonnegative"}])
    assert "integer" in assemble(new) and "continuous" in assemble(seg)


def test_format_abstention_is_not_clean():
    class Fake:
        def call(self, *args, **kwargs):
            return "not json"

    out = diagnose(Fake(), "problem", ["V1"])
    assert not out["valid"] and out["diagnosis"] is None and len(out["attempts"]) == 2

    class Truncated:
        def call(self, *args, **kwargs):
            raise RuntimeError("Output remains truncated after one concise retry")

    out = diagnose(Truncated(), "problem", ["V1"])
    assert not out["valid"] and "truncated" in out["error"] and out["attempts"] == [None, None]


def test_detector_failure_keeps_initial_answer(monkeypatch, tmp_path):
    import or_semantic_detector.pipeline as pipeline

    model = "## Variables:\nx: continuous, nonnegative\n## Objective:\nmaximize x\n## Constraints:\nC1: x <= 3\nEND OF MODEL"

    class Writer:
        def call(self, prompt, kind="generation", **kwargs):
            return model

    class Broken:
        def call(self, *args, **kwargs):
            raise RuntimeError("Output remains truncated after one concise retry")

    monkeypatch.setattr(
        pipeline, "generate_and_solve", lambda c, f, w: {"status": "ok", "objective": 3}
    )
    out = pipeline.run_case("Maximize x.", Writer(), None, Broken(), tmp_path)
    assert out["final"] == out["initial"] == {"status": "ok", "objective": 3}
    assert not out["edited"] and not out["detector"]["valid"]


def test_scoring_requires_optimality_and_finite_values():
    assert correct({"status": "ok", "objective": 10}, 10)
    assert not correct({"status": "time_limit", "objective": 10}, 10)
    assert not correct({"status": "ok", "objective": float("nan")}, 10)
    assert not correct({"status": "ok", "objective": 10}, None)


def test_one_round_pipeline_uses_diagnosis_without_reference(monkeypatch, tmp_path):
    import or_semantic_detector.pipeline as pipeline

    model = "## Sets:\nI={1}\n## Parameters:\nb=3\n## Variables:\nx: continuous, nonnegative\n## Objective:\nmaximize x\n## Constraints:\nC1: x <= b\nEND OF MODEL"

    class Fake:
        def __init__(self, answers):
            self.answers = iter(answers)
            self.kinds = []

        def call(self, prompt, kind="generation", **kwargs):
            self.kinds.append(kind)
            return next(self.answers)

    writer = Fake([model, '{"replace":{"V1":"x: integer, nonnegative"}}'])
    detector = Fake(['{"errors":[{"id":"V1","type":"var_to_int"}],"missing":false}'])
    calls = []

    def solve(client, formulation, work):
        calls.append(formulation)
        return {"status": "ok", "objective": 3}

    monkeypatch.setattr(pipeline, "generate_and_solve", solve)
    out = pipeline.run_case("Choose whole units.", writer, None, detector, tmp_path)
    assert out["edited"] and len(calls) == 2 and "integer" in calls[-1]
    assert writer.kinds == ["initial_math", "math_repair"] and detector.kinds == ["detector"]


def test_solver_certifies_integer_optimum_and_rejects_print_only(tmp_path):
    from or_semantic_detector.execution import execute

    code = """import pyomo.environ as p
model = p.ConcreteModel()
model.x = p.Var(domain=p.NonNegativeIntegers, bounds=(0, 3.5))
model.obj = p.Objective(expr=model.x, sense=p.maximize)
"""
    result = execute(code, tmp_path / "integer")
    assert result["status"] == "ok" and result["objective"] == 3
    result = execute('print("HARNESS_OBJ: 3")', tmp_path / "printed")
    assert result["status"] == "no_model" and result["objective"] is None


def test_client_rejects_persistent_truncation():
    from types import SimpleNamespace
    from or_semantic_detector.client import Client

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="partial"))
            ],
            usage=None,
        )

    client = Client.__new__(Client)
    client.api = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.model, client.log, client.temperature, client.max_tokens, client.events = (
        "fake",
        None,
        0,
        10,
        [],
    )
    with pytest.raises(RuntimeError, match="truncated"):
        client.call("question")
    assert len(calls) == 2 and len(client.events) == 2


def test_semopt_selects_reward_not_reference(monkeypatch, tmp_path):
    from baselines.semopt import search

    class Fake:
        def call(self, prompt, kind):
            return "initial" if kind == "direct" else '{"scores":[0.5]}'

    def solve(client, problem, code, work):
        return dict(ok=True, code=code, objective=1 if code == "initial" else 2, status=2)

    monkeypatch.setattr(search, "solve", solve)

    def scorer(problem, code):
        return 0.1 if code == "initial" else 0.9

    engine = search.Search(Fake(), scorer, tmp_path, simulations=1, breadth=1)
    monkeypatch.setattr(engine, "generate", lambda node: "candidate" if node.depth == 2 else {})
    out = engine.run("Synthetic question")
    assert not out["gated"] and out["selected"]["objective"] == 2
    assert len(out["candidates"]) == 2
