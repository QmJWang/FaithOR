"""Unit test for --sample-weights in training/sft.py (no model download needed)."""

import importlib.util
import math
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location(
    "sft", Path(__file__).resolve().parents[1] / "training/sft.py"
)
sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft)


def test_sample_weight_lookup():
    t = {"procedural/*": 2, "optmath": 1, "procedural/implicit_domain": 3}
    assert sft.sample_weight("optmath", t) == 1
    assert sft.sample_weight("procedural/ratio_contract", t) == 2
    assert sft.sample_weight("procedural/implicit_domain", t) == 3  # exact beats wildcard
    assert sft.sample_weight("resocratic", t) == 1  # default


def test_all_ones_matches_old_formula():
    n, accum = 11, 4
    m = sft.accum_multipliers([1.0] * n, accum)
    for i in range(n):
        scale = min(accum, n - (i // accum) * accum)  # old implementation
        assert m[i] == 1.0 / scale, (i, m[i], scale)


def test_weight_two_gives_double_gradient():
    # two synthetic samples in one accumulation window: loss_i = theta * x_i
    for w in ([1.0, 1.0], [1.0, 2.0]):
        theta = torch.zeros(1, requires_grad=True)
        m = sft.accum_multipliers(w, accum=2)
        contribs = []
        for i, x in enumerate([1.0, 1.0]):
            theta.grad = None
            (theta * x * m[i]).backward()
            contribs.append(theta.grad.item())
        if w == [1.0, 1.0]:
            assert contribs[0] == contribs[1] == 0.5
        else:
            assert math.isclose(contribs[1] / contribs[0], 2.0, abs_tol=1e-6), contribs
            assert math.isclose(
                sum(contribs), 1.0, abs_tol=1e-6
            )  # weighted average, not weighted sum


if __name__ == "__main__":
    test_sample_weight_lookup()
    test_all_ones_matches_old_formula()
    test_weight_two_gives_double_gradient()
    print("SAMPLE_WEIGHT_TESTS_PASS")
