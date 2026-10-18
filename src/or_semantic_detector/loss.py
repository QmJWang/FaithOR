"""Response-only decision and value token weighting for SFT."""

import json
import re


def weights_for_response(target, offsets, choice_weight=10.0, value_weight=3.0):
    start = target.find("{")
    assert start >= 0
    obj, end = json.JSONDecoder().raw_decode(target[start:])
    assert isinstance(obj, dict)
    assert isinstance(obj["errors"], list) and type(obj["missing"]) is bool
    match = re.search(r'"errors"\s*:\s*\[\s*', target[start:])
    assert match
    array_start = start + target[start:].find("[", match.start())
    _, length = json.JSONDecoder().raw_decode(target[array_start:])
    array_end = array_start + length
    decision = start + match.end()
    weights = [1.0] * len(offsets)

    def mark(a, b, weight):
        found = False
        for i, (left, right) in enumerate(offsets):
            if left < b and right > a:
                weights[i] = max(weights[i], weight)
                found = True
        assert found, (a, b, target[a:b])

    mark(decision, decision + 1, choice_weight)
    for m in re.finditer(r'"(?:id|type)"\s*:\s*"([^"\\]*)"', target[array_start:array_end]):
        mark(array_start + m.start(1), array_start + m.end(1), value_weight)
    missing = re.search(r'"missing"\s*:\s*(true|false)', target[start:])
    assert missing
    mark(start + missing.start(1), start + missing.end(1), value_weight)
    return weights


def weighted_response_loss(logits, targets, weights):
    """Logits are already aligned next-token response predictions; prompt excluded."""
    import torch

    assert logits.ndim == 2 and len(logits) == len(targets) == len(weights)
    ce = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="none")
    w = torch.as_tensor(weights, device=ce.device, dtype=ce.dtype)
    assert bool(torch.isfinite(w).all()) and bool((w >= 1).all())
    return (ce * w).sum() / w.sum()
