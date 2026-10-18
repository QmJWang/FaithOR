# verl reward adapter using annotated component labels, with no LLM judge.
# Weights from env: GEN_W_TYPE (0.2), GEN_R_MISS_POS (-0.5), GEN_R_NEG_OK (1), GEN_R_NEG_BAD (-1), GEN_R_FMT (-1).
import json
import os
from or_semantic_detector.protocol import parse_out, reward

W = dict(
    w_type=float(os.environ.get("GEN_W_TYPE", 0.2)),
    r_miss_pos=float(os.environ.get("GEN_R_MISS_POS", -0.5)),
    r_neg_ok=float(os.environ.get("GEN_R_NEG_OK", 1.0)),
    r_neg_bad=float(os.environ.get("GEN_R_NEG_BAD", -1.0)),
    r_fmt=float(os.environ.get("GEN_R_FMT", -1.0)),
)
print("[gen_reward]", W, flush=True)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kw):
    d = json.loads(extra_info["gold"])
    pred = parse_out(solution_str, d["comp_ids"])
    s, info = reward(pred, d, **W)
    info["score"] = float(s)
    info["resp_chars"] = float(len(solution_str))
    return info
