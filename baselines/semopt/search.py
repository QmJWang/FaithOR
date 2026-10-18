"""Reward-guided search reconstructed from the SemOPT paper."""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from .common import DIRECT, code_only, json_only, solve, append_event
from .prompts import STRATEGY, MATH, CODE


@dataclass
class Node:
    depth: int
    value: object
    parent: object = None
    children: list = field(default_factory=list)
    visits: int = 0
    q: float = 0.0
    prior: float = 0.0
    rollout: object = None

    def path(self):
        result = []
        x = self
        while x is not None:
            result.append(x)
            x = x.parent
        return result[::-1]


class Search:
    def __init__(self, relay, scorer, work, simulations=16, breadth=4, omega=1.414, threshold=0.6):
        self.relay = relay
        self.scorer = scorer
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.simulations = simulations
        self.breadth = breadth
        self.omega = omega
        self.threshold = threshold
        self.candidates = []
        self.counter = 0

    def generate(self, node):
        path = node.path()
        problem = path[0].value
        if node.depth == 0:
            return json_only(
                self.relay.call(STRATEGY.format(problem_description=problem), "strategy")
            )
        strategy = path[1].value
        if node.depth == 1:
            return json_only(
                self.relay.call(MATH.format(original_problem=problem, **strategy), "math")
            )
        return code_only(
            self.relay.call(
                CODE.format(
                    original_problem=problem,
                    problem_type=strategy["problem_type"],
                    decision_variables=strategy["decision_variables"],
                    math_model=json.dumps(node.value),
                ),
                "solver_code",
            )
        )

    def priors(self, problem, candidates):
        # The paper specifies LLM priors but supplies no prior-scoring prompt.
        prompt = (
            'Rate the semantic faithfulness and promise of each partial optimization model for the problem. Return JSON {"scores": [numbers between 0 and 1]}, one score per candidate, in order.\nProblem:\n'
            + problem
            + "\nCandidates:\n"
            + json.dumps(candidates)
        )
        try:
            scores = json_only(self.relay.call(prompt, "prior"))["scores"]
            if len(scores) != len(candidates):
                raise ValueError("prior length mismatch")
            return [max(0.0, min(1.0, float(x))) for x in scores]
        except (ValueError, KeyError, TypeError):
            return [0.0] * len(candidates)

    def evaluate(self, problem, code):
        self.counter += 1
        result = solve(self.relay, problem, code, self.work / f"candidate_{self.counter}")
        score_start = time.perf_counter()
        result["reward"] = float(self.scorer(problem, result["code"])) if result["ok"] else 0.0
        result["reward_wall_seconds"] = time.perf_counter() - score_start
        append_event(
            self.work / "reward_usage.jsonl",
            dict(
                candidate=self.counter,
                wall_seconds=result["reward_wall_seconds"],
                scored=result["ok"],
            ),
        )
        result["candidate"] = self.counter
        self.candidates.append(result)
        return result

    def run(self, problem):
        initial = self.evaluate(
            problem,
            code_only(self.relay.call(DIRECT.format(problem=problem), "direct")),
        )
        if initial["reward"] > self.threshold:
            return dict(gated=True, selected=initial, candidates=self.candidates, simulations=0)
        root = Node(0, problem)
        for _ in range(self.simulations):
            node = root
            while node.children:

                def uct(x):
                    if x.visits == 0:
                        return float("inf") + x.prior
                    return x.q + self.omega * math.sqrt(
                        math.log(max(1, node.visits)) / (x.visits + 1e-8)
                    )

                unvisited = [x for x in node.children if x.visits == 0]
                node = (
                    max(unvisited, key=lambda x: x.prior)
                    if unvisited
                    else max(node.children, key=uct)
                )
            try:
                if node.depth < 3:
                    children = []
                    for _ in range(self.breadth):
                        try:
                            children.append(Node(node.depth + 1, self.generate(node), node))
                        except (ValueError, KeyError, TypeError):
                            continue
                    if not children:
                        node.visits += 1
                        continue
                    scores = self.priors(problem, [x.value for x in children])
                    for x, s in zip(children, scores):
                        x.prior = s
                    node.children = children
                    node = max(children, key=lambda x: x.prior)
                rollout = node
                while rollout.depth < 3:
                    rollout = Node(rollout.depth + 1, self.generate(rollout), rollout)
                result = self.evaluate(problem, rollout.value)
            except (ValueError, KeyError, TypeError):
                result = dict(
                    ok=False,
                    reward=0.0,
                    code="",
                    objective=None,
                    status="invalid_model_output",
                )
            node.rollout = result
            cur = node
            while cur is not None:
                cur.visits += 1
                cur.q += (result["reward"] - cur.q) / cur.visits
                cur = cur.parent
        # Report the reward-selected BoN answer, and preserve Algorithm 1's greedy-Q
        # trajectory separately because rollout materialization is underspecified.
        selected = max(self.candidates, key=lambda x: x["reward"])
        node = root
        while node.children:
            visited = [x for x in node.children if x.visits]
            if not visited:
                break
            node = max(visited, key=lambda x: x.q)
        return dict(
            gated=False,
            selected=selected,
            greedy_q_selected=node.rollout,
            candidates=self.candidates,
            simulations=self.simulations,
        )
