"""Solver execution utilities for the SemOPT paper reconstruction."""

import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from or_semantic_detector.execution import sanitize_feedback


def append_event(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(event) + "\n")


def code_only(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return blocks[-1].strip() if blocks else text.strip()


def json_only(text):
    for m in re.finditer(r"\{", text):
        try:
            d, _ = json.JSONDecoder().raw_decode(text[m.start() :])
            if isinstance(d, dict):
                return d
        except ValueError:
            pass
    raise ValueError("no JSON object in response")


def execute(code, work, timeout=120, reference=False):
    started = time.time()
    clock = time.perf_counter()
    try:
        result = _execute(code, work, timeout, reference)
    except Exception as error:
        append_event(
            Path(work) / "solver_usage.jsonl",
            dict(
                started_at=started,
                ended_at=time.time(),
                wall_seconds=time.perf_counter() - clock,
                status="infrastructure_exception",
                error_type=type(error).__name__,
                reference=reference,
            ),
        )
        raise
    result["wall_seconds"] = time.perf_counter() - clock
    append_event(
        Path(work) / "solver_usage.jsonl",
        dict(
            started_at=started,
            ended_at=time.time(),
            wall_seconds=result["wall_seconds"],
            status=result["status"],
            reference=reference,
        ),
    )
    return result


def _execute(code, work, timeout=120, reference=False):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "code.py").write_text(code)
    # The wrapper reads the solver's actual status/objective, never trusts model-written stdout.
    wrapper = """import json,runpy,traceback
try:
 ns=runpy.run_path('code.py',run_name='__main__')
 models=[]
 for k,v in ns.items():
  if type(v).__name__=='Model' and type(v).__module__.startswith(('gurobipy','copt')):models.append(v)
 if not models:raise RuntimeError('No solver model found')
 model=ns.get('model',models[-1])
 try: status=int(model.Status);objective=float(model.ObjVal) if status==2 else None
 except AttributeError: status=int(model.status);objective=float(model.objval) if status==1 else None
 print();print('SEMOPT_RESULT '+json.dumps(dict(status=status,objective=objective)))
except Exception:
 traceback.print_exc()
 raise
"""
    # COPT uses getVars/getConstrs and status=1, Gurobi uses status=2.
    if reference:
        wrapper = wrapper.replace(
            "try: status=int(model.Status);objective=float(model.ObjVal) if status==2 else None\n except AttributeError: status=int(model.status);objective=float(model.objval) if status==1 else None",
            "status=int(model.status);objective=float(model.objval) if status==1 else None",
        )
    (work / "harness.py").write_text(wrapper)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    try:
        r = subprocess.run(
            [sys.executable, "harness.py"],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return dict(ok=False, status="timeout", objective=None, error="Execution timed out")
    (work / "stdout.txt").write_text(r.stdout)
    (work / "stderr.txt").write_text(r.stderr)
    matches = re.findall(r"^SEMOPT_RESULT (.+)$", r.stdout, re.M)
    if r.returncode == 0 and matches:
        d = json.loads(matches[-1])
        return dict(
            ok=d["objective"] is not None,
            status=d["status"],
            objective=d["objective"],
            error=""
            if d["objective"] is not None
            else "Solver returned non-optimal status " + str(d["status"]),
        )
    error = sanitize_feedback((r.stderr or r.stdout)[-4000:])
    if "size-limited license" in error:
        raise RuntimeError("Gurobi license limit, not a model error")
    return dict(ok=False, status="execution_error", objective=None, error=error)


DIRECT = """You are an expert in operations research and Gurobi. Solve the following optimization problem by writing directly executable Python code. Use the exact data in the problem and print the optimal objective value. Output only Python code.
Problem:
{problem}"""

REPAIR = """The following optimization code did not execute to an optimal solution. Correct it according to the original problem and execution feedback. Output only executable Gurobi Python code.
Problem:
{problem}
Code:
{code}
Feedback:
{error}"""


def solve(relay, problem, code, work, max_repairs=3):
    for r in range(max_repairs + 1):
        result = execute(code, Path(work) / str(r))
        if result["ok"] or r == max_repairs:
            return dict(result, code=code, syntax_repairs=r)
        code = code_only(
            relay.call(
                REPAIR.format(problem=problem, code=code, error=result["error"]),
                "syntax_repair",
            )
        )
