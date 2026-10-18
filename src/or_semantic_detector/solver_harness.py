# Solve harness: exec a generated Pyomo script (which only BUILDS a model), then solve it here.
# Usage: python solver_harness.py code.py (install Pyomo and the required solver).
# Prints one of:  "HARNESS_OBJ: <float>" | "HARNESS_STATUS: infeasible|unbounded|no_model|solver_error|..." plus info lines.
import sys
import time
import traceback
import io
import contextlib
import pyomo.environ as pyo
from pyomo.core.expr.visitor import polynomial_degree


def find_model(ns):
    m = ns.get("model")
    if isinstance(m, pyo.ConcreteModel):
        return m
    cands = [v for v in ns.values() if isinstance(v, pyo.ConcreteModel)]
    return cands[-1] if cands else None


def degree_info(m):
    maxdeg, nonpoly, has_int = 0, False, False
    for v in m.component_data_objects(pyo.Var, active=True):
        if not v.is_continuous():
            has_int = True
    exprs = [o.expr for o in m.component_data_objects(pyo.Objective, active=True)]
    exprs += [c.body for c in m.component_data_objects(pyo.Constraint, active=True)]
    for e in exprs:
        d = polynomial_degree(e)
        if d is None:
            nonpoly = True
        else:
            maxdeg = max(maxdeg, d)
    return maxdeg, nonpoly, has_int


def main():
    path = sys.argv[1]
    src = open(path).read()
    ns = {"__name__": "__main__", "__file__": path}
    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, path, "exec"), ns)
    except SystemExit:
        pass
    except Exception:
        print(buf.getvalue()[-4000:])
        print("HARNESS_EXEC_ERROR")
        traceback.print_exc(limit=3)
        print("HARNESS_STATUS: exec_error")
        return
    print(buf.getvalue()[-4000:])
    m = find_model(ns)
    if m is None:
        print("HARNESS_STATUS: no_model")
        return
    objs = list(m.component_data_objects(pyo.Objective, active=True))
    if len(objs) != 1:
        print(f"HARNESS_STATUS: n_objectives={len(objs)}")
        return
    obj = objs[0]
    try:
        maxdeg, nonpoly, has_int = degree_info(m)
    except Exception:
        traceback.print_exc(limit=2)
        print("HARNESS_STATUS: model_error")
        return
    linear = (not nonpoly) and maxdeg <= 1
    solver_name = "appsi_highs" if linear else "scip"
    print(
        f"HARNESS_INFO: maxdeg={maxdeg} nonpoly={nonpoly} has_int={has_int} solver={solver_name} exec_s={time.time() - t0:.1f}"
    )
    try:
        s = pyo.SolverFactory(solver_name)
        if solver_name == "appsi_highs":
            s.options["mip_rel_gap"] = 1e-9
            s.options["time_limit"] = 45
        else:
            s.options["limits/gap"] = 1e-9
            s.options["limits/time"] = 45
        res = s.solve(m, load_solutions=False)
        tc = res.solver.termination_condition
        print(f"HARNESS_INFO: status={res.solver.status} tc={tc}")
        TC = pyo.TerminationCondition
        if tc in (TC.infeasible,):
            print("HARNESS_STATUS: infeasible")
            return
        if tc in (TC.unbounded,):
            print("HARNESS_STATUS: unbounded")
            return
        if tc in (TC.infeasibleOrUnbounded,):
            print("HARNESS_STATUS: infeasible_or_unbounded")
            return
        if tc not in (TC.optimal, TC.globallyOptimal):
            print(f"HARNESS_STATUS: solver_fail_{tc}")
            return
        m.solutions.load_from(res)
        val = pyo.value(obj)
        print(f"HARNESS_OBJ: {val!r}")
        print("HARNESS_STATUS: optimal")
    except Exception:
        traceback.print_exc(limit=3)
        print("HARNESS_STATUS: solver_error")


if __name__ == "__main__":
    main()
