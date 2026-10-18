"""Solver-verified semantic contracts and controlled model mutations."""

import json
import itertools
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from .protocol import gen_prompt, parse_out


def model_text(sections):
    return "\n".join(
        "## " + name + ":\n" + "\n".join(f"[{cid}] {text}" for cid, text in lines)
        for name, lines in sections
    )


def make_record(
    pid, split, family, variant, question, sections, status, obj, errors, missing, proof
):
    ids = [cid for _, lines in sections for cid, _ in lines]
    result = (
        f"solved to optimality, objective value = {obj:g}"
        if status == "ok"
        else "solver status: infeasible"
    )
    text = (
        "Check the mathematical optimization model below against the problem statement. Every component of the model is labelled with an ID ([S..] sets, [P..] parameters, [V..] decision variables with domains and bounds, [O1] objective, [C..] constraints). Decide which components are wrong and whether a required constraint is missing.\n\n"
        + "### Problem\n"
        + question
        + "\n\n### Solver result of this model\n"
        + result
        + "\n\n### Model\n"
        + model_text(sections)
        + "\n\n### Verdict\n"
    )
    target = f"Checked {len(ids)} components against the problem.\n" + json.dumps(
        dict(errors=[dict(id=cid, type=ty) for cid, ty in errors], missing=missing)
    )
    return dict(
        sid=f"{pid}|{variant}",
        pid=pid,
        split=split,
        family=family,
        variant=variant,
        writer="canonical_v51",
        text=text,
        comp_ids=ids,
        target=target,
        errors=errors,
        missing=missing,
        status=status,
        obj=obj,
        proof=proof,
    )


def lp(c, A, b, integer=False, eq=None):
    c = np.array(c, float)
    A = np.array(A, float)
    b = np.array(b, float)
    if integer:
        if eq is not None:
            E, v = eq
            A = np.vstack([A, E, -np.array(E)])
            b = np.concatenate([b, v, -np.array(v)])
        z = milp(
            -c,
            integrality=np.ones(len(c)),
            bounds=Bounds(0, np.inf),
            constraints=LinearConstraint(A, -np.inf, b),
            options={"mip_rel_gap": 0},
        )
    else:
        z = linprog(
            -c,
            A_ub=A,
            b_ub=b,
            A_eq=None if eq is None else eq[0],
            b_eq=None if eq is None else eq[1],
            bounds=(0, None),
            method="highs",
        )
    assert z.success
    return float(-z.fun), z.x.tolist()


def record(pid, split, fam, var, q, sections, obj, errors=(), missing=False, proof=None):
    r = make_record(
        pid, split, fam, var, q, sections, "ok", obj, list(errors), missing, proof or {}
    )
    r.update(
        question=q,
        source="procedural/" + fam,
        schema_version=1,
        prompt=gen_prompt(r["text"]),
    )
    assert parse_out(r["target"], r["comp_ids"]) is not None
    return r


def full_partial(rng, pid, split, idx):
    words = {
        "train": ["reusable drums", "ribbon reels", "liquid cartridges"],
        "val": ["storage vessels", "fabric rolls"],
        "holdout": ["paint canisters", "coiled wire"],
    }[split]
    noun = words[idx % len(words)]
    n = rng.randint(2, 4)
    while True:
        c = [rng.randint(5, 20) for _ in range(n)]
        A = [[rng.randint(1, 9) for _ in range(n)] for _ in range(2)]
        b = [rng.randint(25, 90) for _ in range(2)]
        cont, cx = lp(c, A, b)
        disc, dx = lp(c, A, b, True)
        if abs(cont - disc) > 0.1:
            break
    result = []
    for partial in [True, False]:
        contract = (
            f"Full or partial {noun} may be used; the amount can be any nonnegative real number, including fractions."
            if partial
            else f"Only sealed whole {noun} may be selected; opening or using part of one is forbidden."
        )
        q = (
            f"A workshop selects {noun} of {n} grades. Per vessel-equivalent, returns are {c}; consumption of the two shared resources by grade is {list(map(list, zip(*A)))}. "
            f"The resource stocks are {b}. "
            + contract
            + " Find quantities that maximize total return."
        )
        for integer in [False, True]:
            typ = "integer" if integer else "continuous"
            wrong = integer == partial
            sections = [
                ("Sets", [("S1", f"I={{1,...,{n}}}")]),
                (
                    "Parameters",
                    [("P1", f"c_i={c}; resource matrix a_ki={A}; limits b_k={b}")],
                ),
                (
                    "Variables",
                    [
                        (
                            "V1",
                            f"x_i: amount of grade i measured in {noun}, i in I; {typ},nonnegative",
                        )
                    ],
                ),
                ("Objective", [("O1", "maximize sum_i c_i x_i")]),
                ("Constraints", [("C1", "sum_i a_ki x_i <= b_k for k in {1,2}")]),
            ]
            result.append(
                record(
                    pid,
                    split,
                    "full_partial",
                    f"{partial}_{integer}",
                    q,
                    sections,
                    disc if integer else cont,
                    [("V1", "var_to_cont" if partial else "var_to_int")] if wrong else [],
                    proof=dict(
                        validator="SciPy_HiGHS_LP_MILP",
                        continuous_optimum=cont,
                        integer_optimum=disc,
                        continuous_x=cx,
                        integer_x=dx,
                    ),
                )
            )
    return result


def role_binding(rng, pid, split, idx):
    # Semantic rows correspond to named resources; mutation swaps roles, not indices by guess.
    a = [rng.randint(1, 6) for _ in range(2)]
    b = [rng.randint(7, 12) for _ in range(2)]
    c = [rng.randint(3, 9) for _ in range(2)]
    while c[0] == c[1]:
        c = [rng.randint(3, 9) for _ in range(2)]
    L = rng.randint(20, 50)
    M = rng.randint(30, 65)
    names = {
        "train": ("press", "scanner"),
        "val": ("dryer", "polisher"),
        "holdout": ("cutter", "welder"),
    }[split]
    q = (
        f"Two operating modes run for arbitrary nonnegative fractional hours. Mode1 earns {c[0]} credits per hour and Mode2 earns {c[1]}. "
        f"The hourly {names[0]} use is {a[0]} for Mode1 and {a[1]} for Mode2; hourly {names[1]} use is {b[0]} and {b[1]}, respectively. "
        f"Available {names[0]} time is {L} and available {names[1]} time is {M}. Maximize earned credits."
    )
    result = []
    for var in ["clean", "parameter_roles", "objective_roles", "equivalent"]:
        aa, bb = (b, a) if var == "parameter_roles" else (a, b)
        cc = c[::-1] if var == "objective_roles" else c
        obj, x = lp(cc, [aa, bb], [L, M])
        o = "maximize " + (" + ".join(f"{cc[i]} x_{i + 1}" for i in range(2)))
        if var == "equivalent":
            o = f"maximize {cc[1]} x_2 + {cc[0]} x_1"
        sections = [
            ("Sets", [("S1", "I={1,2}")]),
            (
                "Parameters",
                [
                    ("P1", f"{names[0]} use a={aa}; {names[1]} use b={bb}"),
                    ("P2", f"credit rates c={c}; capacity L={L},M={M}"),
                ],
            ),
            (
                "Variables",
                [
                    (
                        "V1",
                        "x_1,x_2: hours in Modes1 and2 respectively,continuous,nonnegative",
                    )
                ],
            ),
            ("Objective", [("O1", o)]),
            (
                "Constraints",
                [("C1", "a_1*x_1+a_2*x_2 <= L"), ("C2", "b_1*x_1+b_2*x_2 <= M")],
            ),
        ]
        err = (
            [("P1", "num_value")]
            if var == "parameter_roles"
            else [("O1", "obj_expr")]
            if var == "objective_roles"
            else []
        )
        result.append(
            record(
                pid,
                split,
                "role_binding",
                var,
                q,
                sections,
                obj,
                err,
                proof=dict(validator="SciPy_HiGHS", x=x),
            )
        )
    return result


def bounds_scope(rng, pid, split, idx):
    total = rng.randint(20, 50)
    low = rng.randint(2, 7)
    c = [rng.randint(5, 17), rng.randint(19, 28)]
    q = (
        f"A workshop chooses nonnegative amounts of two divisible blends. Each unit of Blend1 earns {c[0]} credits and each unit of Blend2 earns {c[1]}. "
        f"Together the blends may total at most {total} units. A delivery commitment requires at least {low} units of Blend1. Maximize total credits."
    )
    result = []
    for var in ["clean", "direction", "separate_caps", "equivalent"]:
        A = [[1, 1], [-1, 0]]
        b = [total, -low]
        cons = [("C1", f"x_1+x_2 <= {total}"), ("C2", f"x_1 >= {low}")]
        if var == "direction":
            A[1] = [1, 0]
            b[1] = low
            cons[1] = ("C2", f"x_1 <= {low}")
        if var == "separate_caps":
            A = [[1, 0], [0, 1], [-1, 0]]
            b = [total, total, -low]
            cons = [("C1", f"x_i <= {total} for i in {{1,2}}"), cons[1]]
        if var == "equivalent":
            cons[1] = ("C2", f"-x_1 <= {-low}")
        obj, x = lp(c, A, b)
        sections = [
            ("Sets", [("S1", "I={1,2}")]),
            (
                "Parameters",
                [("P1", f"credits c={c}; total limit={total}; Blend1 minimum={low}")],
            ),
            ("Variables", [("V1", "x_i: amount of Blend i,continuous,nonnegative")]),
            ("Objective", [("O1", f"maximize {c[0]}*x_1+{c[1]}*x_2")]),
            ("Constraints", cons),
        ]
        err = (
            [("C2", "con_sense")]
            if var == "direction"
            else [("C1", "con_structure")]
            if var == "separate_caps"
            else []
        )
        result.append(
            record(
                pid,
                split,
                "bounds_scope",
                var,
                q,
                sections,
                obj,
                err,
                proof=dict(validator="SciPy_HiGHS", x=x),
            )
        )
    return result


def ratio_contract(rng, pid, split, idx):
    ratio = rng.randint(2, 5)
    cap = rng.randint(30, 90)
    a = rng.randint(2, 7)
    b = rng.randint(1, 5)
    c = [rng.randint(2, 7), rng.randint(9, 18)]
    out = []
    for exact in [True, False]:
        phrase = (
            f"Quantity Y must be exactly {ratio} times quantity X."
            if exact
            else f"Quantity Y must be at least {ratio} times quantity X."
        )
        q = (
            f"A plant makes divisible liquids X and Y. Unit earnings are {c[0]} and {c[1]} credits, and unit resource use is {a} and {b}. "
            f"There are {cap} units of the resource. "
            + phrase
            + " Fractional quantities are allowed. Maximize total earnings."
        )
        for equality in [True, False]:
            A = [[a, b]]
            rhs = [cap]
            E = None
            if equality:
                E = ([[-ratio, 1]], [0])
            else:
                A.append([ratio, -1])
                rhs.append(0)
            obj, x = lp(c, A, rhs, eq=E)
            sections = [
                ("Sets", [("S1", "I={X,Y}")]),
                (
                    "Parameters",
                    [
                        (
                            "P1",
                            f"returns={c}; resource use={[a, b]}; stock={cap}; ratio={ratio}",
                        )
                    ],
                ),
                (
                    "Variables",
                    [
                        (
                            "V1",
                            "x,y: produced quantities of X and Y,continuous,nonnegative",
                        )
                    ],
                ),
                ("Objective", [("O1", f"maximize {c[0]}*x+{c[1]}*y")]),
                (
                    "Constraints",
                    [
                        ("C1", f"{a}*x+{b}*y <= {cap}"),
                        ("C2", f"y {'=' if equality else '>='} {ratio}*x"),
                    ],
                ),
            ]
            out.append(
                record(
                    pid,
                    split,
                    "ratio_contract",
                    f"{exact}_{equality}",
                    q,
                    sections,
                    obj,
                    [("C2", "con_sense")] if equality != exact else [],
                    proof=dict(validator="SciPy_HiGHS", x=x),
                )
            )
    return out


def prerequisites(rng, pid, split, idx):
    n = rng.randint(5, 8)
    cost = [rng.randint(2, 8) for _ in range(n)]
    value = [rng.randint(3, 18) for _ in range(n)]
    cap = rng.randint(10, 18)
    edges = [(1, 0), (3, 2)]
    noun = {"train": "device", "val": "permit", "holdout": "feature"}[split]
    q = (
        f"Choose a subset of {n} optional {noun}s, numbered1 through{n}. Each can be selected at most once. Costs are {cost}; benefits are {value}; total cost is at most {cap}. "
        f"{noun.capitalize()}2 may be selected only if {noun}1 is also selected. {noun.capitalize()}4 may be selected only if {noun}3 is also selected. Maximize benefit."
    )
    out = []
    for var in ["clean", "missing", "reversed", "equivalent"]:
        actual = (
            edges[1:] if var == "missing" else [(0, 1), edges[1]] if var == "reversed" else edges
        )
        feasible = [
            p
            for p in itertools.product([0, 1], repeat=n)
            if sum(c * v for c, v in zip(cost, p)) <= cap and all(p[i] <= p[j] for i, j in actual)
        ]
        obj, p = max((sum(v * z for v, z in zip(value, p)), p) for p in feasible)
        # A witness proves the omitted/reversed link changes the feasible set.
        witness = [0] * n
        witness[1] = 1
        assert cost[1] <= cap
        cons = [("C1", f"sum_i cost_i*z_i <= {cap}")]
        for k, (i, j) in enumerate(actual):
            text = f"z_{i + 1} <= z_{j + 1}"
            if var == "equivalent":
                text = f"z_{i + 1}-z_{j + 1} <= 0"
            # Renumber consecutively: a gap in the constraint IDs would reveal the
            # omitted link and let a detector predict missing=true without reading.
            cons.append((f"C{k + 2}", text))
        sections = [
            ("Sets", [("S1", f"I={{1,...,{n}}}")]),
            ("Parameters", [("P1", f"cost_i={cost}; benefit_i={value}; budget={cap}")]),
            ("Variables", [("V1", f"z_i:1 if {noun}i selected,0 otherwise,binary")]),
            ("Objective", [("O1", "maximize sum_i benefit_i*z_i")]),
            ("Constraints", cons),
        ]
        out.append(
            record(
                pid,
                split,
                "prerequisites",
                var,
                q,
                sections,
                obj,
                [("C2", "con_structure")] if var == "reversed" else [],
                var == "missing",
                proof=dict(
                    validator="exhaustive_binary_enumeration",
                    x=p,
                    dependency_witness=witness,
                ),
            )
        )
    return out


def solve(c, A, b, integer, maximize):
    z = -np.array(c, float) if maximize else np.array(c, float)
    if integer:
        r = milp(
            z,
            integrality=np.ones(len(c)),
            bounds=Bounds(0, np.inf),
            constraints=LinearConstraint(A, -np.inf, b),
            options={"mip_rel_gap": 0},
        )
    else:
        r = linprog(z, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    assert r.success
    return float(-r.fun if maximize else r.fun), r.x.tolist()


def domain_group(rng, pid, split, index):
    # Positive bounded packing and covering LPs/MILPs; reject no-gap instances.
    n = rng.randint(2, 4)
    maximize = index % 2 == 0
    while True:
        cost = [rng.randint(3, 27) / 2 for _ in range(n)]
        raw = np.array([[rng.randint(1, 13) for _ in range(n)] for _ in range(3)], float)
        rhs = np.array([rng.randint(35, 180) for _ in range(3)], float)
        A = raw if maximize else -raw
        b = rhs if maximize else -rhs
        lp, lx = solve(cost, A, b, False, maximize)
        ip, ix = solve(cost, A, b, True, maximize)
        if abs(lp - ip) > 0.15:
            break
    whole, bulk = CONTEXTS[split][index % len(CONTEXTS[split])]
    result = []
    for sem, item in [("count", whole), ("divisible", bulk)]:
        if maximize:
            clauses = [
                f"{item} of option {i + 1} yields {cost[i]:g} credits and uses "
                + ", ".join(f"{raw[k, i]:g} minutes of workstation {k + 1}" for k in range(3))
                for i in range(n)
            ]
            goal = "maximize total credits"
            limits = "Workstation time budgets are " + str(rhs.astype(int).tolist()) + " minutes."
        else:
            clauses = [
                f"Option {i + 1} costs {cost[i]:g} credits per "
                + item
                + " and supplies "
                + ", ".join(f"{raw[k, i]:g} units of service category {k + 1}" for k in range(3))
                for i in range(n)
            ]
            goal = "minimize total cost"
            limits = "Required minimum service amounts are " + str(rhs.astype(int).tolist()) + "."
        # Counts intentionally omit explicit "integer"/"whole" training shortcuts.
        intro = f"An operations planner chooses how many {item} of each of {n} options to use. "
        if sem == "divisible":
            intro = f"An operations planner chooses amounts measured in {item} for {n} options. Fractional amounts are permitted. "
        if split == "train":
            q = (
                intro
                + "; ".join(clauses)
                + ". "
                + limits
                + " Determine quantities to "
                + goal
                + "."
            )
        elif split == "val":
            q = (
                "Determine a plan to "
                + goal
                + ". "
                + limits
                + " Option specifications: "
                + "; ".join(clauses)
                + ". "
                + intro
            )
        else:
            q = (
                intro
                + limits
                + " The following option specifications apply: "
                + "; ".join(clauses)
                + ". Which quantities "
                + goal
                + "?"
            )
        for integer in [False, True]:
            indexed = index % 2 == 0
            variables = (
                [
                    (
                        "V1",
                        "x_i: "
                        + ("number" if sem == "count" else "amount")
                        + f" of {item} of option i, i in I; "
                        + ("integer" if integer else "continuous")
                        + ", nonnegative",
                    )
                ]
                if indexed
                else [
                    (
                        f"V{i + 1}",
                        f"x_{i + 1}: "
                        + ("number" if sem == "count" else "amount")
                        + f" of {item} of option {i + 1}; "
                        + ("integer" if integer else "continuous")
                        + ", nonnegative",
                    )
                    for i in range(n)
                ]
            )
            obj = (
                ("maximize" if maximize else "minimize")
                + " "
                + " + ".join(f"{cost[i]:g} x_{i + 1}" for i in range(n))
            )
            sense = "<=" if maximize else ">="
            sections = [
                ("Sets", [("S1", f"I = {{1,...,{n}}}")]),
                (
                    "Parameters",
                    [
                        ("P1", f"unit values/costs = {cost}"),
                        (
                            "P2",
                            f"resource/service coefficients by category and option = {raw.astype(int).tolist()}",
                        ),
                        ("P3", f"budgets/requirements = {rhs.astype(int).tolist()}"),
                    ],
                ),
                ("Variables", variables),
                ("Objective", [("O1", obj)]),
                (
                    "Constraints",
                    [
                        (
                            f"C{k + 1}",
                            " + ".join(f"{raw[k, i]:g} x_{i + 1}" for i in range(n))
                            + f" {sense} {rhs[k]:g}",
                        )
                        for k in range(3)
                    ],
                ),
            ]
            # Indexed expressions use x_1 etc. as instances of x_i; no hidden domain.
            wrong = integer != (sem == "count")
            ty = "var_to_int" if sem == "count" else "var_to_cont"
            errors = [(cid, ty) for cid, _ in variables] if wrong else []
            r = make_record(
                pid,
                split,
                "implicit_domain",
                sem + ("_integer" if integer else "_continuous"),
                q,
                sections,
                "ok",
                ip if integer else lp,
                errors,
                False,
                dict(
                    validator="independent_SciPy_HiGHS_LP_and_MILP",
                    continuous_optimum=lp,
                    integer_optimum=ip,
                    lp_x=lx,
                    integer_x=ix,
                    A=A.tolist(),
                    b=b.tolist(),
                    cost=cost,
                    context=item,
                ),
            )
            r.update(
                question=q,
                source="procedural/implicit_domain",
                schema_version=1,
                prompt=gen_prompt(r["text"]),
            )
            result.append(r)
    return result


def scope_group(rng, pid, split, index):
    # No narrative from diagnostic examples: generic device operating time and costs.
    n = 4
    rates = np.array([rng.randint(2, 9) for _ in range(n)], float)
    prices = np.array([rng.randint(15, 25) for _ in range(n)], float)
    op = np.array([rng.randint(1, 4) for _ in range(n)], float)
    inputs = np.array([rng.randint(2, 5) for _ in range(n)], float)
    inputprice = rng.randint(2, 4)
    net = prices - op - inputs * inputprice
    cap = rng.randint(45, 95)
    sub = rng.randint(4, 12)
    limit = rng.randint(12, 25)
    q = (
        f"A facility operates four devices for nonnegative amounts of time; fractional hours are allowed. Device rates are {rates.astype(int).tolist()} units per hour. "
        f"Selling prices per output unit are {prices.astype(int).tolist()} credits. Handling costs per unit are {op.astype(int).tolist()} credits. "
        f"Each output unit consumes {inputs.astype(int).tolist()} units of purchased input, respectively, at {inputprice} credits per input unit. "
        f"At most {cap} total operating hours are available. Devices1 and3 together may run at most {sub} hours. Each device may run at most {limit} hours. "
        "Choose operating hours to maximize net profit after all listed costs."
    )
    result = []
    for variant in ["clean", "scope", "cost", "equivalent"]:
        coef = (prices - op) * rates if variant == "cost" else net * rates
        chosen = [0, 1, 2, 3] if variant == "scope" else [0, 2]
        A = np.vstack([np.ones(n), [int(i in chosen) for i in range(n)], np.eye(n)])
        b = np.array([cap, sub] + [limit] * n, float)
        obj, x = solve(coef, A, b, False, True)
        expr = " + ".join(f"{coef[i]:g} x_{i + 1}" for i in range(n))
        if variant == "equivalent":
            expr = " + ".join(
                f"({prices[i]:g}-{op[i]:g}-{inputs[i]:g}*{inputprice})*{rates[i]:g}*x_{i + 1}"
                for i in range(n)
            )
        sections = [
            ("Sets", [("S1", "I = {1,2,3,4}")]),
            (
                "Parameters",
                [
                    (
                        "P1",
                        f"rates={rates.tolist()}; prices={prices.tolist()}; handling={op.tolist()}; inputs={inputs.tolist()}; input price={inputprice}",
                    )
                ],
            ),
            (
                "Variables",
                [
                    (
                        "V1",
                        "x_i: operating hours of device i in I, continuous, nonnegative",
                    )
                ],
            ),
            ("Objective", [("O1", "maximize " + expr)]),
            (
                "Constraints",
                [
                    ("C1", f"x_1+x_2+x_3+x_4 <= {cap}"),
                    ("C2", " + ".join(f"x_{i + 1}" for i in chosen) + f" <= {sub}"),
                    ("C3", f"x_i <= {limit} for each i in I"),
                ],
            ),
        ]
        errors = (
            [("C2", "con_structure")]
            if variant == "scope"
            else [("O1", "obj_expr")]
            if variant == "cost"
            else []
        )
        r = make_record(
            pid,
            split,
            "scope_cost",
            variant,
            q,
            sections,
            "ok",
            obj,
            errors,
            False,
            dict(
                validator="SciPy_HiGHS_exact_linear_model",
                A=A.tolist(),
                b=b.tolist(),
                coefficients=coef.tolist(),
                x=x,
            ),
        )
        r.update(
            question=q,
            source="procedural/scope_cost",
            schema_version=1,
            prompt=gen_prompt(r["text"]),
        )
        result.append(r)
    return result


CONTEXTS = {
    "train": [
        ("sensor modules", "litres of liquid resin"),
        ("printed booklets", "metres of textile"),
        ("sealed test kits", "kilograms of loose granules"),
        ("lighting fixtures", "hours of machine operation"),
        ("delivery visits", "litres of cleaning solution"),
        ("ceramic tiles", "square metres of coated surface"),
    ],
    "val": [
        ("rental bicycles", "kilograms of bulk pigment"),
        ("scheduled appointments", "metres of cable"),
    ],
    "holdout": [
        ("ticket holders", "litres of blended oil"),
        ("portable generators", "hectares of cultivated area"),
    ],
}
