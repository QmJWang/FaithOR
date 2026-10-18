"""Three published expansion prompts from Appendix I, arXiv:2609.37361v1."""

STRATEGY = """You are an expert in Operations Research. Your task is to determine the modeling strategy for the given problem.
Problem Description:
{problem_description}
Your Task:
1. Identify the specific optimization framework (e.g., LP, MILP, NLP).
2. Define the decision variables, specifying their types (Continuous, Integer, Binary) and semantic meaning.
3. List implicit constraints and key modeling assumptions.
Output Format:
Return a strictly valid JSON object with the following keys:
- "problem_type": The algorithmic framework (e.g., "Mixed Integer Linear Programming").
- "decision_variables": A list of objects, each with "name", "type", and "description".
- "assumptions": A list of strings clarifying ambiguities or implicit logic (e.g., "X must be non-negative").
- "strategy_summary": A concise high-level approach (no formulas).
Guidelines:
- Variable Types: Be rigorous. Use INTEGER/BINARY for discrete entities (people, counts) and CONTINUOUS for rates/amounts.
- Framework: Choose MILP if *any* discrete variables or logical constraints (if-then) are present.
- Output ONLY the JSON object."""
MATH = """You are an expert in mathematical optimization modeling. Translate the provided strategy into a clean LaTeX-style math model.
Context:
- Original problem: {original_problem}
- Problem type: {problem_type}
- Decision Variables: {decision_variables}
- Assumptions: {assumptions}
- Strategy Summary: {strategy_summary}
Task Definition:
1. Define all sets, parameters, variable declarations, objective, and constraint expressions based strictly on the provided strategy.
2. You must strictly adhere to the "Decision Variables" defined in the strategy snapshot. Do not change their types or meanings.
3. Preserve LaTeX notation, maintain consistent units, and ensure every implied constraint appears explicitly.
Output Format:
Provide the model as a single JSON object with exactly these five keys: "sets", "parameters", "variables", "objective", and "constraints".
Key Reminders:
- Output only the JSON object---no explanatory text.
- Keep each field non-empty; add "None" explicitly if nothing applies.
- Ensure notation and units stay consistent throughout the math model."""
CODE = """You are an expert in optimization programming with Gurobi. Your task is to implement the provided math model as executable Python code.
Context:
- Original Problem: {original_problem}
- Strategy Snapshot:
  - Problem Type: {problem_type}
  - Decision Variables: {decision_variables}
- Math Model: {math_model}
Task Definition:
1. Import necessary libraries (gurobipy).
2. Define sample data for all parameters and sets using reasonable example values. The code must be directly executable.
3. Create the Gurobi model and define decision variables, strictly adhering to the types (Continuous, Binary, Integer) specified in the Strategy and Math Model.
4. Implement the objective function and all constraints exactly as formulated in the Math Model.
5. Perform the optimization (model.optimize()).
6. Include error handling for GRB.INFEASIBLE or GRB.UNBOUNDED statuses.
7. CRITICAL: If the solution is optimal, print the objective value using EXACTLY this format: "Optimal objective value: " + str(model.objVal).
Output Format:
Provide ONLY the Python code. Do not include markdown formatting or explanatory prose."""
