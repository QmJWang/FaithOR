"""Execute model-building code and read certified solver outcomes."""

import math
import re
import subprocess
import sys
from pathlib import Path


def sanitize_feedback(text):
    text = re.sub(r'File "[^"]+", line', 'File "/work/model.py", line', text)
    return re.sub(r"(?<!\w)/(?:[\w.~-]+/)+[\w.~-]+", "/work/artifact", text)


def execute(code, work, timeout=120):
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    path = work / "model.py"
    path.write_text(code)
    harness = Path(__file__).with_name("solver_harness.py")
    try:
        result = subprocess.run(
            [sys.executable, str(harness), str(path)],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "objective": None, "error": "Execution timed out"}
    (work / "stdout.txt").write_text(result.stdout)
    (work / "stderr.txt").write_text(result.stderr)
    objectives = re.findall(r"^HARNESS_OBJ:\s*(\S+)", result.stdout, re.M)
    statuses = re.findall(r"^HARNESS_STATUS:\s*(\S+)", result.stdout, re.M)
    if result.returncode == 0 and objectives and statuses and statuses[-1] == "optimal":
        try:
            obj = float(objectives[-1])
            if math.isfinite(obj):
                return {"status": "ok", "objective": obj, "error": ""}
        except ValueError:
            pass
    return {
        "status": statuses[-1] if statuses else "execution_error",
        "objective": None,
        "error": sanitize_feedback((result.stderr + "\n" + result.stdout)[-4000:]),
    }
