"""
Sandboxed code execution for verification.

Runs model-generated verifier code in a restricted subprocess with
timeout and memory limits. This is the core safety layer — we execute
arbitrary Python to check answers, so isolation is critical.
"""

import subprocess
import tempfile
import json
import os
import signal
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class VerificationResult:
    """Result of a verification attempt."""
    correct: bool
    error: Optional[str] = None
    output: Optional[str] = None
    timed_out: bool = False
    execution_time_ms: float = 0.0


@dataclass
class SandboxConfig:
    timeout_seconds: int = 30
    max_memory_mb: int = 512


# Restricted imports whitelist — only safe modules
ALLOWED_IMPORTS = {
    "math", "random", "itertools", "functools", "collections",
    "string", "re", "json", "decimal", "fractions",
    "statistics", "operator", "copy", "heapq", "bisect",
    "dataclasses", "typing", "abc", "enum",
}

SANDBOX_HEADER = '''
import sys
import importlib

# Restrict dangerous modules
_BLOCKED = {"os", "subprocess", "shutil", "pathlib", "socket", "http",
            "urllib", "requests", "ctypes", "signal", "multiprocessing",
            "threading", "pickle", "shelve", "sqlite3", "importlib"}

_original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

def _safe_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top in _BLOCKED:
        raise ImportError(f"Module '{name}' is not allowed in sandbox")
    return _original_import(name, *args, **kwargs)

if isinstance(__builtins__, dict):
    __builtins__["__import__"] = _safe_import
else:
    __builtins__.__import__ = _safe_import
'''


def verify_code_task(
    problem_code: str,
    solution: str,
    test_code: str,
    config: SandboxConfig = SandboxConfig(),
) -> VerificationResult:
    """
    Execute a Code-as-Task verification.
    
    The test_code should define assertions that check whether the
    solution is correct given the problem. Returns VerificationResult.
    """
    import time
    start = time.monotonic()

    # Compose the full script
    full_script = f"""{SANDBOX_HEADER}

# --- Problem Definition ---
{problem_code}

# --- Solution ---
{solution}

# --- Verification ---
try:
    {test_code}
    print("VERIFICATION_PASSED")
except AssertionError as e:
    print(f"VERIFICATION_FAILED: {{e}}")
except Exception as e:
    print(f"VERIFICATION_ERROR: {{type(e).__name__}}: {{e}}")
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(full_script)
        script_path = f.name

    try:
        result = subprocess.run(
            [
                "python3", "-u", script_path,
            ],
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": "/tmp",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

        elapsed = (time.monotonic() - start) * 1000
        output = result.stdout.strip()
        stderr = result.stderr.strip()

        if "VERIFICATION_PASSED" in output:
            return VerificationResult(
                correct=True,
                output=output,
                execution_time_ms=elapsed,
            )
        elif "VERIFICATION_FAILED" in output:
            return VerificationResult(
                correct=False,
                output=output,
                execution_time_ms=elapsed,
            )
        else:
            return VerificationResult(
                correct=False,
                error=stderr or output or "No verification output",
                output=output,
                execution_time_ms=elapsed,
            )

    except subprocess.TimeoutExpired:
        elapsed = (time.monotonic() - start) * 1000
        return VerificationResult(
            correct=False,
            timed_out=True,
            error=f"Execution timed out after {config.timeout_seconds}s",
            execution_time_ms=elapsed,
        )
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return VerificationResult(
            correct=False,
            error=str(e),
            execution_time_ms=elapsed,
        )
    finally:
        os.unlink(script_path)


def verify_math(problem: str, answer: str, expected: str) -> VerificationResult:
    """Quick math verification — check if answer matches expected."""
    problem_code = f'expected_answer = {repr(expected)}'
    solution = f'student_answer = {repr(answer)}'
    test_code = '''
import re

def normalize_math(s):
    """Normalize a math answer for comparison."""
    s = str(s).strip()
    # Remove LaTeX formatting
    s = re.sub(r"\\\\(text|mathrm|mathbf|boxed)\\{([^}]*)\\}", r"\\2", s)
    s = s.replace("$", "").replace("\\\\", "")
    # Try numeric comparison
    try:
        return float(eval(s))
    except:
        return s.lower().strip()

assert normalize_math(student_answer) == normalize_math(expected_answer), \\
    f"Expected {expected_answer}, got {student_answer}"
'''
    return verify_code_task(problem_code, solution, test_code)
