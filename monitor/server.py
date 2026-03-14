"""
Prometheus Monitor Server
FastAPI backend — reads training logs, serves GPU stats, streams updates.
Run: uvicorn monitor.server:app --host 0.0.0.0 --port 8080 --reload
"""

import json
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Prometheus Monitor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"


def parse_log(log_path: Path) -> dict:
    """Parse a training log file into structured data."""
    if not log_path.exists():
        return {"steps": [], "error": "Log file not found"}

    text = log_path.read_text(errors="replace")
    steps = []

    # Parse each step block
    step_pattern = re.compile(
        r"--- Step (\d+) \| (\d+)s elapsed \| (\d+)s remaining ---"
    )
    loss_pattern = re.compile(r"GRPO Loss:\s*([\d.]+)")
    acc_pattern = re.compile(r"Rollout accuracy:\s*([\d.]+)%\s*\((\d+)/(\d+)\)")
    domain_pattern = re.compile(r"Domain:\s*(\w+)\s*\|")
    problem_pattern = re.compile(r"📝 Problem:\s*(.+?)(?:\.\.\.|$)", re.MULTILINE)
    skip_pattern = re.compile(r"⏭️")

    lines = text.split("\n")
    current_step = None

    for i, line in enumerate(lines):
        step_m = step_pattern.search(line)
        if step_m:
            if current_step:
                steps.append(current_step)
            current_step = {
                "step": int(step_m.group(1)),
                "elapsed": int(step_m.group(2)),
                "remaining": int(step_m.group(3)),
                "loss": None,
                "accuracy": None,
                "domain": None,
                "problem": None,
                "skipped": False,
                "trained": False,
            }
            continue

        if current_step is None:
            continue

        loss_m = loss_pattern.search(line)
        if loss_m:
            current_step["loss"] = float(loss_m.group(1))
            current_step["trained"] = True

        acc_m = acc_pattern.search(line)
        if acc_m:
            current_step["accuracy"] = float(acc_m.group(1))
            current_step["correct"] = int(acc_m.group(2))
            current_step["total"] = int(acc_m.group(3))

        domain_m = domain_pattern.search(line)
        if domain_m:
            current_step["domain"] = domain_m.group(1)

        problem_m = problem_pattern.search(line)
        if problem_m:
            current_step["problem"] = problem_m.group(1).strip()

        if skip_pattern.search(line) and "too easy" in line:
            current_step["skipped"] = True

    if current_step:
        steps.append(current_step)

    # Parse header info
    time_budget = None
    tb_m = re.search(r"Time budget:\s*(\d+) minutes", text)
    if tb_m:
        time_budget = int(tb_m.group(1)) * 60

    group_size = None
    gs_m = re.search(r"Group size:\s*(\d+)", text)
    if gs_m:
        group_size = int(gs_m.group(1))

    model_name = None
    mn_m = re.search(r"Loading model:\s*(.+?)\s*\(backend", text)
    if mn_m:
        model_name = mn_m.group(1).strip()

    completed = "Experiment Complete" in text
    crashed = "Traceback" in text and not completed

    return {
        "steps": steps,
        "time_budget": time_budget,
        "group_size": group_size,
        "model_name": model_name,
        "completed": completed,
        "crashed": crashed,
        "total_steps": len(steps),
        "trained_steps": sum(1 for s in steps if s.get("trained")),
        "losses": [s["loss"] for s in steps if s.get("loss") is not None],
        "accuracies": [s["accuracy"] for s in steps if s.get("accuracy") is not None],
    }


def get_gpu_stats() -> dict:
    """Get GPU utilization and memory via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {
                "gpu_util": int(parts[0]),
                "mem_used_mb": int(parts[1]),
                "mem_total_mb": int(parts[2]),
                "temp_c": int(parts[3]),
                "name": parts[4],
                "mem_pct": round(int(parts[1]) / int(parts[2]) * 100, 1),
            }
    except Exception:
        pass
    return {"gpu_util": 0, "mem_used_mb": 0, "mem_total_mb": 0, "temp_c": 0, "name": "N/A", "mem_pct": 0}


def get_process_status() -> dict:
    """Check if training process is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train.py"], capture_output=True, text=True
        )
        running = result.returncode == 0
        pid = result.stdout.strip().split("\n")[0] if running else None
        return {"running": running, "pid": pid}
    except Exception:
        return {"running": False, "pid": None}


def list_experiments() -> list[dict]:
    """List all experiment directories with summary info."""
    exps = []
    if not EXPERIMENTS_DIR.exists():
        return exps
    for d in sorted(EXPERIMENTS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        log = d / "train.log"
        result = d / "results.json"
        info = {"name": d.name, "has_log": log.exists(), "has_results": result.exists()}
        if result.exists():
            try:
                r = json.loads(result.read_text())
                info.update({
                    "steps": r.get("steps", 0),
                    "train_steps": r.get("train_steps", 0),
                    "accuracy": r.get("overall_accuracy", 0),
                    "avg_loss": r.get("avg_loss", 0),
                    "elapsed": r.get("elapsed_seconds", 0),
                    "timestamp": r.get("timestamp", ""),
                })
            except Exception:
                pass
        exps.append(info)
    return exps


# ── API Routes ─────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    """Full dashboard data in one call."""
    experiments = list_experiments()
    process = get_process_status()
    gpu = get_gpu_stats()

    # Find the most recent active run log
    active_log = None
    for d in sorted(EXPERIMENTS_DIR.iterdir(), reverse=True) if EXPERIMENTS_DIR.exists() else []:
        if d.is_dir() and (d / "train.log").exists():
            active_log = d / "train.log"
            active_run = d.name
            break

    parsed = parse_log(active_log) if active_log else {}
    steps = parsed.get("steps", [])
    latest_step = steps[-1] if steps else None

    # Log tail (last 40 lines, filtered)
    log_tail = []
    if active_log and active_log.exists():
        lines = active_log.read_text(errors="replace").split("\n")
        skip = {"Loading weights", "it/s", "torch_dtype", "fast path", "deprecated",
                "Loading token", "Loading model weight"}
        log_tail = [l for l in lines if not any(s in l for s in skip)][-50:]

    return JSONResponse({
        "process": process,
        "gpu": gpu,
        "active_run": active_run if active_log else None,
        "experiments": experiments[:8],
        "current": {
            "step": latest_step.get("step") if latest_step else 0,
            "elapsed": latest_step.get("elapsed") if latest_step else 0,
            "remaining": latest_step.get("remaining") if latest_step else 0,
            "latest_loss": latest_step.get("loss") if latest_step else None,
            "latest_accuracy": latest_step.get("accuracy") if latest_step else None,
            "domain": latest_step.get("domain") if latest_step else None,
            "problem": latest_step.get("problem") if latest_step else None,
            "trained_steps": parsed.get("trained_steps", 0),
            "total_steps": parsed.get("total_steps", 0),
            "completed": parsed.get("completed", False),
            "crashed": parsed.get("crashed", False),
        },
        "chart": {
            "steps": [s["step"] for s in steps if s.get("loss") is not None],
            "losses": [s["loss"] for s in steps if s.get("loss") is not None],
            "accuracies": [s.get("accuracy", 0) for s in steps if s.get("loss") is not None],
        },
        "log_tail": log_tail,
        "timestamp": datetime.utcnow().isoformat(),
    })


@app.get("/api/run/{run_name}")
def api_run(run_name: str):
    log = EXPERIMENTS_DIR / run_name / "train.log"
    return JSONResponse(parse_log(log))


@app.get("/", response_class=HTMLResponse)
def index():
    html = (Path(__file__).parent / "index.html").read_text()
    return HTMLResponse(html)
