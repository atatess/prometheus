"""
Prometheus — Layer 3 Meta-Loop Orchestrator.

This is the outer agent that watches training, detects problems, and intervenes.
It does NOT replace train.py — it runs ALONGSIDE it, monitoring and adapting.

Architecture:
  train.py      → inner loop (GRPO self-play, runs as a subprocess)
  prometheus.py → outer loop (monitors, detects issues, triggers actions)

What it does every 5 minutes:
  1. Parse train.log for latest loss curve
  2. Detect plateau (loss not improving >5% in last 20 steps)
  3. Detect explosion (loss >10x baseline)
  4. Trigger domain expansion if curriculum saturates
  5. Run evaluate.py benchmark every 50 training steps
  6. Adjust LR and restart if needed
  7. Log all decisions to experiments/meta_loop.log

Usage:
  # Start training in background first, then run meta-loop alongside it:
  PROMETHEUS_BACKEND=cuda python train.py --config configs/cuda.toml \\
      --experiment-dir experiments/cuda_run_007 &

  PROMETHEUS_BACKEND=cuda python prometheus.py \\
      --experiment-dir experiments/cuda_run_007 \\
      --config configs/cuda.toml \\
      --model /root/models/qwen3.5-4b
"""

import argparse
import json
import logging
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
_backend_env = os.environ.get("PROMETHEUS_BACKEND", "").lower()
if _backend_env == "cuda":
    _USE_CUDA = True
elif _backend_env == "mlx":
    _USE_CUDA = False
else:
    _USE_CUDA = platform.system() != "Darwin"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(log_path: Path) -> logging.Logger:
    """Set up a logger that writes to both console and meta_loop.log."""
    logger = logging.getLogger("prometheus.meta")
    logger.setLevel(logging.DEBUG)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [META] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    # File handler — everything, with full timestamps
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_path), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# MetaLoop
# ---------------------------------------------------------------------------

class MetaLoop:
    """
    Outer orchestration loop for Prometheus.

    Monitors the training process, detects pathological loss behaviour,
    triggers domain expansion, and periodically benchmarks the model.
    """

    def __init__(
        self,
        experiment_dir: str,
        config_path: str,
        model: str,
        tokenizer=None,           # Not used directly here — passed to evaluate subprocess
        poll_interval: int = 300, # 5 minutes in seconds
    ):
        self.experiment_dir = Path(experiment_dir)
        self.config_path = config_path
        self.model = model
        self.poll_interval = poll_interval

        # Paths
        self.train_log_path = self.experiment_dir / "train.log"
        self.meta_log_path = Path("experiments") / "meta_loop.log"
        self.benchmarks_dir = Path("benchmarks")
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.log = setup_logger(self.meta_log_path)

        # State
        self.baseline_loss: Optional[float] = None   # First measured loss
        self.loss_history: list[float] = []          # All losses seen so far
        self.step_history: list[int] = []            # Step numbers
        self.last_benchmark_step: int = 0            # Step when we last ran eval
        self.training_pid: Optional[int] = None      # PID of the training process
        self.meta_steps: int = 0                     # How many meta-loop iterations

        # Intervention state — avoid thrashing
        self.last_lr_adjust_step: int = -100
        self.last_expansion_step: int = -100
        self.current_lr: Optional[float] = None

        # Load config to read current LR
        try:
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)
            self.current_lr = cfg.get("training", {}).get("learning_rate", 5e-6)
        except Exception:
            self.current_lr = 5e-6

        self.log.info(f"MetaLoop initialized")
        self.log.info(f"  experiment_dir : {self.experiment_dir}")
        self.log.info(f"  config         : {self.config_path}")
        self.log.info(f"  model          : {self.model}")
        self.log.info(f"  poll_interval  : {self.poll_interval}s")
        self.log.info(f"  initial_lr     : {self.current_lr}")

    # ─────────────────────────────────────────────────────────────────────────
    # Monitoring
    # ─────────────────────────────────────────────────────────────────────────

    def monitor_training(self) -> dict:
        """
        Parse the current train.log and return a metrics dict.

        Looks for lines in two formats (both emitted by train.py):
          "📉 GRPO Loss: 0.4321"       (step-level loss lines)
          "step=N loss=X.XXXX ..."     (structured log lines, if added later)

        Returns:
          {
            "steps": [list of step numbers],
            "losses": [list of floats],
            "latest_loss": float or None,
            "latest_step": int or None,
            "curriculum_status": dict or None,
          }
        """
        metrics = {
            "steps": [],
            "losses": [],
            "latest_loss": None,
            "latest_step": None,
            "curriculum_status": None,
        }

        if not self.train_log_path.exists():
            self.log.debug(f"train.log not found at {self.train_log_path}")
            return metrics

        try:
            text = self.train_log_path.read_text()
        except OSError as e:
            self.log.warning(f"Could not read train.log: {e}")
            return metrics

        # Pattern 1: "📉 GRPO Loss: 0.4321" (preceded by "--- Step N ---" line)
        step_pattern = re.compile(r"---\s*Step\s+(\d+)")
        loss_pattern = re.compile(r"GRPO Loss:\s*([\d.]+)")
        # Pattern 2: structured "step=N loss=X"
        structured_pattern = re.compile(r"step=(\d+).*?loss=([\d.]+)")

        steps = []
        losses = []

        # Walk through the log associating losses with steps
        current_step = None
        for line in text.splitlines():
            m = step_pattern.search(line)
            if m:
                current_step = int(m.group(1))
                continue
            m = loss_pattern.search(line)
            if m and current_step is not None:
                steps.append(current_step)
                losses.append(float(m.group(1)))
                current_step = None  # Reset — one loss per step
                continue
            m = structured_pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))

        if steps:
            metrics["steps"] = steps
            metrics["losses"] = losses
            metrics["latest_loss"] = losses[-1]
            metrics["latest_step"] = steps[-1]

        # Try to parse curriculum status from results.json (written periodically)
        results_path = self.experiment_dir / "results.json"
        if results_path.exists():
            try:
                results = json.loads(results_path.read_text())
                metrics["curriculum_status"] = results.get("curriculum_status")
            except (json.JSONDecodeError, OSError):
                pass

        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # Anomaly detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_plateau(self, metrics: dict) -> bool:
        """
        Return True if training has plateaued.

        Definition: loss has NOT improved by more than 5% over the last 20 steps.
        Requires at least 20 data points.
        """
        losses = metrics.get("losses", [])
        if len(losses) < 20:
            return False

        recent = losses[-20:]
        best_recent = min(recent)
        worst_recent = max(recent)

        if worst_recent == 0:
            return False

        # Relative improvement: how much did the best beat the worst?
        # If < 5%, it's flat.
        improvement = (worst_recent - best_recent) / worst_recent
        if improvement < 0.05:
            self.log.debug(
                f"Plateau check: best={best_recent:.4f} worst={worst_recent:.4f} "
                f"improvement={improvement:.1%} < 5%"
            )
            return True

        return False

    def detect_explosion(self, metrics: dict) -> bool:
        """
        Return True if loss is exploding.

        Definition: latest loss > 10× the baseline loss.
        Baseline is the first loss we ever saw.
        """
        losses = metrics.get("losses", [])
        if not losses:
            return False

        # Set baseline on first call
        if self.baseline_loss is None:
            self.baseline_loss = losses[0]
            self.log.info(f"Baseline loss set: {self.baseline_loss:.4f}")
            return False

        latest = losses[-1]
        threshold = self.baseline_loss * 10.0

        if latest > threshold:
            self.log.warning(
                f"Loss explosion detected: {latest:.4f} > 10× baseline ({self.baseline_loss:.4f})"
            )
            return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Interventions
    # ─────────────────────────────────────────────────────────────────────────

    def trigger_domain_expansion(self):
        """
        Run the verification factory to add a new reasoning domain.

        This calls src/verification_factory.py logic via a small helper script.
        In practice, domain expansion requires the model, so we run it as a
        subprocess using the same backend.
        """
        self.log.info("🌱 Triggering domain expansion via VerificationFactory...")

        # Build a small inline script that does the expansion
        script = (
            f"import sys; sys.path.insert(0, '.')\n"
            f"from src.verification_factory import VerificationFactory, EXPANSION_CANDIDATES\n"
            f"from src.curriculum import Curriculum\n"
            f"import json, pathlib\n"
            f"factory = VerificationFactory()\n"
            f"existing = factory.get_registered_domains()\n"
            f"candidate = factory.get_next_candidate(existing)\n"
            f"if candidate:\n"
            f"    print('CANDIDATE:', json.dumps(candidate))\n"
            f"else:\n"
            f"    print('NO_CANDIDATE')\n"
        )

        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(Path(__file__).parent),
            )
            if "CANDIDATE:" in result.stdout:
                line = [l for l in result.stdout.splitlines() if l.startswith("CANDIDATE:")][0]
                candidate = json.loads(line[len("CANDIDATE:"):].strip())
                self.log.info(
                    f"  Domain candidate: {candidate['domain']} — {candidate['description']}"
                )
                self.log.info(
                    "  ℹ️  Full factory expansion (model-generated verifier) requires "
                    "running inside train.py's model context. Logged for next training run."
                )
            else:
                self.log.info("  No expansion candidates remaining — all domains registered.")

            if result.returncode != 0:
                self.log.warning(f"  Factory script error: {result.stderr[:300]}")

        except subprocess.TimeoutExpired:
            self.log.warning("  Domain expansion script timed out.")
        except Exception as e:
            self.log.warning(f"  Domain expansion failed: {e}")

    def adjust_and_restart(self, new_lr: float):
        """
        Write a new config with the given learning rate, kill the current
        training process, and restart it.

        After restarting, update self.training_pid.
        """
        self.log.info(f"⚙️  Adjusting LR: {self.current_lr} → {new_lr} and restarting training...")

        # 1. Read current config
        try:
            with open(self.config_path, "rb") as f:
                config_text = Path(self.config_path).read_text()
        except OSError as e:
            self.log.error(f"  Cannot read config: {e}")
            return

        # 2. Replace learning_rate in config text (TOML format)
        new_config = re.sub(
            r"(learning_rate\s*=\s*)[\d.e\-+]+",
            f"\\g<1>{new_lr:.2e}",
            config_text,
        )

        # Write back (overwrites config — intentional, the meta-loop owns this)
        try:
            Path(self.config_path).write_text(new_config)
            self.log.info(f"  Config updated: learning_rate = {new_lr:.2e}")
        except OSError as e:
            self.log.error(f"  Cannot write config: {e}")
            return

        self.current_lr = new_lr

        # 3. Kill current training process
        if self.training_pid is not None:
            try:
                os.kill(self.training_pid, signal.SIGTERM)
                self.log.info(f"  Sent SIGTERM to training process (PID {self.training_pid})")
                time.sleep(3)  # Give it time to die gracefully
            except ProcessLookupError:
                self.log.info(f"  Training process (PID {self.training_pid}) already gone")
            except PermissionError:
                self.log.warning(f"  No permission to kill PID {self.training_pid}")
        else:
            self.log.info("  No training PID tracked — will not kill (may not be running)")

        # 4. Restart training
        self.log.info("  Restarting training subprocess...")
        try:
            env = dict(os.environ)
            env["PROMETHEUS_BACKEND"] = "cuda" if _USE_CUDA else "mlx"

            proc = subprocess.Popen(
                [
                    sys.executable, "train.py",
                    "--config", self.config_path,
                    "--experiment-dir", str(self.experiment_dir),
                ],
                env=env,
                stdout=open(str(self.train_log_path), "a"),
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent),
            )
            self.training_pid = proc.pid
            self.log.info(f"  Training restarted with PID {proc.pid}")

            # Reset explosion detection state
            self.baseline_loss = None
            self.loss_history = []
            self.step_history = []

        except Exception as e:
            self.log.error(f"  Failed to restart training: {e}")

    def run_benchmark(self) -> dict:
        """
        Run evaluate.py as a subprocess and return parsed results dict.

        Results are also saved to benchmarks/ by evaluate.py itself.
        Returns an empty dict on failure.
        """
        self.log.info("📊 Running benchmark evaluation...")

        env = dict(os.environ)
        env["PROMETHEUS_BACKEND"] = "cuda" if _USE_CUDA else "mlx"

        # Build checkpoint path if it exists
        checkpoint_path = self.experiment_dir / "checkpoint.npz"
        cmd = [
            sys.executable, "evaluate.py",
            "--model", self.model,
        ]
        if checkpoint_path.exists():
            cmd += ["--checkpoint", str(checkpoint_path)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,   # 10 minute timeout for benchmarks
                env=env,
                cwd=str(Path(__file__).parent),
            )
        except subprocess.TimeoutExpired:
            self.log.warning("  Benchmark timed out after 10 minutes")
            return {}
        except Exception as e:
            self.log.warning(f"  Benchmark subprocess error: {e}")
            return {}

        if result.returncode != 0:
            self.log.warning(f"  Benchmark failed (exit {result.returncode}): {result.stderr[:300]}")
            return {}

        # Parse the summary line from stdout for quick logging
        for line in result.stdout.splitlines():
            if line.startswith("📊 SUMMARY"):
                self.log.info(f"  {line}")

        # Find the most recently written eval_results file
        eval_files = sorted(Path("benchmarks").glob("eval_results_*.json"))
        if eval_files:
            try:
                return json.loads(eval_files[-1].read_text())
            except (json.JSONDecodeError, OSError):
                pass

        return {}

    # ─────────────────────────────────────────────────────────────────────────
    # Main step
    # ─────────────────────────────────────────────────────────────────────────

    def step(self):
        """
        Single meta-loop iteration — called every poll_interval seconds.

        Decision tree:
          1. Monitor → get current metrics
          2. Check explosion → adjust LR if needed
          3. Check plateau → trigger domain expansion if curriculum saturated
          4. Check benchmark cadence → run eval every 50 steps
          5. Log decisions
        """
        self.meta_steps += 1
        self.log.info(f"━━━ Meta-step {self.meta_steps} ━━━")

        # 1. Monitor
        metrics = self.monitor_training()
        latest_loss = metrics.get("latest_loss")
        latest_step = metrics.get("latest_step", 0)
        losses = metrics.get("losses", [])
        curriculum_status = metrics.get("curriculum_status")

        if latest_loss is not None:
            self.log.info(
                f"  Training state: step={latest_step} loss={latest_loss:.4f} "
                f"(n_loss_points={len(losses)})"
            )
        else:
            self.log.info("  No training data yet — waiting for train.py to emit loss lines")

        # Update internal history (deduplicate by step)
        seen_steps = set(self.step_history)
        for s, l in zip(metrics["steps"], metrics["losses"]):
            if s not in seen_steps:
                self.step_history.append(s)
                self.loss_history.append(l)
                seen_steps.add(s)

        # 2. Explosion check — highest priority
        if latest_loss is not None and self.detect_explosion(metrics):
            if latest_step - self.last_lr_adjust_step > 10:  # Don't thrash
                new_lr = (self.current_lr or 5e-6) / 5.0
                self.log.warning(f"  🚨 INTERVENTION: Loss explosion — reducing LR to {new_lr:.2e}")
                self.adjust_and_restart(new_lr)
                self.last_lr_adjust_step = latest_step
                self._log_decision("lr_reduce", {
                    "trigger": "explosion",
                    "old_lr": self.current_lr * 5.0,
                    "new_lr": new_lr,
                    "latest_loss": latest_loss,
                    "baseline_loss": self.baseline_loss,
                    "step": latest_step,
                })
            else:
                self.log.info("  ℹ️  Explosion detected but LR was adjusted recently — waiting")
            return  # Skip other checks this iteration

        # 3. Plateau + curriculum saturation check
        if latest_loss is not None and self.detect_plateau(metrics):
            self.log.info("  📉 Plateau detected in last 20 steps")

            # Check if curriculum is also saturated
            frontier_ratio = 0.0
            needs_expansion = False
            if curriculum_status:
                frontier_ratio = curriculum_status.get("frontier_ratio", 1.0)
                needs_expansion = curriculum_status.get("needs_expansion", False)
                self.log.info(
                    f"  Curriculum: frontier_ratio={frontier_ratio:.2f} "
                    f"needs_expansion={needs_expansion}"
                )

            if needs_expansion or frontier_ratio == 0.0:
                if latest_step - self.last_expansion_step > 20:
                    self.log.info("  🌱 INTERVENTION: Curriculum saturated — triggering domain expansion")
                    self.trigger_domain_expansion()
                    self.last_expansion_step = latest_step
                    self._log_decision("domain_expansion", {
                        "trigger": "plateau+saturation",
                        "frontier_ratio": frontier_ratio,
                        "step": latest_step,
                    })
                else:
                    self.log.info("  ℹ️  Expansion triggered recently — waiting")
            else:
                self.log.info(
                    f"  ℹ️  Plateau detected but curriculum still has frontier "
                    f"(frontier_ratio={frontier_ratio:.2f}) — no expansion needed"
                )

        # 4. Benchmark cadence — every 50 training steps
        if latest_step is not None and latest_step >= self.last_benchmark_step + 50:
            self.log.info(
                f"  ⏱️  Benchmark trigger: step {latest_step} ≥ "
                f"last_benchmark_step {self.last_benchmark_step} + 50"
            )
            bench_results = self.run_benchmark()
            if bench_results:
                overall_acc = bench_results.get("overall_accuracy", "?")
                self.log.info(f"  📊 Benchmark complete: overall_accuracy={overall_acc:.1%}")
                self._log_decision("benchmark", {
                    "step": latest_step,
                    "overall_accuracy": bench_results.get("overall_accuracy"),
                    "domain_accuracy": bench_results.get("domain_accuracy"),
                })
            self.last_benchmark_step = latest_step

        self.log.debug(f"  Meta-step {self.meta_steps} complete")

    def _log_decision(self, action: str, details: dict):
        """Append a structured decision log entry to meta_loop.log."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "meta_step": self.meta_steps,
            "action": action,
            **details,
        }
        # Append JSON line to meta_loop.log for easy parsing
        decisions_log = Path("experiments") / "meta_decisions.jsonl"
        decisions_log.parent.mkdir(parents=True, exist_ok=True)
        with open(str(decisions_log), "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.log.info(f"  📝 Decision logged: {action} — {details}")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prometheus Meta-Loop Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The meta-loop runs alongside train.py, monitoring and intervening.

Start training first:
  PROMETHEUS_BACKEND=cuda python train.py --config configs/cuda.toml \\
      --experiment-dir experiments/cuda_run_007 > experiments/cuda_run_007/train.log 2>&1 &

Then start meta-loop:
  PROMETHEUS_BACKEND=cuda python prometheus.py \\
      --experiment-dir experiments/cuda_run_007 \\
      --config configs/cuda.toml \\
      --model /root/models/qwen3.5-4b
        """,
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="experiments/cuda_run_007",
        help="Directory containing train.log, checkpoint.npz, results.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to TOML config (meta-loop may update this)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/root/models/qwen3.5-4b",
        help="Model path (used for benchmark evaluation)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=300,
        help="Seconds between meta-loop iterations (default: 300 = 5 minutes)",
    )
    parser.add_argument(
        "--training-pid",
        type=int,
        default=None,
        help="PID of the training process (optional — used for restart after LR adjustment)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one meta-step and exit (useful for testing / cron)",
    )
    args = parser.parse_args()

    # Ensure experiment dir exists
    exp_dir = Path(args.experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    meta = MetaLoop(
        experiment_dir=args.experiment_dir,
        config_path=args.config,
        model=args.model,
        poll_interval=args.poll_interval,
    )

    if args.training_pid:
        meta.training_pid = args.training_pid
        meta.log.info(f"Tracking training PID: {args.training_pid}")

    print("=" * 60)
    print("🔥 PROMETHEUS — Meta-Loop Orchestrator")
    print("=" * 60)
    print(f"  Experiment : {args.experiment_dir}")
    print(f"  Config     : {args.config}")
    print(f"  Model      : {args.model}")
    print(f"  Poll       : every {args.poll_interval}s")
    print(f"  Backend    : {'cuda' if _USE_CUDA else 'mlx'}")
    print(f"  Log        : experiments/meta_loop.log")
    print("=" * 60)
    print()

    if args.once:
        meta.log.info("--once flag set: running single meta-step")
        meta.step()
        return

    # Run the meta-loop indefinitely
    meta.log.info("Starting continuous meta-loop. Press Ctrl+C to stop.")
    try:
        while True:
            meta.step()
            meta.log.info(f"  Sleeping {args.poll_interval}s until next check...")
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        meta.log.info("Meta-loop stopped by user (KeyboardInterrupt).")
    except Exception as e:
        meta.log.error(f"Meta-loop crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
