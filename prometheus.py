"""
Prometheus — Main Orchestration Loop.

This is the top-level entry point that runs the full Prometheus system:
1. Self-play training (inner loop)
2. Verification factory (domain expansion)
3. Benchmark evaluation (progress tracking)
4. Experiment management (keep/revert via git)
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from src.curriculum import Curriculum, CurriculumConfig
from src.verification_factory import VerificationFactory


def load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def run_training(config_path: str, experiment_dir: str) -> dict:
    """Run a single training experiment as a subprocess."""
    result = subprocess.run(
        ["uv", "run", "python", "train.py",
         "--config", config_path,
         "--experiment-dir", experiment_dir],
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute safety timeout
    )
    
    if result.returncode != 0:
        print(f"❌ Training failed:\n{result.stderr}")
        return None
    
    # Load results
    results_path = Path(experiment_dir) / "results.json"
    if results_path.exists():
        return json.loads(results_path.read_text())
    return None


def run_benchmark(config: dict) -> dict:
    """Run benchmark evaluation."""
    result = subprocess.run(
        ["uv", "run", "python", "evaluate.py",
         "--model", config["model"]["name"]],
        capture_output=True,
        text=True,
        timeout=300,
    )
    
    if result.returncode != 0:
        print(f"⚠️  Benchmark failed: {result.stderr[:200]}")
        return {}
    
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def git_checkpoint(message: str):
    """Create a git checkpoint."""
    subprocess.run(["git", "add", "-A"], capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        capture_output=True,
    )


def git_revert_train():
    """Revert train.py to last committed version."""
    subprocess.run(
        ["git", "checkout", "HEAD", "--", "train.py"],
        capture_output=True,
    )


def main():
    parser = argparse.ArgumentParser(description="🔥 Prometheus — Self-Evolving Training")
    parser.add_argument(
        "--config", type=str, default="configs/default.toml",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--max-experiments", type=int, default=-1,
        help="Max experiments to run (-1 = unlimited)",
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    max_experiments = args.max_experiments
    if max_experiments < 0:
        max_experiments = config.get("meta", {}).get("max_experiments", -1)
    
    # Initialize components
    factory = VerificationFactory()
    benchmark_interval = config.get("meta", {}).get("benchmark_interval", 10)
    checkpoint_interval = config.get("meta", {}).get("checkpoint_interval", 5)
    
    # State tracking
    experiment_num = 0
    best_accuracy = 0.0
    best_benchmark = {}
    log_path = Path("experiments") / "log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔥 PROMETHEUS — Self-Evolving Training System")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Model: {config['model']['name']}")
    print(f"Max experiments: {max_experiments if max_experiments > 0 else '∞'}")
    print(f"Benchmark every {benchmark_interval} experiments")
    print()
    
    # Initial benchmark
    print("📊 Running initial benchmark...")
    baseline = run_benchmark(config)
    if baseline:
        print(f"   Baseline: {json.dumps(baseline, indent=2)}")
        best_benchmark = baseline
    
    git_checkpoint("prometheus: initial state")
    
    # --- Main Loop ---
    while max_experiments < 0 or experiment_num < max_experiments:
        experiment_num += 1
        exp_dir = f"experiments/exp_{experiment_num:04d}"
        
        print(f"\n{'=' * 60}")
        print(f"🔬 Experiment #{experiment_num}")
        print(f"{'=' * 60}")
        
        # Run training experiment
        results = run_training(args.config, exp_dir)
        
        if results is None:
            print("⚠️  Experiment failed, skipping")
            continue
        
        # Log results
        log_entry = {
            "experiment": experiment_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **results,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Check if we should expand domains
        curriculum_status = results.get("curriculum_status", {})
        if curriculum_status.get("needs_expansion", False):
            print("\n⚡ Triggering Verification Factory...")
            existing = curriculum_status.get("domains", [])
            candidate = factory.get_next_candidate(existing)
            
            if candidate:
                print(f"   Candidate domain: {candidate['domain']}")
                print(f"   Description: {candidate['description']}")
                # The verification factory expansion would be triggered here
                # In Phase 1, we just log it
                print(f"   (Verification Factory expansion not yet implemented)")
        
        # Periodic benchmark
        if experiment_num % benchmark_interval == 0:
            print(f"\n📊 Running benchmark (every {benchmark_interval} experiments)...")
            current_benchmark = run_benchmark(config)
            
            if current_benchmark:
                improved = any(
                    current_benchmark.get(k, 0) > best_benchmark.get(k, 0)
                    for k in current_benchmark
                )
                
                if improved:
                    print(f"   ✅ Improvement detected!")
                    best_benchmark = current_benchmark
                    git_checkpoint(
                        f"prometheus: exp {experiment_num} — improved benchmarks"
                    )
                else:
                    print(f"   ➡️  No improvement over best")
        
        # Periodic checkpoint
        if experiment_num % checkpoint_interval == 0:
            git_checkpoint(f"prometheus: checkpoint at exp {experiment_num}")
        
        print(f"\n   Accuracy: {results.get('overall_accuracy', 0):.1%}")
        print(f"   Steps: {results.get('steps', 0)}")
    
    print(f"\n{'=' * 60}")
    print(f"🏁 Prometheus Complete — {experiment_num} experiments")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
