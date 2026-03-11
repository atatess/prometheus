"""
Prometheus Training Loop — Self-Play via GRPO on MLX.

This is the file that gets modified by the meta-agent (autoresearch outer loop).
It contains the core training loop: generate problems → solve → verify → train.
"""

import argparse
import json
import time
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import mlx.core as mx
from mlx_lm import load

from src.grpo import GRPOConfig, GRPOTrainer, generate_rollouts
from src.proposer import build_proposer_prompt, parse_proposed_problem
from src.solver import build_solver_prompt, parse_solution
from src.verifier import verify_code_task, verify_math, SandboxConfig
from src.curriculum import Curriculum, CurriculumConfig


def load_config(config_path: str) -> dict:
    """Load TOML configuration."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def run_experiment(config: dict, experiment_dir: Path):
    """Run a single training experiment within the time budget."""
    
    print("=" * 60)
    print("🔥 PROMETHEUS — Self-Play Training")
    print("=" * 60)
    
    # --- Load Model ---
    model_name = config["model"]["name"]
    print(f"\n📦 Loading model: {model_name}")
    model, tokenizer = load(model_name)
    
    # --- Setup GRPO ---
    grpo_config = GRPOConfig(
        group_size=config["training"]["group_size"],
        learning_rate=config["training"]["learning_rate"],
        max_new_tokens=config["training"]["max_new_tokens"],
        lora_rank=config["model"].get("lora_rank", 16),
        lora_alpha=config["model"].get("lora_alpha", 32),
        lora_targets=config["model"].get("lora_targets", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        max_seq_len=config["training"].get("max_seq_len", 2048),
    )
    
    trainer = GRPOTrainer(model, tokenizer, grpo_config)
    
    # --- Setup Curriculum ---
    curriculum_config = CurriculumConfig(
        initial_domains=config["curriculum"]["initial_domains"],
        frontier_band=tuple(config["curriculum"]["frontier_band"]),
        problems_per_round=config["curriculum"]["problems_per_round"],
    )
    curriculum = Curriculum(curriculum_config)
    
    # Load existing curriculum state if available
    curriculum_path = experiment_dir / "curriculum_state.json"
    if curriculum_path.exists():
        curriculum.load(str(curriculum_path))
        print(f"📚 Loaded curriculum state: {curriculum.get_status()}")
    
    # --- Setup Sandbox ---
    sandbox_config = SandboxConfig(
        timeout_seconds=config["sandbox"]["timeout_seconds"],
        max_memory_mb=config["sandbox"]["max_memory_mb"],
    )
    
    # --- Training Loop ---
    time_budget = config["training"]["time_budget_minutes"] * 60
    start_time = time.monotonic()
    step = 0
    total_correct = 0
    total_attempts = 0
    losses = []
    
    print(f"\n⏱️  Time budget: {config['training']['time_budget_minutes']} minutes")
    print(f"🎯 Domains: {curriculum.domains}")
    print(f"🔄 Group size: {grpo_config.group_size}")
    print()
    
    while (time.monotonic() - start_time) < time_budget:
        step += 1
        elapsed = time.monotonic() - start_time
        remaining = time_budget - elapsed
        
        print(f"--- Step {step} | {elapsed:.0f}s elapsed | {remaining:.0f}s remaining ---")
        
        # 1. Select domain and difficulty
        domain = curriculum.select_domain()
        if domain == "__needs_expansion__":
            print("⚡ All domains saturated — needs expansion!")
            break
        
        difficulty = curriculum.select_difficulty(domain)
        print(f"  Domain: {domain} | Difficulty: {difficulty}")
        
        # 2. Generate a problem (Proposer)
        proposer_prompt = build_proposer_prompt(domain, difficulty)
        raw_problem = generate_rollouts(model, tokenizer, proposer_prompt, grpo_config)[0]
        problem = parse_proposed_problem(domain, raw_problem)
        
        if problem is None:
            print(f"  ⚠️  Failed to parse problem, skipping")
            continue
        
        print(f"  📝 Problem: {problem.prompt[:80]}...")
        
        # 3. Generate solutions (Solver) — multiple rollouts for GRPO
        solver_prompt = build_solver_prompt(problem)
        responses = generate_rollouts(model, tokenizer, solver_prompt, grpo_config)
        solutions = [parse_solution(r, domain) for r in responses]
        
        # 4. Verify each solution
        rewards = []
        for sol in solutions:
            if domain == "math" and "expected_answer" in problem.metadata:
                result = verify_math(
                    problem.prompt, sol.answer, problem.metadata["expected_answer"]
                )
            else:
                result = verify_code_task(
                    problem.problem_code,
                    f"student_answer = {json.dumps(sol.answer)}",
                    problem.test_code,
                    sandbox_config,
                )
            
            reward = 1.0 if result.correct else 0.0
            rewards.append(reward)
            total_attempts += 1
            if result.correct:
                total_correct += 1
        
        curriculum.record_attempt(domain, any(r > 0 for r in rewards))
        
        accuracy = sum(rewards) / len(rewards)
        print(f"  ✅ Rollout accuracy: {accuracy:.1%} ({sum(rewards):.0f}/{len(rewards)})")
        
        # 5. GRPO training step (only if we have signal)
        if sum(rewards) > 0 and sum(rewards) < len(rewards):
            # Need both positive and negative examples for GRPO
            loss = trainer.train_step(
                [solver_prompt],
                [rewards],
                [responses],
            )
            losses.append(loss)
            print(f"  📉 Loss: {loss:.4f}")
        else:
            print(f"  ⏭️  Skipping train step (all same reward)")
        
        # Check time
        if (time.monotonic() - start_time) > time_budget * 0.95:
            print(f"\n⏰ Time budget nearly exhausted, stopping.")
            break
    
    # --- Save Results ---
    elapsed_total = time.monotonic() - start_time
    overall_accuracy = total_correct / max(total_attempts, 1)
    avg_loss = sum(losses) / max(len(losses), 1)
    
    results = {
        "steps": step,
        "elapsed_seconds": elapsed_total,
        "total_attempts": total_attempts,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "avg_loss": avg_loss,
        "curriculum_status": curriculum.get_status(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Save experiment results
    results_path = experiment_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    
    # Save curriculum state
    curriculum.save(str(curriculum_path))
    
    # Save LoRA checkpoint
    checkpoint_path = experiment_dir / "lora_weights.npz"
    trainer.save_checkpoint(str(checkpoint_path))
    
    print(f"\n{'=' * 60}")
    print(f"🏁 Experiment Complete")
    print(f"   Steps: {step}")
    print(f"   Time: {elapsed_total:.1f}s")
    print(f"   Accuracy: {overall_accuracy:.1%}")
    print(f"   Avg Loss: {avg_loss:.4f}")
    print(f"   Curriculum: {json.dumps(curriculum.get_status(), indent=2)}")
    print(f"{'=' * 60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Prometheus Self-Play Training")
    parser.add_argument(
        "--config", type=str, default="configs/default.toml",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--experiment-dir", type=str, default=None,
        help="Directory for experiment outputs",
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Create experiment directory
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
    else:
        exp_dir = Path("experiments") / time.strftime("%Y%m%d_%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config for reproducibility
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    
    run_experiment(config, exp_dir)


if __name__ == "__main__":
    main()
