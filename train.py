"""
Prometheus Training Loop — Self-Play via GRPO.

Core loop: generate problems → solve → verify → GRPO train.
This file can be modified by the meta-agent (autoresearch outer loop).

Backend selection
-----------------
Set the environment variable PROMETHEUS_BACKEND=cuda to use the PyTorch/CUDA
backend (grpo_cuda.py, model_utils_cuda.py).  On non-Darwin hosts the CUDA
backend is chosen automatically if PROMETHEUS_BACKEND is not set.

  export PROMETHEUS_BACKEND=cuda   # force CUDA
  export PROMETHEUS_BACKEND=mlx    # force MLX (default on macOS)
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# ---------------------------------------------------------------------------
# Backend detection — must happen before any framework-specific imports
# ---------------------------------------------------------------------------
_backend_env = os.environ.get("PROMETHEUS_BACKEND", "").lower()
if _backend_env == "cuda":
    _USE_CUDA = True
elif _backend_env == "mlx":
    _USE_CUDA = False
else:
    # Auto-detect: use CUDA on non-macOS hosts
    _USE_CUDA = platform.system() != "Darwin"

if _USE_CUDA:
    # ---------- CUDA / PyTorch imports -------------------------------------
    import torch
    from src.grpo_cuda import GRPOConfig, GRPOTrainer
    from src.model_utils_cuda import chat_generate, raw_generate, strip_thinking
    from src.load_model_cuda import load_model_cuda as load
else:
    # ---------- MLX imports (macOS / Apple Silicon) -----------------------
    import mlx.core as mx
    from mlx_lm import load
    from src.grpo import GRPOConfig, GRPOTrainer
    from src.model_utils import chat_generate, strip_thinking
from src.proposer import build_proposer_prompt, parse_proposed_problem
from src.template_proposer import generate_problem as template_generate_problem
from src.solver import build_solver_prompt, parse_solution
from src.verifier import verify_code_task, verify_math, SandboxConfig
from src.curriculum import Curriculum, CurriculumConfig
from src.seed_problems import SEED_PROBLEMS
from src.verification_factory import VerificationFactory, EXPANSION_CANDIDATES


def load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def run_experiment(config: dict, experiment_dir: Path):
    """Run a single training experiment within the time budget."""
    
    print("=" * 60)
    print("🔥 PROMETHEUS — Self-Play Training")
    print("=" * 60)
    
    # --- Load Model ---
    model_name = config["model"]["name"]
    print(f"\n📦 Loading model: {model_name}  (backend: {'cuda' if _USE_CUDA else 'mlx'})")
    if _USE_CUDA:
        # load_model_cuda returns (model, tokenizer) — same tuple as mlx_lm.load()
        model, tokenizer = load(model_name, device="cuda")
    else:
        model, tokenizer = load(model_name)
    print(f"✅ Model loaded")
    
    # --- Setup GRPO ---
    grpo_config = GRPOConfig(
        group_size=config["training"]["group_size"],
        learning_rate=config["training"]["learning_rate"],
        max_new_tokens=config["training"].get("max_new_tokens", 2000),
        max_seq_len=config["training"].get("max_seq_len", 2048),
        gradient_accumulation_steps=config["training"].get("grad_accumulation_steps", 4),
    )
    
    time_budget = config["training"]["time_budget_minutes"] * 60
    estimated_steps = max(10, int(time_budget / 30))  # ~30s per step
    trainer = GRPOTrainer(model, tokenizer, grpo_config, total_steps=estimated_steps)
    print(f"✅ GRPO trainer initialized (LoRA + single model for 32GB)")
    
    # --- Setup Curriculum ---
    curriculum_config = CurriculumConfig(
        initial_domains=config["curriculum"]["initial_domains"],
        frontier_band=tuple(config["curriculum"]["frontier_band"]),
        problems_per_round=config["curriculum"]["problems_per_round"],
    )
    curriculum = Curriculum(curriculum_config)
    
    curriculum_path = experiment_dir / "curriculum_state.json"
    if curriculum_path.exists():
        curriculum.load(str(curriculum_path))

    # --- Setup Verification Factory (Layer 2) ---
    factory = VerificationFactory(generated_dir="src/domains/generated")
    # Wire any already-registered factory domains into the curriculum
    for factory_domain in factory.get_registered_domains():
        if factory_domain not in curriculum.domains:
            curriculum.add_domain(factory_domain)
            print(f"  🏭 Loaded factory domain: {factory_domain}")
    print(f"✅ Verification Factory ready ({len(factory.get_registered_domains())} registered domains)")
    
    # --- Setup Sandbox ---
    sandbox_config = SandboxConfig(
        timeout_seconds=config["sandbox"]["timeout_seconds"],
        max_memory_mb=config["sandbox"]["max_memory_mb"],
    )
    
    # --- Training Loop ---
    start_time = time.monotonic()
    step = 0
    total_correct = 0
    total_attempts = 0
    losses = []
    train_steps_done = 0
    seed_idx = 0  # Index into seed problems
    
    import random
    random.shuffle(SEED_PROBLEMS)
    
    # Cap seed problems if configured (forces proposer to kick in earlier)
    max_seeds = config["curriculum"].get("max_seed_problems", len(SEED_PROBLEMS))
    seed_pool = SEED_PROBLEMS[:max_seeds]
    
    print(f"\n⏱️  Time budget: {config['training']['time_budget_minutes']} minutes")
    print(f"🎯 Domains: {curriculum.domains}")
    print(f"🔄 Group size: {grpo_config.group_size}")
    print(f"🌱 Seed problems: {len(seed_pool)}/{len(SEED_PROBLEMS)} (then proposer takes over)")
    print()
    
    while (time.monotonic() - start_time) < time_budget:
        step += 1
        elapsed = time.monotonic() - start_time
        remaining = time_budget - elapsed
        
        print(f"\n--- Step {step} | {elapsed:.0f}s elapsed | {remaining:.0f}s remaining ---")
        
        # 1. Select domain and difficulty
        domain = curriculum.select_domain()
        if domain == "__needs_expansion__":
            print("⚡ All domains saturated — triggering Layer 2 expansion!")
            # Try to add a new domain via the Verification Factory
            candidate = factory.get_next_candidate(curriculum.domains)
            if candidate is None:
                print("⚡ No more expansion candidates — training complete.")
                break

            print(f"🏭 Layer 2: generating verifier for '{candidate['domain']}'...")

            # Build a generate_fn that wraps the current model
            def _make_generate_fn(mdl, tok):
                def _generate(prompt_text: str) -> str:
                    from src.verification_factory import FACTORY_MAX_TOKENS
                    messages = [{"role": "user", "content": prompt_text}]
                    if _USE_CUDA:
                        import torch
                        formatted = tok.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = tok(formatted, return_tensors="pt").to("cuda")
                        with torch.no_grad():
                            out = mdl.generate(
                                **inputs,
                                max_new_tokens=FACTORY_MAX_TOKENS,
                                do_sample=True,
                                temperature=0.6,
                                top_p=0.95,
                                pad_token_id=tok.eos_token_id,
                            )
                        generated = out[0][inputs["input_ids"].shape[1]:]
                        return tok.decode(generated, skip_special_tokens=True)
                    else:
                        return chat_generate(mdl, tok, prompt_text, max_tokens=FACTORY_MAX_TOKENS)
                return _generate

            generate_fn = _make_generate_fn(model, tokenizer)
            new_verifier = factory.run_factory_with_model(
                candidate["domain"], candidate["description"],
                generate_fn=generate_fn,
                max_retries=2,
            )
            if new_verifier is not None:
                factory.register_verifier(new_verifier)
                curriculum.add_domain(new_verifier.domain)
                print(f"✅ Layer 2 success! Added domain: {new_verifier.domain}")
            else:
                print(f"❌ Layer 2 failed for '{candidate['domain']}' — will try next step")
            # Continue training (don't break)
            continue
        
        difficulty = curriculum.select_difficulty(domain)
        print(f"  Domain: {domain} | Difficulty: {difficulty}")
        
        # 2. Get a problem — seed problems first, then model-generated
        if seed_idx < len(seed_pool):
            problem = seed_pool[seed_idx]
            seed_idx += 1
            print(f"  🌱 Seed problem [{problem.domain}]")
        else:
            proposer_prompt = build_proposer_prompt(domain, difficulty)
            # IMPORTANT: Use raw_generate (NOT chat_generate) for proposer.
            # Qwen3.5-4B embeds its PROBLEM:/ANSWER: output inside <think> blocks.
            # chat_generate strips thinking → loses the problem content entirely.
            # raw_generate returns the full output including thinking, and
            # parse_proposed_problem already handles <think> content correctly.
            if _USE_CUDA:
                raw_problem = raw_generate(model, tokenizer, proposer_prompt, max_tokens=1500)
            else:
                raw_problem = chat_generate(model, tokenizer, proposer_prompt, max_tokens=2000)
            problem = parse_proposed_problem(domain, raw_problem)
            
            if problem is None:
                # Proposer parse failed — use template generator (guaranteed valid).
                # Template proposer generates problems programmatically from
                # parameterized templates (no LLM, no parse failure possible).
                problem = template_generate_problem(domain, difficulty)
                print(f"  🔄 Proposer failed, template fallback [{problem.domain}]: {problem.prompt[:50]}")
        
        print(f"  📝 Problem: {problem.prompt[:80]}...")
        
        # 3. Probe first — quick single rollout to check if problem is in goldilocks zone
        solver_prompt = build_solver_prompt(problem)
        probe_responses = trainer.generate_rollouts_n(solver_prompt, n=1)
        probe_clean = strip_thinking(probe_responses[0])
        probe_sol = parse_solution(probe_clean, problem.domain)
        
        expected = problem.metadata.get("expected_answer", "")
        probe_result = verify_math(problem.prompt, probe_sol.answer, expected)
        
        if probe_result.correct:
            # Probe passed — likely too easy. Do one more to confirm
            probe2 = trainer.generate_rollouts_n(solver_prompt, n=1)
            probe2_clean = strip_thinking(probe2[0])
            probe2_sol = parse_solution(probe2_clean, problem.domain)
            probe2_result = verify_math(problem.prompt, probe2_sol.answer, expected)
            
            if probe2_result.correct:
                # Both probes correct — skip, too easy
                print(f"  ⏭️  Probe: 2/2 correct — too easy, skipping")
                curriculum.record_attempt(problem.domain, True)
                continue
            else:
                # Mixed — do full rollouts starting with what we have
                remaining = trainer.generate_rollouts_n(solver_prompt, n=grpo_config.group_size - 2)
                responses = probe_responses + probe2 + remaining
        else:
            # Probe failed — might be goldilocks zone, do full rollouts
            remaining = trainer.generate_rollouts_n(solver_prompt, n=grpo_config.group_size - 1)
            responses = probe_responses + remaining
        
        # Parse answers from RAW responses — parse_solution searches for FINAL_ANSWER: / ANSWER:
        # in the full text including thinking blocks.
        solutions = [parse_solution(r, problem.domain) for r in responses]
        # Use RAW responses for GRPO loss computation.
        # Computing log_probs on stripped thinking ("15") fails — the model can't predict
        # a bare "15" without its reasoning context, making log_probs near-random.
        # With full responses, log_probs are meaningful and gradients are useful.
        # (Memory cost is ~2x but still fine on 48GB VRAM.)
        train_responses = responses

        # 4. Verify each solution
        actual_domain = problem.domain  # Use problem's domain, not curriculum's
        expected = problem.metadata.get("expected_answer", "?")
        print(f"  📊 Expected: {expected}")
        for i, (sol, raw) in enumerate(zip(solutions, responses)):
            print(f"     Rollout {i}: answer='{sol.answer}' (from: '{raw[:60]}...')")
        
        rewards = []
        for sol in solutions:
            # Priority 1: factory-registered verifier for this domain
            if actual_domain in factory.verifiers:
                correct = factory.call_verifier(actual_domain, problem.metadata, sol.answer)
            # Priority 2: math/string expected_answer
            elif "expected_answer" in problem.metadata:
                result = verify_math(
                    problem.prompt, sol.answer, problem.metadata["expected_answer"]
                )
                correct = result.correct
            # Priority 3: code sandbox (problem_code + test_code)
            else:
                result = verify_code_task(
                    problem.problem_code,
                    f"student_answer = {json.dumps(sol.answer)}",
                    problem.test_code,
                    sandbox_config,
                )
                correct = result.correct

            reward = 1.0 if correct else 0.0
            rewards.append(reward)
            total_attempts += 1
            if correct:
                total_correct += 1
        
        curriculum.record_attempt(actual_domain, any(r > 0 for r in rewards))
        
        accuracy = sum(rewards) / len(rewards)
        print(f"  ✅ Rollout accuracy: {accuracy:.1%} ({int(sum(rewards))}/{len(rewards)})")
        
        # 5. GRPO training step
        if sum(rewards) > 0 and sum(rewards) < len(rewards):
            # Need both positive and negative examples for GRPO
            # Use stripped responses (without thinking) to save memory
            loss = trainer.train_step(solver_prompt, train_responses, rewards)
            losses.append(loss)
            train_steps_done += 1
            print(f"  📉 GRPO Loss: {loss:.4f}")
            # Save checkpoint after every training update
            checkpoint_path = experiment_dir / "checkpoint.npz"
            trainer.save_checkpoint(str(checkpoint_path))
            print(f"  💾 Checkpoint saved")
        elif sum(rewards) == len(rewards):
            print(f"  ⏭️  All correct — too easy, no training signal")
        else:
            print(f"  ⏭️  All wrong — too hard, no training signal")
        
        # Aggressive memory cleanup between steps
        import gc
        if _USE_CUDA:
            torch.cuda.empty_cache()
        else:
            mx.clear_cache()
        gc.collect()
        
        # Time check
        if (time.monotonic() - start_time) > time_budget * 0.95:
            print(f"\n⏰ Time budget nearly exhausted, stopping.")
            break
    
    # --- Save Results ---
    elapsed_total = time.monotonic() - start_time
    overall_accuracy = total_correct / max(total_attempts, 1)
    avg_loss = sum(losses) / max(len(losses), 1)
    
    results = {
        "steps": step,
        "train_steps": train_steps_done,
        "elapsed_seconds": elapsed_total,
        "total_attempts": total_attempts,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "avg_loss": avg_loss,
        "losses": losses,
        "curriculum_status": curriculum.get_status(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    results_path = experiment_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    
    curriculum.save(str(curriculum_path))
    
    checkpoint_path = experiment_dir / "checkpoint.npz"
    trainer.save_checkpoint(str(checkpoint_path))
    
    print(f"\n{'=' * 60}")
    print(f"🏁 Experiment Complete")
    print(f"   Steps: {step} ({train_steps_done} training updates)")
    print(f"   Time: {elapsed_total:.1f}s")
    print(f"   Accuracy: {overall_accuracy:.1%}")
    print(f"   Avg Loss: {avg_loss:.4f}")
    print(f"   Curriculum: {json.dumps(curriculum.get_status(), indent=2)}")
    print(f"{'=' * 60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Prometheus Self-Play Training")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--experiment-dir", type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
    else:
        exp_dir = Path("experiments") / time.strftime("%Y%m%d_%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    
    run_experiment(config, exp_dir)


if __name__ == "__main__":
    main()
