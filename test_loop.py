"""Quick test of the Proposer → Solver → Verifier pipeline."""

import time
from mlx_lm import load

from src.model_utils import chat_generate
from src.proposer import build_proposer_prompt, parse_proposed_problem
from src.solver import build_solver_prompt, parse_solution
from src.verifier import verify_code_task, verify_math, SandboxConfig


def main():
    print("=" * 60)
    print("🧪 Testing Proposer → Solver → Verifier Pipeline")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading model...")
    model, tokenizer = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    print("✅ Model loaded")
    
    # Test 1: Math problem
    print("\n--- Test 1: Math Problem Generation ---")
    proposer_prompt = build_proposer_prompt("math", "easy")
    
    t0 = time.monotonic()
    raw_problem = chat_generate(model, tokenizer, proposer_prompt, max_tokens=2000)
    t1 = time.monotonic()
    print(f"Raw problem output ({t1-t0:.1f}s):")
    print(raw_problem[:500])
    
    problem = parse_proposed_problem("math", raw_problem)
    if problem:
        print(f"\n✅ Parsed problem: {problem.prompt[:100]}")
        print(f"   Expected answer: {problem.metadata.get('expected_answer', 'N/A')}")
        
        # Solve it
        print("\n--- Solving ---")
        solver_prompt = build_solver_prompt(problem)
        raw_solution = chat_generate(model, tokenizer, solver_prompt, max_tokens=2000)
        solution = parse_solution(raw_solution, "math")
        print(f"Solution answer: '{solution.answer[:200]}'")
        
        # Verify it
        print("\n--- Verifying ---")
        if "expected_answer" in problem.metadata:
            result = verify_math(problem.prompt, solution.answer, problem.metadata["expected_answer"])
        else:
            result = verify_code_task(
                problem.problem_code,
                f"student_answer = {repr(solution.answer)}",
                problem.test_code,
                SandboxConfig(timeout_seconds=10),
            )
        print(f"Correct: {result.correct}")
        if result.error:
            print(f"Error: {result.error}")
        print(f"Verify time: {result.execution_time_ms:.0f}ms")
    else:
        print("❌ Failed to parse problem from model output")
    
    # Test 2: Direct math verification
    print("\n--- Test 2: Direct Math Verification ---")
    result = verify_math("What is 7*13?", "91", "91")
    print(f"91 == 91: {result.correct}")
    
    result = verify_math("What is 7*13?", "92", "91")
    print(f"92 == 91: {result.correct}")
    
    print("\n✅ Pipeline test complete!")

if __name__ == "__main__":
    main()
