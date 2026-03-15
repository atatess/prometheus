"""
Prometheus Benchmark Evaluation Suite.

Evaluates the model on 50 fixed hold-out problems to measure training progress.
Problems are NEVER used in training — they exist purely for measurement.

Problems are designed to test REASONING not memorization:
  - 15 math (arithmetic, algebra, number theory, combinatorics)
  - 10 logic (syllogisms, truth tables, word problems)
  - 10 code (trace output of code snippets — no writing code)
  - 10 science (physics/chemistry/biology calculations)
  - 5 hard (multi-step, cross-domain)

Usage:
  PROMETHEUS_BACKEND=cuda python evaluate.py \\
      --model /root/models/qwen3.5-4b \\
      --checkpoint experiments/cuda_run_007/checkpoint.npz

Output:
  benchmarks/eval_results_<timestamp>.json
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

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
# 50 Fixed Hold-Out Problems
# Each: {"prompt": str, "answer": str, "domain": str, "difficulty": str}
# ---------------------------------------------------------------------------
HOLDOUT_PROBLEMS = [
    # ─────────────────────────────────────────────────────────────────────────
    # MATH (15 problems)
    # ─────────────────────────────────────────────────────────────────────────

    # Arithmetic / Number Theory
    {
        "prompt": (
            "What is the sum of all integers from 1 to 100 inclusive? "
            "Show your reasoning, then give ANSWER: <number>."
        ),
        "answer": "5050",
        "domain": "math",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "A train travels 360 km in 4 hours. "
            "How many kilometers will it travel in 7 hours at the same speed? "
            "ANSWER: <number>"
        ),
        "answer": "630",
        "domain": "math",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "What is the greatest common divisor (GCD) of 84 and 36? "
            "Show steps, then ANSWER: <number>."
        ),
        "answer": "12",
        "domain": "math",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "How many prime numbers are there between 1 and 50? "
            "List them and count. ANSWER: <number>."
        ),
        "answer": "15",
        "domain": "math",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "Solve for x: 3x + 7 = 22. ANSWER: <number>."
        ),
        "answer": "5",
        "domain": "math",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "A rectangle has perimeter 46 cm. Its length is 3 more than twice its width. "
            "What is the area of the rectangle in cm²? ANSWER: <number>."
        ),
        "answer": "120",
        "domain": "math",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "What is the remainder when 2^10 is divided by 7? "
            "Think step by step. ANSWER: <number>."
        ),
        "answer": "2",
        "domain": "math",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "The sum of three consecutive odd integers is 57. "
            "What is the largest of the three integers? ANSWER: <number>."
        ),
        "answer": "21",
        "domain": "math",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "In how many ways can you arrange the letters in the word 'MATH'? "
            "ANSWER: <number>."
        ),
        "answer": "24",
        "domain": "math",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "How many ways can a committee of 3 people be chosen from a group of 8? "
            "ANSWER: <number>."
        ),
        "answer": "56",
        "domain": "math",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "If f(x) = 2x² - 3x + 1, what is f(4)? ANSWER: <number>."
        ),
        "answer": "21",
        "domain": "math",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "A geometric sequence has first term 3 and common ratio 2. "
            "What is the sum of the first 6 terms? ANSWER: <number>."
        ),
        "answer": "189",
        "domain": "math",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "What is the smallest positive integer that is divisible by both 12 and 18? "
            "ANSWER: <number>."
        ),
        "answer": "36",
        "domain": "math",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "A number when divided by 5 leaves remainder 3, and when divided by 7 leaves remainder 4. "
            "What is the smallest such positive integer? ANSWER: <number>."
        ),
        "answer": "53",
        "domain": "math",
        "difficulty": "hard",
    },
    {
        "prompt": (
            "There are 10 people in a room. Every person shakes hands with every other person exactly once. "
            "How many handshakes happen in total? ANSWER: <number>."
        ),
        "answer": "45",
        "domain": "math",
        "difficulty": "medium",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # LOGIC (10 problems)
    # ─────────────────────────────────────────────────────────────────────────

    {
        "prompt": (
            "All mammals are warm-blooded. All whales are mammals. "
            "Is the following conclusion valid: 'All whales are warm-blooded'? "
            "Answer YES or NO, then explain. ANSWER: <YES or NO>."
        ),
        "answer": "YES",
        "domain": "logic",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "If today is not Wednesday, then it is Monday. "
            "It is not Monday. "
            "What day is it? ANSWER: <day>."
        ),
        "answer": "Wednesday",
        "domain": "logic",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "Evaluate the truth value of: (True AND False) OR (NOT False). "
            "Answer TRUE or FALSE. ANSWER: <TRUE or FALSE>."
        ),
        "answer": "TRUE",
        "domain": "logic",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "Three friends — Alice, Bob, and Carol — each have a different pet: a cat, a dog, and a fish. "
            "Alice does not have the cat. Bob does not have the dog. Carol has the fish. "
            "Who has the cat? ANSWER: <name>."
        ),
        "answer": "Bob",
        "domain": "logic",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "Five boxes are arranged in a row. The red box is immediately to the left of the blue box. "
            "The green box is two positions to the right of the red box. "
            "The yellow box is the leftmost. The white box is the rightmost. "
            "What position (1-5, left to right) is the blue box? ANSWER: <number>."
        ),
        "answer": "3",
        "domain": "logic",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "On an island, knights always tell the truth and knaves always lie. "
            "Person A says: 'Both of us are knights.' "
            "Person B says nothing. "
            "What is A? ANSWER: <knight or knave>."
        ),
        "answer": "knave",
        "domain": "logic",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "You have a 3-litre jug and a 5-litre jug. Both start empty. "
            "You need exactly 4 litres of water. "
            "What is the minimum number of steps (fill, empty, or pour) to get exactly 4 litres? "
            "ANSWER: <number>."
        ),
        "answer": "6",
        "domain": "logic",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "If all bloops are razzles and all razzles are lazzles, are all bloops definitely lazzles? "
            "Answer YES or NO. ANSWER: <YES or NO>."
        ),
        "answer": "YES",
        "domain": "logic",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "A farmer needs to cross a river with a fox, a chicken, and a bag of grain. "
            "The boat can carry the farmer and one other item. "
            "The fox eats the chicken if left alone; the chicken eats the grain if left alone. "
            "What is the minimum number of river crossings (one-way trips) needed? "
            "ANSWER: <number>."
        ),
        "answer": "7",
        "domain": "logic",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "There are 4 cards on a table, each with a number on one side and a color on the other. "
            "Visible: 3, 8, red, blue. "
            "Rule: If a card has an even number, the other side must be red. "
            "Which cards must you flip to verify the rule? "
            "List the cards (by what's showing) separated by commas. ANSWER: <cards>."
        ),
        "answer": "8, blue",
        "domain": "logic",
        "difficulty": "hard",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # CODE (10 problems — trace the output, don't write code)
    # ─────────────────────────────────────────────────────────────────────────

    {
        "prompt": (
            "What is the output of this Python code?\n"
            "```python\n"
            "x = [1, 2, 3, 4, 5]\n"
            "print(sum(x[1:4]))\n"
            "```\n"
            "ANSWER: <number>"
        ),
        "answer": "9",
        "domain": "code",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "What is the output of this Python code?\n"
            "```python\n"
            "def f(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return f(n-1) + f(n-2)\n"
            "print(f(7))\n"
            "```\n"
            "ANSWER: <number>"
        ),
        "answer": "13",
        "domain": "code",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "What is the output of this Python code?\n"
            "```python\n"
            "result = 0\n"
            "for i in range(1, 6):\n"
            "    result += i * i\n"
            "print(result)\n"
            "```\n"
            "ANSWER: <number>"
        ),
        "answer": "55",
        "domain": "code",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "What is the output of this Python code?\n"
            "```python\n"
            "s = 'hello world'\n"
            "print(s.count('l'))\n"
            "```\n"
            "ANSWER: <number>"
        ),
        "answer": "3",
        "domain": "code",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "What is the output of this Python code?\n"
            "```python\n"
            "d = {'a': 1, 'b': 2, 'c': 3}\n"
            "d['b'] = d['a'] + d['c']\n"
            "print(sum(d.values()))\n"
            "```\n"
            "ANSWER: <number>"
        ),
        "answer": "8",
        "domain": "code",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "What is the output of this Python code?\n"
            "```python\n"
            "nums = [3, 1, 4, 1, 5, 9, 2, 6]\n"
            "nums.sort()\n"
            "print(nums[3])\n"
            "```\n"
            "ANSWER: <number>"
        ),
        "answer": "4",
        "domain": "code",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "What is the output of this Python code?\n"
            "```python\n"
            "def mystery(lst):\n"
            "    return [x for x in lst if x % 2 == 0]\n"
            "print(len(mystery([1,2,3,4,5,6,7,8])))\n"
            "```\n"
            "ANSWER: <number>"
        ),
        "answer": "4",
        "domain": "code",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "What is the output of this Python code?\n"
            "```python\n"
            "a = [1, 2, 3]\n"
            "b = a\n"
            "b.append(4)\n"
            "print(len(a))\n"
            "```\n"
            "ANSWER: <number>"
        ),
        "answer": "4",
        "domain": "code",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "What is the output of this Python code?\n"
            "```python\n"
            "x = 5\n"
            "result = x ** 2 - 3 * x + 2\n"
            "print(result)\n"
            "```\n"
            "ANSWER: <number>"
        ),
        "answer": "12",
        "domain": "code",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "What is the output of this Python code?\n"
            "```python\n"
            "stack = []\n"
            "for i in [1, 2, 3, 4, 5]:\n"
            "    stack.append(i)\n"
            "total = 0\n"
            "for _ in range(3):\n"
            "    total += stack.pop()\n"
            "print(total)\n"
            "```\n"
            "ANSWER: <number>"
        ),
        "answer": "12",
        "domain": "code",
        "difficulty": "medium",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # SCIENCE (10 problems)
    # ─────────────────────────────────────────────────────────────────────────

    {
        "prompt": (
            "An object accelerates from rest at 3 m/s². "
            "How fast is it moving after 6 seconds? "
            "Use v = u + at. ANSWER: <number> m/s."
        ),
        "answer": "18",
        "domain": "science",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "A ball is dropped from a height of 80 m. "
            "Using g = 10 m/s² and ignoring air resistance, "
            "how long does it take to hit the ground? "
            "Use h = ½gt². ANSWER: <number> seconds."
        ),
        "answer": "4",
        "domain": "science",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "Water boils at 100°C. What is this temperature in Kelvin? "
            "Use K = C + 273. ANSWER: <number>."
        ),
        "answer": "373",
        "domain": "science",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "A 500 W heater runs for 2 hours. "
            "How much energy (in kilojoules) does it consume? "
            "1 kWh = 3600 kJ. ANSWER: <number> kJ."
        ),
        "answer": "3600",
        "domain": "science",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "What is the molecular weight of water (H₂O)? "
            "Atomic weights: H=1, O=16. ANSWER: <number>."
        ),
        "answer": "18",
        "domain": "science",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "How many moles of NaCl are in 117 grams? "
            "Molecular weight of NaCl: Na=23, Cl=35.5. ANSWER: <number>."
        ),
        "answer": "2",
        "domain": "science",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "Ohm's Law: V = IR. A resistor with R = 10 ohms carries a current of 2.5 A. "
            "What is the voltage across it? ANSWER: <number> V."
        ),
        "answer": "25",
        "domain": "science",
        "difficulty": "easy",
    },
    {
        "prompt": (
            "A car has kinetic energy of 90,000 J and is moving at 30 m/s. "
            "What is the mass of the car in kg? "
            "Use KE = ½mv². ANSWER: <number> kg."
        ),
        "answer": "200",
        "domain": "science",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "The half-life of a radioactive element is 10 years. "
            "Starting with 80 grams, how many grams remain after 30 years? "
            "ANSWER: <number> g."
        ),
        "answer": "10",
        "domain": "science",
        "difficulty": "medium",
    },
    {
        "prompt": (
            "Light travels at approximately 3 × 10⁸ m/s. "
            "How many seconds does it take for light to travel from the Sun to Earth, "
            "a distance of 1.5 × 10¹¹ m? ANSWER: <number> seconds."
        ),
        "answer": "500",
        "domain": "science",
        "difficulty": "medium",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # HARD (5 multi-step problems)
    # ─────────────────────────────────────────────────────────────────────────

    {
        "prompt": (
            "A snail is at the bottom of a 20-foot well. "
            "Each day it climbs 5 feet, but each night it slides back 3 feet. "
            "On which day does it reach the top? "
            "Think step by step. ANSWER: <number>."
        ),
        "answer": "9",
        "domain": "hard",
        "difficulty": "hard",
    },
    {
        "prompt": (
            "You have two ropes. Each rope takes exactly 60 minutes to burn completely, "
            "but burns at a non-uniform rate. "
            "How can you measure exactly 45 minutes? "
            "Describe the steps, then state how many distinct lighting events occur. "
            "ANSWER: <number>."
        ),
        "answer": "3",
        "domain": "hard",
        "difficulty": "hard",
    },
    {
        "prompt": (
            "In a room of 30 people, what is the probability (as a percentage, rounded to nearest whole number) "
            "that at least two people share the same birthday? "
            "Hint: compute P(no shared birthday) = 365/365 × 364/365 × ... × 336/365, then subtract from 1. "
            "ANSWER: <number>%."
        ),
        "answer": "71",
        "domain": "hard",
        "difficulty": "hard",
    },
    {
        "prompt": (
            "A 6×6 grid is filled with numbers 1–36 in row-major order "
            "(row 1: 1–6, row 2: 7–12, ..., row 6: 31–36). "
            "What is the sum of all numbers on the main diagonal (top-left to bottom-right)? "
            "ANSWER: <number>."
        ),
        "answer": "111",
        "domain": "hard",
        "difficulty": "hard",
    },
    {
        "prompt": (
            "Three missionaries and three cannibals need to cross a river. "
            "The boat holds at most 2 people. "
            "Cannibals must never outnumber missionaries on either bank (or the boat) "
            "or the missionaries will be eaten. "
            "What is the minimum number of one-way crossings to get everyone across? "
            "ANSWER: <number>."
        ),
        "answer": "11",
        "domain": "hard",
        "difficulty": "hard",
    },
]

# Sanity check at import time
assert len(HOLDOUT_PROBLEMS) == 50, f"Expected 50 problems, got {len(HOLDOUT_PROBLEMS)}"


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def generate_answer(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """Generate a model response for a given prompt.

    Works with both CUDA (HuggingFace) and MLX backends.
    Returns the raw text string.
    """
    if _USE_CUDA:
        return _cuda_generate(model, tokenizer, prompt, max_tokens)
    else:
        return _mlx_generate(model, tokenizer, prompt, max_tokens)


def _cuda_generate(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """HuggingFace/PyTorch inference."""
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,       # greedy — deterministic eval
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _mlx_generate(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """MLX inference."""
    from mlx_lm import generate
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def normalize_answer(raw: str) -> str:
    """Light normalisation: strip whitespace, lowercase, remove trailing punctuation."""
    s = raw.strip().lower()
    # Remove trailing period or percent sign for numeric answers
    s = s.rstrip(".%").strip()
    return s


def _single_answer_match(p: str, e: str) -> bool:
    """Compare two normalized single-value strings."""
    if p == e:
        return True
    try:
        pf = float(p.replace(",", ""))
        ef = float(e.replace(",", ""))
        return abs(pf - ef) < 0.5
    except ValueError:
        return False


def answers_match(predicted: str, expected: str) -> bool:
    """Compare predicted vs expected answer (no recursion)."""
    p = normalize_answer(predicted)
    e = normalize_answer(expected)

    if _single_answer_match(p, e):
        return True

    # Multi-value answers like "8, blue" — split and compare parts (non-recursive)
    p_parts = [normalize_answer(x) for x in predicted.split(",")]
    e_parts = [normalize_answer(x) for x in expected.split(",")]
    if len(p_parts) == len(e_parts) and len(p_parts) > 1:
        return all(_single_answer_match(pp, ep) for pp, ep in zip(p_parts, e_parts))

    return False


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(model, tokenizer, problems: list[dict]) -> dict:
    """Run all 50 problems and return per-problem and aggregate results."""
    from src.solver import parse_solution

    results_per_problem = []
    domain_stats: dict[str, dict] = {}

    print(f"\n{'─' * 60}")
    print(f"  Running {len(problems)} benchmark problems...")
    print(f"{'─' * 60}")

    for i, problem in enumerate(problems):
        domain = problem["domain"]
        difficulty = problem["difficulty"]
        expected = problem["answer"]

        # Generate model response — wrap with FINAL_ANSWER instruction
        eval_prompt = (
            f"{problem['prompt']}\n\n"
            f"End your response with exactly this line (replace X with your answer):\n"
            f"FINAL_ANSWER: X"
        )
        t0 = time.time()
        raw_response = generate_answer(model, tokenizer, eval_prompt)
        elapsed = time.time() - t0

        # Extract answer using the project's standard parser
        solution = parse_solution(raw_response, domain)
        predicted = solution.answer

        correct = answers_match(predicted, expected)

        # Track domain stats
        if domain not in domain_stats:
            domain_stats[domain] = {"correct": 0, "total": 0}
        domain_stats[domain]["total"] += 1
        if correct:
            domain_stats[domain]["correct"] += 1

        # Log per-problem result
        result_entry = {
            "index": i,
            "domain": domain,
            "difficulty": difficulty,
            "prompt": problem["prompt"][:120] + "...",  # truncate for readability
            "expected": expected,
            "predicted": predicted,
            "raw_response_snippet": raw_response[:200],
            "correct": correct,
            "latency_s": round(elapsed, 2),
        }
        results_per_problem.append(result_entry)

        status = "✅" if correct else "❌"
        print(
            f"  [{i+1:2d}/50] {status} [{domain:8s}/{difficulty:6s}] "
            f"expected={expected!r:10s} got={predicted!r:10s}"
        )

    # Compute per-domain accuracy
    domain_accuracy = {
        d: stats["correct"] / stats["total"]
        for d, stats in domain_stats.items()
    }

    total_correct = sum(s["correct"] for s in domain_stats.values())
    overall_accuracy = total_correct / len(problems)

    print(f"\n{'─' * 60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'─' * 60}")
    for domain, acc in sorted(domain_accuracy.items()):
        stats = domain_stats[domain]
        print(f"  {domain:10s}: {acc:.1%}  ({stats['correct']}/{stats['total']})")
    print(f"{'─' * 60}")
    print(f"  OVERALL   : {overall_accuracy:.1%}  ({total_correct}/{len(problems)})")
    print(f"{'─' * 60}\n")

    return {
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_problems": len(problems),
        "domain_accuracy": domain_accuracy,
        "domain_stats": domain_stats,
        "per_problem": results_per_problem,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prometheus Benchmark Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MLX backend (macOS)
  python evaluate.py --model mlx-community/Qwen3.5-4B-MLX-4bit

  # CUDA backend with checkpoint
  PROMETHEUS_BACKEND=cuda python evaluate.py \\
      --model /root/models/qwen3.5-4b \\
      --checkpoint experiments/cuda_run_007/checkpoint.npz
        """,
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or local path")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .npz checkpoint to apply on top of base model (LoRA weights)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks",
        help="Directory to save eval results JSON (default: benchmarks/)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per model response (default: 512)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="*",
        default=None,
        help="Only run problems for these domains (e.g. --domains math logic)",
    )
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backend_label = "cuda" if _USE_CUDA else "mlx"

    print("=" * 60)
    print("📊 PROMETHEUS — Benchmark Evaluation")
    print("=" * 60)
    print(f"  Model   : {args.model}")
    print(f"  Backend : {backend_label}")
    print(f"  Chkpoint: {args.checkpoint or 'none (base model)'}")
    print(f"  Problems: {len(HOLDOUT_PROBLEMS)}")
    print()

    # --- Load model ---
    print("📦 Loading model...")
    try:
        if _USE_CUDA:
            from src.load_model_cuda import load_model_cuda
            model, tokenizer = load_model_cuda(args.model, device="cuda")
        else:
            from mlx_lm import load
            model, tokenizer = load(args.model)
    except Exception as e:
        print(f"❌ Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)
    print("✅ Model loaded\n")

    # --- Apply checkpoint (LoRA weights) if provided ---
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found: {args.checkpoint} — evaluating base model")
        else:
            print(f"🔧 Loading checkpoint: {args.checkpoint}")
            try:
                if _USE_CUDA:
                    import numpy as np
                    import torch
                    weights = np.load(str(checkpoint_path), allow_pickle=True)
                    # Try to load LoRA adapter weights if present
                    state_dict = {k: torch.tensor(weights[k]).to(model.device) for k in weights.files}
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    print(f"   Loaded {len(state_dict) - len(unexpected)} weight tensors "
                          f"({len(missing)} missing, {len(unexpected)} unexpected)")
                else:
                    import mlx.core as mx
                    import mlx.nn as nn
                    weights = mx.load(str(checkpoint_path))
                    model.load_weights(list(weights.items()), strict=False)
                    mx.eval(model.parameters())
                print("✅ Checkpoint applied\n")
            except Exception as e:
                print(f"⚠️  Checkpoint load failed ({e}) — evaluating base model\n")

    # --- Filter problems if requested ---
    problems = HOLDOUT_PROBLEMS
    if args.domains:
        problems = [p for p in HOLDOUT_PROBLEMS if p["domain"] in args.domains]
        print(f"ℹ️  Filtered to domains {args.domains}: {len(problems)} problems\n")

    # --- Run evaluation ---
    eval_results = run_evaluation(model, tokenizer, problems)

    # --- Save results ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_results_{timestamp}.json"

    full_results = {
        "timestamp": timestamp,
        "model": args.model,
        "checkpoint": args.checkpoint,
        "backend": backend_label,
        "num_problems": len(problems),
        **eval_results,
    }

    output_path.write_text(json.dumps(full_results, indent=2))
    print(f"💾 Results saved to: {output_path}")

    # Also print compact summary for easy piping/grepping
    print(f"\n📊 SUMMARY: overall={eval_results['overall_accuracy']:.1%} "
          f"| {eval_results['total_correct']}/{eval_results['total_problems']} correct")
    for domain, acc in sorted(eval_results["domain_accuracy"].items()):
        print(f"   {domain}: {acc:.1%}")

    # Return 0 (success) — exit code useful in scripting
    return 0


if __name__ == "__main__":
    sys.exit(main())
