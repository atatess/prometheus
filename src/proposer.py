"""
Problem Proposer — generates Code-as-Task problems.

The Proposer generates problems expressed as executable Python with
built-in test assertions. This makes ANY domain automatically verifiable.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import random


@dataclass
class Problem:
    """A Code-as-Task problem."""
    domain: str
    difficulty: str  # "easy", "medium", "hard"
    prompt: str  # Natural language problem statement
    problem_code: str  # Python setup code
    test_code: str  # Python assertions for verification
    solution_hint: Optional[str] = None  # Optional hint for the answer format
    metadata: dict = field(default_factory=dict)


# --- Prompt Templates ---

MATH_PROPOSER_PROMPT = """Solve this task: generate a {difficulty} math problem about {topic} and compute its answer.

Step 1 — Write a problem. Step 2 — Solve it. Step 3 — Output ONLY this JSON with real values:
{{"prompt": "<your problem here>", "expected_answer": "<the number you computed>"}}

Example of correct output:
{{"prompt": "A train travels 120 km in 2 hours. What is its speed in km/h?", "expected_answer": "60"}}

Topic: {topic}, Difficulty: {difficulty}. Output JSON only.
"""

CODE_PROPOSER_PROMPT = """You are a coding challenge generator. Generate a Python programming
problem that can be automatically tested.

Difficulty: {difficulty}
Topic: {topic}

Output EXACTLY in this JSON format:
{{
    "prompt": "Problem description",
    "problem_code": "def solution_template():\\n    # Student implements this\\n    pass",
    "test_code": "assert solution(input1) == expected1\\nassert solution(input2) == expected2",
    "function_name": "The function to implement"
}}

Include 3-5 test cases in test_code.
"""

LOGIC_PROPOSER_PROMPT = """Solve this task: generate a {difficulty} logic puzzle and compute its answer.

Step 1 — Write a self-contained puzzle. Step 2 — Solve it. Step 3 — Output ONLY this JSON:
{{"prompt": "<your puzzle here>", "expected_answer": "<the number you computed>"}}

Example of correct output:
{{"prompt": "Alice is twice Bob's age. Bob is 15. How old is Alice?", "expected_answer": "30"}}

Difficulty: {difficulty}. Output JSON only.
"""

SPATIAL_PROPOSER_PROMPT = """Solve this task: generate a {difficulty} spatial reasoning problem and compute its answer.

Topics: faces/edges/vertices of shapes, painted cube cuts, grid paths, nets, rotations.
Step 1 — Write a problem. Step 2 — Solve it. Step 3 — Output ONLY this JSON:
{{"prompt": "<your problem here>", "expected_answer": "<the number you computed>"}}

Example of correct output:
{{"prompt": "How many faces does a triangular prism have?", "expected_answer": "5"}}

Difficulty: {difficulty}. Output JSON only.
"""

# Domain registry — maps domain names to their proposer prompts
DOMAIN_PROMPTS = {
    "math": MATH_PROPOSER_PROMPT,
    "code": CODE_PROPOSER_PROMPT,
    "logic": LOGIC_PROPOSER_PROMPT,
    "spatial": SPATIAL_PROPOSER_PROMPT,
}

MATH_TOPICS = [
    "algebra", "number theory", "combinatorics", "probability",
    "geometry", "sequences", "modular arithmetic", "polynomials",
    "inequalities", "functions",
]

CODE_TOPICS = [
    "arrays", "strings", "dynamic programming", "graphs",
    "sorting", "searching", "recursion", "trees",
    "hash maps", "greedy algorithms",
]


def build_proposer_prompt(domain: str, difficulty: str = "medium") -> str:
    """Build a prompt for the proposer model to generate a problem."""
    template = DOMAIN_PROMPTS.get(domain)
    if template is None:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_PROMPTS.keys())}")

    topic = ""
    if domain == "math":
        topic = random.choice(MATH_TOPICS)
    elif domain == "code":
        topic = random.choice(CODE_TOPICS)

    return template.format(difficulty=difficulty, topic=topic)


def parse_proposed_problem(domain: str, raw_output: str) -> Optional[Problem]:
    """Parse the model's raw output into a Problem object."""
    try:
        # Extract JSON from the response (handle thinking tags and code blocks)
        text = raw_output.strip()
        
        # Strip <think>...</think> tags (Qwen3.5 thinking model)
        if "<think>" in text:
            # Take everything after the last </think>
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()
            else:
                # Thinking didn't close — try to find JSON after it
                pass
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        data = json.loads(text)
        
        # Must be a dict with prompt key
        if not isinstance(data, dict):
            return None

        # Validate — reject template echoes and placeholder values
        prompt = data.get("prompt", "")
        expected = str(data.get("expected_answer", ""))
        
        bad_prompts = [
            "your math question", "your question", "problem text",
            "problem statement", "the logic puzzle", "puzzle description",
            "spatial problem", "...", "insert problem", "example problem",
        ]
        bad_answers = ["answer", "...", "expected", "result", "value", ""]
        
        if any(b in prompt.lower() for b in bad_prompts):
            return None
        if expected.lower() in bad_answers or len(expected) == 0:
            return None
        if len(prompt) < 15:  # Too short to be a real problem
            return None

        return Problem(
            domain=domain,
            difficulty=data.get("difficulty", "medium"),
            prompt=prompt,
            problem_code=data.get("problem_code", f"expected = {data.get('expected_answer', '')}"),
            test_code=data.get("test_code", "assert str(student_answer).strip() == str(expected).strip()"),
            metadata=data,
        )
    except (json.JSONDecodeError, KeyError) as e:
        return None


def register_domain(name: str, prompt_template: str, topics: list[str] = None):
    """Register a new domain for problem generation."""
    DOMAIN_PROMPTS[name] = prompt_template
    if topics:
        # Store topics for the domain
        globals()[f"{name.upper()}_TOPICS"] = topics
