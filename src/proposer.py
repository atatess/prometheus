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

MATH_PROPOSER_PROMPT = """You are a math problem generator. Generate a novel math problem
that can be verified programmatically.

Difficulty: {difficulty}
Topic: {topic}

Output EXACTLY in this JSON format:
{{
    "prompt": "The problem statement in natural language",
    "problem_code": "Python code that sets up the problem (define variables, expected answer)",
    "test_code": "Python assertions that verify the solution",
    "expected_answer": "The correct answer"
}}

The test_code should check if a variable called `student_answer` equals the expected answer.
Make the problem novel — don't use textbook examples.
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

LOGIC_PROPOSER_PROMPT = """You are a logic puzzle generator. Generate a logic/reasoning problem
that can be verified by running a Python program.

Difficulty: {difficulty}

Output EXACTLY in this JSON format:
{{
    "prompt": "The logic puzzle in natural language",
    "problem_code": "Python code that encodes the constraints",
    "test_code": "Python assertions that verify the solution satisfies all constraints",
    "expected_answer": "The correct answer"
}}

The key insight: express the logic puzzle as CONSTRAINTS in Python.
For example, a scheduling problem becomes constraint checking code.
"""

# Domain registry — maps domain names to their proposer prompts
DOMAIN_PROMPTS = {
    "math": MATH_PROPOSER_PROMPT,
    "code": CODE_PROPOSER_PROMPT,
    "logic": LOGIC_PROPOSER_PROMPT,
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
        # Extract JSON from the response (handle markdown code blocks)
        text = raw_output.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        data = json.loads(text)

        return Problem(
            domain=domain,
            difficulty=data.get("difficulty", "medium"),
            prompt=data["prompt"],
            problem_code=data["problem_code"],
            test_code=data["test_code"],
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
