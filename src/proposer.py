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

MATH_PROPOSER_PROMPT = """You are a math teacher writing exam questions. Write one {difficulty} math problem about {topic} and solve it.

Here are three examples of the format I want:

PROBLEM: A train travels 120 km in 2 hours. What is its speed in km/h?
ANSWER: 60

PROBLEM: How many prime numbers are less than 10?
ANSWER: 4

PROBLEM: What is 15% of 80?
ANSWER: 12

Now write a NEW problem about {topic} at {difficulty} difficulty. Do NOT copy the examples above. Solve it yourself.

PROBLEM:"""

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

LOGIC_PROPOSER_PROMPT = """You are a logic teacher writing puzzles. Write one {difficulty} logic or reasoning puzzle and solve it.

Here are two examples of the format I want:

PROBLEM: Alice is twice Bob's age. Bob is 15. How old is Alice?
ANSWER: 30

PROBLEM: If it takes 3 workers 6 days to dig a ditch, how many days would 9 workers take?
ANSWER: 2

Now write a NEW puzzle at {difficulty} difficulty. Do NOT copy the examples. Solve it yourself.

PROBLEM:"""

SPATIAL_PROPOSER_PROMPT = """You are a geometry teacher writing problems. Write one {difficulty} spatial reasoning problem and solve it.

Topics: edges/faces/vertices of 3D shapes, painted cube cuts, grid paths, symmetry.

Here are two examples of the format I want:

PROBLEM: How many edges does a triangular prism have?
ANSWER: 9

PROBLEM: A cube is painted red on all faces then cut into 8 equal pieces. How many pieces have exactly 3 painted faces?
ANSWER: 8

Now write a NEW problem at {difficulty} difficulty. Do NOT copy the examples. Solve it yourself.

PROBLEM:"""

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
    """Parse the model's raw output into a Problem object.
    
    Supports two formats:
    1. PROBLEM: ... / ANSWER: ... (new plain-text format)
    2. JSON {"prompt": ..., "expected_answer": ...} (legacy)
    """
    import re
    
    # Strip thinking tags first
    text = raw_output.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    
    # --- Try plain-text PROBLEM/ANSWER format first ---
    prob_match = re.search(r'PROBLEM:\s*(.+?)(?=\nANSWER:|\Z)', text, re.DOTALL | re.IGNORECASE)
    ans_match = re.search(r'ANSWER:\s*(.+?)(?=\n[A-Z]|\Z)', text, re.DOTALL | re.IGNORECASE)
    
    if prob_match and ans_match:
        prompt = prob_match.group(1).strip()
        expected = ans_match.group(1).strip().split('\n')[0].strip()  # first line only
        
        if _is_valid_problem(prompt, expected):
            return Problem(
                domain=domain,
                difficulty="medium",
                prompt=prompt,
                problem_code=f"expected = {repr(expected)}",
                test_code="assert str(student_answer).strip() == str(expected).strip()",
                metadata={"expected_answer": expected},
            )
    
    # --- Fallback: try JSON format ---
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        # Find JSON object
        json_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if isinstance(data, dict):
                prompt = data.get("prompt", "")
                expected = str(data.get("expected_answer", ""))
                if _is_valid_problem(prompt, expected):
                    return Problem(
                        domain=domain,
                        difficulty=data.get("difficulty", "medium"),
                        prompt=prompt,
                        problem_code=data.get("problem_code", f"expected = {repr(expected)}"),
                        test_code=data.get("test_code", "assert str(student_answer).strip() == str(expected).strip()"),
                        metadata=data,
                    )
    except (json.JSONDecodeError, KeyError):
        pass
    
    return None


def _is_valid_problem(prompt: str, expected: str) -> bool:
    """Validate that a problem/answer pair is not a template echo."""
    bad_prompts = [
        "your math question", "your question", "problem text", "problem statement",
        "the logic puzzle", "puzzle description", "spatial problem", "example problem",
        "insert problem", "problem here", "your problem", "your puzzle",
    ]
    bad_answers = [
        "answer", "...", "expected", "result", "value", "number",
        "the number you computed", "computed", "solution",
    ]
    
    if len(prompt) < 15:
        return False
    if any(b in prompt.lower() for b in bad_prompts):
        return False
    if expected.lower() in bad_answers or len(expected) == 0:
        return False
    if expected in ["...", "<", ">", "?"]:
        return False
    
    return True


def register_domain(name: str, prompt_template: str, topics: list[str] = None):
    """Register a new domain for problem generation."""
    DOMAIN_PROMPTS[name] = prompt_template
    if topics:
        # Store topics for the domain
        globals()[f"{name.upper()}_TOPICS"] = topics
