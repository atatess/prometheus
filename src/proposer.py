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

MATH_PROPOSER_PROMPT = """Write one math problem about {topic} and its answer.

Output ONLY these two lines, nothing else:
PROBLEM: [your problem statement here]
ANSWER: [the numeric answer here]

Examples:
PROBLEM: A train travels 120 km in 2 hours. What is its speed?
ANSWER: 60

PROBLEM: How many primes are less than 10?
ANSWER: 4

PROBLEM: What is 15% of 80?
ANSWER: 12

Your new {difficulty} problem about {topic}:
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

LOGIC_PROPOSER_PROMPT = """Write one logic/reasoning puzzle and its answer.

Output ONLY these two lines, nothing else:
PROBLEM: [your puzzle here]
ANSWER: [the numeric answer here]

Examples:
PROBLEM: Alice is twice Bob's age. Bob is 15. How old is Alice?
ANSWER: 30

PROBLEM: 3 workers dig a ditch in 6 days. How many days for 9 workers?
ANSWER: 2

Your new {difficulty} puzzle:
PROBLEM:"""

SPATIAL_PROPOSER_PROMPT = """Write one spatial/geometry problem and its answer.

Output ONLY these two lines, nothing else:
PROBLEM: [your problem here]
ANSWER: [the numeric answer here]

Examples:
PROBLEM: How many edges does a triangular prism have?
ANSWER: 9

PROBLEM: A cube painted on all faces, cut into 8 equal pieces. How many pieces have exactly 3 painted faces?
ANSWER: 8

Your new {difficulty} problem:
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

    Search strategy: scan the FULL raw output (including thinking blocks) for
    PROBLEM/ANSWER patterns and JSON. The model often puts the answer inside
    the <think> block or Thinking Process section — we want all of it.

    Supports:
    1. PROBLEM: ... / ANSWER: ... (plain-text format — most domains)
    2. JSON {"prompt": ..., "expected_answer": ...} (code domain)
    3. Raw continuation: model directly outputs problem text + ANSWER:
    """
    import re

    # Work on the full text — don't strip thinking, the answer lives inside it
    text = raw_output.strip()
    # Remove <think> tags but keep the content inside
    text = text.replace("<think>", "").replace("</think>", "")
    # Remove "Thinking Process:" header but keep content
    text = re.sub(r'^Thinking Process:\s*\n?', '', text, flags=re.MULTILINE)

    # --- 1. PROBLEM: ... / ANSWER: ... format ---
    # Search the whole text — the model may write it in the thinking section
    prob_match = re.search(r'PROBLEM:\s*(.+?)(?=\nANSWER:)', text, re.DOTALL | re.IGNORECASE)
    ans_match  = re.search(r'ANSWER:\s*(.+?)(?=\n(?:PROBLEM:|NOTE:|$)|\Z)', text, re.DOTALL | re.IGNORECASE)

    if prob_match and ans_match:
        prompt   = prob_match.group(1).strip()
        expected = ans_match.group(1).strip().split('\n')[0].strip()
        if _is_valid_problem(prompt, expected):
            return Problem(
                domain=domain,
                difficulty="medium",
                prompt=prompt,
                problem_code=f"expected = {repr(expected)}",
                test_code="assert str(student_answer).strip() == str(expected).strip()",
                metadata={"expected_answer": expected},
            )

    # The math/logic prompts end with "PROBLEM:" so the model just continues
    # without repeating the keyword — match "text\nANSWER: N" directly
    direct_match = re.search(r'^(.+?)\nANSWER:\s*(.+?)(?:\n|\Z)', text, re.DOTALL | re.IGNORECASE)
    if direct_match:
        prompt   = direct_match.group(1).strip()
        expected = direct_match.group(2).strip().split('\n')[0].strip()
        if _is_valid_problem(prompt, expected):
            return Problem(
                domain=domain,
                difficulty="medium",
                prompt=prompt,
                problem_code=f"expected = {repr(expected)}",
                test_code="assert str(student_answer).strip() == str(expected).strip()",
                metadata={"expected_answer": expected},
            )

    # --- 2. JSON format (code domain + others) ---
    # Use a proper brace-balanced scanner — handles nested JSON
    def _find_json(src: str):
        for m in re.finditer(r'\{', src):
            depth, end, in_str, esc = 0, -1, False, False
            for i, ch in enumerate(src[m.start():]):
                if esc:             esc = False; continue
                if ch == '\\' and in_str: esc = True; continue
                if ch == '"':       in_str = not in_str; continue
                if in_str:          continue
                if ch == '{':       depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:  end = m.start() + i + 1; break
            if end == -1: continue
            try:
                data = json.loads(src[m.start():end])
                if isinstance(data, dict) and "prompt" in data:
                    return data
            except json.JSONDecodeError:
                continue
        return None

    # Try markdown fences first, then full text
    for block in [text]:
        if "```json" in block:
            block = block.split("```json")[1].split("```")[0]
        elif "```" in block:
            block = block.split("```")[1].split("```")[0]
        data = _find_json(block)
        if data:
            prompt   = data.get("prompt", "")
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
