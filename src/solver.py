"""
Solver — generates solutions to Code-as-Task problems.

The Solver takes a problem prompt and attempts to produce a solution.
It can optionally use tool calls (Python execution) to work through
multi-step problems.
"""

from dataclasses import dataclass
from typing import Optional
from .proposer import Problem


SOLVER_PROMPT = """You are solving a problem. Think step by step, then provide your final answer.

{problem_prompt}

{format_hint}

Think carefully and show your work. Put your final answer in the format requested.
"""

CODE_SOLVER_PROMPT = """You are solving a coding problem. Write a Python function that solves it.

{problem_prompt}

Write ONLY the function implementation. Do not include test cases.
"""


@dataclass
class Solution:
    """A solution attempt."""
    raw_response: str
    answer: str
    reasoning: Optional[str] = None
    code: Optional[str] = None


def build_solver_prompt(problem: Problem) -> str:
    """Build a prompt for the solver model."""
    if problem.domain == "code":
        return CODE_SOLVER_PROMPT.format(
            problem_prompt=problem.prompt,
        )
    
    format_hint = ""
    if problem.solution_hint:
        format_hint = f"Answer format: {problem.solution_hint}"
    
    return SOLVER_PROMPT.format(
        problem_prompt=problem.prompt,
        format_hint=format_hint,
    )


def parse_solution(raw_response: str, domain: str) -> Solution:
    """Parse the model's raw response into a Solution object."""
    text = raw_response.strip()
    
    # Strip <think>...</think> tags
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text:
        # Thinking didn't close — try to find answer after thinking markers
        pass
    
    # Try to extract a boxed answer (common in math)
    answer = text
    if "\\boxed{" in text:
        try:
            start = text.index("\\boxed{") + 7
            depth = 1
            end = start
            while depth > 0 and end < len(text):
                if text[end] == "{":
                    depth += 1
                elif text[end] == "}":
                    depth -= 1
                end += 1
            answer = text[start : end - 1]
        except (ValueError, IndexError):
            pass
    
    # Try to extract code blocks
    code = None
    if "```python" in text:
        try:
            code = text.split("```python")[1].split("```")[0].strip()
        except IndexError:
            pass
    elif "```" in text:
        try:
            code = text.split("```")[1].split("```")[0].strip()
        except IndexError:
            pass
    
    # For code problems, the code IS the answer
    if domain == "code" and code:
        answer = code
    
    return Solution(
        raw_response=text,
        answer=answer,
        code=code,
    )
