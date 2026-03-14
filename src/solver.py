"""
Solver — generates solutions to Code-as-Task problems.

The Solver takes a problem prompt and attempts to produce a solution.
It can optionally use tool calls (Python execution) to work through
multi-step problems.
"""

from dataclasses import dataclass
from typing import Optional
from .proposer import Problem


SOLVER_PROMPT = """Solve this problem. Think carefully, then give your answer.

{problem_prompt}

{format_hint}

FINAL_ANSWER: [write only the answer here, nothing else]
"""

# Code domain uses the same answer format — "what does this print?" = numeric answer.
# Do NOT use a code-writing prompt for code domain seed problems.


@dataclass
class Solution:
    """A solution attempt."""
    raw_response: str
    answer: str
    reasoning: Optional[str] = None
    code: Optional[str] = None


def build_solver_prompt(problem: Problem) -> str:
    """Build a prompt for the solver model.
    
    All domains (including code) use SOLVER_PROMPT — our code problems are
    'what does this print?' questions with a numeric answer, not code-writing tasks.
    """
    format_hint = ""
    if problem.solution_hint:
        format_hint = f"Answer format: {problem.solution_hint}"
    
    return SOLVER_PROMPT.format(
        problem_prompt=problem.prompt,
        format_hint=format_hint,
    )


def parse_solution(raw_response: str, domain: str) -> Solution:
    """Parse the model's raw response into a Solution object."""
    import re
    text = raw_response.strip()

    # Pattern 0: Search for ANSWER: in the raw text.
    # Take the LAST match — the model often echoes "ANSWER: " from the prompt
    # instructions inside its thinking ("Put your answer after ANSWER: on its own line"),
    # so the first hit is the instruction echo, the last hit is the actual answer.
    # Also exclude lines where ANSWER: is preceded by other words (instruction context).
    # Pattern 0a: FINAL_ANSWER: marker (our primary, distinctive format)
    fa_matches = re.findall(r'FINAL_ANSWER:\s*\[?([^\]\n\[]+?)\]?(?:\n|$)', text)
    if fa_matches:
        for candidate in reversed(fa_matches):
            candidate = candidate.strip()
            if candidate and 'write only' not in candidate and candidate != '...':
                return Solution(raw_response=raw_response, answer=candidate)

    # Pattern 0b: ANSWER: marker (legacy, take last non-instruction match)
    answer_matches = re.findall(r'(?:^|\n)ANSWER:\s*([^\n"(]+?)(?:\n|$)', text)
    if answer_matches:
        for candidate in reversed(answer_matches):
            candidate = candidate.strip()
            if candidate and candidate not in ('...', '[number]', 'X', 'the answer', 'your answer'):
                return Solution(raw_response=raw_response, answer=candidate)

    # Strip thinking blocks to get the final response
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text:
        # Truncated thinking — look for a number in the visible content
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return Solution(raw_response=raw_response, answer=numbers[-1])
    elif "Thinking Process:" in text:
        # Alternative thinking format — take last meaningful line
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        # Look for "Answer:" or similar near the end
        for line in reversed(lines[-5:]):
            m = re.match(r'^(?:Answer|Result|Therefore|So)[:\s]+(.+)', line, re.IGNORECASE)
            if m:
                return Solution(raw_response=raw_response, answer=m.group(1).strip())
        # Fallback: last standalone number
        numbers = re.findall(r'(?:^|\s)(-?\d+\.?\d*)(?:\s|$)', text)
        if numbers:
            return Solution(raw_response=raw_response, answer=numbers[-1].strip())

    # Try to extract answer from common patterns
    answer = text

    # Pattern 1: "ANSWER: X" (already searched above, belt-and-suspenders)
    answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text)
    if answer_match:
        answer = answer_match.group(1).strip()
        return Solution(raw_response=raw_response, answer=answer)
    
    # Pattern 2: "The answer is X"
    answer_match = re.search(r'[Tt]he (?:final )?answer is[:\s]*([^\n.]+)', text)
    if answer_match:
        answer = answer_match.group(1).strip().rstrip('.')
        return Solution(raw_response=raw_response, answer=answer)
    
    # Pattern 3: boxed answer (common in math)
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
    
    # NOTE: code domain answers are NUMBERS (output of code), not code blocks.
    # Do not use extracted code blocks as the answer for any domain.
    
    # Fallback: extract last number/fraction from text (for math)
    if domain in ("math", "logic", "spatial", "science", "data") and answer == text:
        # Try fractions first (e.g., 1/6, 5/14)
        fracs = re.findall(r'\d+/\d+', text)
        if fracs:
            answer = fracs[-1]
        else:
            numbers = re.findall(r'-?\d+\.?\d*', text)
            if numbers:
                answer = numbers[-1]
    
    return Solution(
        raw_response=raw_response,
        answer=answer,
        code=code,
    )
