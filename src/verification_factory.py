"""
Verification Factory — the novel contribution.

This module enables the model to autonomously expand the space of
verifiable rewards by writing domain-specific Python verifiers.

The key insight: verification is always easier than generation.
A model that can't solve a problem CAN often write a program that
checks whether a given answer is correct.
"""

import json
import importlib.util
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from .verifier import verify_code_task, SandboxConfig
from .proposer import register_domain


@dataclass
class DomainVerifier:
    """A generated verifier for a new domain."""
    domain: str
    description: str
    verifier_code: str
    proposer_template: str
    test_examples: list[dict] = field(default_factory=list)
    validated: bool = False
    accuracy_on_tests: float = 0.0


FACTORY_PROMPT = """You are designing a verification system for a new reasoning domain.

Domain: {domain}
Description: {description}

Your job is to create:
1. A Python verifier function that checks if an answer is correct
2. A proposer template that generates problems in this domain
3. 5 test examples (problem + correct answer + incorrect answer)

The verifier must be PROGRAMMATIC — it runs Python code to check answers,
not just string matching. Express the domain's logic as executable constraints.

Output EXACTLY in this JSON format:
{{
    "domain": "{domain}",
    "description": "{description}",
    "verifier_code": "def verify(problem_data, answer):\\n    # Check answer correctness\\n    ...",
    "proposer_template": "You are a {domain} problem generator. Generate a problem...\\n\\nOutput in JSON with: prompt, problem_code, test_code, expected_answer",
    "test_examples": [
        {{
            "problem_code": "# Setup code",
            "correct_answer": "the right answer",
            "incorrect_answer": "a wrong answer",
            "test_code": "# Assertions that check the answer"
        }},
        ...
    ]
}}

CRITICAL: The verifier must use pure Python (math, itertools, re, collections, etc).
No external APIs, no file I/O, no network access.
"""

# Token budget for factory generation — needs more room than regular inference
# (full JSON with verifier code + 5 test examples is 800-1200 tokens)
FACTORY_MAX_TOKENS = 2500

# Candidate domains for expansion
EXPANSION_CANDIDATES = [
    {
        "domain": "logic",
        "description": "Propositional and predicate logic, truth tables, logical deduction",
    },
    {
        "domain": "spatial",
        "description": "Spatial reasoning — positions, directions, relative locations",
    },
    {
        "domain": "planning",
        "description": "Sequential planning — ordering steps, scheduling, resource allocation",
    },
    {
        "domain": "constraint",
        "description": "Constraint satisfaction — Sudoku-like puzzles, scheduling constraints",
    },
    {
        "domain": "probability",
        "description": "Probability and statistics — compute probabilities, expected values",
    },
    {
        "domain": "graph",
        "description": "Graph theory — shortest paths, connectivity, coloring, matching",
    },
    {
        "domain": "data_analysis",
        "description": "Data analysis — given data, compute statistics, find patterns",
    },
    {
        "domain": "physics",
        "description": "Classical physics — kinematics, forces, energy, circuits",
    },
]


class VerificationFactory:
    """Creates and validates new domain verifiers."""

    def __init__(self, generated_dir: str = "src/domains/generated"):
        self.generated_dir = Path(generated_dir)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.verifiers: dict[str, DomainVerifier] = {}
        self._load_existing()

    def _load_existing(self):
        """Load previously generated verifiers."""
        for f in self.generated_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                verifier = DomainVerifier(**data)
                if verifier.validated:
                    self.verifiers[verifier.domain] = verifier
            except (json.JSONDecodeError, TypeError):
                continue

    def get_next_candidate(self, existing_domains: list[str]) -> Optional[dict]:
        """Get the next domain candidate for expansion."""
        for candidate in EXPANSION_CANDIDATES:
            if candidate["domain"] not in existing_domains:
                return candidate
        return None

    def build_factory_prompt(self, domain: str, description: str) -> str:
        """Build the prompt for the model to generate a verifier."""
        return FACTORY_PROMPT.format(domain=domain, description=description)

    def parse_verifier(self, raw_output: str) -> Optional[DomainVerifier]:
        """Parse the model's output into a DomainVerifier.

        Robust parsing strategy:
        1. Strip all thinking wrappers (<think> tags, "Thinking Process:" headers)
           to work on the full flattened text
        2. Try markdown code fences first (```json...```)
        3. Try ALL JSON objects in the text (scan every '{', pick largest valid one)
        4. Fall back to regex field extraction for truncated output
        """
        import re as _re

        # --- Flatten text: remove all thinking wrappers ---
        text = raw_output.strip()
        text = text.replace("<think>", "").replace("</think>", "")
        # Strip "Thinking Process:" header lines — keep the content
        text = _re.sub(r'^Thinking Process:\s*\n', '', text, flags=_re.MULTILINE)

        def _try_parse(candidate: str) -> Optional[DomainVerifier]:
            """Try to parse a JSON string into a DomainVerifier."""
            try:
                data = json.loads(candidate)
                if "verifier_code" not in data:
                    return None
                return DomainVerifier(
                    domain=data.get("domain", "unknown"),
                    description=data.get("description", ""),
                    verifier_code=data["verifier_code"],
                    proposer_template=data.get("proposer_template", ""),
                    test_examples=data.get("test_examples", []),
                )
            except (json.JSONDecodeError, KeyError):
                return None

        # 1. Markdown code fences (highest confidence)
        if "```json" in text:
            candidate = text.split("```json")[1].split("```")[0].strip()
            result = _try_parse(candidate)
            if result:
                return result
        if "```" in text:
            for block in text.split("```")[1::2]:  # odd-indexed = code blocks
                result = _try_parse(block.strip())
                if result:
                    return result

        # 2. Scan ALL '{' positions — try each as root JSON, keep largest valid
        best: Optional[DomainVerifier] = None
        best_len = 0
        for m in _re.finditer(r'\{', text):
            candidate = text[m.start():]
            # Walk forward to find the matching closing brace
            depth = 0
            end = -1
            in_string = False
            escape = False
            for i, ch in enumerate(candidate):
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_string:
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end == -1:
                continue
            result = _try_parse(candidate[:end])
            if result and end > best_len:
                best = result
                best_len = end

        if best:
            return best

        # 3. Truncated JSON recovery — regex field extraction
        print("  ⚠️  JSON parse failed, attempting field extraction...")
        try:
            domain_m = _re.search(r'"domain"\s*:\s*"([^"]+)"', text)
            desc_m = _re.search(r'"description"\s*:\s*"([^"]+)"', text)
            code_m = _re.search(r'"verifier_code"\s*:\s*"((?:[^"\\]|\\.)*)"', text, _re.DOTALL)
            tmpl_m = _re.search(r'"proposer_template"\s*:\s*"((?:[^"\\]|\\.)*)"', text, _re.DOTALL)

            if not (domain_m and code_m):
                print("  ❌ Could not extract required fields from output")
                return None

            verifier_code = code_m.group(1).replace("\\n", "\n").replace('\\"', '"')
            proposer_template = tmpl_m.group(1).replace("\\n", "\n").replace('\\"', '"') if tmpl_m else ""

            print(f"  ✅ Extracted verifier for '{domain_m.group(1)}' via regex (no test examples)")
            return DomainVerifier(
                domain=domain_m.group(1),
                description=desc_m.group(1) if desc_m else domain_m.group(1),
                verifier_code=verifier_code,
                proposer_template=proposer_template,
                test_examples=[],
            )
        except Exception as e:
            print(f"  ❌ Field extraction failed: {e}")
            return None

    def validate_verifier(
        self, verifier: DomainVerifier, sandbox_config: SandboxConfig = SandboxConfig()
    ) -> bool:
        """Validate a generated verifier against its test examples.
        
        A verifier is valid if:
        1. It correctly accepts all correct answers
        2. It correctly rejects all incorrect answers
        3. It doesn't crash or timeout
        """
        if not verifier.test_examples:
            return False

        correct_count = 0
        total = len(verifier.test_examples) * 2  # Each example tests correct + incorrect

        for example in verifier.test_examples:
            # Test with correct answer
            result_correct = verify_code_task(
                problem_code=example.get("problem_code", ""),
                solution=f'student_answer = {json.dumps(example["correct_answer"])}',
                test_code=example.get("test_code", "assert False"),
                config=sandbox_config,
            )
            if result_correct.correct:
                correct_count += 1

            # Test with incorrect answer
            result_incorrect = verify_code_task(
                problem_code=example.get("problem_code", ""),
                solution=f'student_answer = {json.dumps(example["incorrect_answer"])}',
                test_code=example.get("test_code", "assert False"),
                config=sandbox_config,
            )
            if not result_incorrect.correct:  # Should FAIL for wrong answer
                correct_count += 1

        accuracy = correct_count / max(total, 1)
        verifier.accuracy_on_tests = accuracy
        verifier.validated = accuracy >= 0.8  # Require 80%+ accuracy

        return verifier.validated

    def register_verifier(self, verifier: DomainVerifier):
        """Save and register a validated verifier."""
        if not verifier.validated:
            raise ValueError("Cannot register an unvalidated verifier")

        # Save to disk
        path = self.generated_dir / f"{verifier.domain}.json"
        path.write_text(json.dumps({
            "domain": verifier.domain,
            "description": verifier.description,
            "verifier_code": verifier.verifier_code,
            "proposer_template": verifier.proposer_template,
            "test_examples": verifier.test_examples,
            "validated": verifier.validated,
            "accuracy_on_tests": verifier.accuracy_on_tests,
        }, indent=2))

        # Register in the proposer system
        register_domain(verifier.domain, verifier.proposer_template)
        self.verifiers[verifier.domain] = verifier

    def get_registered_domains(self) -> list[str]:
        """Get all registered (validated) domain names."""
        return list(self.verifiers.keys())
