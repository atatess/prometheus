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
        """Parse the model's output into a DomainVerifier."""
        try:
            text = raw_output.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text)
            return DomainVerifier(
                domain=data["domain"],
                description=data["description"],
                verifier_code=data["verifier_code"],
                proposer_template=data["proposer_template"],
                test_examples=data.get("test_examples", []),
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse verifier output: {e}")
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
