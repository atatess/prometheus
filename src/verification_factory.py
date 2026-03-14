"""
Verification Factory — the novel contribution.

This module enables the model to autonomously expand the space of
verifiable rewards by writing domain-specific Python verifiers.

The key insight: verification is always easier than generation.
A model that can't solve a problem CAN often write a program that
checks whether a given answer is correct.

Layer 2 Pipeline:
  1. Model generates a verify(problem_data, answer) -> bool function
  2. We exec() it and test against simple correct/incorrect pairs
  3. If accuracy >= 60% → register domain in curriculum
  4. Training loop uses the verifier as a reward signal
"""

import json
import time
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field

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


# ─────────────────────────────────────────────────────────────────────────────
# Prompt: kept intentionally short so small (4B) models don't lose the format
# ─────────────────────────────────────────────────────────────────────────────

FACTORY_PROMPT = """Write a Python verifier for a reasoning domain. Output ONLY valid JSON.

The verifier function signature:
  def verify(problem_data: dict, answer: str) -> bool
  - problem_data ALWAYS has an "expected_answer" key
  - answer is the model's answer (string)
  - Return True if correct, False otherwise
  - Use ONLY stdlib: math, re, json, itertools

COMPLETE EXAMPLE (arithmetic domain):
{{"domain":"arithmetic","description":"Basic arithmetic","verifier_code":"def verify(problem_data, answer):\\n    exp = str(problem_data.get('expected_answer','')).strip()\\n    ans = str(answer).strip()\\n    try:\\n        return abs(float(ans) - float(exp)) < 0.01\\n    except:\\n        return ans.lower() == exp.lower()","proposer_template":"Generate an arithmetic problem.\\nPROBLEM: <problem statement>\\nANSWER: <numeric answer>","test_examples":[{{"correct_answer":"42","incorrect_answer":"41"}},{{"correct_answer":"7","incorrect_answer":"8"}},{{"correct_answer":"100","incorrect_answer":"99"}}]}}

Now write a verifier for:
Domain: {domain}
Description: {description}

Requirements:
- verifier_code must define: def verify(problem_data, answer) -> bool
- test_examples must have "correct_answer" and "incorrect_answer" (at least 3 pairs)
- proposer_template must show how to generate problems for this domain

Output ONLY the JSON object, nothing else:"""

FACTORY_RETRY_PROMPT = """Your verifier for "{domain}" failed: {accuracy:.0%} accuracy on {total} checks.

Failing tests (correct_answer that verify() MUST accept, incorrect_answer it MUST reject):
{failing_cases}

Current verifier_code:
{verifier_code}

The most common bugs:
  - Forgot .strip() → "42 " != "42"
  - Float compare with == → use abs(float(a)-float(b)) < 0.01
  - Wrong key name → always use problem_data.get('expected_answer', '')
  - Crashes on wrong input → add try/except, return False on error

Rewrite the entire JSON with a fixed verifier_code:
{{"domain":"{domain}","description":"{description}","verifier_code":"def verify(problem_data, answer):\\n    ...","proposer_template":"...","test_examples":[...]}}"""

# Token budget — factory JSON needs ~800-1500 tokens of output
FACTORY_MAX_TOKENS = 2000

# Candidate domains for expansion
EXPANSION_CANDIDATES = [
    {
        "domain": "planning",
        "description": "Sequential planning — ordering steps, choosing optimal sequences",
    },
    {
        "domain": "causal_reasoning",
        "description": "Causal reasoning — cause-and-effect chains, counterfactuals",
    },
    {
        "domain": "analogy",
        "description": "Analogical reasoning — A is to B as C is to D patterns",
    },
    {
        "domain": "logic",
        "description": "Propositional and predicate logic, truth tables, syllogisms",
    },
    {
        "domain": "spatial",
        "description": "Spatial reasoning — positions, directions, 2D grid navigation",
    },
    {
        "domain": "probability",
        "description": "Probability — compute likelihoods, expected values, combinatorics",
    },
    {
        "domain": "constraint",
        "description": "Constraint satisfaction — Sudoku-like puzzles, scheduling constraints",
    },
    {
        "domain": "graph",
        "description": "Graph theory — shortest paths, connectivity, coloring",
    },
]


class VerificationFactory:
    """Creates, validates, and registers new domain verifiers."""

    def __init__(self, generated_dir: str = "src/domains/generated"):
        self.generated_dir = Path(generated_dir)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.verifiers: dict[str, DomainVerifier] = {}
        # Cache for compiled verify() functions — avoid re-exec on every call
        self._fn_cache: dict[str, Callable] = {}
        self._load_existing()

    def _load_existing(self):
        """Load previously generated (and validated) verifiers from disk."""
        for f in self.generated_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                verifier = DomainVerifier(**{
                    k: v for k, v in data.items()
                    if k in DomainVerifier.__dataclass_fields__
                })
                if verifier.validated:
                    self.verifiers[verifier.domain] = verifier
                    print(f"  📂 Loaded registered verifier: {verifier.domain}")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  ⚠️  Failed to load {f.name}: {e}")

    def get_verify_fn(self, domain: str) -> Optional[Callable]:
        """Get (and cache) the compiled verify() function for a domain."""
        if domain in self._fn_cache:
            return self._fn_cache[domain]
        
        verifier = self.verifiers.get(domain)
        if not verifier:
            return None
        
        try:
            namespace: dict = {}
            exec(compile(verifier.verifier_code, f"<verifier:{domain}>", "exec"), namespace)
            fn = namespace.get("verify")
            if callable(fn):
                self._fn_cache[domain] = fn
                return fn
        except Exception as e:
            print(f"  ⚠️  Failed to compile verifier for {domain}: {e}")
        return None

    def call_verifier(self, domain: str, problem_data: dict, answer: str) -> bool:
        """Call a registered verifier. Returns False on any error."""
        fn = self.get_verify_fn(domain)
        if fn is None:
            return False
        try:
            return bool(fn(problem_data, str(answer)))
        except Exception:
            return False

    def get_next_candidate(self, existing_domains: list[str]) -> Optional[dict]:
        """Get the next domain candidate not yet in the curriculum."""
        for candidate in EXPANSION_CANDIDATES:
            if candidate["domain"] not in existing_domains:
                return candidate
        return None

    def build_factory_prompt(self, domain: str, description: str) -> str:
        return FACTORY_PROMPT.format(domain=domain, description=description)

    def parse_verifier(self, raw_output: str) -> Optional[DomainVerifier]:
        """Parse model output into a DomainVerifier.

        Strategy:
        1. Strip <think> wrappers
        2. Try ```json fences
        3. Scan all '{' positions for largest valid JSON object
        4. Regex field extraction as last resort
        """
        import re as _re

        text = raw_output.strip()
        # Strip thinking wrappers
        text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()
        text = _re.sub(r'^Thinking Process:\s*\n', '', text, flags=_re.MULTILINE)

        def _try_parse(candidate: str) -> Optional[DomainVerifier]:
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
            except (json.JSONDecodeError, KeyError, TypeError):
                return None

        # 1. Markdown code fences
        for marker in ["```json", "```"]:
            if marker in text:
                parts = text.split(marker)
                for i in range(1, len(parts), 2):
                    result = _try_parse(parts[i].strip())
                    if result:
                        return result

        # 2. Scan all '{' positions — keep largest valid parse
        best: Optional[DomainVerifier] = None
        best_len = 0
        for m in _re.finditer(r'\{', text):
            candidate = text[m.start():]
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

        # 3. Regex field extraction for truncated output
        print("  ⚠️  JSON parse failed, attempting field extraction...")
        try:
            domain_m = _re.search(r'"domain"\s*:\s*"([^"]+)"', text)
            desc_m = _re.search(r'"description"\s*:\s*"([^"]+)"', text)
            code_m = _re.search(r'"verifier_code"\s*:\s*"((?:[^"\\]|\\.)*)"', text, _re.DOTALL)
            tmpl_m = _re.search(r'"proposer_template"\s*:\s*"((?:[^"\\]|\\.)*)"', text, _re.DOTALL)

            if not (domain_m and code_m):
                print("  ❌ Could not extract required fields")
                return None

            verifier_code = code_m.group(1).replace("\\n", "\n").replace('\\"', '"')
            proposer_template = (
                tmpl_m.group(1).replace("\\n", "\n").replace('\\"', '"') if tmpl_m else ""
            )
            print(f"  ✅ Extracted verifier for '{domain_m.group(1)}' via regex")
            return DomainVerifier(
                domain=domain_m.group(1),
                description=desc_m.group(1) if desc_m else "",
                verifier_code=verifier_code,
                proposer_template=proposer_template,
                test_examples=[],
            )
        except Exception as e:
            print(f"  ❌ Field extraction failed: {e}")
            return None

    def validate_verifier(self, verifier: DomainVerifier) -> bool:
        """Validate a generated verifier by directly executing it.

        For each test example, we check:
          - verify({"expected_answer": correct_answer}, correct_answer) → True
          - verify({"expected_answer": correct_answer}, incorrect_answer) → False

        Threshold: 60% of checks must pass.
        """
        if not verifier.test_examples:
            print("  ❌ No test examples — cannot validate")
            verifier.validated = False
            return False

        # Compile the verifier function
        try:
            namespace: dict = {}
            exec(compile(verifier.verifier_code, f"<verifier:{verifier.domain}>", "exec"), namespace)
            verify_fn = namespace.get("verify")
            if not callable(verify_fn):
                print("  ❌ No callable 'verify' function in generated code")
                verifier.validated = False
                return False
        except Exception as e:
            print(f"  ❌ Verifier compile/exec error: {e}")
            verifier.validated = False
            return False

        correct_count = 0
        total = len(verifier.test_examples) * 2
        results_detail = []

        for ex in verifier.test_examples:
            correct_ans = ex.get("correct_answer", "")
            wrong_ans = ex.get("incorrect_answer", "")
            problem_data = {"expected_answer": correct_ans}

            # Check 1: correct answer should be accepted
            accepted = False
            try:
                accepted = bool(verify_fn(problem_data, str(correct_ans)))
                if accepted:
                    correct_count += 1
            except Exception:
                pass

            # Check 2: wrong answer should be rejected
            rejected = False
            try:
                result = verify_fn(problem_data, str(wrong_ans))
                rejected = not bool(result)
                if rejected:
                    correct_count += 1
            except Exception:
                # Exception on wrong answer = implicit rejection
                rejected = True
                correct_count += 1

            results_detail.append({
                "correct_answer": correct_ans,
                "incorrect_answer": wrong_ans,
                "accepted": accepted,
                "rejected": rejected,
            })

        accuracy = correct_count / max(total, 1)
        verifier.accuracy_on_tests = accuracy
        verifier.validated = accuracy >= 0.6

        status = "✅" if verifier.validated else "❌"
        print(f"  {status} Validation: {accuracy:.0%} ({correct_count}/{total} checks passed)")
        for r in results_detail:
            accept_icon = "✓" if r["accepted"] else "✗"
            reject_icon = "✓" if r["rejected"] else "✗"
            print(
                f"     correct='{r['correct_answer']}' [{accept_icon}]  "
                f"wrong='{r['incorrect_answer']}' [{reject_icon}]"
            )

        return verifier.validated

    def get_failing_cases(self, verifier: DomainVerifier) -> list[dict]:
        """Return cases where the verifier gave wrong results."""
        try:
            namespace: dict = {}
            exec(compile(verifier.verifier_code, "<verifier>", "exec"), namespace)
            verify_fn = namespace.get("verify")
        except Exception:
            return verifier.test_examples

        failing = []
        for ex in verifier.test_examples:
            correct_ans = ex.get("correct_answer", "")
            wrong_ans = ex.get("incorrect_answer", "")
            problem_data = {"expected_answer": correct_ans}

            try:
                accepted = bool(verify_fn(problem_data, str(correct_ans)))
            except Exception:
                accepted = False

            try:
                rejected = not bool(verify_fn(problem_data, str(wrong_ans)))
            except Exception:
                rejected = True

            if not accepted or not rejected:
                failing.append({
                    **ex,
                    "correct_passed": accepted,
                    "incorrect_rejected": rejected,
                })
        return failing

    def build_retry_prompt(self, verifier: DomainVerifier, failing_cases: list[dict]) -> str:
        cases_text = "\n".join(
            f"  correct='{c.get('correct_answer','')}'  wrong='{c.get('incorrect_answer','')}'"
            f"  → accepted={c.get('correct_passed',False)}, rejected={c.get('incorrect_rejected',False)}"
            for c in failing_cases
        )
        total = len(verifier.test_examples) * 2
        return FACTORY_RETRY_PROMPT.format(
            domain=verifier.domain,
            description=verifier.description,
            accuracy=verifier.accuracy_on_tests,
            total=total,
            failing_cases=cases_text,
            verifier_code=verifier.verifier_code,
        )

    def register_verifier(self, verifier: DomainVerifier):
        """Save a validated verifier to disk and register it in the curriculum."""
        if not verifier.validated:
            raise ValueError("Cannot register an unvalidated verifier")

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

        register_domain(verifier.domain, verifier.proposer_template)
        self.verifiers[verifier.domain] = verifier
        # Invalidate cache so next call recompiles fresh
        self._fn_cache.pop(verifier.domain, None)
        print(f"  💾 Registered verifier: {verifier.domain} (saved to {path})")

    def get_registered_domains(self) -> list[str]:
        return list(self.verifiers.keys())

    def run_factory_with_model(
        self,
        domain: str,
        description: str,
        generate_fn,          # Callable[[str], str]  — takes prompt, returns raw text
        max_retries: int = 2,
    ) -> Optional[DomainVerifier]:
        """End-to-end: prompt → parse → validate → (retry) → return verifier or None.

        generate_fn should accept a prompt string and return the model's raw output.
        """
        print(f"\n{'─'*50}")
        print(f"🏭 Factory: generating verifier for '{domain}'")
        print(f"   {description}")

        prompt = self.build_factory_prompt(domain, description)
        print(f"   Prompt: {len(prompt)} chars | max_retries={max_retries}")

        verifier = None
        for attempt in range(max_retries + 1):
            if attempt == 0:
                current_prompt = prompt
                print(f"\n   ⟳ Attempt 1: generating...")
            else:
                if verifier is None:
                    print(f"   ⟳ Attempt {attempt+1}: previous parse failed, retrying from scratch")
                    current_prompt = prompt
                else:
                    failing = self.get_failing_cases(verifier)
                    current_prompt = self.build_retry_prompt(verifier, failing)
                    print(f"   ⟳ Attempt {attempt+1}: retry with {len(failing)} failing cases")

            t0 = time.monotonic()
            raw_output = generate_fn(current_prompt)
            elapsed = time.monotonic() - t0
            print(f"   Generated {len(raw_output)} chars in {elapsed:.1f}s")

            verifier = self.parse_verifier(raw_output)
            if verifier is None:
                print("   ❌ Parse failed")
                continue

            verifier.domain = domain  # Ensure domain matches what was requested
            verifier.description = description
            print(f"   Parsed: {len(verifier.test_examples)} test examples")

            if self.validate_verifier(verifier):
                print(f"   🎉 Verifier validated! accuracy={verifier.accuracy_on_tests:.0%}")
                return verifier
            else:
                print(f"   ↩️  Validation failed ({verifier.accuracy_on_tests:.0%}) — retrying")

        print(f"   ❌ All {max_retries+1} attempts failed for '{domain}'")
        return None
