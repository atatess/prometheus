"""
Self-Play Curriculum Manager.

Implements the frontier-based curriculum from Agent0: problems are selected
at the edge of the model's ability (self-consistency band), ensuring the
model always trains on optimally difficult problems.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import random
from pathlib import Path

from .proposer import Problem


@dataclass
class CurriculumConfig:
    initial_domains: list[str] = field(default_factory=lambda: ["math", "code"])
    frontier_band: tuple[float, float] = (0.3, 0.8)
    problems_per_round: int = 32
    max_domain_saturation: float = 0.85


@dataclass
class DomainStats:
    """Track performance statistics per domain."""
    domain: str
    total_attempts: int = 0
    correct: int = 0
    problems_generated: int = 0
    
    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total_attempts, 1)
    
    @property
    def is_saturated(self) -> bool:
        """Domain is saturated when accuracy is consistently high."""
        return self.accuracy > 0.85 and self.total_attempts > 50


class Curriculum:
    """Manages the self-play curriculum across domains."""

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.domains = list(config.initial_domains)
        self.stats: dict[str, DomainStats] = {
            d: DomainStats(domain=d) for d in self.domains
        }
        self.problem_pool: list[Problem] = []
        self.history: list[dict] = []

    def select_domain(self) -> str:
        """Select the next domain to generate problems for.
        
        Prioritizes domains where the model is in the frontier band
        (not too easy, not too hard). Avoids saturated domains.
        """
        # Filter out saturated domains
        active_domains = [
            d for d in self.domains
            if not self.stats.get(d, DomainStats(domain=d)).is_saturated
        ]
        
        if not active_domains:
            # All domains saturated — signal for expansion
            return "__needs_expansion__"
        
        # Weight by inverse accuracy (focus on harder domains)
        weights = []
        for d in active_domains:
            acc = self.stats.get(d, DomainStats(domain=d)).accuracy
            # Frontier weighting: prefer domains near the frontier band
            if self.config.frontier_band[0] <= acc <= self.config.frontier_band[1]:
                weights.append(3.0)  # Strong preference for frontier
            elif acc < self.config.frontier_band[0]:
                weights.append(1.0)  # Too hard, still attempt
            else:
                weights.append(0.5)  # Too easy, deprioritize
        
        return random.choices(active_domains, weights=weights, k=1)[0]

    def select_difficulty(self, domain: str) -> str:
        """Select problem difficulty based on current domain performance."""
        acc = self.stats.get(domain, DomainStats(domain=domain)).accuracy
        
        if acc < 0.3:
            return random.choice(["easy", "easy", "medium"])
        elif acc < 0.6:
            return random.choice(["easy", "medium", "hard"])
        elif acc < 0.8:
            return random.choice(["medium", "hard", "hard"])
        else:
            return "hard"

    def record_attempt(self, domain: str, correct: bool):
        """Record a solution attempt."""
        if domain not in self.stats:
            self.stats[domain] = DomainStats(domain=domain)
        
        self.stats[domain].total_attempts += 1
        if correct:
            self.stats[domain].correct += 1
        
        self.history.append({
            "domain": domain,
            "correct": correct,
        })

    def add_domain(self, domain: str):
        """Add a new domain to the curriculum."""
        if domain not in self.domains:
            self.domains.append(domain)
            self.stats[domain] = DomainStats(domain=domain)

    def get_frontier_ratio(self) -> float:
        """What fraction of domains are in the frontier band."""
        if not self.domains:
            return 0.0
        in_frontier = sum(
            1 for d in self.domains
            if self.config.frontier_band[0]
            <= self.stats.get(d, DomainStats(domain=d)).accuracy
            <= self.config.frontier_band[1]
        )
        return in_frontier / len(self.domains)

    def get_status(self) -> dict:
        """Get curriculum status for logging."""
        return {
            "domains": self.domains,
            "stats": {
                d: {
                    "accuracy": s.accuracy,
                    "attempts": s.total_attempts,
                    "saturated": s.is_saturated,
                }
                for d, s in self.stats.items()
            },
            "frontier_ratio": self.get_frontier_ratio(),
            "needs_expansion": all(
                s.is_saturated for s in self.stats.values()
            ),
        }

    def save(self, path: str):
        """Save curriculum state."""
        state = {
            "domains": self.domains,
            "stats": {
                d: {
                    "total_attempts": s.total_attempts,
                    "correct": s.correct,
                    "problems_generated": s.problems_generated,
                }
                for d, s in self.stats.items()
            },
        }
        Path(path).write_text(json.dumps(state, indent=2))

    def load(self, path: str):
        """Load curriculum state."""
        state = json.loads(Path(path).read_text())
        self.domains = state["domains"]
        for d, s in state["stats"].items():
            self.stats[d] = DomainStats(
                domain=d,
                total_attempts=s["total_attempts"],
                correct=s["correct"],
                problems_generated=s["problems_generated"],
            )
