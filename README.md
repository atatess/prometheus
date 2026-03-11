# 🔥 Prometheus

**Self-evolving LLM training through autonomous verification expansion — on consumer hardware.**

Prometheus is a recursive self-improvement system that trains language models to get better at reasoning by manufacturing its own verification. Unlike existing approaches that are limited to math and code (where answers can be checked automatically), Prometheus autonomously expands the space of verifiable tasks by writing its own verifiers as Python programs.

> *"One day, frontier AI research used to require million-dollar GPU clusters. That era is ending."*

## The Core Idea

Existing self-play training systems (Absolute Zero, Agent0, R-Zero) only work in domains where you can automatically check answers — math equations have definitive solutions, code has test suites. This limits self-improvement to a narrow slice of intelligence.

**Prometheus breaks this ceiling.** It teaches the model to write Python programs that verify answers in *any* domain — logic, spatial reasoning, planning, data analysis, science. Every new verifier unlocks an entire domain for reinforcement learning.

```
┌─────────────────────────────────────────────┐
│  LAYER 3: Meta-Loop (outer agent)           │
│  • Monitors training progress               │
│  • Evolves training script (autoresearch)    │
│  • Searches for new problem domains          │
│  • Decides when to expand frontier           │
├─────────────────────────────────────────────┤
│  LAYER 2: Verification Factory              │
│  • Model writes Python verifiers for new     │
│    domains                                   │
│  • Tests verifiers against known examples    │
│  • Adds verified domains to curriculum       │
│  • THE NOVEL CONTRIBUTION                    │
├─────────────────────────────────────────────┤
│  LAYER 1: Self-Play Training (MLX-GRPO)     │
│  • Proposer generates Code-as-Task problems  │
│  • Solver attempts solutions with tool use   │
│  • Verifier checks via code execution        │
│  • GRPO updates model weights via LoRA       │
│  • Runs natively on Apple Silicon            │
└─────────────────────────────────────────────┘
```

## What Makes This Different

| Approach | Self-Play | Verifiable Domains | Consumer HW | Expands Verification |
|----------|-----------|-------------------|-------------|---------------------|
| DeepSeek R1 | ❌ Fixed data | Math, Code | ❌ GPU cluster | ❌ |
| Absolute Zero | ✅ | Math, Code | ❌ GPU cluster | ❌ |
| Agent0 | ✅ | Math + Tool Use | ❌ GPU cluster | ❌ |
| **Prometheus** | ✅ | **Any domain** | ✅ **Apple Silicon** | ✅ **Autonomous** |

## How It Works

### Phase 1: Self-Play Training
The model plays two roles — **Proposer** (generates problems) and **Solver** (attempts solutions). Problems are expressed as Code-as-Task: executable Python with built-in test assertions. The model trains via GRPO on correct/incorrect signals.

### Phase 2: Verification Factory
When the model plateaus on known domains, it encounters new task types and writes Python verifiers for them:

```python
# The model generates a verifier for spatial reasoning:
def verify_spatial(problem, answer):
    """Check if spatial reasoning answer is correct."""
    objects = parse_positions(problem)
    claimed = parse_answer(answer)
    # Simulate spatial relationships
    for rel in claimed:
        if not check_relation(objects, rel):
            return False
    return True
```

### Phase 3: Domain Expansion
The system actively seeks new problem domains, converts them to Code-as-Task format, and adds them to the training curriculum — preventing the information saturation that causes existing self-play systems to plateau.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4, tested on M2 Max 32GB)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
# Clone
git clone https://github.com/atatess/prometheus.git
cd prometheus

# Install dependencies
uv sync

# Download base model (Qwen3.5-4B)
uv run python scripts/download_model.py

# Run baseline evaluation
uv run python evaluate.py

# Start self-play training loop
uv run python train.py

# Start the full Prometheus loop (self-play + verification expansion)
uv run python prometheus.py
```

## Project Structure

```
prometheus/
├── prometheus.py          # Main orchestration loop
├── train.py               # GRPO training (modified by the meta-loop)
├── evaluate.py            # Benchmark evaluation suite
├── program.md             # Instructions for the meta-agent
├── src/
│   ├── proposer.py        # Problem generation (Code-as-Task)
│   ├── solver.py          # Solution generation with tool use
│   ├── verifier.py        # Verification execution sandbox
│   ├── verification_factory.py  # Autonomous verifier creation
│   ├── curriculum.py      # Self-play curriculum management
│   ├── grpo.py            # GRPO implementation for MLX
│   └── domains/           # Domain-specific verifiers
│       ├── math.py
│       ├── code.py
│       ├── logic.py
│       └── generated/     # Auto-generated verifiers
├── benchmarks/
│   ├── math_eval.py       # MATH-500, GSM8K
│   ├── code_eval.py       # HumanEval
│   ├── reasoning_eval.py  # BBH, MMLU subsets
│   └── runner.py          # Unified benchmark runner
├── scripts/
│   ├── download_model.py
│   └── convert_model.py
├── configs/
│   ├── default.toml       # Default training config
│   ├── smoke_test.toml    # Quick iteration config
│   └── overnight.toml     # Full overnight run
└── experiments/           # Experiment logs (gitignored)
```

## Configuration

```toml
# configs/default.toml
[model]
name = "Qwen/Qwen3.5-4B"
quantize = "4bit"

[training]
method = "grpo"
group_size = 4
learning_rate = 5e-6
lora_rank = 16
max_new_tokens = 512
time_budget_minutes = 5

[curriculum]
initial_domains = ["math", "code"]
enable_verification_factory = true
frontier_band = [0.3, 0.8]  # Self-consistency range for optimal difficulty

[meta]
enable_domain_expansion = true
enable_autoresearch = false  # Enable for outer loop
benchmark_interval = 10  # Evaluate every N experiments
```

## Research Context

This project builds on and combines insights from:

- **[Absolute Zero](https://arxiv.org/abs/2505.03335)** — Reinforced self-play reasoning with zero data
- **[Agent0](https://arxiv.org/abs/2511.16043)** — Self-evolving LLMs through co-evolution
- **[Self-Play Only Evolves When...](https://arxiv.org/abs/2603.02218)** (ICML 2026) — Triadic framework for sustained self-evolution
- **[autoresearch](https://github.com/karpathy/autoresearch)** — Karpathy's autonomous ML research loop
- **[MLX-GRPO](https://github.com/Doriandarko/MLX-GRPO)** — GRPO training on Apple Silicon
- **[Code-as-Task](https://arxiv.org/abs/2508.01223)** (NeurIPS 2025) — Self-challenging agents

The novel contribution: **autonomous verification expansion** — a system that manufactures its own reward signals by writing domain-specific verifiers, enabling self-improvement beyond math and code.

## Status

🚧 **Active Development** — Phase 1 (self-play training on MLX) in progress.

## License

MIT
