# 🔥 Prometheus

**Self-evolving LLM training through autonomous verification expansion.**

Prometheus is a recursive self-improvement system that trains language models to get better at reasoning by manufacturing its own verification. Unlike existing approaches that are limited to math and code (where answers can be checked automatically), Prometheus autonomously expands the space of verifiable tasks by writing its own verifiers as Python programs.

> *"One day, frontier AI research used to require million-dollar GPU clusters. That era is ending."*

## The Core Idea

Existing self-play training systems (Absolute Zero, Agent0, R-Zero) only work in domains where you can automatically check answers — math equations have definitive solutions, code has test suites. This limits self-improvement to a narrow slice of intelligence.

**Prometheus breaks this ceiling.** It teaches the model to write Python programs that verify answers in *any* domain — logic, spatial reasoning, planning, data analysis, science. Every new verifier unlocks an entire domain for reinforcement learning.

```
┌─────────────────────────────────────────────┐
│  LAYER 3: Meta-Loop (outer agent)           │
│  • Monitors training progress               │
│  • Evolves training script (autoresearch)   │
│  • Searches for new problem domains         │
│  • Decides when to expand frontier          │
├─────────────────────────────────────────────┤
│  LAYER 2: Verification Factory              │
│  • Model writes Python verifiers for new    │
│    domains                                  │
│  • Tests verifiers against known examples   │
│  • Adds verified domains to curriculum      │
│  • ← THE NOVEL CONTRIBUTION                 │
├─────────────────────────────────────────────┤
│  LAYER 1: Self-Play Training (GRPO)         │
│  • Proposer generates Code-as-Task problems │
│  • Solver attempts solutions                │
│  • Verifier checks via code execution       │
│  • GRPO updates model weights via LoRA      │
│  • Runs on Apple Silicon OR NVIDIA GPU      │
└─────────────────────────────────────────────┘
```

## What Makes This Different

| Approach | Self-Play | Verifiable Domains | Expands Verification |
|----------|-----------|-------------------|---------------------|
| DeepSeek R1 | ❌ Fixed data | Math, Code | ❌ |
| Absolute Zero | ✅ | Math, Code | ❌ |
| Agent0 | ✅ | Math + Tool Use | ❌ |
| **Prometheus** | ✅ | **Any domain** | ✅ **Autonomous** |

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
    for rel in claimed:
        if not check_relation(objects, rel):
            return False
    return True
```

### Phase 3: Domain Expansion
The system actively seeks new problem domains, converts them to Code-as-Task format, and adds them to the training curriculum — preventing the information saturation that causes existing self-play systems to plateau.

## Hardware Support

Prometheus runs on two backends:

| Backend | Hardware | Framework | Config |
|---------|----------|-----------|--------|
| **MLX** | Apple Silicon (M1/M2/M3/M4) | mlx + mlx_lm | `configs/default.toml` |
| **CUDA** | NVIDIA GPU (tested: L40S 48GB) | PyTorch + PEFT | `configs/cuda_default.toml` |

Auto-detected at runtime. Override with `PROMETHEUS_BACKEND=cuda` or `PROMETHEUS_BACKEND=mlx`.

## Requirements

**Apple Silicon (MLX backend):**
- macOS with Apple Silicon (tested: M2 Max 32GB)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

**NVIDIA GPU (CUDA backend):**
- CUDA 12.x+, 20GB+ VRAM recommended
- Python 3.11+
- `pip install torch transformers peft accelerate trl`

**Model:** `Qwen/Qwen3.5-4B` (instruct variant, 8GB in bf16)

## Quick Start

### Apple Silicon (MLX)

```bash
git clone https://github.com/atatess/prometheus.git
cd prometheus

# Install dependencies
uv sync

# Download model
uv run python scripts/download_model.py

# Run training
uv run python train.py --config configs/default.toml
```

### NVIDIA GPU (CUDA)

```bash
git clone https://github.com/atatess/prometheus.git
cd prometheus

# Install CUDA deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers peft accelerate trl datasets bitsandbytes huggingface_hub tomli

# Download model
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-4B', local_dir='./models/qwen3.5-4b')
"

# Update configs/cuda_default.toml → model.name = "./models/qwen3.5-4b"

# Run training
PROMETHEUS_BACKEND=cuda python train.py --config configs/cuda_default.toml
```

## Project Structure

```
prometheus/
├── train.py                    # GRPO training loop (modifiable by meta-loop)
├── evaluate.py                 # Benchmark evaluation suite
├── src/
│   ├── proposer.py             # Problem generation (Code-as-Task)
│   ├── solver.py               # Solution generation
│   ├── verifier.py             # Verification sandbox
│   ├── verification_factory.py # Autonomous verifier creation ← Layer 2
│   ├── curriculum.py           # Curriculum management
│   ├── seed_problems.py        # Bootstrap problem set
│   │
│   ├── grpo.py                 # GRPO trainer — MLX backend
│   ├── model_utils.py          # Model utilities — MLX backend
│   │
│   ├── grpo_cuda.py            # GRPO trainer — CUDA/PyTorch backend
│   ├── model_utils_cuda.py     # Model utilities — CUDA backend
│   ├── load_model_cuda.py      # Model loader — CUDA backend
│   │
│   └── domains/                # Domain-specific verifiers
│       ├── math.py
│       ├── code.py
│       └── generated/          # Auto-generated verifiers (Layer 2 output)
├── configs/
│   ├── default.toml            # MLX default (Apple Silicon)
│   ├── cuda_default.toml       # CUDA default (NVIDIA GPU)
│   └── smoke_test.toml         # Quick iteration config
├── scripts/
│   └── download_model.py
└── experiments/                # Experiment logs (gitignored)
```

## Configuration

### MLX (Apple Silicon)
```toml
# configs/default.toml
[model]
name = "mlx-community/Qwen3.5-4B-MLX-4bit"
quantize = "4bit"

[training]
group_size = 4
learning_rate = 5e-6
max_new_tokens = 512
time_budget_minutes = 5
```

### CUDA (NVIDIA GPU)
```toml
# configs/cuda_default.toml
[model]
name = "/path/to/models/qwen3.5-4b"   # local path or HF repo

[training]
group_size = 6          # More rollouts thanks to GPU throughput
learning_rate = 5e-6
max_new_tokens = 1024
time_budget_minutes = 10
grad_accumulation_steps = 2
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

**Phase 1 — Self-Play Training Loop:** ✅ Working on both backends
- MLX: Qwen3.5-4B-MLX-4bit on Apple Silicon
- CUDA: Qwen3.5-4B bf16 on NVIDIA L40S (48GB), PyTorch + PEFT LoRA

**Phase 2 — Verification Factory:** 🔄 In progress
- Verifier generation pipeline scaffolded
- Domain expansion trigger logic in curriculum

**Phase 3 — Meta-Loop:** ⏳ Planned
- Outer agent monitors loss curves
- Rewrites train.py to evolve training strategy

## License

MIT
