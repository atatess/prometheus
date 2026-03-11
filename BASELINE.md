# Baseline — Qwen3.5-4B-MLX-4bit

Official benchmark scores from [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) model card.
These are our starting point — any improvement from Prometheus training is measured against these.

## Language & Reasoning

| Benchmark | Score | Notes |
|-----------|-------|-------|
| MMLU-Pro | 79.1 | Knowledge & STEM |
| MMLU-Redux | 88.8 | Knowledge |
| GPQA Diamond | 76.2 | Graduate-level QA |
| SuperGPQA | 52.9 | Super hard QA |
| HMMT Feb 25 | 74.0 | Math competition |
| HMMT Nov 25 | 76.8 | Math competition |
| LiveCodeBench v6 | 55.8 | Code generation |
| OJBench | 24.1 | Online judge |
| IFEval | 89.8 | Instruction following |
| IFBench | 59.2 | Instruction following |
| C-Eval | 85.1 | Chinese knowledge |

## Targets for Prometheus

Classic benchmarks (likely near-ceiling already but good for measuring):
- GSM8K: Expected ~90%+ (not listed, likely saturated for this model)
- MATH-500: Expected ~80%+ (similar tier to HMMT scores)
- HumanEval: Expected ~75%+ (interpolating from LiveCodeBench)

Hard benchmarks (room for improvement):
- GPQA Diamond: 76.2 → target 80+
- SuperGPQA: 52.9 → target 57+
- LiveCodeBench: 55.8 → target 60+
- HMMT: 74-77 → target 80+

## Quantization Note

We use the 4-bit quantized MLX version (`mlx-community/Qwen3.5-4B-MLX-4bit`).
Quantization typically costs 1-3% on benchmarks vs full precision.
Our actual baseline may be slightly below these official scores.

## Running Our Own Baseline

```bash
# TODO: Implement benchmark runners
uv run python evaluate.py --model mlx-community/Qwen3.5-4B-MLX-4bit
```
