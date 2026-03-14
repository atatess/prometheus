"""
CUDA model loader for Prometheus.

Simple helper that loads a HuggingFace causal LM onto CUDA with bfloat16.
Equivalent to mlx_lm.load() but for CUDA/PyTorch.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_cuda(model_name: str, device: str = "cuda"):
    """Load a causal LM and tokenizer onto the given device.

    Args:
        model_name: HuggingFace model name or local path (e.g. "/root/models/qwen3.5-4b")
        device: Target device string — typically "cuda" or "cuda:0"

    Returns:
        (model, tokenizer) tuple, with model in eval mode on `device`
    """
    print(f"  Loading tokenizer from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Many models (Qwen, LLaMA) don't set a pad token — use eos so
    # batch tokenisation doesn't crash.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  Loading model weights (bfloat16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,   # bfloat16 — good precision / memory trade-off
        device_map=device,             # put all layers on the requested GPU
        trust_remote_code=True,
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Model loaded: {total_params:.2f}B parameters on {device}")

    return model, tokenizer
