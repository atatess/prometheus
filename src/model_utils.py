"""
Model utilities for Prometheus.

Handles chat template formatting and thinking-model quirks (Qwen3.5).
"""

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


def chat_generate(
    model,
    tokenizer,
    user_msg: str,
    max_tokens: int = 1500,
    temp: float = 0.7,
) -> str:
    """Generate a response using the chat template.
    
    Handles Qwen3.5 thinking models by:
    1. Generating with enough tokens for thinking + answer
    2. Stripping <think>...</think> tags from the output
    """
    sampler = make_sampler(temp=temp)
    messages = [{"role": "user", "content": user_msg}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    raw = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )
    
    return strip_thinking(raw)


def strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks from model output."""
    if "<think>" in text and "</think>" in text:
        # Take everything after the last </think>
        return text.split("</think>")[-1].strip()
    elif "<think>" in text:
        # Thinking never closed — try to find JSON or answer in the text
        # Look for JSON object
        import re
        # Find the last JSON-like object in the text
        json_match = re.findall(r'\{[^{}]*\}', text)
        if json_match:
            return json_match[-1]
        # Look for a number at the end
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('*') and not line.startswith('#'):
                return line
    return text.strip()
