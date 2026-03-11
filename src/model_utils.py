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
    """Strip thinking blocks from model output.
    
    Handles:
    - <think>...</think> (standard Qwen3.5 tags)
    - "Thinking Process:..." (text-based thinking)
    - Finding JSON, ANSWER:, or numeric answers in thinking output
    """
    import re
    
    # Format 1: <think>...</think> — clean extraction
    if "<think>" in text and "</think>" in text:
        after = text.split("</think>")[-1].strip()
        if after:
            return after
    
    # Format 2: Look for JSON in the text (for proposer outputs)
    # Find JSON objects that look like problem definitions
    json_matches = re.findall(r'\{[^{}]*"prompt"[^{}]*\}', text)
    if json_matches:
        return json_matches[-1]
    # Also try generic JSON
    json_matches = re.findall(r'\{[^{}]*"[^"]+"\s*:[^{}]*\}', text)
    if json_matches:
        return json_matches[-1]
    
    # Format 3: Find ANSWER: pattern — but only when it looks like a real answer
    # (a number or short value, not a quoted description)
    answer_matches = re.findall(r'ANSWER:\s*(\d+(?:\.\d+)?(?:/\d+)?)', text)
    if answer_matches:
        return answer_matches[-1]  # Last match is usually the final answer
    
    # Format 4: "The answer is X" — extract the value
    answer_matches = re.findall(r'(?:answer|result) (?:is|=)\s*[:\s]*(\d+(?:\.\d+)?(?:/\d+)?)', text, re.IGNORECASE)
    if answer_matches:
        return answer_matches[-1]
    
    # Format 5: "$X$" boxed math — last standalone number
    boxed = re.findall(r'\$(\d+(?:\.\d+)?)\$', text)
    if boxed:
        return boxed[-1]
    
    # Format 6: Last line that's just a number
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip().rstrip('.')
        if re.match(r'^-?\d+(?:\.\d+)?(?:/\d+)?$', line):
            return line
    
    # Fallback: return the original text
    return text.strip()
