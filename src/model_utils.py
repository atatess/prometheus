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
    
    # Number pattern: integers, decimals, fractions, negative numbers
    NUM = r'-?\d+(?:\.\d+)?(?:/\d+)?'
    
    # Format 3: Find ANSWER: pattern
    answer_matches = re.findall(rf'ANSWER:\s*({NUM})', text)
    if answer_matches:
        return answer_matches[-1]
    
    # Format 4: "The answer is X"
    answer_matches = re.findall(rf'(?:answer|result)\s+(?:is|=)\s*:?\s*({NUM})', text, re.IGNORECASE)
    if answer_matches:
        return answer_matches[-1]
    
    # Format 5: Fractions like "1/6" or "5/14" — find standalone fractions
    fractions = re.findall(r'(?:^|\s)(\d+/\d+)(?:\s|$|\.)', text)
    if fractions:
        return fractions[-1]
    
    # Format 6: "$X$" boxed math
    boxed = re.findall(r'\$({NUM})\$'.replace('NUM', r'-?\d+(?:\.\d+)?(?:/\d+)?'), text)
    if boxed:
        return boxed[-1]
    
    # Format 7: \\boxed{X}
    boxed2 = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed2:
        return boxed2[-1].strip()
    
    # Format 8: Last line that's just a number/fraction
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip().rstrip('.')
        if re.match(rf'^{NUM}$', line):
            return line
    
    # Fallback: return the original text
    return text.strip()
