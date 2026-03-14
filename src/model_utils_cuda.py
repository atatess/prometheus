"""
Model utilities for Prometheus — CUDA / PyTorch backend.

Drop-in replacement for model_utils.py.
Same public API: chat_generate(), strip_thinking()
"""

import re
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def strip_thinking(text: str) -> str:
    """Strip thinking blocks from model output (pure Python, no framework deps)."""
    # Format 1: <think>...</think> (standard Qwen3.5 instruct tags)
    if "<think>" in text and "</think>" in text:
        after = text.split("</think>")[-1].strip()
        if after:
            return after

    # Format 1b: <think> present but </think> missing (cut off by max_new_tokens)
    # Extract the partial thinking and look for an answer in it
    if text.strip().startswith("<think>"):
        text = text.replace("<think>", "").strip()

    # Format 2: "Thinking Process:\n..." — Qwen3.5 base-style thinking
    # Find the actual answer after the thinking block
    if text.strip().startswith("Thinking Process:") or "Thinking Process:" in text[:50]:
        import re
        # Look for "Answer:" or a final standalone number/word after the thinking
        answer_match = re.search(r'(?:^|\n)(?:Answer|Result|Therefore|So)[:\s]+(.+?)(?:\n|$)', text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        # Last non-empty line that looks like an answer
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        if lines:
            return lines[-1]
    json_matches = re.findall(r'\{[^{}]*"prompt"[^{}]*\}', text)
    if json_matches:
        return json_matches[-1]
    json_matches = re.findall(r'\{[^{}]*"[^"]+"\s*:[^{}]*\}', text)
    if json_matches:
        return json_matches[-1]
    NUM = r'-?\d+(?:\.\d+)?(?:/\d+)?'
    answer_matches = re.findall(rf'ANSWER:\s*({NUM})', text)
    if answer_matches:
        return answer_matches[-1]
    answer_matches = re.findall(rf'(?:answer|result)\s+(?:is|=)\s*:?\s*({NUM})', text, re.IGNORECASE)
    if answer_matches:
        return answer_matches[-1]
    fractions = re.findall(r'(?:^|\s)(\d+/\d+)(?:\s|$|\.)', text)
    if fractions:
        return fractions[-1]
    boxed2 = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed2:
        return boxed2[-1].strip()
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip().rstrip('.')
        if re.match(rf'^{NUM}$', line):
            return line
    return text.strip()


def chat_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    user_msg: str,
    max_tokens: int = 1500,
    temp: float = 0.7,
) -> str:
    """Generate a response using the chat template (CUDA / HuggingFace version).

    Behaviour is identical to the MLX version in model_utils.py:
    1. Apply chat template to format the user message.
    2. Greedy / temperature-sampled generation via model.generate().
    3. Strip <think>...</think> blocks before returning.

    Args:
        model:      A loaded HuggingFace CausalLM (should be on CUDA).
        tokenizer:  Matching tokenizer.
        user_msg:   Raw user-facing message string.
        max_tokens: Maximum *new* tokens to generate.
        temp:       Sampling temperature (0 = greedy).

    Returns:
        Model response string with thinking tags removed.
    """
    messages = [{"role": "user", "content": user_msg}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenise the prompt; keep tensors on the model's device.
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=(temp > 0),
            temperature=temp if temp > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt prefix).
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, prompt_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return strip_thinking(raw)
