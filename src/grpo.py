"""
GRPO (Group Relative Policy Optimization) implementation for MLX.

Based on DeepSeek-R1's approach: generate multiple rollouts per prompt,
compute relative advantages within each group, update toward higher-reward responses.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.lora import LoRALinear
from dataclasses import dataclass
from typing import Optional
import math
import json
import time


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    group_size: int = 4
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_new_tokens: int = 512
    temperature: float = 0.8
    kl_coeff: float = 0.04
    clip_range: float = 0.2
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: list = None
    grad_accumulation_steps: int = 4
    max_seq_len: int = 2048

    def __post_init__(self):
        if self.lora_targets is None:
            self.lora_targets = ["q_proj", "v_proj", "k_proj", "o_proj"]


def apply_lora(model, config: GRPOConfig):
    """Apply LoRA adapters to the model."""
    lora_params = {}
    for name, module in model.named_modules():
        if any(target in name for target in config.lora_targets):
            if isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_linear(
                    module,
                    r=config.lora_rank,
                    alpha=config.lora_alpha,
                )
                # Set the LoRA layer in the model
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], lora_layer)
                lora_params[name] = lora_layer
    return lora_params


def compute_grpo_loss(
    model,
    tokenizer,
    prompts: list[str],
    rewards: list[list[float]],
    responses: list[list[str]],
    config: GRPOConfig,
) -> mx.array:
    """
    Compute GRPO loss for a batch of prompts.
    
    For each prompt, we have `group_size` responses with corresponding rewards.
    The advantage is computed relative to the group mean (no critic needed).
    """
    total_loss = mx.array(0.0)
    n_samples = 0

    for prompt, group_rewards, group_responses in zip(prompts, rewards, responses):
        # Compute group-relative advantages
        mean_reward = sum(group_rewards) / len(group_rewards)
        std_reward = max(
            math.sqrt(sum((r - mean_reward) ** 2 for r in group_rewards) / len(group_rewards)),
            1e-8,
        )
        advantages = [(r - mean_reward) / std_reward for r in group_rewards]

        for response, advantage in zip(group_responses, advantages):
            if advantage <= 0:
                continue  # Only train on above-average responses

            # Tokenize the full sequence
            full_text = prompt + response
            tokens = tokenizer.encode(full_text)
            if len(tokens) > config.max_seq_len:
                tokens = tokens[: config.max_seq_len]

            input_ids = mx.array([tokens[:-1]])
            target_ids = mx.array([tokens[1:]])

            # Forward pass
            logits = model(input_ids)
            
            # Cross-entropy loss on response tokens only
            prompt_len = len(tokenizer.encode(prompt))
            log_probs = nn.losses.cross_entropy(
                logits[:, prompt_len - 1 :, :].reshape(-1, logits.shape[-1]),
                target_ids[:, prompt_len - 1 :].reshape(-1),
                reduction="mean",
            )

            # Weight by advantage
            total_loss = total_loss + log_probs * mx.array(advantage)
            n_samples += 1

    if n_samples > 0:
        total_loss = total_loss / n_samples

    return total_loss


def generate_rollouts(
    model,
    tokenizer,
    prompt: str,
    config: GRPOConfig,
) -> list[str]:
    """Generate multiple rollout responses for a prompt."""
    sampler = make_sampler(temp=config.temperature)
    responses = []
    for _ in range(config.group_size):
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=config.max_new_tokens,
            sampler=sampler,
        )
        responses.append(response)
    return responses


class GRPOTrainer:
    """GRPO trainer for MLX models with LoRA."""

    def __init__(self, model, tokenizer, config: GRPOConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Apply LoRA
        self.lora_params = apply_lora(model, config)
        
        # Optimizer (only LoRA params)
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def train_step(
        self,
        prompts: list[str],
        rewards: list[list[float]],
        responses: list[list[str]],
    ) -> float:
        """Execute one GRPO training step."""
        loss_and_grad_fn = nn.value_and_grad(
            self.model,
            lambda m, p, rew, resp: compute_grpo_loss(
                m, self.tokenizer, p, rew, resp, self.config
            ),
        )

        loss, grads = loss_and_grad_fn(
            self.model, prompts, rewards, responses
        )

        # Update only LoRA parameters
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)

        return loss.item()

    def save_checkpoint(self, path: str):
        """Save LoRA weights."""
        # Collect LoRA state
        lora_state = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                lora_state[name] = {
                    "lora_a": module.lora_a,
                    "lora_b": module.lora_b,
                }
        mx.savez(path, **{
            f"{k}.{p}": v
            for k, params in lora_state.items()
            for p, v in params.items()
        })
        print(f"Saved LoRA checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load LoRA weights."""
        state = mx.load(path)
        # Reconstruct and apply LoRA state
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                a_key = f"{name}.lora_a"
                b_key = f"{name}.lora_b"
                if a_key in state and b_key in state:
                    module.lora_a = state[a_key]
                    module.lora_b = state[b_key]
        mx.eval(self.model.parameters())
        print(f"Loaded LoRA checkpoint from {path}")
