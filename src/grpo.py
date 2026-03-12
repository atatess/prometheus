"""
GRPO (Group Relative Policy Optimization) for MLX.

Based on DeepSeek-R1's approach adapted from Doriandarko/MLX-GRPO.
Three models: trainable policy (π_θ), frozen rollout policy (π_old), frozen reference (π_ref).
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam, cosine_decay
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers

from .model_utils import strip_thinking


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    group_size: int = 4
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_new_tokens: int = 2000  # High for thinking models
    temperature: float = 0.7
    kl_coeff: float = 0.04
    clip_eps: float = 0.2
    max_seq_len: int = 2048
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.1
    warmup_ratio: float = 0.1


def calculate_log_probs(model, tokenizer, prompt: str, completion: str, max_seq_len: int = 384) -> mx.array:
    """Compute log p(completion | prompt) for a single sample.
    
    Memory-efficient: truncates total sequence to max_seq_len tokens
    to fit in Metal GPU memory on 32GB M2 Max.
    """
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    
    # Truncate to fit in memory — keep as much completion as possible
    # but cap total at max_seq_len
    total = len(prompt_tokens) + len(completion_tokens)
    if total > max_seq_len:
        # Prioritize keeping the prompt + trimming completion
        max_completion = max(32, max_seq_len - len(prompt_tokens))
        completion_tokens = completion_tokens[:max_completion]
        # If prompt is too long too, truncate from the front
        if len(prompt_tokens) > max_seq_len - 32:
            prompt_tokens = prompt_tokens[-(max_seq_len - 32):]
    
    full_tokens = prompt_tokens + completion_tokens
    input_ids = mx.array(full_tokens, dtype=mx.int32)[None, :]
    
    # Forward pass
    logits = model(input_ids)
    log_probs_full = nn.log_softmax(logits, axis=-1)
    
    # Extract log probs for completion tokens only
    prompt_len = len(prompt_tokens)
    completion_len = len(completion_tokens)
    
    completion_log_probs = []
    for i in range(completion_len):
        pos = prompt_len - 1 + i
        if pos < len(full_tokens) - 1:
            next_token_id = full_tokens[pos + 1]
            log_prob = log_probs_full[0, pos, next_token_id]
            completion_log_probs.append(log_prob)
    
    if len(completion_log_probs) > 0:
        return mx.sum(mx.stack(completion_log_probs))
    else:
        return mx.array(0.0)


def grpo_loss_single(model, tokenizer, prompt, completion, advantage, config):
    """Compute GRPO loss for a single (prompt, completion, advantage) triple.
    
    L = -advantage * log_prob(completion | prompt; model)
    
    KL penalty disabled to save memory (no reference model copy needed).
    """
    log_prob = calculate_log_probs(model, tokenizer, prompt, completion)
    pg_loss = -advantage * log_prob
    return pg_loss


class GRPOTrainer:
    """GRPO trainer for MLX models."""

    def __init__(self, model, tokenizer, config: GRPOConfig, total_steps: int = 100):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Apply LoRA adapters for training (enables backward pass on quantized model)
        lora_config = {
            'rank': 8,
            'alpha': 16,
            'scale': 2.0,
            'keys': ['gate_proj', 'up_proj', 'down_proj'],
        }
        linear_to_lora_layers(model, num_layers=32, config=lora_config)
        model.train()
        
        # Count trainable params
        trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
        total = sum(p.size for _, p in tree_flatten(model.parameters()))
        print(f"  LoRA: {trainable/1e6:.2f}M trainable / {total/1e6:.0f}M total ({trainable/total*100:.3f}%)")
        
        # No reference model copy — saves ~3GB on 32GB machine
        # KL penalty disabled (kl_coeff=0) to save memory
        self.ref_model = None
        
        # Optimizer
        schedule = cosine_decay(config.learning_rate, total_steps)
        self.optimizer = Adam(learning_rate=schedule)
        
        # State
        self.step = 0
        self._accum_grads = None
        self._accum_count = 0
    
    def generate_rollouts(self, prompt: str) -> list[str]:
        """Generate group_size responses."""
        return self.generate_rollouts_n(prompt, self.config.group_size)
    
    def generate_rollouts_n(self, prompt: str, n: int) -> list[str]:
        """Generate n responses using the current model."""
        import gc
        sampler = make_sampler(temp=self.config.temperature)
        responses = []
        for i in range(n):
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            raw = generate(
                self.model, self.tokenizer,
                prompt=formatted,
                max_tokens=self.config.max_new_tokens,
                sampler=sampler,
            )
            responses.append(raw)
            # Free KV cache between generations
            if i % 2 == 1:
                mx.clear_cache()
                gc.collect()
        return responses

    def train_step(
        self,
        prompt: str,
        responses: list[str],
        rewards: list[float],
    ) -> float:
        """Execute one GRPO training step.
        
        Args:
            prompt: The original problem prompt
            responses: List of model responses (group_size)
            rewards: List of reward scores (group_size)
        
        Returns:
            Loss value
        """
        # Compute group-relative advantages
        mean_r = sum(rewards) / len(rewards)
        std_r = max(
            math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / len(rewards)),
            1e-8,
        )
        advantages = [(r - mean_r) / std_r for r in rewards]
        
        # Format prompt with chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        total_loss = 0.0
        n_updates = 0
        
        for response, advantage in zip(responses, advantages):
            if advantage <= 0:
                continue  # Only train on above-average responses
            
            # Compute loss and gradients
            loss_fn = nn.value_and_grad(
                self.model,
                lambda m: grpo_loss_single(
                    m, self.tokenizer,
                    formatted_prompt, response, advantage, self.config
                ),
            )
            
            loss, grads = loss_fn(self.model)
            
            # Gradient accumulation
            if self._accum_grads is None:
                self._accum_grads = grads
            else:
                self._accum_grads = tree_map(
                    lambda a, b: a + b, self._accum_grads, grads
                )
            self._accum_count += 1
            
            # Apply accumulated gradients
            if self._accum_count >= self.config.gradient_accumulation_steps:
                # Average gradients
                scale = 1.0 / self._accum_count
                avg_grads = tree_map(lambda g: g * scale, self._accum_grads)
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    norms = [mx.sqrt(mx.sum(g * g)) for _, g in tree_flatten(avg_grads)]
                    total_norm = mx.sqrt(sum(n * n for n in norms)) if norms else mx.array(0.0)
                    clip_coef = min(1.0, self.config.max_grad_norm / (total_norm.item() + 1e-6))
                    if clip_coef < 1.0:
                        avg_grads = tree_map(lambda g: g * clip_coef, avg_grads)
                
                # Update
                self.optimizer.update(self.model, avg_grads)
                mx.eval(self.model.parameters(), self.optimizer.state)
                
                self._accum_grads = None
                self._accum_count = 0
            
            total_loss += loss.item()
            n_updates += 1
        
        self.step += 1
        
        # Sync rollout model every 10 steps
        if self.step % 10 == 0:
            self._sync_rollout_model()
        
        return total_loss / max(n_updates, 1)
    
    def _sync_rollout_model(self):
        """No-op — we generate rollouts from the trainable model directly."""
        pass

    def save_checkpoint(self, path: str):
        """Save model weights (all trainable params)."""
        weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.savez(path, **weights)
        print(f"Saved checkpoint to {path} ({len(weights)} tensors)")

    def load_checkpoint(self, path: str):
        """Load model weights."""
        state = mx.load(path)
        self.model.load_weights(list(state.items()), strict=False)
        mx.eval(self.model.parameters())
        print(f"Loaded checkpoint from {path}")
