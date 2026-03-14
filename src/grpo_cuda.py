"""
GRPO (Group Relative Policy Optimization) — CUDA / PyTorch backend.

Drop-in replacement for grpo.py.  Identical public API:
  - GRPOConfig  (dataclass, same fields)
  - GRPOTrainer (same methods: __init__, generate_rollouts,
                 generate_rollouts_n, train_step,
                 save_checkpoint, load_checkpoint)

Key design choices:
  - LoRA via PEFT (frozen base model, only adapters are trained).
  - No reference model / no KL penalty (same as the MLX version).
  - Log-prob computed by a single forward pass over [prompt + completion],
    then slicing the logit positions that correspond to completion tokens.
  - Only responses with advantage > 0 contribute to the loss.
  - Gradient accumulation + AdamW + cosine LR schedule.
"""

import gc
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    """Configuration for GRPO training — identical fields to the MLX version."""
    group_size: int = 4
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_new_tokens: int = 2000
    temperature: float = 0.7
    kl_coeff: float = 0.04          # Kept for API parity; not used (no ref model)
    clip_eps: float = 0.2           # Kept for API parity; not used (vanilla PG)
    max_seq_len: int = 2048
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.1
    warmup_ratio: float = 0.1


# ---------------------------------------------------------------------------
# Log-probability helper
# ---------------------------------------------------------------------------

def calculate_log_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    completion: str,
    max_seq_len: int = 2048,
    device: torch.device = None,
) -> torch.Tensor:
    """Compute sum of log p(completion_token | context) over all completion tokens.

    Algorithm:
      1. Tokenise [prompt + completion] into a single sequence.
      2. Forward pass → logits of shape [1, seq_len, vocab_size].
      3. Apply log_softmax over vocab dimension.
      4. For each completion position i, the "input" token at position
         (prompt_len - 1 + i) predicts the token at position
         (prompt_len + i).  Gather those log-probs and sum them.

    Memory guard: total sequence is capped at max_seq_len tokens, trimming
    the completion first, then the prompt front if necessary.
    """
    if device is None:
        device = next(model.parameters()).device

    # Tokenise separately so we know the prompt boundary.
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)

    # ----- Truncation (mirrors MLX version) --------------------------------
    total = len(prompt_ids) + len(completion_ids)
    if total > max_seq_len:
        max_comp = max(32, max_seq_len - len(prompt_ids))
        completion_ids = completion_ids[:max_comp]
        if len(prompt_ids) > max_seq_len - 32:
            prompt_ids = prompt_ids[-(max_seq_len - 32):]
    # -----------------------------------------------------------------------

    full_ids = prompt_ids + completion_ids
    input_tensor = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    # shape: [1, seq_len]

    # Forward pass (no gradient here; gradients flow through the outer call
    # when this function is called inside train_step).
    logits = model(input_tensor).logits
    # shape: [1, seq_len, vocab_size]

    # Log-softmax over the vocabulary axis.
    log_probs = F.log_softmax(logits, dim=-1)
    # shape: [1, seq_len, vocab_size]

    # ---- Gather completion token log-probs ---------------------------------
    # For token at index `t` in full_ids, the model's prediction is at
    # logits position t-1 (the model predicts the *next* token).
    # Completion tokens start at index `prompt_len` in full_ids.
    prompt_len = len(prompt_ids)
    comp_len = len(completion_ids)

    if comp_len == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Logit positions that predict each completion token:
    #   completion token 0  (full_ids[prompt_len])     → logit at prompt_len-1
    #   completion token k  (full_ids[prompt_len + k]) → logit at prompt_len-1+k
    logit_positions = torch.arange(
        prompt_len - 1, prompt_len - 1 + comp_len, device=device
    )  # shape: [comp_len]

    # Token IDs we want to look up:
    target_ids = torch.tensor(
        completion_ids, dtype=torch.long, device=device
    )  # shape: [comp_len]

    # Gather: log_probs[0, logit_positions, target_ids]
    gathered = log_probs[0, logit_positions, target_ids]
    # shape: [comp_len]

    return gathered.sum()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """GRPO trainer — CUDA / PyTorch backend.

    Wraps a HuggingFace model with PEFT LoRA adapters and trains it using
    Group Relative Policy Optimisation.

    Usage mirrors the MLX GRPOTrainer exactly so that train.py needs only
    a conditional import to switch backends.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: GRPOConfig,
        total_steps: int = 100,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.step = 0

        # Determine device from where the model lives.
        self.device = next(model.parameters()).device

        # ------------------------------------------------------------------
        # Apply LoRA adapters via PEFT.
        # The base model weights are frozen; only the adapter parameters are
        # updated during training.
        # ------------------------------------------------------------------
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, lora_cfg)
        self.model.train()

        # Log trainable parameter count.
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in self.model.parameters())
        print(
            f"  LoRA: {trainable/1e6:.2f}M trainable / {total_p/1e6:.0f}M total "
            f"({trainable/total_p*100:.3f}%)"
        )

        # ------------------------------------------------------------------
        # Optimizer: AdamW with cosine LR schedule.
        # ------------------------------------------------------------------
        warmup_steps = max(1, int(total_steps * config.warmup_ratio))
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_cycles=total_steps,
        )

        # Gradient accumulation state.
        self._accum_count = 0
        self.optimizer.zero_grad()

    # ------------------------------------------------------------------
    # Rollout generation
    # ------------------------------------------------------------------

    def generate_rollouts(self, prompt: str) -> list[str]:
        """Generate group_size responses for `prompt`."""
        return self.generate_rollouts_n(prompt, self.config.group_size)

    def generate_rollouts_n(self, prompt: str, n: int) -> list[str]:
        """Generate `n` independent responses for `prompt`.

        The model is temporarily put in eval mode and torch.no_grad() is
        used so no activations are stored, keeping VRAM usage low.
        """
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        responses = []
        self.model.eval()  # Disable dropout during generation.

        with torch.no_grad():
            for i in range(n):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=(self.config.temperature > 0),
                    temperature=self.config.temperature if self.config.temperature > 0 else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                new_tokens = output_ids[0, prompt_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                responses.append(text)

                # Free CUDA cache every other generation.
                if i % 2 == 1:
                    torch.cuda.empty_cache()
                    gc.collect()

        self.model.train()  # Restore training mode.
        return responses

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        prompt: str,
        responses: list[str],
        rewards: list[float],
    ) -> float:
        """Execute one GRPO update.

        Args:
            prompt:    The solver prompt (already formatted for the model).
            responses: Decoded response strings (length == group_size).
            rewards:   Scalar reward for each response.

        Returns:
            Mean loss over the positive-advantage responses.
        """
        # ---- Compute normalised advantages --------------------------------
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_t.mean().item()
        std_r = max(rewards_t.std(unbiased=False).item(), 1e-8)
        advantages = [(r - mean_r) / std_r for r in rewards]

        total_loss = 0.0
        n_updates = 0

        for response, advantage in zip(responses, advantages):
            if advantage <= 0:
                # Only train on above-average responses (positive advantage).
                continue

            # ---- Forward pass to get log p(completion | prompt) -----------
            log_prob = calculate_log_probs(
                self.model,
                self.tokenizer,
                prompt,
                response,
                max_seq_len=self.config.max_seq_len,
                device=self.device,
            )

            # GRPO policy-gradient loss: L = -A * log π(y|x)
            loss = -advantage * log_prob

            # Scale loss for gradient accumulation so the effective gradient
            # magnitude is independent of how many samples we accumulate.
            scaled_loss = loss / self.config.gradient_accumulation_steps
            scaled_loss.backward()

            total_loss += loss.item()
            n_updates += 1
            self._accum_count += 1

            # ---- Apply accumulated gradients when buffer is full ----------
            if self._accum_count >= self.config.gradient_accumulation_steps:
                self._apply_gradients()

        self.step += 1
        return total_loss / max(n_updates, 1)

    def _apply_gradients(self):
        """Clip gradients and step the optimizer."""
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self._accum_count = 0

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        """Save LoRA adapter weights to `path` (treated as a directory).

        Uses PEFT's save_pretrained so the checkpoint can be reloaded
        with PeftModel.from_pretrained().
        """
        self.model.save_pretrained(path)
        print(f"Saved LoRA checkpoint to {path}/")

    def load_checkpoint(self, path: str):
        """Load LoRA adapter weights from a save_pretrained directory.

        The base model is left frozen; only the adapter weights are replaced.
        """
        # PeftModel.from_pretrained expects the *base* model, not the wrapped one.
        base_model = self.model.base_model.model
        self.model = PeftModel.from_pretrained(base_model, path)
        self.model.to(self.device)
        self.model.train()
        print(f"Loaded LoRA checkpoint from {path}/")
