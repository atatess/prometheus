# Prometheus Research Log

*Updated after every run. Goal: track what we learn, what breaks, and what to try next.*

---

## Infrastructure Setup (2026-03-14)

**Hardware:** DigitalOcean L40S GPU (48GB VRAM, Toronto `tor1`)
**Cost:** $1.57/hr — destroy between sessions
**Model:** `Qwen/Qwen3.5-4B` (instruct, 8.4GB in bf16) — confirmed this IS the instruct variant
**Stack:** PyTorch 2.10 + CUDA 12.8 + PEFT LoRA (rank 8) + transformers

**Key discovery:** `Qwen/Qwen3.5-4B` outputs two thinking formats:
- `<think>...</think>` — standard, parseable
- `Thinking Process:\n1. ...` — alternative format, JSON ends up inside the thinking block
- `strip_thinking` must NOT be applied before factory parsing — destroys context

---

## Run 001–003 (Aborted — infrastructure bugs)

**Problems fixed:**
- `train.py` had MLX imports at module level (crashed on Linux)
- Sequential rollout generation (4 min/step) → fixed with batched `model.generate()`
- `UnboundLocalError: re` — local import inside conditional shadowed module-level import
- `num_training_cycles` → `num_training_steps` (wrong kwarg for cosine scheduler)

**Lesson:** Always test import chain before a long run. Add smoke test to CI.

---

## Run 004 — First Complete Run (10 min, `cuda_default.toml`)

**Config:** group_size=6, max_new_tokens=1024, time_budget=10min
**Results:** 5 steps, 3 GRPO updates, 16.7% accuracy, avg loss 178
**Step time:** ~2 min/step

**What went wrong:**
- 1024 tokens × 6 rollouts × sequential = ~4 min/step (most of budget wasted on generation)
- Loss of 178 looks alarming but is expected: GRPO loss = `-advantage × sum_log_prob(completion)`. With 1000 tokens of thinking, log_prob sum ≈ -250, so loss ≈ 250. The absolute value means nothing.
- `strip_thinking` returning garbage for "Thinking Process:" format (answer extracted as `"."`)

**What worked:**
- LoRA applies correctly (10.6M trainable / 4.2B total = 0.25%)
- Curriculum probe logic works (skips too-easy problems)
- GRPO update executes without OOM (8.4GB base + training overhead = ~20GB, well within 48GB)

**Changes made:**
- Batched generation: 6x speedup
- max_new_tokens: 1024 → 768 → 512 (iterating down)
- group_size: 6 → 4

---

## Run 005 — Proper 60-min Run (IN PROGRESS)

**Config:** group_size=4, max_new_tokens=512, time_budget=60min
**Step time:** ~75 sec/step → expect ~48 steps total

**Live results so far (step 25):**

| Step | Accuracy | GRPO Loss | Notes |
|------|----------|-----------|-------|
| 4    | 25%      | 55.0      | First update |
| 5    | 25%      | 31.1      | Trending down ✅ |
| 11   | 50%      | 11.8      | Good step |
| 13   | 75%      | 7.4       | Best so far |
| 15   | 25%      | 11.8      | Back up (normal variance) |
| 22   | 50%      | 14.4      | Stable range |

**Observations:**
- Loss trend: 55 → 31 → 11.8 → 7.4 → 11.8 → 14.4 — general downward trend with noise ✅
- Many steps have 0% accuracy (all wrong) → no training signal, curriculum skips
- Curriculum correctly identifies math as hard (0% domain acc) and logic/data as easy (100%)
- ~30-50% of steps produce no training update (all wrong or all right)

**What to watch:**
- Does loss continue trending down past step 30?
- Does accuracy improve on math domain over time?
- Do we hit curriculum saturation before 60 min?

**Extended (steps 22-27, ~33 min in):**

| Step | Accuracy | GRPO Loss | Notes |
|------|----------|-----------|-------|
| 22   | 50%      | 14.4      | |
| 23   | 0%       | —         | no signal |
| 24   | 0%       | —         | no signal |
| 25   | 0%       | —         | no signal |
| 26   | 25%      | 44.3      | loss spike — normal RL variance |
| 27   | TBD      | —         | |

**Observation:** Loss is NOT consistently trending down — it spikes to 44 at step 26 after being at 14. This is normal for RL (high variance). Need 50+ training updates to see a real trend, not just 10.

**Planned changes after this run:**
- [ ] Add loss-per-step JSON tracking to results file (currently only avg)
- [ ] Log which specific problems are being sampled (helps diagnose curriculum)
- [ ] Increase time budget to 3-4 hours for a real training signal
- [ ] Consider `group_size=8` to get more diverse rollouts per step (L40S has headroom)

---

## Layer 2 — Verification Factory (2026-03-14)

**Test v1 (failed — wrong approach):**
- Only 96 chars output → max_tokens was too low somewhere
- `strip_thinking` applied before `parse_verifier` → destroyed the JSON context

**Test v2 (running now):**
- 3000 token budget for factory generation
- Raw output passed directly to `parse_verifier` (no strip_thinking)
- Improved `parse_verifier`: scans ALL `{` positions, picks largest valid JSON

**Logic domain result (v1 + v2 — same outcome):**
- Generated 5066-char Python verifier on first attempt ✅
- Verifier has 5 test examples ✅
- Validation: **50% accuracy** ❌ (need 80%) — same result in both runs
- Root cause 1: verifier uses complex `problem_type` dispatch — too many code paths, bugs in each
- Root cause 2: Generated Python has truncated lines e.g. `str(answer).s` — code cut off mid-token
- This suggests the verifier code itself hits max_tokens and gets truncated

**Spatial domain result (v2 — still generating):**
- v1: parse failed (JSON inside "Thinking Process:" text) ❌ — fixed in v2
- v2: parsing now works, result pending

**Root cause of 50% validation:**
Two separate issues:
1. **Code quality**: verifier logic is too complex for one-shot generation (truth tables, nested dispatch)
2. **Code truncation**: even at 3000 tokens, long verifier code gets cut mid-line

**What Layer 2 needs:**
1. **Constrained verifier design** — one problem type per verifier, simple input/output contract
2. **Increased token budget** — 4000+ tokens for factory generation
3. **Retry loop with error feedback** — show failing test cases, ask model to fix specific bugs
4. **Multi-shot prompting** — 1-2 working verifier examples in the prompt
5. **Domain difficulty ordering** — start with counting/sequences before spatial/logic
6. **Lower initial validation bar** — 60% to register; 80% to "promote" to primary curriculum

---

## Key Architectural Decisions & Lessons

### GRPO Training
- **LoRA rank 8 is fine** — 10.6M params, trains quickly, no OOM
- **No KL term** — saves memory, works for early training. Add it back if mode collapse appears
- **group_size=4** is the sweet spot for L40S: enough diversity, reasonable step time
- **Batch generation** is essential — sequential was 6x slower
- **Only train on positive-advantage responses** — correct in principle but means ~50% of steps are wasted when all rollouts fail

### Thinking Model Quirks
- Model uses `<think>` OR `Thinking Process:` unpredictably (temperature-driven)
- `strip_thinking` is only for extracting final answers; NEVER apply before structural parsing (factory, proposer)
- The thinking block contains useful context — don't throw it away prematurely

### Speed vs Quality
- 75 sec/step is acceptable for 60-min runs
- For overnight runs (8h), we get ~385 steps — should be enough to see clear learning signal
- Flash Attention would help (not installed — 15-20% speedup available)

### Infrastructure
- Droplet is stateless — always commit checkpoints to repo or download before destroying
- `nohup` + `PYTHONUNBUFFERED=1` + redirect to file = reliable background training
- Running two processes (train + layer2 test) is fine on 48GB: ~26GB total VRAM

---

## Next Session Priorities

1. **Analyze run_005 results** — did loss converge? accuracy improve on math?
2. **Fix Layer 2 with simpler domain + retry loop**
3. **Start overnight run** (8h, group_size=6, max_new_tokens=512)
4. **Flash Attention install** — `pip install flash-attn --no-build-isolation`
5. **Benchmark** — run `evaluate.py` before and after training to measure actual improvement

---

*Log maintained by Clawd 🦞 — updated after every run*
