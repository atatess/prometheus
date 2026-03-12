# 🔬 Prometheus — Research Notes & Evolution Log

> Living document. Updated after every experiment. Goal: turn Prometheus into a genuine research contribution.

---

## Current Status
- **Phase:** Layer 1 (Self-play GRPO) working. Moving to proposer-driven curriculum.
- **Model:** Qwen3.5-4B-MLX-4bit (M2 Max 32GB)
- **Best run:** smoke_016 — 8 training updates, 50% accuracy on hard problems

---

## Experiment Log

### smoke_001–011 — Iterations before v2
- Pipeline broke on: thinking tag stripping, fraction parsing (1/6 → 6), OOM kills
- Root causes: `max_seq_len` too large, no memory cleanup between steps
- Fix: `mx.clear_cache()` + `gc.collect()` + `max_seq_len=512`

### smoke_012–014 — False OOM kills
- SIGTERM at model load looked like OOM but memory pressure was GREEN (11.5GB cached)
- Root cause: exec tool 30-second timeout was killing the process
- Fix: run with background=true, no timeout

### smoke_015 — Output buffering hang
- Process ran at 67% CPU for 10+ minutes with no output
- Root cause: Python stdout block-buffered when piped, `print()` never flushed
- Fix: `python -u` (unbuffered)

### smoke_016 — First clean 30-min run ✅
- **28 steps, 8 training updates (29% utilization), avg loss 30.82**
- Model accuracy on triggered problems: 50% overall
- Per-domain: math 100%, science 100%, data 100%, spatial 100%, logic 86%
- Key insight: **72% of seed problems skipped as "too easy"** — model is already strong

---

## Findings & Analysis

### 🔴 Critical Weaknesses

#### 1. Seed Floor Problem
- Qwen3.5-4B has near-perfect accuracy on our "hard" seed problems
- 20/28 steps skipped — 72% wasted compute doing 2-probe inference with no training signal
- Root cause: we designed seeds for GPT-3.5 level, not a Qwen3.5 thinking model
- **Fix:** (a) replace seeds with competition-level math (AMC/AIME), (b) enable proposer earlier

#### 2. Answer Parsing Fragility
- "7^2025 last two digits" → model answers "1/5" (treating it as probability!) — fraction extraction matches `/` and grabs the fraction
- Some rollouts return `"[number]" on a separate line.` — prompt template leakage
- Cube nets: model gets right reasoning but can output "27." with a period
- **Fix:** Smarter answer extraction — try multiple patterns, normalize before comparison

#### 3. GRPO Loss Scale
- Avg loss 30.82 is unusually high (typical GRPO is 0.1–2.0)
- Possible cause: rewards are {0, 1} binary with no normalization; GRPO loss = -log(prob) * advantage, advantage variance matters
- Could also be group_size=6 with extreme reward variance (0/6 or 5/6 is common)
- **Fix:** Normalize rewards within group, add reward shaping (partial credit for near-misses), check loss computation

#### 4. Proposer Never Triggered
- 30-min budget exhausted before 36 seed problems used up (only 28 steps reached)
- The whole point of Prometheus (self-generating curriculum) never ran
- **Fix:** (a) set `max_seed_problems` config param, (b) proposer kicks in after N seeds

#### 5. Single Domain Training
- Curriculum locked to "math" in config despite multi-domain seed problems
- Domain stats show science/logic/spatial/data getting attempted via seeds but model only trains on math
- **Fix:** Enable all domains in curriculum from the start

#### 6. No Benchmark Tracking
- We know training is happening but zero visibility into whether it helps
- Without pre/post benchmark, we can't measure improvement
- **Fix:** Run GSM8K or MATH-500 subset before and after training

---

### 🟢 What's Working Well

1. **End-to-end pipeline is stable** — no crashes in smoke_016
2. **Probe-first efficiency** — correctly skips trivially easy problems, saves full rollout compute
3. **Checkpoint-every-update** — partial runs recoverable
4. **Memory management** — OOM issues resolved, 30-min run completed cleanly
5. **Thinking model handling** — `<think>` tag stripping works
6. **GRPO signal quality** — when the model IS in goldilocks zone (e.g., cube nets 33%), loss is low (19.5) and training makes sense

---

## Research Directions (Inspired by Prior Work)

### From Absolute Zero (2505.03335)
- **Self-referential bootstrap:** Model generates both problems AND solutions in a single pass
- We should try "propose + solve in one generation" to reduce inference calls
- Their key finding: proposer and solver co-evolve, proposer learns to write problems the solver ALMOST gets right
- **TODO:** Log proposer problem difficulty distribution over time — does it adapt?

### From Agent0 (2511.16043)
- **Co-evolutionary curriculum:** Agent0 uses two separate model copies (proposer vs solver)
- We use same weights for both (memory constraint) — this may limit co-evolution
- Workaround: alternate LoRA adapters (train one, freeze other) to create gradient separation
- **TODO:** Measure if proposer-generated problems are harder or easier than seeds

### From ICML 2026 Triadic Framework (2603.02218)
Three conditions for sustained self-evolution:
1. **Asymmetric co-evolution** — proposer and solver must have different objectives
2. **Capacity growth** — model must be able to get harder (LoRA rank expansion?)
3. **Proactive info seeking** — agent must search for new domains, not wait
- We satisfy (1) partially (same weights, different prompts)
- We DON'T satisfy (3) yet — no active domain search
- **TODO:** Implement domain scout that searches for "interesting problems" online

### From Karpathy's autoresearch
- **Key idea:** The training script itself is a hypothesis. The meta-loop tests hypotheses.
- `train.py` should be versioned and modified by the outer loop based on experiment results
- Concrete: if loss > threshold, auto-adjust learning rate. If skip rate > 70%, auto-increase difficulty.
- **TODO:** Implement `meta_loop.py` that reads experiment results and patches `train.py`

### From Code-as-Task (NeurIPS 2025)
- Expressing problems as Python code with assertions is the key insight
- We're using it for verification but not fully for generation
- The model should learn to WRITE the test_code itself, not just solve the problem
- **TODO:** Add "write your own test" training objective — model writes verifiable problem + verifier simultaneously

### Novel: Autonomous Verification Expansion (Prometheus's core claim)
- **The gap we're filling:** Absolute Zero et al. are stuck in math/code because those have natural verifiers
- Our Verification Factory should write Python verifiers for domains like:
  - Spatial reasoning (SVG/coordinate geometry checks)
  - Planning (BFS/DFS to verify optimal solutions)
  - Scientific reasoning (numerical simulation checks)
- **Status:** `verification_factory.py` exists but untested
- **Next:** Run VF on a simple domain (chess positions? map navigation?) and validate

---

## Improvement Roadmap

### Short Term (next 3–5 runs)
- [ ] Add `max_seed_problems` config to cap seeds and force proposer to kick in
- [ ] Enable multi-domain curriculum (math, logic, spatial)
- [ ] Add competition-level seed problems (AMC 10/12 style)
- [ ] Fix answer extraction — fuzzy match, normalize fractions, strip punctuation
- [ ] Run GSM8K-50 mini benchmark pre/post each long run to measure actual improvement

### Medium Term
- [ ] Implement `meta_loop.py` — reads experiment results, auto-tunes config
  - If skip_rate > 60%: increase difficulty bias in proposer prompt
  - If loss > 25: halve learning rate
  - If loss < 1 for 5 steps: expand to new domain
- [ ] Reward shaping: partial credit for correct reasoning with wrong format
- [ ] Two-adapter approach: separate proposer and solver LoRA weights
- [ ] Verification Factory first test: write a verifier for "river crossing" logic puzzles

### Long Term (the actual thesis)
- [ ] Demonstrate GPQA Diamond improvement: 76.2% → 78%+ (measurable edge over baseline)
- [ ] Show verification expansion: start with math-only, autonomously add 3+ new domains
- [ ] Meta-loop modifying train.py in real-time (Karpathy's autoresearch realized)
- [ ] Publish results: "Self-improving 4B model on consumer hardware via autonomous verification expansion"

---

## Open Questions

1. **Does GRPO actually work at LoRA rank 8 for a 4B model?** Loss numbers are weird (30.82 avg). Need to compare against known-good GRPO implementations.

2. **Thinking tokens and GRPO:** We strip `<think>` before computing loss. Is this correct? Thinking tokens shouldn't contribute to GRPO gradient — but are we sure we're not accidentally penalizing reasoning?

3. **Is 4B big enough?** The Verification Factory requires the model to write Python verifiers. Does a 4B model have enough capacity to (a) solve problems AND (b) write verifiers for novel domains?

4. **LoRA for self-improvement:** We train with rank 8, but the proposer and solver share weights. After training, the LoRA shifts the model toward "being a better solver." Does this degrade proposer quality?

5. **Evaluation validity:** We measure accuracy on our own training distribution. That's not a real eval. Need external benchmark to know if we're actually improving reasoning or just overfitting to our problem format.

---

## Technical Reference

### Config tuning cheat sheet
| Goal | Config change |
|------|--------------|
| Force proposer early | `max_seed_problems = 10` |
| More gradient signal | `group_size = 8` (more rollouts per problem) |
| Reduce loss scale | normalize rewards: `rewards = (r - mean) / (std + 1e-8)` |
| Longer run | `time_budget_minutes = 120` |
| Multi-domain | `initial_domains = ["math", "logic", "spatial"]` |

### Known gotchas
- `python -u` required for unbuffered output (otherwise looks like a hang)
- `mx.clear_cache()` must be called between steps or Metal OOM on step 3+
- Fraction answers need special parsing: "1/6" != "0.1667"
- Thinking model: always strip `<think>.*</think>` before extracting answer
- SIGTERM 143 from exec tool timeout — always background long runs

---

*Last updated: 2026-03-12 | Run: smoke_016 | Next: v3_001 (proposer enabled)*
