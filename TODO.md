# Prometheus — TODO & Roadmap

*Last updated: 2026-03-15*
*Rule: never leave a session without checking this. Never stop improving.*

---

## 🔥 Immediate (next 24h)

- [ ] **Run baseline benchmark** — `evaluate.py` on base model BEFORE training ends
  - SSH: `source .venv/bin/activate && PROMETHEUS_BACKEND=cuda python evaluate.py --tag baseline`
  - Need this to prove improvement after run_016 completes

- [ ] **Run post-training benchmark** — after run_016 finishes, run evaluate.py again
  - Compare baseline vs trained, compute delta per domain
  - If delta is meaningful → we have proof the system works

- [ ] **Wire autonomous Layer 2 into meta-loop** — currently requires manual `run_layer2.py`
  - `prometheus.py` should detect plateau → trigger factory → register → continue training
  - No human intervention = the actual revolutionary claim

- [ ] **Analyze meta_decisions.jsonl** — check what the meta-loop has decided so far
  - `cat experiments/meta_decisions.jsonl | python3 -c "import sys,json; [print(json.loads(l)) for l in sys.stdin]"`

- [ ] **Commit RESEARCH_LOG.md entry for run_016** after it completes
  - Include: accuracy trend, loss trend, Layer 2 outcomes, benchmark delta

---

## 🧠 Make It Actually Revolutionary

- [ ] **Prove the recursive loop** — train → model improves → model writes BETTER verifiers → train on new domains → repeat
  - Currently only cycled once (math/code → writing verifiers for planning/causal/analogy)
  - Need to show iteration 2: can the trained model write more sophisticated verifiers?

- [ ] **Better verifiers for causal_reasoning and analogy**
  - Current: generic string equality (not domain-specific)
  - Goal: verifiers that check actual causal structure / relational patterns
  - The planning verifier is good — use it as the bar

- [ ] **Add template proposer coverage for new domains**
  - `src/template_proposer.py` only covers math/code/logic
  - Need templates for: planning (sequence ordering), causal_reasoning (if-then), analogy (A:B::C:?)
  - Without good problems, new domains can't train meaningfully

- [ ] **Hard domain verifiers** — stretch goal
  - `moral_reasoning`, `scientific_hypothesis`, `creative_analogy`
  - These are the domains nobody can auto-verify — if Prometheus can bootstrap verifiers here, that's the paper

- [ ] **Retry loop for factory** — `FACTORY_RETRY_PROMPT` exists but loop logic not fully written
  - Currently max_retries=2 with generic retry; should use specific feedback on what failed

---

## 📊 Measurement & Validation

- [ ] **Pre/post benchmark delta** — primary proof of concept (see Immediate above)

- [ ] **Verifier quality score** — measure how domain-specific generated verifiers are
  - Test: does the planning verifier actually reject logically impossible sequences?
  - Test: does the causal_reasoning verifier distinguish correlation from causation?

- [ ] **Domain expansion tracking** — log which domains get added when, and what accuracy effect they have

- [ ] **Flash Attention** — confirm if installed (`cat /tmp/flash_attn.log`)
  - Would meaningfully speed up generation (longer contexts, faster rollouts)

---

## 🏗️ Infrastructure

- [ ] **Monitor dashboard accuracy history** — currently shows live stats, no trend chart for accuracy
  - Add accuracy-over-steps chart alongside loss chart

- [ ] **Auto-push from droplet** — droplet can't push to GitHub (no auth)
  - Solution: pull from local after each run, or set up deploy key on droplet

- [ ] **Save LoRA adapter after training** — currently checkpoints mid-run but unclear if final adapter is saved cleanly
  - After run_016: check `experiments/cuda_run_016/checkpoints/` for final adapter

- [ ] **Evaluation script for new domains** — `evaluate.py` only tests math/logic/code/science
  - Add planning/causal_reasoning/analogy eval problems once those domains have training data

---

## 📄 Documentation & Research

- [ ] **RESEARCH_LOG.md** — document every run with: hypothesis, result, lesson learned
  - Run 016 entry pending completion

- [ ] **README.md** — needs update with Layer 2 working, Layer 3 active, benchmark results (once we have them)

- [ ] **Paper draft** — outline the Prometheus contribution vs Absolute Zero / Agent0 / R-Zero
  - Core claim: "We extend self-play RL to arbitrary domains by teaching the model to write its own verifiers"
  - Baseline: Absolute Zero (only math/code verifiable)
  - Our contribution: Layer 2 breaks the verifiability ceiling

---

## ✅ Done (for reference)

- [x] Port training to CUDA (PyTorch GRPO, no Unsloth, no trl)
- [x] Fix 6 critical training bugs (log_prob, chat format, answer extraction, etc.)
- [x] Layer 1: GRPO training stable, accuracy improving (7.5% → 28% over 60 steps)
- [x] Layer 2: Verification factory works — 3 verifiers registered (planning, causal_reasoning, analogy)
- [x] Layer 3: Meta-loop active — monitors loss, will trigger interventions
- [x] Template proposer — guaranteed problem generation, zero LLM dependency
- [x] Monitor dashboard — live at http://138.197.141.102:8080
- [x] Cron monitor — auto-restart on crash, duplicate kill, 15m health checks
- [x] evaluate.py — 50 hold-out benchmark problems

---

*The goal: a system that genuinely improves itself across domains, provably, without human intervention.*
*Every session: check this list. Pick the highest-impact unchecked item. Do it.*
