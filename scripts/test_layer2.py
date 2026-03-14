"""
Layer 2 Validation — Verification Factory smoke test.

Tests whether the model can autonomously write Python verifiers
for new domains. This is the core novel contribution of Prometheus.

Run: PROMETHEUS_BACKEND=cuda python scripts/test_layer2.py
"""

import sys
import os
import json

sys.path.insert(0, ".")

os.environ.setdefault("PROMETHEUS_BACKEND", "cuda" if sys.platform != "darwin" else "mlx")
USE_CUDA = os.environ.get("PROMETHEUS_BACKEND") == "cuda"

print("=" * 60)
print("🏭 PROMETHEUS — Layer 2: Verification Factory Test")
print("=" * 60)

# Load model
if USE_CUDA:
    from src.load_model_cuda import load_model_cuda
    from src.model_utils_cuda import strip_thinking
    print("\n📦 Loading model (CUDA)...")
    model, tokenizer = load_model_cuda("/root/models/qwen3.5-4b")
else:
    from mlx_lm import load
    from src.model_utils import strip_thinking
    print("\n📦 Loading model (MLX)...")
    model, tokenizer = load("mlx-community/Qwen3.5-4B-MLX-4bit")

from src.verification_factory import VerificationFactory, EXPANSION_CANDIDATES
from src.verifier import SandboxConfig

factory = VerificationFactory()
sandbox = SandboxConfig(timeout_seconds=10, max_memory_mb=256)

# Test domains
test_candidates = [
    {
        "domain": "logic",
        "description": "Propositional logic, truth tables, syllogisms, and logical deduction",
    },
    {
        "domain": "spatial",
        "description": "Spatial reasoning — relative positions, directions, and 2D grid navigation",
    },
]

results = []

for candidate in test_candidates:
    domain = candidate["domain"]
    description = candidate["description"]
    print(f"\n{'─'*50}")
    print(f"🧪 Testing domain: {domain}")
    print(f"   {description}")

    # Build the factory prompt
    prompt = factory.build_factory_prompt(domain, description)
    print(f"   Prompt length: {len(prompt)} chars")

    # Generate — needs LOTS of tokens for the full JSON
    print("   Generating verifier (this takes ~60s)...")
    messages = [{"role": "user", "content": prompt}]

    if USE_CUDA:
        import torch
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=3000,   # Factory needs a LOT of tokens
                do_sample=True,
                temperature=0.3,       # Lower temp for structured JSON output
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        raw = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    else:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        raw = generate(model, tokenizer, prompt=formatted,
                      max_tokens=3000, sampler=make_sampler(temp=0.3))

    print(f"   Raw output length: {len(raw)} chars")
    print(f"   First 200 chars: {repr(raw[:200])}")

    # Strip thinking before parsing
    cleaned = strip_thinking(raw)
    print(f"   After strip_thinking: {len(cleaned)} chars")

    # Parse
    verifier = factory.parse_verifier(cleaned)
    if verifier is None:
        # Try on raw too (thinking block might contain the JSON)
        verifier = factory.parse_verifier(raw)

    if verifier is None:
        print(f"   ❌ FAILED to parse verifier")
        results.append({"domain": domain, "status": "parse_failed", "output_len": len(raw)})
        continue

    print(f"   ✅ Parsed verifier for '{verifier.domain}'")
    print(f"   Verifier code ({len(verifier.verifier_code)} chars):")
    print("   " + verifier.verifier_code[:300].replace("\n", "\n   "))
    print(f"   Test examples: {len(verifier.test_examples)}")

    # Validate against test examples
    if verifier.test_examples:
        print("   Running validation...")
        valid = factory.validate_verifier(verifier, sandbox)
        print(f"   Validation: {'✅ PASSED' if valid else '❌ FAILED'} ({verifier.accuracy_on_tests:.0%} accuracy)")
        if valid:
            factory.register_verifier(verifier)
            print(f"   💾 Registered domain '{domain}' to curriculum")
    else:
        print("   ⚠️  No test examples — skipping validation")
        valid = False

    results.append({
        "domain": domain,
        "status": "validated" if valid else "generated",
        "verifier_chars": len(verifier.verifier_code),
        "test_examples": len(verifier.test_examples),
        "accuracy": verifier.accuracy_on_tests,
    })

print(f"\n{'='*60}")
print("🏁 Layer 2 Test Results")
for r in results:
    status = "✅" if r["status"] == "validated" else ("⚠️" if r["status"] == "generated" else "❌")
    print(f"   {status} {r['domain']}: {r['status']}")
print(f"{'='*60}")

# Save results
os.makedirs("experiments", exist_ok=True)
with open("experiments/layer2_test.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to experiments/layer2_test.json")
