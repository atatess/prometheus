"""
Layer 2: Verification Factory — standalone runner.

Generates, validates, and registers Python verifiers for new domains.
The model writes its own reward functions — the revolutionary part of Prometheus.

Usage (on GPU droplet):
    cd /root/prometheus
    PROMETHEUS_BACKEND=cuda python scripts/run_layer2.py

Options:
    --domains planning causal_reasoning analogy   (default: all three)
    --model   /root/models/qwen3.5-4b             (default)
    --retries 2                                    (max retry attempts per domain)
    --dry-run                                      (parse only, don't register)
"""

import argparse
import os
import sys
import json
import time

# Allow import from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────────────────────
# Backend detection
# ─────────────────────────────────────────────────────────────────────────────
_backend_env = os.environ.get("PROMETHEUS_BACKEND", "").lower()
import platform
if _backend_env == "cuda":
    USE_CUDA = True
elif _backend_env == "mlx":
    USE_CUDA = False
else:
    USE_CUDA = platform.system() != "Darwin"

BACKEND_NAME = "CUDA/PyTorch" if USE_CUDA else "MLX"

print("=" * 60)
print("🏭 PROMETHEUS — Layer 2: Verification Factory")
print(f"   Backend: {BACKEND_NAME}")
print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Verification Factory Runner")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["planning", "causal_reasoning", "analogy"],
        help="Domains to generate verifiers for",
    )
    parser.add_argument(
        "--model",
        default="/root/models/qwen3.5-4b",
        help="Model path (CUDA) or HF model id (MLX)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Max retry attempts per domain if validation fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and validate but do not register to curriculum",
    )
    parser.add_argument(
        "--output-dir",
        default="src/domains/generated",
        help="Where to save validated verifiers",
    )
    return parser.parse_args()


def load_model(model_path: str):
    """Load model + tokenizer for the current backend."""
    print(f"\n📦 Loading model: {model_path}")
    t0 = time.monotonic()

    if USE_CUDA:
        from src.load_model_cuda import load_model_cuda
        model, tokenizer = load_model_cuda(model_path, device="cuda")
    else:
        from mlx_lm import load
        model, tokenizer = load(model_path)

    elapsed = time.monotonic() - t0
    print(f"✅ Model loaded in {elapsed:.1f}s")
    return model, tokenizer


def make_generate_fn(model, tokenizer, max_tokens: int = 2000):
    """Return a generate_fn(prompt: str) -> str for the current backend."""
    if USE_CUDA:
        import torch

        def generate_cuda(prompt_text: str) -> str:
            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_ids = out[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generate_cuda
    else:
        from src.model_utils import chat_generate

        def generate_mlx(prompt_text: str) -> str:
            return chat_generate(model, tokenizer, prompt_text, max_tokens=max_tokens)

        return generate_mlx


# Candidate metadata — descriptions for our target domains
DOMAIN_DESCRIPTIONS = {
    "planning": "Sequential planning — ordering steps, choosing optimal action sequences, scheduling tasks in the right order",
    "causal_reasoning": "Causal reasoning — identifying cause-and-effect relationships, counterfactuals ('if X hadn't happened, Y would...')",
    "analogy": "Analogical reasoning — completing A:B::C:? patterns, identifying structural similarities between domains",
    "logic": "Propositional and predicate logic, truth tables, deductive syllogisms",
    "spatial": "Spatial reasoning — relative positions, directions, 2D grid navigation, rotations",
    "probability": "Probability — compute likelihoods, expected values, basic combinatorics",
    "constraint": "Constraint satisfaction — Sudoku-like puzzles, scheduling with constraints",
    "graph": "Graph theory — shortest paths, connectivity, coloring problems",
}


def main():
    args = parse_args()

    # Load model
    model, tokenizer = load_model(args.model)
    generate_fn = make_generate_fn(model, tokenizer, max_tokens=2000)

    # Init factory
    from src.verification_factory import VerificationFactory, EXPANSION_CANDIDATES

    factory = VerificationFactory(generated_dir=args.output_dir)
    already_registered = factory.get_registered_domains()
    if already_registered:
        print(f"\n📂 Already registered: {already_registered}")

    # Summary tracking
    summary = {
        "total": len(args.domains),
        "succeeded": [],
        "failed": [],
        "skipped": [],
    }

    for domain in args.domains:
        # Skip if already registered (unless we want to re-run)
        if domain in already_registered:
            print(f"\n⏭️  {domain}: already registered, skipping")
            summary["skipped"].append(domain)
            continue

        description = DOMAIN_DESCRIPTIONS.get(
            domain,
            # Fallback: look up from EXPANSION_CANDIDATES
            next(
                (c["description"] for c in EXPANSION_CANDIDATES if c["domain"] == domain),
                f"Reasoning domain: {domain}",
            ),
        )

        verifier = factory.run_factory_with_model(
            domain=domain,
            description=description,
            generate_fn=generate_fn,
            max_retries=args.retries,
        )

        if verifier is not None:
            if args.dry_run:
                print(f"\n  [DRY RUN] Would register '{domain}' (accuracy={verifier.accuracy_on_tests:.0%})")
                # Save verifier JSON anyway (but don't call register_domain)
                path = factory.generated_dir / f"{domain}_dryrun.json"
                path.write_text(json.dumps({
                    "domain": verifier.domain,
                    "verifier_code": verifier.verifier_code,
                    "proposer_template": verifier.proposer_template,
                    "test_examples": verifier.test_examples,
                    "accuracy_on_tests": verifier.accuracy_on_tests,
                    "validated": verifier.validated,
                }, indent=2))
                print(f"  Saved dry-run JSON: {path}")
            else:
                factory.register_verifier(verifier)
            summary["succeeded"].append({
                "domain": domain,
                "accuracy": verifier.accuracy_on_tests,
                "num_examples": len(verifier.test_examples),
            })
        else:
            summary["failed"].append(domain)

        # GPU memory cleanup between domains
        if USE_CUDA:
            try:
                import torch, gc
                torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass

    # ─── Final Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("🏁 Layer 2 Complete — Summary")
    print(f"{'='*60}")
    print(f"  Total domains attempted : {summary['total']}")
    print(f"  ✅ Succeeded            : {len(summary['succeeded'])}")
    print(f"  ❌ Failed               : {len(summary['failed'])}")
    print(f"  ⏭️  Skipped (existing)   : {len(summary['skipped'])}")

    if summary["succeeded"]:
        print("\n  Registered verifiers:")
        for s in summary["succeeded"]:
            print(f"    ✅ {s['domain']:20s}  accuracy={s['accuracy']:.0%}  examples={s['num_examples']}")

    if summary["failed"]:
        print(f"\n  Failed domains: {summary['failed']}")
        print("  → Tip: try --retries 3 or check model output with --dry-run")

    if not args.dry_run and summary["succeeded"]:
        print(f"\n  Verifiers saved to: {args.output_dir}/")
        print("  Next: git commit + push on local machine, then git pull on droplet")
        print("  The running train.py will pick up new domains on next curriculum check.")

    print(f"\n{'='*60}")
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
