"""Download and optionally quantize the base model for Prometheus."""

import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download base model")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3.5-4B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: HF cache)",
    )
    args = parser.parse_args()
    
    print(f"📦 Downloading {args.model}...")
    path = snapshot_download(
        args.model,
        local_dir=args.output,
    )
    print(f"✅ Downloaded to: {path}")
    print(f"\nTo use with MLX, you can quantize with:")
    print(f"  python -m mlx_lm.convert --hf-path {args.model} -q")


if __name__ == "__main__":
    main()
