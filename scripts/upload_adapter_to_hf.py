"""
Run locally after downloading adapter files:

    export HF_TOKEN="<hf_write_token>"
    export HF_REPO_ID="<username>/<repo_name>"
    python upload_adapter_to_hf.py

Optional:
    python upload_adapter_to_hf.py --adapter-dir ./adapter --repo-id <username>/<repo_name>
    python upload_adapter_to_hf.py --no-private   # create/update repo as public

This script uploads only inference-time adapter/tokenizer artifacts from the adapter folder,
and skips training checkpoint files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local LoRA adapter artifacts to Hugging Face")
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="./adapter",
        help="Local folder containing downloaded adapter files",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=os.environ.get("HF_REPO_ID", ""),
        help="Hugging Face model repo id, e.g. username/qwen2.5-7b-swiss-law-adapter",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face write token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the repo as private (default: true). Use --no-private for public.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.repo_id:
        raise ValueError("Missing repo id. Pass --repo-id or set HF_REPO_ID.")
    if not args.token:
        raise ValueError("Missing HF token. Pass --token or set HF_TOKEN.")

    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    if not adapter_dir.exists() or not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is not installed. Install with: pip install huggingface_hub"
        ) from e

    # Keep upload focused on files needed for inference.
    allow_patterns = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "README.md",
    ]

    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing_required = [name for name in required if not (adapter_dir / name).exists()]
    if missing_required:
        raise FileNotFoundError(
            "Missing required adapter files in adapter dir: " + ", ".join(missing_required)
        )

    existing_for_upload = [name for name in allow_patterns if (adapter_dir / name).exists()]
    print("Adapter directory:", adapter_dir)
    print("Repo id:", args.repo_id)
    print("Visibility:", "private" if args.private else "public")
    print("Files to upload:")
    for name in existing_for_upload:
        size = (adapter_dir / name).stat().st_size
        print(f"  - {name} ({size} bytes)")

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(adapter_dir),
        path_in_repo=".",
        allow_patterns=existing_for_upload,
        commit_message="Upload LoRA adapter artifacts",
    )

    print("Upload complete.")
    print(f"Model repo: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
