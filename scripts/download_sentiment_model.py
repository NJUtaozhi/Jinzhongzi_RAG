"""Download the Task 7.1 checkpoint for offline server transfer.

Run this script on a machine that can reach Hugging Face. The resulting
``models/chinese-emotion`` directory can be copied to the server together with
the repository; production inference never downloads files at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL = "LXDaugh/chinese-6-emotion-model"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "models" / "chinese-emotion"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model,
        local_dir=str(args.output),
        allow_patterns=[
            "*.json", "*.txt", "*.model", "*.safetensors", "*.bin",
            "tokenizer.*", "special_tokens_map.json", "vocab.*", "merges.txt",
        ],
    )
    total = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    print(f"Downloaded {args.model} to {args.output}")
    print(f"Total size: {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

