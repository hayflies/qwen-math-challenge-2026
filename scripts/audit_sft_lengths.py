"""Audit E001 direct-answer token lengths with the exact cached Qwen tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from qwen_math_challenge.config import find_project_root, load_config
from qwen_math_challenge.training.sft import (
    SFTError,
    audit_token_lengths,
    load_sft_settings,
    load_training_tokenizer,
    validate_official_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the E001 YAML config.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the versioned audit path from the config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        project_root = find_project_root(config.source_path)
        settings = load_sft_settings(config, project_root)
        split = validate_official_split(settings.data)
        tokenizer, tokenizer_commit = load_training_tokenizer(settings)
        audit = audit_token_lengths(
            split.train,
            tokenizer,
            settings,
            tokenizer_commit=tokenizer_commit,
        )
        output = (
            args.output.expanduser().resolve()
            if args.output is not None
            else settings.data.length_audit_path
        )
        analysis_root = (project_root / "analysis").resolve()
        if analysis_root not in output.parents:
            raise SFTError("Token audit output must be a versioned file below analysis/.")
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
        )
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(output)
    except (SFTError, OSError) as exc:
        print(f"E001 token audit failed: {exc}", file=sys.stderr)
        return 2
    print(output)
    print(hashlib.sha256(payload).hexdigest())
    print(json.dumps(audit["total_token_length"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
