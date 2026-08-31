"""Reconstruct and checksum the exact deadline submission without a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qwen_math_challenge.config import find_project_root, load_config
from qwen_math_challenge.inference.reproduction import reproduce_exact_final_submission
from qwen_math_challenge.inference.submission import (
    FinalSubmissionError,
    load_final_submission_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--flag-file", required=True, type=Path)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/final_submission"),
        help="Directory containing partial_989/, shard_0/, and shard_1/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_submission_reproduced"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        project_root = find_project_root(config.source_path)
        settings = load_final_submission_settings(
            config,
            project_root,
            test_path=args.test_file,
            flag_path=args.flag_file,
        )
        artifact_dir = (
            args.artifact_dir.resolve()
            if args.artifact_dir.is_absolute()
            else (project_root / args.artifact_dir).resolve()
        )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir.is_absolute()
            else (project_root / args.output_dir).resolve()
        )
        result = reproduce_exact_final_submission(
            config,
            settings,
            artifact_directory=artifact_dir,
            output_directory=output_dir,
        )
    except (FinalSubmissionError, OSError) as exc:
        print(f"Exact final-submission reproduction failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
