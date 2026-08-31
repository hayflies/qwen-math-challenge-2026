"""Plan, merge, and validate emergency final-inference shards without a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qwen_math_challenge.config import find_project_root, load_config
from qwen_math_challenge.inference.shards import (
    merge_final_shards,
    plan_final_shards,
    validate_merged_submission,
)
from qwen_math_challenge.inference.submission import (
    FinalSubmissionError,
    load_final_submission_settings,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--flag-file", required=True, type=Path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Validate and print a deterministic shard plan.")
    _common(plan)
    plan.add_argument("--existing-run", required=True, type=Path)
    plan.add_argument("--num-shards", required=True, type=int)

    merge = subparsers.add_parser("merge", help="Merge completed disjoint shard runs.")
    _common(merge)
    merge.add_argument("--existing-run", required=True, type=Path)
    merge.add_argument("--shard-run", required=True, action="append", type=Path)
    merge.add_argument("--output-dir", required=True, type=Path)

    validate = subparsers.add_parser("validate", help="Validate official submission output.")
    _common(validate)
    validate.add_argument("--submission", required=True, type=Path)
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
        if args.command == "plan":
            result = plan_final_shards(
                config,
                settings,
                existing_run=args.existing_run,
                num_shards=args.num_shards,
            ).report
        elif args.command == "merge":
            result = merge_final_shards(
                config,
                settings,
                existing_run=args.existing_run,
                shard_runs=tuple(args.shard_run),
                output_dir=args.output_dir,
            )
        else:
            result = validate_merged_submission(settings, args.submission)
    except (FinalSubmissionError, OSError) as exc:
        print(f"Final shard operation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
