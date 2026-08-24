"""Inspect and prepare the versioned official dataset for Phase 1."""

from __future__ import annotations

import argparse

from qwen_math_challenge.config import load_config
from qwen_math_challenge.data import run_official_data_pipeline
from qwen_math_challenge.run_context import start_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a Phase 1 data YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    with start_run(config) as run:
        result = run_official_data_pipeline(config, project_root=run.project_root)
        run.write_json_artifact("phase1_result.json", result.as_dict())
        run.record_metrics(
            {
                "raw_train_rows": result.raw_train_rows,
                "mandatory_exclusion_rows": result.mandatory_exclusion_rows,
                "clean_train_rows": result.clean_train_rows,
                "leaderboard_rows": result.leaderboard_rows,
                "clean_train_sha256": result.clean_train_sha256,
            }
        )
        run.logger.info("Clean train rows: %d", result.clean_train_rows)
        run.logger.info("Clean train SHA-256: %s", result.clean_train_sha256)
        run_dir = run.run_dir

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
