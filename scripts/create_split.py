"""Create the Phase 2 leakage-safe internal train/validation split."""

from __future__ import annotations

import argparse

from qwen_math_challenge.config import load_config
from qwen_math_challenge.data.split import run_split_pipeline
from qwen_math_challenge.run_context import start_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a Phase 2 split YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    with start_run(config) as run:
        result = run_split_pipeline(config, project_root=run.project_root)
        run.write_json_artifact("phase2_result.json", result.as_dict())
        run.record_metrics(result.as_dict())
        run.logger.info("Train rows: %d", result.train_rows)
        run.logger.info("Validation rows: %d", result.val_rows)
        run.logger.info("Validation ratio: %.8f", result.actual_val_ratio)
        run.logger.info("Train SHA-256: %s", result.train_sha256)
        run.logger.info("Validation SHA-256: %s", result.val_sha256)
        run_dir = run.run_dir

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
