"""Run a Phase 0 infrastructure smoke test without loading data or a model."""

from __future__ import annotations

import argparse

from qwen_math_challenge.config import load_config
from qwen_math_challenge.reproducibility import deterministic_probe, seed_everything
from qwen_math_challenge.run_context import start_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a Phase 0 YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    with start_run(config) as run:
        seed_report = seed_everything(config.seed, deterministic=config.deterministic)
        probe = deterministic_probe()
        run.write_json_artifact(
            "smoke_result.json",
            {
                "experiment_id": config.experiment_id,
                "phase": config.phase,
                "seed_report": seed_report,
                "deterministic_probe": probe,
            },
        )
        run.record_metrics(
            {
                "smoke_test_passed": True,
                "deterministic_probe_sha256": probe["sha256"],
            }
        )
        run.logger.info("Deterministic probe SHA-256: %s", probe["sha256"])
        run_dir = run.run_dir

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
