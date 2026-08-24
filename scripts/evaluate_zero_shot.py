"""Run the Phase 3 E000 Qwen2.5-3B-Instruct zero-shot baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qwen_math_challenge.config import find_project_root, load_config
from qwen_math_challenge.evaluation.zero_shot import (
    ModelUnavailableError,
    TransformersBatchGenerator,
    build_run_manifest_fields,
    load_zero_shot_settings,
    resolve_local_model_snapshot,
    run_zero_shot_evaluation,
    select_device,
)
from qwen_math_challenge.run_context import start_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the E000 YAML config.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N validation rows.")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Compatible prior E000 run directory whose completed prefix should be reused.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    project_root = find_project_root(config.source_path)
    settings = load_zero_shot_settings(config, project_root)
    try:
        snapshot = resolve_local_model_snapshot(settings.model)
        device = select_device(settings.runtime.device, settings.runtime.dtype)
        generator = TransformersBatchGenerator(
            settings=settings,
            snapshot=snapshot,
            device_spec=device,
        )
    except ModelUnavailableError as exc:
        print(f"E000 model unavailable: {exc}", file=sys.stderr)
        return 2

    resume_from = str(args.resume.resolve()) if args.resume is not None else None
    with start_run(config, project_root=project_root) as run:
        result = run_zero_shot_evaluation(
            config,
            project_root=project_root,
            run_dir=run.run_dir,
            generator=generator,
            limit=args.limit,
            resume_from=args.resume,
        )
        run.manifest.update(
            build_run_manifest_fields(
                settings,
                generator,
                result,
                limit=args.limit,
                resumed_from=resume_from,
            )
        )
        for path in (
            result.predictions_path,
            result.failures_path,
            result.metrics_path,
            result.resume_identity_path,
        ):
            run.register_artifact(path.name)
        run.record_metrics(
            {
                "total": result.metrics["total"],
                "correct": result.metrics["correct"],
                "incorrect": result.metrics["incorrect"],
                "parse_failures": result.metrics["parse_failures"],
                "accuracy": result.metrics["accuracy"],
                "parse_failure_rate": result.metrics["parse_failure_rate"],
                "total_wall_clock_sec": result.metrics["total_wall_clock_sec"],
            }
        )
        run.logger.info("E000 accuracy: %.8f", result.metrics["accuracy"])
        run.logger.info("Parse failures: %d", result.metrics["parse_failures"])
        run_dir = run.run_dir

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
