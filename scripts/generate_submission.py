"""Run crash-safe canonical E000 inference for the official final-test CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qwen_math_challenge.config import find_project_root, load_config
from qwen_math_challenge.evaluation.zero_shot import (
    EvaluationError,
    ModelUnavailableError,
    TransformersBatchGenerator,
    resolve_local_model_snapshot,
    select_device,
)
from qwen_math_challenge.inference.submission import (
    FinalSubmissionError,
    final_manifest_fields,
    load_final_input,
    load_final_submission_settings,
    run_final_submission,
)
from qwen_math_challenge.run_context import start_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to final E000 YAML config.")
    parser.add_argument(
        "--test-file",
        type=Path,
        help="Official test_submission.csv path; overrides config data.test_path.",
    )
    parser.add_argument(
        "--flag-file",
        type=Path,
        help="Official test_flag.csv path; overrides config data.flag_path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Smoke-test prefix only. A limited run never creates submission.csv.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Prior compatible full-run directory whose completed prefix should be reused.",
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
        # Validate both official files before spending time loading model weights.
        load_final_input(settings.data)
        snapshot = resolve_local_model_snapshot(settings.model)
        device = select_device(settings.runtime.device, settings.runtime.dtype)
        generator = TransformersBatchGenerator(
            settings=settings,
            snapshot=snapshot,
            device_spec=device,
        )
        if generator.adapter_identity is not None:
            raise FinalSubmissionError("Final E000 must not load an adapter.")
        resume_from = str(args.resume.expanduser().resolve()) if args.resume else None
        with start_run(config, project_root=project_root) as run:
            result = run_final_submission(
                config,
                settings,
                run_dir=run.run_dir,
                generator=generator,
                limit=args.limit,
                resume_from=args.resume,
            )
            run.manifest.update(
                final_manifest_fields(
                    settings,
                    generator,
                    result,
                    resume_from=resume_from,
                )
            )
            for path in (
                result.predictions_path,
                result.input_identity_path,
                result.resume_identity_path,
                result.metrics_path,
                result.submission_path,
                result.submission_sha256_path,
            ):
                if path is not None:
                    run.register_artifact(path.name)
            run.record_metrics(
                {
                    "total_rows": result.metrics["total_rows"],
                    "flagged_rows": result.metrics["flagged_rows"],
                    "scoring_eligible_rows": result.metrics["scoring_eligible_rows"],
                    "parse_failures": result.metrics["parse_failures"],
                    "max_new_tokens_hits": result.metrics["max_new_tokens_hits"],
                    "canonical_full_submission": result.metrics["canonical_full_submission"],
                    "submission_sha256": result.metrics.get("submission_sha256"),
                }
            )
            run.logger.info(
                "Final rows=%d flagged=%d parse_failures=%d",
                result.metrics["total_rows"],
                result.metrics["flagged_rows"],
                result.metrics["parse_failures"],
            )
            run_dir = run.run_dir
    except (
        EvaluationError,
        FinalSubmissionError,
        ModelUnavailableError,
        OSError,
    ) as exc:
        print(f"Final submission inference failed: {exc}", file=sys.stderr)
        return 2

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
