"""Evaluate an E001 adapter with the canonical E000 protocol and paired comparison."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from qwen_math_challenge.analysis.e000 import extract_canonical_archive, sha256_file
from qwen_math_challenge.config import find_project_root, load_config
from qwen_math_challenge.evaluation.sft import (
    E001_EVALUATION_PIPELINE_VERSION,
    compare_e000_e001,
    load_e001_evaluation_settings,
)
from qwen_math_challenge.evaluation.zero_shot import (
    EvaluationError,
    ModelUnavailableError,
    TransformersBatchGenerator,
    build_run_manifest_fields,
    load_prediction_records,
    resolve_local_model_snapshot,
    run_zero_shot_evaluation,
    select_device,
)
from qwen_math_challenge.run_context import start_run
from qwen_math_challenge.training.sft import (
    SFTError,
    validate_adapter_compatibility,
    validate_official_split,
)

EXPECTED_E000_ARCHIVE_SHA256 = "dbb1110d42a6f153e2af47e10b458ef8a981717d3afba8f6f6c1c4d1c8e64e7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the canonical E001 config.")
    parser.add_argument("--adapter", required=True, type=Path, help="Final E001 adapter directory.")
    parser.add_argument(
        "--e000-archive",
        required=True,
        type=Path,
        help="Immutable canonical E000 artifact archive used for paired comparison.",
    )
    parser.add_argument("--limit", type=int, help="Non-canonical evaluation smoke prefix.")
    parser.add_argument("--resume", type=Path, help="Compatible prior E001 evaluation run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        project_root = find_project_root(config.source_path)
        sft_settings, evaluation_settings = load_e001_evaluation_settings(config, project_root)
        validate_official_split(sft_settings.data)
        adapter_validation = validate_adapter_compatibility(args.adapter, sft_settings)
        archive = args.e000_archive.expanduser().resolve()
        if sha256_file(archive) != EXPECTED_E000_ARCHIVE_SHA256:
            raise EvaluationError("Canonical E000 archive SHA-256 mismatch.")
        snapshot = resolve_local_model_snapshot(evaluation_settings.model)
        device = select_device(
            evaluation_settings.runtime.device,
            evaluation_settings.runtime.dtype,
        )
        generator = TransformersBatchGenerator(
            settings=evaluation_settings,
            snapshot=snapshot,
            device_spec=device,
            adapter_path=args.adapter,
        )
        if generator.adapter_identity is None:
            raise EvaluationError("E001 adapter was not loaded; refusing base-only evaluation.")
        generator.adapter_identity.update(adapter_validation)
        resumed_from = str(args.resume.resolve()) if args.resume is not None else None
        with start_run(config, project_root=project_root) as run:
            result = run_zero_shot_evaluation(
                config,
                project_root=project_root,
                run_dir=run.run_dir,
                generator=generator,
                limit=args.limit,
                resume_from=args.resume,
                settings=evaluation_settings,
                pipeline_version=E001_EVALUATION_PIPELINE_VERSION,
                experiment_id="E001",
            )
            comparison = None
            if args.limit is None:
                with tempfile.TemporaryDirectory(prefix="e001-e000-compare-") as temporary:
                    e000_run, _ = extract_canonical_archive(archive, Path(temporary))
                    e000_records = load_prediction_records(e000_run / "predictions.csv")
                    e001_records = load_prediction_records(result.predictions_path)
                    comparison = compare_e000_e001(e000_records, e001_records)
                run.write_json_artifact("comparison_e000.json", comparison)
            run.manifest.update(
                build_run_manifest_fields(
                    evaluation_settings,
                    generator,
                    result,
                    limit=args.limit,
                    resumed_from=resumed_from,
                    pipeline_version=E001_EVALUATION_PIPELINE_VERSION,
                    no_training=False,
                )
            )
            run.manifest.update(
                {
                    "adapter_loaded": True,
                    "adapter_sha256": adapter_validation["adapter_sha256"],
                    "adapter_training_identity_sha256": adapter_validation[
                        "training_identity_sha256"
                    ],
                    "canonical_e000_archive_sha256": EXPECTED_E000_ARCHIVE_SHA256,
                    "paired_comparison_completed": comparison is not None,
                }
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
                    "accuracy": result.metrics["accuracy"],
                    "parse_failures": result.metrics["parse_failures"],
                    "max_new_tokens_hits": result.metrics["max_new_tokens_hits"],
                    "e000_absolute_delta_percentage_points": (
                        None
                        if comparison is None
                        else comparison["absolute_delta_percentage_points"]
                    ),
                }
            )
            run_dir = run.run_dir
    except (EvaluationError, ModelUnavailableError, SFTError, OSError) as exc:
        print(f"E001 evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
