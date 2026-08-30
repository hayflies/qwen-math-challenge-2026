"""Train Phase 4 E001 official-only direct-answer QLoRA on the frozen split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qwen_math_challenge.config import find_project_root, load_config
from qwen_math_challenge.environment import collect_git_info
from qwen_math_challenge.reproducibility import seed_everything
from qwen_math_challenge.run_context import start_run
from qwen_math_challenge.training.sft import (
    SFTError,
    encode_sft_example,
    inspect_base_model_architecture,
    load_sft_settings,
    load_training_tokenizer,
    run_sft_training,
    training_manifest_fields,
    validate_length_audit,
    validate_official_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the E001 YAML config.")
    parser.add_argument("--limit", type=int, help="Smoke-only number of train/eval rows.")
    parser.add_argument("--max-steps", type=int, help="Smoke-only Trainer step override.")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Compatible checkpoint-N directory containing optimizer/scheduler state.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate split, tokenizer, masking, and architecture without loading weights.",
    )
    return parser.parse_args()


def _preflight(args: argparse.Namespace):
    config = load_config(args.config)
    project_root = find_project_root(config.source_path)
    settings = load_sft_settings(config, project_root)
    split = validate_official_split(settings.data)
    length_audit = validate_length_audit(settings)
    tokenizer, tokenizer_commit = load_training_tokenizer(settings)
    probes = [split.train[0], split.train[-1], split.validation[0]]
    encoded = [
        encode_sft_example(
            tokenizer,
            row,
            settings.prompt,
            max_seq_length=settings.training.max_seq_length,
        )
        for row in probes
    ]
    architecture = inspect_base_model_architecture(settings)
    preflight = {
        "status": "PASS",
        "split": split.report,
        "length_audit_sha256": settings.data.length_audit_sha256,
        "length_audit": length_audit["total_token_length"],
        "tokenizer_commit": tokenizer_commit,
        "probe_ids": [item.sample_id for item in encoded],
        "probe_prompt_tokens": [item.prompt_tokens for item in encoded],
        "probe_supervised_tokens": [item.supervised_tokens for item in encoded],
        "architecture": architecture,
    }
    return config, project_root, settings, split, tokenizer, tokenizer_commit, preflight


def main() -> int:
    args = parse_args()
    try:
        (
            config,
            project_root,
            settings,
            split,
            tokenizer,
            tokenizer_commit,
            preflight,
        ) = _preflight(args)
        if args.validate_only:
            if args.limit is not None or args.max_steps is not None or args.resume is not None:
                raise SFTError("--validate-only cannot be combined with training overrides.")
            print(json.dumps(preflight, indent=2, sort_keys=True))
            return 0
        canonical_full_run = args.limit is None and args.max_steps is None
        if canonical_full_run and settings.require_clean_git_for_full_run:
            git = collect_git_info(project_root)
            if git["commit"] is None or git["dirty"]:
                raise SFTError(
                    "Canonical full E001 requires a clean committed worktree; "
                    "smoke runs may use explicit --limit and --max-steps overrides."
                )
        seed_report = seed_everything(settings.seed, deterministic=config.deterministic)
        resume_from = str(args.resume.resolve()) if args.resume is not None else None
        with start_run(config, project_root=project_root) as run:
            run.write_json_artifact("preflight.json", preflight)
            run.write_json_artifact(
                "token_length_audit.json",
                validate_length_audit(settings),
            )
            run.write_json_artifact("seed_report.json", seed_report)
            metrics = run_sft_training(
                config,
                settings,
                split,
                tokenizer,
                tokenizer_commit=tokenizer_commit,
                run_dir=run.run_dir,
                limit=args.limit,
                max_steps=args.max_steps,
                resume_from=args.resume,
                git_commit=run.manifest["git_commit"],
                git_dirty=run.manifest["git_dirty"],
            )
            run.manifest.update(
                training_manifest_fields(
                    config,
                    settings,
                    split.report,
                    metrics,
                    tokenizer_commit=tokenizer_commit,
                    limit=args.limit,
                    max_steps=args.max_steps,
                    resume_from=resume_from,
                    project_root=project_root,
                )
            )
            for filename in (
                settings.output.training_metrics_filename,
                settings.output.telemetry_filename,
                settings.output.identity_filename,
            ):
                run.register_artifact(filename)
            run.manifest["artifacts"]["adapter"] = settings.output.adapter_directory
            run.record_metrics(
                {
                    "canonical_full_run": metrics["canonical_full_run"],
                    "global_step": metrics["global_step"],
                    "train_loss": metrics["train_metrics"].get("train_loss"),
                    "validation_loss": metrics["final_validation_metrics"].get("eval_loss"),
                    "elapsed_sec": metrics["elapsed_sec"],
                    "peak_gpu_allocated_bytes": metrics["peak_gpu_allocated_bytes"],
                }
            )
            run_dir = run.run_dir
    except (SFTError, OSError, RuntimeError, FloatingPointError) as exc:
        print(f"E001 training failed: {exc}", file=sys.stderr)
        return 2
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
