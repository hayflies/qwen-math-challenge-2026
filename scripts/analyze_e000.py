"""Validate and analyze the immutable canonical E000 archive without re-inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qwen_math_challenge.analysis.e000 import AnalysisError, analyze_canonical_archive

EXPECTED_ARCHIVE_SHA256 = "dbb1110d42a6f153e2af47e10b458ef8a981717d3afba8f6f6c1c4d1c8e64e7b"
PHASE3_FREEZE_COMMIT = "9d77878e7c7802c97825f6757242529a943f4a0b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("E000_20260828_canonical_artifacts.tar.gz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/e000"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = analyze_canonical_archive(
            archive_path=args.archive,
            expected_archive_sha256=EXPECTED_ARCHIVE_SHA256,
            freeze_record_path="experiments/e000_zero_shot.yaml",
            validation_path="data/splits/official_v001/val.csv",
            clean_train_path="data/processed/official_v001/train_clean.csv",
            phase1_config_path="configs/data/official_v001.yaml",
            phase1_audit_path="data/processed/official_v001/audit_report.json",
            output_dir=args.output_dir,
            freeze_commit=PHASE3_FREEZE_COMMIT,
            seed=2026,
        )
    except (AnalysisError, OSError) as exc:
        print(f"E000 analysis failed: {exc}", file=sys.stderr)
        return 2
    print(args.output_dir.resolve())
    print(result["predictions_integrity"]["recomputed_accuracy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
