import io
import tarfile
from pathlib import Path

import pytest

from qwen_math_challenge.analysis.e000 import (
    AnalysisError,
    Prediction,
    analyze_predictions,
    extract_canonical_archive,
)


def _prediction(
    sample_id: str,
    *,
    gold: int,
    parsed: int | None,
    output_tokens: int,
    finish_reason: str = "eos_token",
    parse_status: str = "parsed_explicit",
    category: str = "unknown",
) -> Prediction:
    return Prediction(
        sample_id=sample_id,
        question=f"Question {sample_id}",
        gold_answer=gold,
        raw_output="Final answer: 1",
        parsed_answer=parsed,
        stored_correct=parsed == gold,
        parse_status=parse_status,
        input_tokens=10,
        output_tokens=output_tokens,
        latency_sec=0.5,
        finish_reason=finish_reason,
        truncated=finish_reason == "max_new_tokens",
        derived_category=category,
    )


def test_analysis_metrics_are_based_on_stored_parsed_answer() -> None:
    predictions = [
        _prediction("a", gold=-1, parsed=-1, output_tokens=10, category="algebra"),
        _prediction("b", gold=0, parsed=2, output_tokens=20, category="algebra"),
        _prediction(
            "c",
            gold=3,
            parsed=4,
            output_tokens=30,
            finish_reason="max_new_tokens",
        ),
        _prediction(
            "d",
            gold=5,
            parsed=None,
            output_tokens=30,
            parse_status="parse_failure_conflict",
        ),
    ]

    result = analyze_predictions(predictions, {"b": ["suspect"]}, seed=2026)
    sign = {row["group"]: row for row in result["answer_sign_metrics"]}
    max_token = {row["group"]: row for row in result["max_token_metrics"]}

    assert sign["negative"]["accuracy"] == 1.0
    assert sign["zero"]["accuracy"] == 0.0
    assert max_token["max_token_hit"]["count"] == 1
    assert result["parse_failure_reason_counts"] == {"conflicting_final_answers": 1}
    assert result["phase1_suspect_validation_count"] == 1


def test_max_token_repetition_is_only_a_candidate_label() -> None:
    repeated = _prediction(
        "repeat",
        gold=1,
        parsed=2,
        output_tokens=30,
        finish_reason="max_new_tokens",
    )
    repeated = Prediction(
        **{
            **repeated.__dict__,
            "raw_output": "reasoning " + "28 " * 50,
        }
    )

    normal = _prediction("normal", gold=1, parsed=1, output_tokens=10)
    result = analyze_predictions([repeated, normal], {}, seed=2026)

    assert result["max_token_form_counts"] == {"repetitive_output_candidate": 1}


def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("../escape.txt")
        payload = b"unsafe"
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with tarfile.open(archive_path, "r:gz"):
        with pytest.raises(AnalysisError, match="Unsafe archive member"):
            extract_canonical_archive(archive_path, tmp_path / "out")


def test_archive_requires_all_canonical_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "incomplete.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("run/metrics.json")
        payload = b"{}"
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(AnalysisError, match="missing required artifacts"):
        extract_canonical_archive(archive_path, tmp_path / "out")
