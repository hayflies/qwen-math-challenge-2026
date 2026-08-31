import csv
import hashlib
import io
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from qwen_math_challenge.config import LoadedConfig, load_config
from qwen_math_challenge.evaluation.zero_shot import (
    DeviceSpec,
    GenerationSettings,
    ModelSettings,
    PromptSettings,
    RuntimeSettings,
    parse_integer_answer_v2,
)
from qwen_math_challenge.inference.reproduction import (
    FINAL_EMERGENCY_CODE_COMMIT,
    ExactReproductionExpectations,
    FrozenPredictionSource,
    reproduce_exact_final_submission,
)
from qwen_math_challenge.inference.shards import _shard_resume_identity, plan_final_shards
from qwen_math_challenge.inference.submission import (
    FinalDataSettings,
    FinalOutputSettings,
    FinalPrediction,
    FinalSubmissionError,
    FinalSubmissionSettings,
    _write_predictions,
    build_final_resume_identity,
    load_final_input,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "inference" / "e000_final_submission.yaml"
BIG_INTEGER = "123456789012345678901234567890123456789012345678901234567890"


class IdentityGenerator:
    model_commit = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    tokenizer_commit = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    chat_template = "frozen-test-chat-template"
    device_spec = DeviceSpec("cpu", "float32", "cpu")

    def runtime_metadata(self) -> dict[str, str]:
        return {"device": "cpu"}


@dataclass(frozen=True)
class ReproductionCase:
    config: LoadedConfig
    settings: FinalSubmissionSettings
    artifact_dir: Path
    expectations: ExactReproductionExpectations
    expected_payload: bytes
    expected_answers: dict[str, str]


def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _settings(tmp_path: Path, *, total: int, flags: int) -> FinalSubmissionSettings:
    test_path = tmp_path / "test_submission.csv"
    flag_path = tmp_path / "test_flag.csv"
    test_rows = [
        [f"test-{index:04d}", f"Question {index}\nPreserve exactly", ""] for index in range(total)
    ]
    _write_csv(test_path, ["id", "question", "answer"], test_rows)
    _write_csv(flag_path, ["id", "question"], [row[:2] for row in test_rows[-flags:]])
    return FinalSubmissionSettings(
        model=ModelSettings(
            "Qwen/Qwen2.5-3B-Instruct",
            "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            True,
            False,
        ),
        prompt=PromptSettings(
            "zero_shot_v001",
            "You solve math problems. End your response with the final answer as an integer.",
            "{question}\n\nSolve the problem and end with: Final answer: <integer>",
        ),
        generation=GenerationSettings("greedy_v001", False, 1, 1024, True),
        runtime=RuntimeSettings("auto", "auto", 1),
        data=FinalDataSettings(
            test_path,
            total,
            ("id", "question", "answer"),
            flag_path,
            flags,
            ("id", "question"),
        ),
        output=FinalOutputSettings(
            "predictions.csv",
            "resume_identity.json",
            "input_identity.json",
            "metrics.json",
            "submission.csv",
            "submission.sha256",
        ),
        parser_version="integer_v002_last_explicit_on_conflict",
        seed=2026,
        canonical_e000_config_sha256=(
            "f2d6d851d263466ee32fbdeabf70be326a0ff5a63f1e63e0376bf7bc10daaaea"
        ),
    )


def _prediction(source: object, answer: str) -> FinalPrediction:
    raw = f"Final answer: {answer}"
    parsed = parse_integer_answer_v2(raw)
    return FinalPrediction(
        source_index=source.index,
        sample_id=source.sample_id,
        question=source.question,
        question_sha256=source.question_sha256,
        raw_output=raw,
        parsed_answer=parsed.value,
        parse_status=parsed.status,
        input_tokens=12,
        output_tokens=4,
        latency_sec=0.25,
        finish_reason="eos_token",
        truncated=False,
        is_flagged=source.is_flagged,
    )


def _run_manifest(config: LoadedConfig, settings: FinalSubmissionSettings) -> dict[str, object]:
    return {
        "experiment_id": "FINAL_E000",
        "config_sha256": config.source_sha256,
        "checkpoint": f"{settings.model.name_or_path}@{settings.model.revision}",
        "parser_version": settings.parser_version,
        "lora_config": None,
        "git_commit": FINAL_EMERGENCY_CODE_COMMIT,
    }


def _expected_submission(
    settings: FinalSubmissionSettings,
    answers: dict[str, str],
    fallback: str,
) -> bytes:
    bundle = load_final_input(settings.data)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "question", "answer"])
    for row in bundle.rows:
        writer.writerow([row.sample_id, row.question, answers.get(row.sample_id, fallback)])
    return buffer.getvalue().encode("utf-8")


def _build_case(
    tmp_path: Path,
    *,
    total: int = 8,
    flags: int = 2,
    partial_rows: int = 3,
    shard_rows: tuple[int, int] = (2, 1),
) -> ReproductionCase:
    config = load_config(CONFIG_PATH)
    settings = _settings(tmp_path, total=total, flags=flags)
    bundle = load_final_input(settings.data)
    artifacts = tmp_path / "artifacts"
    partial_dir = artifacts / "partial_989"
    partial_records = [
        _prediction(row, BIG_INTEGER if index == 0 else "7")
        for index, row in enumerate(bundle.rows[:partial_rows])
    ]
    _write_predictions(partial_dir / "predictions.csv", partial_records)
    _write_json(partial_dir / "input_identity.json", bundle.identity)
    _write_json(
        partial_dir / "resume_identity.json",
        build_final_resume_identity(
            config,
            settings,
            IdentityGenerator(),
            bundle.identity,
            limit=None,
        ),
    )
    _write_json(partial_dir / "run_manifest.json", _run_manifest(config, settings))

    plan = plan_final_shards(config, settings, existing_run=partial_dir, num_shards=2)
    all_records = list(partial_records)
    source_specs = [FrozenPredictionSource("partial_989", "partial_989", partial_rows, None)]
    for shard_index, completed_rows in enumerate(shard_rows):
        source_dir = artifacts / f"shard_{shard_index}"
        records = [_prediction(row, "7") for row in plan.shards[shard_index][:completed_rows]]
        _write_predictions(source_dir / "predictions.csv", records)
        _write_json(source_dir / "input_identity.json", bundle.identity)
        _write_json(
            source_dir / "resume_identity.json",
            _shard_resume_identity(
                config,
                settings,
                IdentityGenerator(),
                plan,
                shard_index=shard_index,
            ),
        )
        _write_json(source_dir / "run_manifest.json", _run_manifest(config, settings))
        all_records.extend(records)
        source_specs.append(
            FrozenPredictionSource(
                f"shard_{shard_index}",
                f"shard_{shard_index}",
                completed_rows,
                shard_index,
            )
        )

    answers = {record.sample_id: str(record.parsed_answer) for record in all_records}
    payload = _expected_submission(settings, answers, "7")
    expectations = ExactReproductionExpectations(
        total_rows=total,
        flag_rows=flags,
        real_prediction_rows=len(all_records),
        fallback_rows=total - len(all_records),
        num_shards=2,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        emergency_code_commit=FINAL_EMERGENCY_CODE_COMMIT,
        sources=tuple(source_specs),
    )
    return ReproductionCase(config, settings, artifacts, expectations, payload, answers)


def _reproduce(case: ReproductionCase, output: Path) -> object:
    return reproduce_exact_final_submission(
        case.config,
        case.settings,
        artifact_directory=case.artifact_dir,
        output_directory=output,
        expectations=case.expectations,
    )


def test_exact_2000_ordering_big_integer_fallback_and_flags(tmp_path: Path) -> None:
    case = _build_case(
        tmp_path,
        total=2000,
        flags=120,
        partial_rows=989,
        shard_rows=(382, 332),
    )
    result = _reproduce(case, tmp_path / "result")

    assert result.real_prediction_rows == 1703
    assert result.fallback_rows == 297
    assert result.fallback_value == "7"
    assert result.submission_path.read_bytes() == case.expected_payload
    with result.submission_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == [f"test-{index:04d}" for index in range(2000)]
    assert rows[0]["answer"] == BIG_INTEGER
    assert str(int(rows[0]["answer"])) == BIG_INTEGER
    flagged_ids = {f"test-{index:04d}" for index in range(1880, 2000)}
    assert flagged_ids <= {row["id"] for row in rows}
    assert sum(row["id"] not in case.expected_answers for row in rows) == 297


def test_duplicate_prediction_id_is_rejected(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    shard0 = case.artifact_dir / "shard_0" / "predictions.csv"
    shard1 = case.artifact_dir / "shard_1" / "predictions.csv"
    from qwen_math_challenge.inference.submission import load_final_predictions

    duplicate = load_final_predictions(shard0)[0]
    _write_predictions(shard1, [duplicate])

    with pytest.raises(FinalSubmissionError, match="Duplicate prediction id"):
        _reproduce(case, tmp_path / "result")


def test_unexpected_missing_prediction_is_rejected(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    _write_predictions(case.artifact_dir / "shard_1" / "predictions.csv", [])

    with pytest.raises(FinalSubmissionError, match="must contain exactly 1 rows"):
        _reproduce(case, tmp_path / "result")


def test_non_integer_prediction_is_rejected_without_int64_conversion(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    path = case.artifact_dir / "shard_1" / "predictions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        columns = list(rows[0])
    rows[0]["parsed_answer"] = "42.0"
    _write_csv(path, columns, [[row[column] for column in columns] for row in rows])

    with pytest.raises(FinalSubmissionError, match="malformed typed fields"):
        _reproduce(case, tmp_path / "result")


def test_deterministic_fallback_produces_identical_bytes(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    first = _reproduce(case, tmp_path / "first")
    second = _reproduce(case, tmp_path / "second")

    assert first.fallback_value == second.fallback_value == "7"
    assert first.submission_path.read_bytes() == second.submission_path.read_bytes()
    assert first.sha256 == second.sha256 == case.expectations.expected_sha256


def test_checksum_mismatch_fails_before_writing(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    wrong = replace(case.expectations, expected_sha256="0" * 64)

    with pytest.raises(FinalSubmissionError, match="Exact submission SHA-256 mismatch"):
        reproduce_exact_final_submission(
            case.config,
            case.settings,
            artifact_directory=case.artifact_dir,
            output_directory=tmp_path / "result",
            expectations=wrong,
        )
    assert not (tmp_path / "result" / "submission.csv").exists()


def test_prediction_question_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    path = case.artifact_dir / "shard_1" / "predictions.csv"
    from qwen_math_challenge.inference.submission import load_final_predictions

    record = load_final_predictions(path)[0]
    _write_predictions(path, [replace(record, question="Different question")])

    with pytest.raises(FinalSubmissionError, match="source/parser integrity"):
        _reproduce(case, tmp_path / "result")


def test_missing_frozen_artifacts_fail_with_copy_instruction(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    missing_root = tmp_path / "not-copied"

    with pytest.raises(FinalSubmissionError, match="Copy the exact Kaggle artifact"):
        reproduce_exact_final_submission(
            case.config,
            case.settings,
            artifact_directory=missing_root,
            output_directory=tmp_path / "result",
            expectations=case.expectations,
        )
