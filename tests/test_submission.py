import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from qwen_math_challenge.config import load_config
from qwen_math_challenge.evaluation.zero_shot import (
    DeviceSpec,
    GenerationResult,
    GenerationSettings,
    ModelSettings,
    PromptSettings,
    RuntimeSettings,
)
from qwen_math_challenge.inference.submission import (
    FinalDataSettings,
    FinalOutputSettings,
    FinalSubmissionError,
    FinalSubmissionSettings,
    load_final_input,
    load_final_predictions,
    load_final_submission_settings,
    run_final_submission,
    validate_submission_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_CONFIG = PROJECT_ROOT / "configs" / "inference" / "e000_final_submission.yaml"


def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _settings(tmp_path: Path) -> tuple[object, FinalSubmissionSettings]:
    test_path = tmp_path / "test_submission.csv"
    flag_path = tmp_path / "test_flag.csv"
    _write_csv(
        test_path,
        ["id", "question", "answer"],
        [
            ["test-1", "Question one", ""],
            ["test-2", "Question two", ""],
            ["test-3", "Question three", ""],
        ],
    )
    _write_csv(flag_path, ["id", "question"], [["test-2", "Question two"]])
    config = load_config(FINAL_CONFIG)
    settings = FinalSubmissionSettings(
        model=ModelSettings(
            name_or_path="Qwen/Qwen2.5-3B-Instruct",
            revision="canonical-revision",
            tokenizer_revision="canonical-revision",
            local_files_only=True,
            trust_remote_code=False,
        ),
        prompt=PromptSettings("zero_shot_v001", "System", "{question}"),
        generation=GenerationSettings("greedy_v001", False, 1, 1024, True),
        runtime=RuntimeSettings("auto", "auto", 1),
        data=FinalDataSettings(
            test_path=test_path,
            expected_rows=3,
            columns=("id", "question", "answer"),
            flag_path=flag_path,
            expected_flag_rows=1,
            flag_columns=("id", "question"),
        ),
        output=FinalOutputSettings(
            predictions_filename="predictions.csv",
            resume_identity_filename="resume_identity.json",
            input_identity_filename="input_identity.json",
            metrics_filename="metrics.json",
            submission_filename="submission.csv",
            submission_sha256_filename="submission.sha256",
        ),
        parser_version="integer_v002_last_explicit_on_conflict",
        seed=2026,
        canonical_e000_config_sha256="f2d6d851",
    )
    return config, settings


class FakeGenerator:
    model_commit = "canonical-revision"
    tokenizer_commit = "canonical-revision"
    chat_template = "canonical-chat-template"
    device_spec = DeviceSpec("cpu", "float32", "fake-cpu")

    def __init__(self, outputs: dict[str, str] | None = None, *, fail_after: int | None = None):
        self.outputs = outputs or {
            "Question one": "Final answer: 42",
            "Question two": "Final answer: 2\nFinal answer: -4",
            "Question three": "0",
        }
        self.fail_after = fail_after
        self.calls = 0

    def generate(self, questions: list[str]) -> list[GenerationResult]:
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("simulated interruption")
        self.calls += 1
        raw = self.outputs[questions[0]]
        return [
            GenerationResult(
                raw_output=raw,
                input_tokens=10,
                output_tokens=3,
                latency_sec=0.25,
                finish_reason="eos_token",
                truncated=False,
            )
        ]

    def runtime_metadata(self) -> dict[str, str]:
        return {"device": "cpu", "dtype": "float32"}


def test_final_config_is_exactly_anchored_to_canonical_e000() -> None:
    config = load_config(FINAL_CONFIG)
    settings = load_final_submission_settings(config, PROJECT_ROOT)

    assert settings.model.revision == "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    assert settings.prompt.version == "zero_shot_v001"
    assert settings.generation.max_new_tokens == 1024
    assert settings.generation.do_sample is False
    assert settings.parser_version == "integer_v002_last_explicit_on_conflict"
    assert settings.data.expected_rows == 2000
    assert settings.data.expected_flag_rows == 120


def test_official_inputs_preserve_all_rows_and_validate_flags(tmp_path: Path) -> None:
    _, settings = _settings(tmp_path)
    bundle = load_final_input(settings.data)

    assert [row.sample_id for row in bundle.rows] == ["test-1", "test-2", "test-3"]
    assert [row.is_flagged for row in bundle.rows] == [False, True, False]
    assert bundle.identity["test"]["rows"] == 3
    assert bundle.identity["flags"]["rows"] == 1
    assert bundle.identity["scoring_eligible_rows"] == 2


def test_flag_question_must_match_official_test_exactly(tmp_path: Path) -> None:
    _, settings = _settings(tmp_path)
    _write_csv(settings.data.flag_path, ["id", "question"], [["test-2", "Changed"]])

    with pytest.raises(FinalSubmissionError, match="does not match"):
        load_final_input(settings.data)


def test_nonempty_template_answer_is_rejected(tmp_path: Path) -> None:
    _, settings = _settings(tmp_path)
    _write_csv(
        settings.data.test_path,
        ["id", "question", "answer"],
        [
            ["test-1", "Question one", "42"],
            ["test-2", "Question two", ""],
            ["test-3", "Question three", ""],
        ],
    )

    with pytest.raises(FinalSubmissionError, match="must be empty"):
        load_final_input(settings.data)


def test_full_run_includes_flagged_rows_and_writes_strict_submission(tmp_path: Path) -> None:
    config, settings = _settings(tmp_path)
    result = run_final_submission(
        config,
        settings,
        run_dir=tmp_path / "run",
        generator=FakeGenerator(),
    )

    assert result.submission_path is not None
    assert result.submission_path.name == "submission.csv"
    assert result.submission_sha256_path is not None
    assert result.metrics["total_rows"] == 3
    assert result.metrics["flagged_rows"] == 1
    assert result.metrics["scoring_eligible_rows"] == 2
    assert result.metrics["parse_failures"] == 0
    with result.submission_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["id", "question", "answer"]
    assert [(row["id"], row["question"], row["answer"]) for row in rows] == [
        ("test-1", "Question one", "42"),
        ("test-2", "Question two", "-4"),
        ("test-3", "Question three", "0"),
    ]
    assert load_final_predictions(result.predictions_path)[1].is_flagged is True
    validate_submission_file(result.submission_path, load_final_input(settings.data).rows)


def test_limited_smoke_never_creates_official_submission(tmp_path: Path) -> None:
    config, settings = _settings(tmp_path)
    result = run_final_submission(
        config,
        settings,
        run_dir=tmp_path / "smoke",
        generator=FakeGenerator(),
        limit=1,
    )

    assert result.submission_path is None
    assert result.submission_sha256_path is None
    assert not (tmp_path / "smoke" / "submission.csv").exists()
    assert result.metrics["canonical_full_submission"] is False


def test_interrupted_full_run_resumes_only_the_missing_suffix(tmp_path: Path) -> None:
    config, settings = _settings(tmp_path)
    interrupted = tmp_path / "interrupted"
    first = FakeGenerator(fail_after=2)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_final_submission(config, settings, run_dir=interrupted, generator=first)
    assert len(load_final_predictions(interrupted / "predictions.csv")) == 2

    resumed_generator = FakeGenerator()
    result = run_final_submission(
        config,
        settings,
        run_dir=tmp_path / "resumed",
        generator=resumed_generator,
        resume_from=interrupted,
    )

    assert result.metrics["resumed_predictions"] == 2
    assert resumed_generator.calls == 1
    assert result.submission_path is not None


def test_resume_rejects_changed_test_identity(tmp_path: Path) -> None:
    config, settings = _settings(tmp_path)
    prior = run_final_submission(
        config,
        settings,
        run_dir=tmp_path / "prior",
        generator=FakeGenerator(),
    )
    _write_csv(
        settings.data.test_path,
        ["id", "question", "answer"],
        [
            ["test-1", "Question one changed", ""],
            ["test-2", "Question two", ""],
            ["test-3", "Question three", ""],
        ],
    )
    changed_settings = replace(settings, data=replace(settings.data))

    with pytest.raises(FinalSubmissionError, match="Resume identity mismatch"):
        run_final_submission(
            config,
            changed_settings,
            run_dir=tmp_path / "changed",
            generator=FakeGenerator({"Question one changed": "1"}),
            resume_from=prior.predictions_path.parent,
        )


def test_submission_rejects_noncanonical_answer(tmp_path: Path) -> None:
    _, settings = _settings(tmp_path)
    source = load_final_input(settings.data).rows
    path = tmp_path / "submission.csv"
    _write_csv(
        path,
        ["id", "question", "answer"],
        [
            ["test-1", "Question one", "42.0"],
            ["test-2", "Question two", "-4"],
            ["test-3", "Question three", "0"],
        ],
    )

    with pytest.raises(FinalSubmissionError, match="canonical integer"):
        validate_submission_file(path, source)


def test_resume_identity_records_data_model_config_parser_and_code(tmp_path: Path) -> None:
    config, settings = _settings(tmp_path)
    result = run_final_submission(
        config,
        settings,
        run_dir=tmp_path / "identity",
        generator=FakeGenerator(),
        limit=1,
    )
    identity = json.loads(result.resume_identity_path.read_text(encoding="utf-8"))["identity"]

    assert identity["input_identity"]["test"]["sha256"]
    assert identity["input_identity"]["test"]["ordered_id_sha256"]
    assert identity["input_identity"]["flags"]["sha256"]
    assert identity["model_revision"] == "canonical-revision"
    assert identity["generation"]["max_new_tokens"] == 1024
    assert identity["parser_version"] == "integer_v002_last_explicit_on_conflict"
    assert set(identity["code_sha256"]) == {"submission.py", "zero_shot.py"}
