import csv
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
from qwen_math_challenge.inference.shards import (
    merge_final_shards,
    plan_final_shards,
    run_final_submission_shard,
    validate_merged_submission,
)
from qwen_math_challenge.inference.submission import (
    FinalDataSettings,
    FinalOutputSettings,
    FinalSubmissionSettings,
    _write_predictions,
    load_final_predictions,
    run_final_submission,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "inference" / "e000_final_submission.yaml")


def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _settings(tmp_path: Path) -> FinalSubmissionSettings:
    test = tmp_path / "test_submission.csv"
    flags = tmp_path / "test_flag.csv"
    _write_csv(
        test,
        ["id", "question", "answer"],
        [[f"test-{index}", f"Question {index}", ""] for index in range(8)],
    )
    _write_csv(flags, ["id", "question"], [["test-6", "Question 6"]])
    return FinalSubmissionSettings(
        model=ModelSettings("Qwen/Qwen2.5-3B-Instruct", "rev", "rev", True, False),
        prompt=PromptSettings("zero_shot_v001", "System", "{question}"),
        generation=GenerationSettings("greedy_v001", False, 1, 1024, True),
        runtime=RuntimeSettings("auto", "auto", 1),
        data=FinalDataSettings(
            test,
            8,
            ("id", "question", "answer"),
            flags,
            1,
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
        canonical_e000_config_sha256="canonical-config",
    )


class FakeGenerator:
    model_commit = "rev"
    tokenizer_commit = "rev"
    chat_template = "chat"
    device_spec = DeviceSpec("cpu", "float32", "cpu")

    def __init__(self, fail_after: int | None = None):
        self.calls = 0
        self.fail_after = fail_after

    def generate(self, questions: list[str]) -> list[GenerationResult]:
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("stop")
        self.calls += 1
        answer = questions[0].split()[-1]
        return [GenerationResult(answer, 5, 2, 0.1, "eos_token", False)]

    def runtime_metadata(self) -> dict[str, str]:
        return {"device": "cpu"}


def _nonprefix_existing_run(tmp_path: Path, settings: FinalSubmissionSettings) -> Path:
    run = tmp_path / "original"
    with pytest.raises(RuntimeError, match="stop"):
        run_final_submission(
            CONFIG,
            settings,
            run_dir=run,
            generator=FakeGenerator(fail_after=3),
        )
    records = load_final_predictions(run / "predictions.csv")
    _write_predictions(run / "predictions.csv", (records[0], records[2]))
    return run


def test_id_membership_plan_and_two_shard_merge(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    original = _nonprefix_existing_run(tmp_path, settings)
    plan = plan_final_shards(CONFIG, settings, existing_run=original, num_shards=2)

    assert [record.sample_id for record in plan.existing_predictions] == ["test-0", "test-2"]
    assert plan.report["existing_completed"] == 2
    assert plan.report["remaining"] == 6
    assert plan.report["shard_counts"] == {"0": 3, "1": 3}
    assert plan.report["intersection_existing_shards"] == {"0": 0, "1": 0}
    assert plan.report["intersection_between_shards"] == {"0:1": 0}
    assert plan.report["union_count"] == 8
    assert plan.report["all_official_ids_covered"] is True

    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    run_final_submission_shard(
        CONFIG,
        settings,
        run_dir=shard0,
        generator=FakeGenerator(),
        existing_run=original,
        shard_index=0,
        num_shards=2,
    )
    run_final_submission_shard(
        CONFIG,
        settings,
        run_dir=shard1,
        generator=FakeGenerator(),
        existing_run=original,
        shard_index=1,
        num_shards=2,
    )
    merged = tmp_path / "merged"
    manifest = merge_final_shards(
        CONFIG,
        settings,
        existing_run=original,
        shard_runs=(shard1, shard0),
        output_dir=merged,
    )

    assert manifest["total_rows"] == 8
    assert manifest["flagged_rows"] == 1
    assert manifest["parse_failures"] == 0
    result = validate_merged_submission(settings, merged / "submission.csv")
    assert result["status"] == "PASS"
    with (merged / "submission.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == [f"test-{index}" for index in range(8)]
    assert [row["answer"] for row in rows] == [str(index) for index in range(8)]


def test_shard_resume_reuses_only_its_completed_prefix(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    original = _nonprefix_existing_run(tmp_path, settings)
    interrupted = tmp_path / "interrupted-shard"
    with pytest.raises(RuntimeError, match="stop"):
        run_final_submission_shard(
            CONFIG,
            settings,
            run_dir=interrupted,
            generator=FakeGenerator(fail_after=1),
            existing_run=original,
            shard_index=0,
            num_shards=2,
        )
    assert len(load_final_predictions(interrupted / "predictions.csv")) == 1

    generator = FakeGenerator()
    result = run_final_submission_shard(
        CONFIG,
        settings,
        run_dir=tmp_path / "resumed-shard",
        generator=generator,
        existing_run=original,
        shard_index=0,
        num_shards=2,
        resume_from=interrupted,
    )
    assert result.metrics["resumed_predictions"] == 1
    assert generator.calls == 2
