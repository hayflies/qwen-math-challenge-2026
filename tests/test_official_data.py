import csv
import json
from pathlib import Path

import pytest

from qwen_math_challenge.config import load_config
from qwen_math_challenge.data.official import (
    DatasetValidationError,
    run_official_data_pipeline,
    sha256_file,
)

DEFAULT_TRAIN = [
    ["train-1", "Alpha question with enough text", "1"],
    ["train-2", "Beta question\nwith formatting", "-2"],
    ["train-3", "Gamma question with enough text", "0"],
    ["train-4", "Delta question with enough text", "42"],
]
DEFAULT_LEADERBOARD = [
    ["val-1", "Leaderboard question one"],
    ["val-2", "Leaderboard question two"],
]
DEFAULT_FILTERED = [["train-2", "-2", "Beta question with formatting"]]


def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _build_fixture(
    tmp_path: Path,
    *,
    train_rows: list[list[str]] | None = None,
    train_columns: list[str] | None = None,
    leaderboard_rows: list[list[str]] | None = None,
    filtered_rows: list[list[str]] | None = None,
    processed_dir: str = "data/processed/test_v001",
    expected_clean_rows: int | None = None,
) -> Path:
    train_rows = train_rows if train_rows is not None else [row[:] for row in DEFAULT_TRAIN]
    train_columns = train_columns or ["id", "question", "answer"]
    leaderboard_rows = (
        leaderboard_rows
        if leaderboard_rows is not None
        else [row[:] for row in DEFAULT_LEADERBOARD]
    )
    filtered_rows = (
        filtered_rows if filtered_rows is not None else [row[:] for row in DEFAULT_FILTERED]
    )

    raw_dir = tmp_path / "data" / "raw" / "test_v001"
    train_path = raw_dir / "train.csv"
    leaderboard_path = raw_dir / "leaderboard.csv"
    filtered_path = raw_dir / "filtered.csv"
    _write_csv(train_path, train_columns, train_rows)
    _write_csv(leaderboard_path, ["id", "question"], leaderboard_rows)
    _write_csv(filtered_path, ["id", "answer", "question"], filtered_rows)

    manifest_path = tmp_path / "data" / "manifests" / "test_v001_raw_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "dataset_version": "test_v001",
                "files": {
                    "train": {
                        "path": "data/raw/test_v001/train.csv",
                        "sha256": sha256_file(train_path),
                    },
                    "leaderboard": {
                        "path": "data/raw/test_v001/leaderboard.csv",
                        "sha256": sha256_file(leaderboard_path),
                    },
                    "filtered_ids": {
                        "path": "data/raw/test_v001/filtered.csv",
                        "sha256": sha256_file(filtered_path),
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("fixture rules\n", encoding="utf-8")
    config_path = tmp_path / "configs" / "data" / "test_v001.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    clean_rows = (
        expected_clean_rows
        if expected_clean_rows is not None
        else len(train_rows) - len(filtered_rows)
    )
    config_path.write_text(
        f"""schema_version: 1
experiment:
  experiment_id: test_phase1
  phase: 1
  dataset_version: test_v001
  seed: 7
runtime:
  output_root: outputs
  log_level: INFO
  deterministic: true
dataset:
  dataset_version: test_v001
  raw_manifest: data/manifests/test_v001_raw_manifest.json
  paths:
    train: data/raw/test_v001/train.csv
    leaderboard: data/raw/test_v001/leaderboard.csv
    filtered_ids: data/raw/test_v001/filtered.csv
    processed_dir: {processed_dir}
  schema:
    train: [id, question, answer]
    leaderboard: [id, question]
    filtered_ids: [id, answer, question]
  invariants:
    raw_train_rows: {len(train_rows)}
    leaderboard_rows: {len(leaderboard_rows)}
    mandatory_exclusion_rows: {len(filtered_rows)}
    clean_train_rows: {clean_rows}
  filtering:
    mandatory_exclusion_role: filtered_ids
    key: id
    candidate_exclusion_sources: []
    apply_candidate_exclusions: false
  audit:
    unusually_short_question_chars: 5
    unusually_long_question_chars: 1000
    max_examples_per_reason: 5
""",
        encoding="utf-8",
    )
    return config_path


def _run(config_path: Path, project_root: Path):
    return run_official_data_pipeline(load_config(config_path), project_root=project_root)


def _clean_rows(project_root: Path) -> list[dict[str, str]]:
    path = project_root / "data" / "processed" / "test_v001" / "train_clean.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_valid_schema_and_manifest_are_generated(tmp_path: Path) -> None:
    result = _run(_build_fixture(tmp_path), tmp_path)
    audit = json.loads((tmp_path / result.audit_report_path).read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / result.dataset_manifest_path).read_text(encoding="utf-8"))

    assert result.raw_train_rows == 4
    assert result.mandatory_exclusion_rows == 1
    assert result.clean_train_rows == 3
    assert audit["tables"]["raw_train"]["dtypes"]["answer"] == "integer"
    assert audit["tables"]["leaderboard"]["columns"] == ["id", "question"]
    assert manifest["clean_train_sha256"] == result.clean_train_sha256
    assert manifest["filter_policy"]["candidate_exclusions_applied"] is False


def test_duplicate_id_is_detected(tmp_path: Path) -> None:
    rows = [row[:] for row in DEFAULT_TRAIN] + [DEFAULT_TRAIN[0][:]]
    config_path = _build_fixture(tmp_path, train_rows=rows)

    with pytest.raises(DatasetValidationError, match="duplicate IDs"):
        _run(config_path, tmp_path)


def test_missing_required_column_is_detected(tmp_path: Path) -> None:
    rows = [[row[0], row[1]] for row in DEFAULT_TRAIN]
    config_path = _build_fixture(
        tmp_path,
        train_rows=rows,
        train_columns=["id", "question"],
    )

    with pytest.raises(DatasetValidationError, match="columns must be"):
        _run(config_path, tmp_path)


def test_null_required_value_is_detected(tmp_path: Path) -> None:
    rows = [row[:] for row in DEFAULT_TRAIN]
    rows[0][1] = ""
    config_path = _build_fixture(tmp_path, train_rows=rows)

    with pytest.raises(DatasetValidationError, match="empty required values"):
        _run(config_path, tmp_path)


def test_filtered_id_missing_from_train_is_detected(tmp_path: Path) -> None:
    filtered = [["not-in-train", "7", "Missing"]]
    config_path = _build_fixture(tmp_path, filtered_rows=filtered)

    with pytest.raises(DatasetValidationError, match="missing from train"):
        _run(config_path, tmp_path)


def test_filtering_uses_only_id_not_question(tmp_path: Path) -> None:
    filtered = [["train-2", "-2", DEFAULT_TRAIN[0][1]]]
    _run(_build_fixture(tmp_path, filtered_rows=filtered), tmp_path)
    ids = {row["id"] for row in _clean_rows(tmp_path)}

    assert "train-1" in ids
    assert "train-2" not in ids


def test_question_formatting_difference_does_not_prevent_removal(tmp_path: Path) -> None:
    filtered = [["train-2", "-2", "Beta question with formatting"]]
    _run(_build_fixture(tmp_path, filtered_rows=filtered), tmp_path)
    ids = {row["id"] for row in _clean_rows(tmp_path)}

    assert "train-2" not in ids


def test_no_filtered_id_remains_in_clean_train(tmp_path: Path) -> None:
    _run(_build_fixture(tmp_path), tmp_path)
    clean_ids = {row["id"] for row in _clean_rows(tmp_path)}
    filtered_ids = {row[0] for row in DEFAULT_FILTERED}

    assert clean_ids.isdisjoint(filtered_ids)


def test_raw_files_are_not_modified(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    raw_dir = tmp_path / "data" / "raw" / "test_v001"
    hashes_before = {path.name: sha256_file(path) for path in raw_dir.glob("*.csv")}

    _run(config_path, tmp_path)

    hashes_after = {path.name: sha256_file(path) for path in raw_dir.glob("*.csv")}
    assert hashes_after == hashes_before


def test_output_under_raw_is_rejected(tmp_path: Path) -> None:
    config_path = _build_fixture(
        tmp_path,
        processed_dir="data/raw/test_v001/processed",
    )

    with pytest.raises(DatasetValidationError, match="data/raw"):
        _run(config_path, tmp_path)


def test_manifest_and_clean_hashes_are_reproducible(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    first = _run(config_path, tmp_path)
    first_manifest = json.loads(
        (tmp_path / first.dataset_manifest_path).read_text(encoding="utf-8")
    )
    second = _run(config_path, tmp_path)
    second_manifest = json.loads(
        (tmp_path / second.dataset_manifest_path).read_text(encoding="utf-8")
    )

    assert first.clean_train_sha256 == second.clean_train_sha256
    assert first_manifest["source_file_sha256"] == second_manifest["source_file_sha256"]
    assert first_manifest["raw_manifest"] == second_manifest["raw_manifest"]


def test_clean_row_count_invariant_is_enforced(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path, expected_clean_rows=99)

    with pytest.raises(DatasetValidationError, match="clean train expected 99"):
        _run(config_path, tmp_path)


def test_raw_hash_manifest_mismatch_is_detected(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    train_path = tmp_path / "data" / "raw" / "test_v001" / "train.csv"
    train_path.write_text("changed\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="SHA-256 mismatch"):
        _run(config_path, tmp_path)
