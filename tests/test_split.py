import csv
import hashlib
import json
from pathlib import Path

import pytest

from qwen_math_challenge.config import load_config
from qwen_math_challenge.data.official import sha256_file
from qwen_math_challenge.data.split import (
    SplitValidationError,
    create_duplicate_groups,
    deterministic_group_split,
    load_split_settings,
    read_clean_source,
    run_split_pipeline,
)


def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _default_rows(count: int = 30) -> list[list[str]]:
    rows = []
    answers = ["-7", "0", "11"]
    for index in range(count):
        marker = hashlib.sha256(f"fixture-{index}".encode()).hexdigest()
        rows.append(
            [
                f"train-{index:03d}",
                f"Standalone fixture narrative {marker} has independently distinct wording.",
                answers[index % len(answers)],
            ]
        )
    return rows


def _build_fixture(
    tmp_path: Path,
    *,
    rows: list[list[str]] | None = None,
    columns: list[str] | None = None,
    output_dir: str = "data/splits/test_v001",
    target_val_ratio: float = 0.2,
    source_sha256: str | None = None,
) -> Path:
    rows = [row[:] for row in (rows if rows is not None else _default_rows())]
    columns = columns or ["id", "question", "answer"]
    source_path = tmp_path / "data" / "processed" / "test_v001" / "train_clean.csv"
    _write_csv(source_path, columns, rows)
    leaderboard_path = tmp_path / "data" / "raw" / "test_v001" / "leaderboard.csv"
    _write_csv(
        leaderboard_path,
        ["id", "question"],
        [["val-001", "A leaderboard-only question with no answer field."]],
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("fixture rules\n", encoding="utf-8")
    source_hash = source_sha256 or sha256_file(source_path)
    config_path = tmp_path / "configs" / "data" / "split_test_v001.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""schema_version: 1
experiment:
  experiment_id: test_phase2
  phase: 2
  dataset_version: test_v001
  seed: 2026
runtime:
  output_root: outputs
  log_level: INFO
  deterministic: true
split:
  split_version: test_v001_split_v001
  source_dataset_version: test_v001
  source:
    clean_path: data/processed/test_v001/train_clean.csv
    clean_sha256: "{source_hash}"
    row_count: {len(rows)}
    columns: [{", ".join(columns)}]
  leaderboard_audit:
    enabled: true
    path: data/raw/test_v001/leaderboard.csv
    sha256: "{sha256_file(leaderboard_path)}"
    row_count: 1
    columns: [id, question]
  output_dir: {output_dir}
  target_val_ratio: {target_val_ratio}
  allowed_val_ratio: [0.15, 0.25]
  normalization:
    version: test_normalization_v1
    nfkc: true
    casefold: true
    collapse_whitespace: true
    normalize_math_whitespace: true
    number_placeholder: "<num>"
  near_duplicate:
    enabled: true
    algorithm: rare_token_bigram_blocking_char5_union_find
    version: test_grouping_v1
    character_ngram_size: 5
    rare_token_ngrams_per_document: 8
    max_token_ngram_document_frequency: 32
    max_block_size: 64
    min_template_characters: 15
    min_length_ratio: 0.50
    calibration_thresholds: [0.82, 0.90, 0.94]
    review_threshold: 0.82
    grouping_threshold: 0.90
    threshold_rationale: "Synthetic fixture uses a conservative test threshold."
    metric_weights:
      template_character_jaccard: 0.50
      template_token_bigram_jaccard: 0.25
      template_sequence_ratio: 0.25
    representative_pairs_per_threshold: 3
  category:
    enabled: true
    version: conservative_keyword_v1
    use_for_split: false
    unknown_allowed: true
""",
        encoding="utf-8",
    )
    return config_path


def _settings_and_rows(config_path: Path, project_root: Path):
    config = load_config(config_path)
    settings = load_split_settings(config, project_root)
    return settings, read_clean_source(settings)


def _read_output(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_duplicate_id_fails(tmp_path: Path) -> None:
    rows = _default_rows(10)
    rows.append(rows[0][:])
    config_path = _build_fixture(tmp_path, rows=rows)
    settings = load_split_settings(load_config(config_path), tmp_path)

    with pytest.raises(SplitValidationError, match="duplicate IDs"):
        read_clean_source(settings)


def test_exact_duplicate_questions_share_group(tmp_path: Path) -> None:
    rows = _default_rows(12)
    rows[1][1] = rows[0][1]
    config_path = _build_fixture(tmp_path, rows=rows)
    settings, clean_rows = _settings_and_rows(config_path, tmp_path)
    grouping = create_duplicate_groups(clean_rows, settings.normalization, settings.near_duplicate)

    assert grouping.group_by_index[0] == grouping.group_by_index[1]
    assert grouping.duplicate_audit["exact_question"]["groups"] == 1


def test_normalized_duplicate_questions_share_group(tmp_path: Path) -> None:
    rows = _default_rows(12)
    rows[0][1] = "Ｓｏｌｖｅ   X + 2 = 5"
    rows[1][1] = "solve x+2=5"
    config_path = _build_fixture(tmp_path, rows=rows)
    settings, clean_rows = _settings_and_rows(config_path, tmp_path)
    grouping = create_duplicate_groups(clean_rows, settings.normalization, settings.near_duplicate)

    assert grouping.group_by_index[0] == grouping.group_by_index[1]
    assert grouping.duplicate_audit["normalized_question"]["groups"] == 1


def test_number_changed_template_is_detected(tmp_path: Path) -> None:
    rows = _default_rows(12)
    rows[0][1] = "Solve 3x + 4 = 19 and report the integer x."
    rows[1][1] = "Solve 5x + 7 = 32 and report the integer x."
    config_path = _build_fixture(tmp_path, rows=rows)
    settings, clean_rows = _settings_and_rows(config_path, tmp_path)
    grouping = create_duplicate_groups(clean_rows, settings.normalization, settings.near_duplicate)

    assert grouping.group_by_index[0] == grouping.group_by_index[1]
    assert any("number_template" in pair.candidate_reasons for pair in grouping.candidates)


def test_clearly_different_questions_do_not_share_group(tmp_path: Path) -> None:
    rows = _default_rows(12)
    rows[0][1] = "Find the area of a triangle with base 6 and height 8."
    rows[1][1] = "How many prime divisors does 2310 have?"
    config_path = _build_fixture(tmp_path, rows=rows)
    settings, clean_rows = _settings_and_rows(config_path, tmp_path)
    grouping = create_duplicate_groups(clean_rows, settings.normalization, settings.near_duplicate)

    assert grouping.group_by_index[0] != grouping.group_by_index[1]


def test_every_group_is_wholly_in_one_split(tmp_path: Path) -> None:
    rows = _default_rows(30)
    rows[1][1] = rows[0][1]
    config_path = _build_fixture(tmp_path, rows=rows)
    result = run_split_pipeline(load_config(config_path), project_root=tmp_path)
    groups = _read_output(tmp_path / result.output_dir / "groups.csv")
    splits_by_group: dict[str, set[str]] = {}
    for row in groups:
        splits_by_group.setdefault(row["group_id"], set()).add(row["split"])

    assert all(len(splits) == 1 for splits in splits_by_group.values())


def test_train_val_id_intersection_is_empty(tmp_path: Path) -> None:
    result = run_split_pipeline(load_config(_build_fixture(tmp_path)), project_root=tmp_path)
    train = _read_output(tmp_path / result.output_dir / "train.csv")
    val = _read_output(tmp_path / result.output_dir / "val.csv")

    assert {row["id"] for row in train}.isdisjoint(row["id"] for row in val)


def test_train_val_group_intersection_is_empty(tmp_path: Path) -> None:
    result = run_split_pipeline(load_config(_build_fixture(tmp_path)), project_root=tmp_path)
    groups = _read_output(tmp_path / result.output_dir / "groups.csv")
    train_groups = {row["group_id"] for row in groups if row["split"] == "train"}
    val_groups = {row["group_id"] for row in groups if row["split"] == "validation"}

    assert train_groups.isdisjoint(val_groups)


def test_same_seed_and_config_produce_identical_split_hashes(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    first = run_split_pipeline(load_config(config_path), project_root=tmp_path)
    second = run_split_pipeline(load_config(config_path), project_root=tmp_path)

    assert first.train_sha256 == second.train_sha256
    assert first.val_sha256 == second.val_sha256
    assert first.groups_sha256 == second.groups_sha256
    assert first.manifest_sha256 == second.manifest_sha256


def test_different_seed_can_change_split() -> None:
    groups = {f"group-{index}": (index,) for index in range(40)}

    first = deterministic_group_split(groups, total_rows=40, target_val_ratio=0.2, seed=2026)
    second = deterministic_group_split(groups, total_rows=40, target_val_ratio=0.2, seed=2027)

    assert first != second


def test_target_validation_ratio_is_close(tmp_path: Path) -> None:
    result = run_split_pipeline(load_config(_build_fixture(tmp_path)), project_root=tmp_path)

    assert result.actual_val_ratio == pytest.approx(0.2)


def test_source_sha_mismatch_is_detected(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path, source_sha256="0" * 64)
    settings = load_split_settings(load_config(config_path), tmp_path)

    with pytest.raises(SplitValidationError, match="SHA-256 mismatch"):
        read_clean_source(settings)


@pytest.mark.parametrize(
    "unsafe_output",
    ["data/raw/test_v001/split", "data/processed/test_v001/split"],
)
def test_output_may_not_overwrite_raw_or_processed_source(
    tmp_path: Path, unsafe_output: str
) -> None:
    config_path = _build_fixture(tmp_path, output_dir=unsafe_output)

    with pytest.raises(SplitValidationError, match="may not be data/raw, data/processed"):
        load_split_settings(load_config(config_path), tmp_path)


def test_negative_and_zero_answers_are_preserved(tmp_path: Path) -> None:
    result = run_split_pipeline(load_config(_build_fixture(tmp_path)), project_root=tmp_path)
    train = _read_output(tmp_path / result.output_dir / "train.csv")
    val = _read_output(tmp_path / result.output_dir / "val.csv")
    answers = {row["answer"] for row in [*train, *val]}

    assert "-7" in answers
    assert "0" in answers


def test_manifest_records_algorithm_threshold_seed_and_source(tmp_path: Path) -> None:
    result = run_split_pipeline(load_config(_build_fixture(tmp_path)), project_root=tmp_path)
    manifest = json.loads(
        (tmp_path / result.output_dir / "split_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["grouping_algorithm"] == "rare_token_bigram_blocking_char5_union_find"
    assert manifest["similarity_threshold"] == 0.9
    assert manifest["seed"] == 2026
    assert manifest["source_clean_sha256"] == sha256_file(
        tmp_path / "data" / "processed" / "test_v001" / "train_clean.csv"
    )


def test_near_duplicate_candidate_report_is_created(tmp_path: Path) -> None:
    rows = _default_rows(20)
    rows[0][1] = "Solve 3x + 4 = 19 and report the integer x."
    rows[1][1] = "Solve 5x + 7 = 32 and report the integer x."
    result = run_split_pipeline(
        load_config(_build_fixture(tmp_path, rows=rows)), project_root=tmp_path
    )
    candidates = _read_output(tmp_path / result.output_dir / "near_duplicate_candidates.csv")

    assert candidates
    assert {candidates[0]["left_id"], candidates[0]["right_id"]} == {
        "train-000",
        "train-001",
    }


@pytest.mark.parametrize(
    ("columns", "rows", "message"),
    [
        (["id", "question"], [["train-1", "missing answer"]], "columns must be"),
        (["id", "question", "answer"], [], "source.row_count.*integer"),
        (["id", "question", "answer"], [["train-1", "", "1"]], "empty required"),
    ],
)
def test_empty_or_malformed_source_fails(
    tmp_path: Path, columns: list[str], rows: list[list[str]], message: str
) -> None:
    config_path = _build_fixture(tmp_path, columns=columns, rows=rows)
    with pytest.raises(SplitValidationError, match=message):
        settings = load_split_settings(load_config(config_path), tmp_path)
        read_clean_source(settings)


def test_official_v001_source_contract_if_available() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "data" / "processed" / "official_v001" / "train_clean.csv"
    if not source.is_file():
        pytest.skip("Ignored official clean payload is not present in this environment.")
    config = load_config(project_root / "configs" / "data" / "split_official_v001.yaml")
    settings = load_split_settings(config, project_root)
    rows = read_clean_source(settings)

    assert len(rows) == 16_373
    assert sha256_file(source) == "4c9d3646e4bf06078122ca4436b1a05b916f267d6c61f2040b9bfba891410d1c"
