"""Phase 2 text-only duplicate grouping and deterministic group-safe splitting."""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

from qwen_math_challenge.config import LoadedConfig
from qwen_math_challenge.data.official import sha256_file
from qwen_math_challenge.environment import collect_git_info

PIPELINE_VERSION = "phase2_split_v1"
_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_MATH_WHITESPACE_PATTERN = re.compile(r"\s*([=+*/^<>≤≥(),{}\[\]])\s*")
_NUMBER_PATTERN = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)")
_TOKEN_PATTERN = re.compile(r"<num>|\\[a-z]+|[a-z]+|\w+|[^\w\s]", re.IGNORECASE)


class SplitValidationError(ValueError):
    """Raised when Phase 2 input, configuration, or invariants are invalid."""


@dataclass(frozen=True)
class CleanRow:
    index: int
    sample_id: str
    question: str
    answer_text: str
    answer: int


@dataclass(frozen=True)
class NormalizationSettings:
    version: str
    nfkc: bool
    casefold: bool
    collapse_whitespace: bool
    normalize_math_whitespace: bool
    number_placeholder: str


@dataclass(frozen=True)
class NearDuplicateSettings:
    enabled: bool
    algorithm: str
    version: str
    character_ngram_size: int
    rare_token_ngrams_per_document: int
    max_token_ngram_document_frequency: int
    max_block_size: int
    min_template_characters: int
    min_length_ratio: float
    calibration_thresholds: tuple[float, ...]
    review_threshold: float
    grouping_threshold: float
    threshold_rationale: str
    character_weight: float
    token_weight: float
    sequence_weight: float
    representative_pairs_per_threshold: int


@dataclass(frozen=True)
class SplitSettings:
    split_version: str
    source_dataset_version: str
    source_path: Path
    source_sha256: str
    source_row_count: int
    source_columns: tuple[str, ...]
    leaderboard_audit_enabled: bool
    leaderboard_path: Path
    leaderboard_sha256: str
    leaderboard_row_count: int
    leaderboard_columns: tuple[str, ...]
    output_dir: Path
    target_val_ratio: float
    allowed_val_ratio: tuple[float, float]
    seed: int
    normalization: NormalizationSettings
    near_duplicate: NearDuplicateSettings
    category_enabled: bool
    category_version: str
    category_use_for_split: bool


@dataclass(frozen=True)
class CandidatePair:
    left: int
    right: int
    score: float
    base_character_jaccard: float
    template_character_jaccard: float
    template_token_bigram_jaccard: float
    template_sequence_ratio: float
    candidate_reasons: tuple[str, ...]


@dataclass(frozen=True)
class GroupingResult:
    group_by_index: tuple[str, ...]
    group_members: dict[str, tuple[int, ...]]
    candidates: tuple[CandidatePair, ...]
    duplicate_audit: dict[str, Any]
    candidate_generation: dict[str, Any]
    threshold_calibration: list[dict[str, Any]]


@dataclass(frozen=True)
class Phase2Result:
    split_version: str
    train_rows: int
    val_rows: int
    actual_val_ratio: float
    total_groups: int
    largest_group_size: int
    train_sha256: str
    val_sha256: str
    groups_sha256: str
    candidates_sha256: str
    report_sha256: str
    manifest_sha256: str
    output_dir: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "split_version": self.split_version,
            "train_rows": self.train_rows,
            "val_rows": self.val_rows,
            "actual_val_ratio": self.actual_val_ratio,
            "total_groups": self.total_groups,
            "largest_group_size": self.largest_group_size,
            "train_sha256": self.train_sha256,
            "val_sha256": self.val_sha256,
            "groups_sha256": self.groups_sha256,
            "candidates_sha256": self.candidates_sha256,
            "report_sha256": self.report_sha256,
            "manifest_sha256": self.manifest_sha256,
            "output_dir": self.output_dir,
        }


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SplitValidationError(f"'{label}' must be a mapping.")
    return dict(value)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SplitValidationError(f"'{label}' must be a non-empty string.")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise SplitValidationError(f"'{label}' must be a boolean.")
    return value


def _require_int(value: object, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SplitValidationError(f"'{label}' must be an integer >= {minimum}.")
    return value


def _require_ratio(value: object, label: str, *, include_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SplitValidationError(f"'{label}' must be a numeric ratio.")
    ratio = float(value)
    lower_valid = ratio >= 0 if include_zero else ratio > 0
    if not lower_valid or ratio > 1:
        lower = "0" if include_zero else "0 (exclusive)"
        raise SplitValidationError(f"'{label}' must be between {lower} and 1.")
    return ratio


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise SplitValidationError(f"'{label}' must be a lowercase SHA-256 hex digest.")
    return value


def _require_columns(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise SplitValidationError(f"'{label}' must be a non-empty list of column names.")
    if len(set(value)) != len(value):
        raise SplitValidationError(f"'{label}' must not contain duplicates.")
    return tuple(value)


def _resolve_path(root: Path, value: object, label: str) -> Path:
    text = _require_string(value, label)
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def load_split_settings(config: LoadedConfig, project_root: str | Path) -> SplitSettings:
    """Validate Phase 2 settings and reject source-overwriting output paths."""

    root = Path(project_root).resolve()
    split = _require_mapping(config.raw.get("split"), "split")
    source = _require_mapping(split.get("source"), "split.source")
    leaderboard = _require_mapping(split.get("leaderboard_audit"), "split.leaderboard_audit")
    normalization = _require_mapping(split.get("normalization"), "split.normalization")
    near = _require_mapping(split.get("near_duplicate"), "split.near_duplicate")
    category = _require_mapping(split.get("category"), "split.category")

    dataset_version = _require_string(
        split.get("source_dataset_version"), "split.source_dataset_version"
    )
    if dataset_version != config.experiment.get("dataset_version"):
        raise SplitValidationError(
            "split.source_dataset_version must match experiment.dataset_version."
        )
    if config.phase != 2:
        raise SplitValidationError("Phase 2 split config must set experiment.phase to 2.")

    source_path = _resolve_path(root, source.get("clean_path"), "split.source.clean_path")
    output_dir = _resolve_path(root, split.get("output_dir"), "split.output_dir")
    raw_root = (root / "data" / "raw").resolve()
    processed_root = (root / "data" / "processed").resolve()
    if _is_within(output_dir, raw_root) or _is_within(output_dir, processed_root):
        raise SplitValidationError(
            "split.output_dir may not be data/raw, data/processed, or their descendants."
        )
    if _is_within(source_path, output_dir) or source_path == output_dir:
        raise SplitValidationError("split.output_dir may not contain or overwrite the source.")

    allowed = split.get("allowed_val_ratio")
    if not isinstance(allowed, list) or len(allowed) != 2:
        raise SplitValidationError("split.allowed_val_ratio must contain [minimum, maximum].")
    allowed_ratio = (
        _require_ratio(allowed[0], "split.allowed_val_ratio[0]"),
        _require_ratio(allowed[1], "split.allowed_val_ratio[1]"),
    )
    target_ratio = _require_ratio(split.get("target_val_ratio"), "split.target_val_ratio")
    if not allowed_ratio[0] <= target_ratio <= allowed_ratio[1]:
        raise SplitValidationError("target_val_ratio must be inside allowed_val_ratio.")

    thresholds_value = near.get("calibration_thresholds")
    if not isinstance(thresholds_value, list) or not thresholds_value:
        raise SplitValidationError(
            "split.near_duplicate.calibration_thresholds must be a non-empty list."
        )
    thresholds = tuple(
        sorted(
            {
                _require_ratio(value, "split.near_duplicate.calibration_threshold")
                for value in thresholds_value
            }
        )
    )
    review_threshold = _require_ratio(
        near.get("review_threshold"), "split.near_duplicate.review_threshold"
    )
    grouping_threshold = _require_ratio(
        near.get("grouping_threshold"), "split.near_duplicate.grouping_threshold"
    )
    if review_threshold > grouping_threshold:
        raise SplitValidationError("review_threshold may not exceed grouping_threshold.")
    if grouping_threshold not in thresholds:
        raise SplitValidationError("grouping_threshold must appear in calibration_thresholds.")
    if min(thresholds) < review_threshold:
        raise SplitValidationError("calibration_thresholds may not be lower than review_threshold.")

    weights = _require_mapping(near.get("metric_weights"), "near_duplicate.metric_weights")
    weight_values = (
        _require_ratio(
            weights.get("template_character_jaccard"),
            "near_duplicate.metric_weights.template_character_jaccard",
            include_zero=True,
        ),
        _require_ratio(
            weights.get("template_token_bigram_jaccard"),
            "near_duplicate.metric_weights.template_token_bigram_jaccard",
            include_zero=True,
        ),
        _require_ratio(
            weights.get("template_sequence_ratio"),
            "near_duplicate.metric_weights.template_sequence_ratio",
            include_zero=True,
        ),
    )
    if abs(sum(weight_values) - 1.0) > 1e-9:
        raise SplitValidationError("near_duplicate metric weights must sum to 1.0.")

    normalization_settings = NormalizationSettings(
        version=_require_string(normalization.get("version"), "normalization.version"),
        nfkc=_require_bool(normalization.get("nfkc"), "normalization.nfkc"),
        casefold=_require_bool(normalization.get("casefold"), "normalization.casefold"),
        collapse_whitespace=_require_bool(
            normalization.get("collapse_whitespace"), "normalization.collapse_whitespace"
        ),
        normalize_math_whitespace=_require_bool(
            normalization.get("normalize_math_whitespace"),
            "normalization.normalize_math_whitespace",
        ),
        number_placeholder=_require_string(
            normalization.get("number_placeholder"), "normalization.number_placeholder"
        ),
    )
    near_settings = NearDuplicateSettings(
        enabled=_require_bool(near.get("enabled"), "near_duplicate.enabled"),
        algorithm=_require_string(near.get("algorithm"), "near_duplicate.algorithm"),
        version=_require_string(near.get("version"), "near_duplicate.version"),
        character_ngram_size=_require_int(
            near.get("character_ngram_size"), "near_duplicate.character_ngram_size", minimum=2
        ),
        rare_token_ngrams_per_document=_require_int(
            near.get("rare_token_ngrams_per_document"),
            "near_duplicate.rare_token_ngrams_per_document",
        ),
        max_token_ngram_document_frequency=_require_int(
            near.get("max_token_ngram_document_frequency"),
            "near_duplicate.max_token_ngram_document_frequency",
            minimum=2,
        ),
        max_block_size=_require_int(
            near.get("max_block_size"), "near_duplicate.max_block_size", minimum=2
        ),
        min_template_characters=_require_int(
            near.get("min_template_characters"),
            "near_duplicate.min_template_characters",
            minimum=1,
        ),
        min_length_ratio=_require_ratio(
            near.get("min_length_ratio"), "near_duplicate.min_length_ratio"
        ),
        calibration_thresholds=thresholds,
        review_threshold=review_threshold,
        grouping_threshold=grouping_threshold,
        threshold_rationale=_require_string(
            near.get("threshold_rationale"), "near_duplicate.threshold_rationale"
        ),
        character_weight=weight_values[0],
        token_weight=weight_values[1],
        sequence_weight=weight_values[2],
        representative_pairs_per_threshold=_require_int(
            near.get("representative_pairs_per_threshold"),
            "near_duplicate.representative_pairs_per_threshold",
        ),
    )

    category_enabled = _require_bool(category.get("enabled"), "category.enabled")
    category_use_for_split = _require_bool(category.get("use_for_split"), "category.use_for_split")
    if category_use_for_split:
        raise SplitValidationError(
            "Phase 2 conservative derived categories are report-only and may not stratify."
        )
    if not _require_bool(category.get("unknown_allowed"), "category.unknown_allowed"):
        raise SplitValidationError("Phase 2 category heuristics must allow unknown.")

    source_columns = _require_columns(source.get("columns"), "source.columns")
    if source_columns != ("id", "question", "answer"):
        raise SplitValidationError(
            "split.source.columns must be exactly ['id', 'question', 'answer']."
        )
    leaderboard_columns = _require_columns(leaderboard.get("columns"), "leaderboard_audit.columns")
    if leaderboard_columns != ("id", "question"):
        raise SplitValidationError(
            "split.leaderboard_audit.columns must be exactly ['id', 'question']; "
            "answer is forbidden."
        )

    return SplitSettings(
        split_version=_require_string(split.get("split_version"), "split.split_version"),
        source_dataset_version=dataset_version,
        source_path=source_path,
        source_sha256=_require_sha256(source.get("clean_sha256"), "source.clean_sha256"),
        source_row_count=_require_int(source.get("row_count"), "source.row_count"),
        source_columns=source_columns,
        leaderboard_audit_enabled=_require_bool(
            leaderboard.get("enabled"), "leaderboard_audit.enabled"
        ),
        leaderboard_path=_resolve_path(root, leaderboard.get("path"), "leaderboard_audit.path"),
        leaderboard_sha256=_require_sha256(leaderboard.get("sha256"), "leaderboard_audit.sha256"),
        leaderboard_row_count=_require_int(
            leaderboard.get("row_count"), "leaderboard_audit.row_count"
        ),
        leaderboard_columns=leaderboard_columns,
        output_dir=output_dir,
        target_val_ratio=target_ratio,
        allowed_val_ratio=allowed_ratio,
        seed=config.seed,
        normalization=normalization_settings,
        near_duplicate=near_settings,
        category_enabled=category_enabled,
        category_version=_require_string(category.get("version"), "category.version"),
        category_use_for_split=category_use_for_split,
    )


def normalize_question(question: str, settings: NormalizationSettings) -> str:
    """Conservatively normalize presentation without removing mathematical symbols."""

    value = unicodedata.normalize("NFKC", question) if settings.nfkc else question
    if settings.casefold:
        value = value.casefold()
    if settings.collapse_whitespace:
        value = _WHITESPACE_PATTERN.sub(" ", value).strip()
    if settings.normalize_math_whitespace:
        value = _MATH_WHITESPACE_PATTERN.sub(r"\1", value)
    return value


def number_template(question: str, settings: NormalizationSettings) -> str:
    """Replace numeric literals in the normalized representation with one placeholder."""

    return _NUMBER_PATTERN.sub(settings.number_placeholder, normalize_question(question, settings))


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(value))


def _token_bigrams(value: str) -> frozenset[str]:
    tokens = _tokens(value)
    if len(tokens) < 2:
        return frozenset(tokens)
    return frozenset(f"{left}\u241f{right}" for left, right in itertools.pairwise(tokens))


def _character_ngrams(value: str, size: int) -> frozenset[str]:
    if len(value) <= size:
        return frozenset({value}) if value else frozenset()
    return frozenset(value[index : index + size] for index in range(len(value) - size + 1))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union_size = len(left | right)
    return len(left & right) / union_size if union_size else 0.0


def _duplicate_summary(values: Sequence[str]) -> dict[str, Any]:
    counts = Counter(values)
    groups = sorted(count for count in counts.values() if count > 1)
    return {
        "groups": len(groups),
        "rows_in_duplicate_groups": sum(groups),
        "extra_rows": sum(count - 1 for count in groups),
        "largest_group_size": max(groups, default=1),
    }


def _read_csv(path: Path, expected_columns: Sequence[str], role: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise SplitValidationError(f"{role} file does not exist: {path}")
    source = path.read_bytes()
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SplitValidationError(f"{role} must be valid UTF-8.") from exc
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        columns = tuple(next(reader))
    except StopIteration as exc:
        raise SplitValidationError(f"{role} CSV is empty.") from exc
    if columns != tuple(expected_columns):
        raise SplitValidationError(
            f"{role} columns must be {list(expected_columns)}, got {list(columns)}."
        )
    rows: list[dict[str, str]] = []
    for logical_row, values in enumerate(reader, start=2):
        if len(values) != len(columns):
            raise SplitValidationError(
                f"{role} logical row {logical_row} has {len(values)} fields; "
                f"expected {len(columns)}."
            )
        rows.append(dict(zip(columns, values, strict=True)))
    if not rows:
        raise SplitValidationError(f"{role} CSV has no data rows.")
    return rows


def read_clean_source(settings: SplitSettings) -> tuple[CleanRow, ...]:
    """Verify source hash/schema/count and return validated immutable rows."""

    actual_hash = sha256_file(settings.source_path) if settings.source_path.is_file() else None
    if actual_hash != settings.source_sha256:
        raise SplitValidationError(
            f"clean source SHA-256 mismatch: expected {settings.source_sha256}, got {actual_hash}."
        )
    raw_rows = _read_csv(settings.source_path, settings.source_columns, "clean source")
    if len(raw_rows) != settings.source_row_count:
        raise SplitValidationError(
            f"clean source expected {settings.source_row_count} rows, got {len(raw_rows)}."
        )

    rows: list[CleanRow] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for index, row in enumerate(raw_rows):
        missing = [column for column in settings.source_columns if row[column] == ""]
        if missing:
            raise SplitValidationError(
                f"clean source row {index + 2} has empty required values: {missing}."
            )
        sample_id = row["id"]
        if sample_id in seen_ids:
            duplicate_ids.append(sample_id)
        seen_ids.add(sample_id)
        answer_text = row["answer"]
        if not _INTEGER_PATTERN.fullmatch(answer_text):
            raise SplitValidationError(
                f"clean source row {index + 2} answer is not an integer: {answer_text!r}."
            )
        rows.append(
            CleanRow(
                index=index,
                sample_id=sample_id,
                question=row["question"],
                answer_text=answer_text,
                answer=int(answer_text),
            )
        )
    if duplicate_ids:
        raise SplitValidationError(
            f"clean source has duplicate IDs: {sorted(set(duplicate_ids))[:20]}."
        )
    return tuple(rows)


def _add_bucket_pairs(
    buckets: Mapping[str, list[int]],
    reasons_by_pair: dict[tuple[int, int], set[str]],
    *,
    reason: str,
    max_block_size: int,
    lengths: Sequence[int],
    min_length_ratio: float,
) -> tuple[int, int]:
    accepted_blocks = 0
    skipped_blocks = 0
    for members in buckets.values():
        unique_members = sorted(set(members))
        if len(unique_members) < 2:
            continue
        if len(unique_members) > max_block_size:
            if reason in {"exact_question", "normalized_question"}:
                accepted_blocks += 1
                anchor = unique_members[0]
                for member in unique_members[1:]:
                    reasons_by_pair[(anchor, member)].add(reason)
            else:
                skipped_blocks += 1
            continue
        accepted_blocks += 1
        for left, right in itertools.combinations(unique_members, 2):
            longer = max(lengths[left], lengths[right])
            length_ratio = min(lengths[left], lengths[right]) / longer if longer else 1.0
            if reason != "exact_question" and length_ratio < min_length_ratio:
                continue
            reasons_by_pair[(left, right)].add(reason)
    return accepted_blocks, skipped_blocks


def _candidate_pairs(
    rows: Sequence[CleanRow],
    normalized: Sequence[str],
    templates: Sequence[str],
    template_token_bigrams: Sequence[frozenset[str]],
    settings: NearDuplicateSettings,
) -> tuple[dict[tuple[int, int], set[str]], dict[str, Any]]:
    reasons_by_pair: dict[tuple[int, int], set[str]] = defaultdict(set)
    lengths = [len(value) for value in templates]
    exact_buckets: dict[str, list[int]] = defaultdict(list)
    normalized_buckets: dict[str, list[int]] = defaultdict(list)
    template_buckets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        exact_buckets[row.question].append(index)
        normalized_buckets[normalized[index]].append(index)
        if len(templates[index]) >= settings.min_template_characters:
            template_buckets[templates[index]].append(index)

    block_stats: dict[str, dict[str, int]] = {}
    for reason, buckets in (
        ("exact_question", exact_buckets),
        ("normalized_question", normalized_buckets),
        ("number_template", template_buckets),
    ):
        accepted, skipped = _add_bucket_pairs(
            buckets,
            reasons_by_pair,
            reason=reason,
            max_block_size=settings.max_block_size,
            lengths=lengths,
            min_length_ratio=settings.min_length_ratio,
        )
        block_stats[reason] = {"accepted_blocks": accepted, "oversize_blocks_skipped": skipped}

    document_frequency: Counter[str] = Counter()
    for features in template_token_bigrams:
        document_frequency.update(features)
    rare_buckets: dict[str, list[int]] = defaultdict(list)
    for index, features in enumerate(template_token_bigrams):
        selected = sorted(
            (
                feature
                for feature in features
                if 2 <= document_frequency[feature] <= settings.max_token_ngram_document_frequency
            ),
            key=lambda feature: (document_frequency[feature], feature),
        )[: settings.rare_token_ngrams_per_document]
        for feature in selected:
            rare_buckets[feature].append(index)
    accepted, skipped = _add_bucket_pairs(
        rare_buckets,
        reasons_by_pair,
        reason="shared_rare_token_bigram",
        max_block_size=settings.max_block_size,
        lengths=lengths,
        min_length_ratio=settings.min_length_ratio,
    )
    block_stats["shared_rare_token_bigram"] = {
        "accepted_blocks": accepted,
        "oversize_blocks_skipped": skipped,
    }
    return reasons_by_pair, {
        "all_pairs_if_quadratic": len(rows) * (len(rows) - 1) // 2,
        "candidate_pairs": len(reasons_by_pair),
        "candidate_fraction_of_all_pairs": round(
            len(reasons_by_pair) / max(1, len(rows) * (len(rows) - 1) // 2), 10
        ),
        "unique_template_token_bigrams": len(document_frequency),
        "blocking": block_stats,
    }


def _evaluate_candidates(
    reasons_by_pair: Mapping[tuple[int, int], set[str]],
    normalized: Sequence[str],
    templates: Sequence[str],
    template_token_bigrams: Sequence[frozenset[str]],
    settings: NearDuplicateSettings,
) -> tuple[CandidatePair, ...]:
    base_char_cache: dict[int, frozenset[str]] = {}
    template_char_cache: dict[int, frozenset[str]] = {}

    def base_chars(index: int) -> frozenset[str]:
        if index not in base_char_cache:
            base_char_cache[index] = _character_ngrams(
                normalized[index], settings.character_ngram_size
            )
        return base_char_cache[index]

    def template_chars(index: int) -> frozenset[str]:
        if index not in template_char_cache:
            template_char_cache[index] = _character_ngrams(
                templates[index], settings.character_ngram_size
            )
        return template_char_cache[index]

    candidates: list[CandidatePair] = []
    for (left, right), reasons in sorted(reasons_by_pair.items()):
        base_score = _jaccard(base_chars(left), base_chars(right))
        template_score = _jaccard(template_chars(left), template_chars(right))
        token_score = _jaccard(template_token_bigrams[left], template_token_bigrams[right])
        sequence_score = SequenceMatcher(
            None, templates[left], templates[right], autojunk=False
        ).ratio()
        weighted_template_score = (
            settings.character_weight * template_score
            + settings.token_weight * token_score
            + settings.sequence_weight * sequence_score
        )
        score = max(base_score, weighted_template_score)
        if "exact_question" in reasons or "normalized_question" in reasons:
            score = 1.0
        elif "number_template" in reasons and templates[left] == templates[right]:
            score = 1.0
        if score + 1e-12 < settings.review_threshold:
            continue
        candidates.append(
            CandidatePair(
                left=left,
                right=right,
                score=score,
                base_character_jaccard=base_score,
                template_character_jaccard=template_score,
                template_token_bigram_jaccard=token_score,
                template_sequence_ratio=sequence_score,
                candidate_reasons=tuple(sorted(reasons)),
            )
        )
    return tuple(sorted(candidates, key=lambda pair: (-pair.score, pair.left, pair.right)))


def _components(
    rows: Sequence[CleanRow], candidates: Sequence[CandidatePair], threshold: float
) -> tuple[dict[str, tuple[int, ...]], tuple[str, ...]]:
    union_find = _UnionFind(len(rows))
    for pair in candidates:
        if pair.score + 1e-12 >= threshold:
            union_find.union(pair.left, pair.right)
    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        members_by_root[union_find.find(index)].append(index)

    group_members: dict[str, tuple[int, ...]] = {}
    group_by_index = [""] * len(rows)
    for members in members_by_root.values():
        ordered = tuple(sorted(members))
        member_ids = "\n".join(sorted(rows[index].sample_id for index in ordered))
        group_id = f"grp-{hashlib.sha256(member_ids.encode('utf-8')).hexdigest()[:16]}"
        group_members[group_id] = ordered
        for index in ordered:
            group_by_index[index] = group_id
    return dict(sorted(group_members.items())), tuple(group_by_index)


def _group_size_summary(group_members: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    sizes = [len(members) for members in group_members.values()]
    distribution = Counter(sizes)
    return {
        "total_groups": len(sizes),
        "singleton_groups": sum(size == 1 for size in sizes),
        "multi_member_groups": sum(size > 1 for size in sizes),
        "largest_group_size": max(sizes, default=0),
        "group_size_distribution": {str(size): distribution[size] for size in sorted(distribution)},
    }


def create_duplicate_groups(
    rows: Sequence[CleanRow],
    normalization: NormalizationSettings,
    settings: NearDuplicateSettings,
) -> GroupingResult:
    """Build exact/normalized/near-duplicate connected components for every row."""

    normalized = tuple(normalize_question(row.question, normalization) for row in rows)
    templates = tuple(number_template(row.question, normalization) for row in rows)
    token_bigrams = tuple(_token_bigrams(value) for value in templates)
    duplicate_audit = {
        "id": _duplicate_summary([row.sample_id for row in rows]),
        "exact_question": _duplicate_summary([row.question for row in rows]),
        "normalized_question": _duplicate_summary(normalized),
        "normalization_version": normalization.version,
        "mathematical_symbols_removed": False,
    }
    if duplicate_audit["id"]["groups"]:
        raise SplitValidationError("duplicate IDs must be rejected before grouping.")

    reasons_by_pair, generation = _candidate_pairs(
        rows, normalized, templates, token_bigrams, settings
    )
    candidates = (
        _evaluate_candidates(reasons_by_pair, normalized, templates, token_bigrams, settings)
        if settings.enabled
        else tuple()
    )
    calibration: list[dict[str, Any]] = []
    for threshold in settings.calibration_thresholds:
        threshold_groups, _ = _components(rows, candidates, threshold)
        pair_samples = sorted(
            (pair for pair in candidates if pair.score + 1e-12 >= threshold),
            key=lambda pair: (pair.score - threshold, pair.left, pair.right),
        )[: settings.representative_pairs_per_threshold]
        calibration.append(
            {
                "threshold": threshold,
                "candidate_pair_count": sum(pair.score + 1e-12 >= threshold for pair in candidates),
                **_group_size_summary(threshold_groups),
                "representative_pairs": [
                    {
                        "left_id": rows[pair.left].sample_id,
                        "right_id": rows[pair.right].sample_id,
                        "score": round(pair.score, 6),
                        "left_question": rows[pair.left].question,
                        "right_question": rows[pair.right].question,
                    }
                    for pair in pair_samples
                ],
            }
        )

    group_members, group_by_index = _components(rows, candidates, settings.grouping_threshold)
    return GroupingResult(
        group_by_index=group_by_index,
        group_members=group_members,
        candidates=candidates,
        duplicate_audit=duplicate_audit,
        candidate_generation=generation,
        threshold_calibration=calibration,
    )


def deterministic_group_split(
    group_members: Mapping[str, Sequence[int]],
    *,
    total_rows: int,
    target_val_ratio: float,
    seed: int,
) -> set[str]:
    """Choose whole groups using a seeded order and closest attainable target size."""

    target_rows = round(total_rows * target_val_ratio)
    group_ids = sorted(group_members)
    random.Random(seed).shuffle(group_ids)
    selected: set[str] = set()
    current_rows = 0
    for group_id in group_ids:
        size = len(group_members[group_id])
        if current_rows + size <= target_rows:
            selected.add(group_id)
            current_rows += size
        if current_rows == target_rows:
            break
    if current_rows != target_rows:
        remaining = [group_id for group_id in group_ids if group_id not in selected]
        if remaining:
            best = min(
                remaining,
                key=lambda group_id: (
                    abs(target_rows - (current_rows + len(group_members[group_id]))),
                    group_id,
                ),
            )
            if abs(target_rows - (current_rows + len(group_members[best]))) < abs(
                target_rows - current_rows
            ):
                selected.add(best)
    return selected


_CATEGORY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "geometry": tuple(
        re.compile(pattern)
        for pattern in (
            r"\btriangle\b",
            r"\bcircle\b",
            r"\brectangle\b",
            r"\bpolygon\b",
            r"\bangle\b",
            r"\bperimeter\b",
            r"\bhypotenuse\b",
            r"\borthocenter\b",
        )
    ),
    "probability": tuple(
        re.compile(pattern)
        for pattern in (r"\bprobability\b", r"\bodds\b", r"\brandomly\b", r"\bdice\b")
    ),
    "combinatorics": tuple(
        re.compile(pattern)
        for pattern in (r"\bpermutations?\b", r"\bcombinations?\b", r"\barrangements?\b")
    ),
    "calculus": tuple(
        re.compile(pattern)
        for pattern in (r"\bderivatives?\b", r"\bintegrals?\b", r"\bdifferentiat", r"\blimit as\b")
    ),
    "number_theory": tuple(
        re.compile(pattern)
        for pattern in (
            r"\bprime\b",
            r"\bdivisib",
            r"\bremainder\b",
            r"\bgreatest common divisor\b",
            r"\bleast common multiple\b",
            r"\bmodulo\b",
            r"\bcongruent\b",
        )
    ),
    "algebra": tuple(
        re.compile(pattern)
        for pattern in (
            r"\bpolynomial\b",
            r"\bequations?\b",
            r"\binequalit",
            r"\bfunctions?\b",
            r"\bsolve for\b",
            r"\bsystem of\b",
        )
    ),
    "word_problem": tuple(
        re.compile(pattern)
        for pattern in (
            r"\bdollars?\b",
            r"\bmiles?\b",
            r"\bgallons?\b",
            r"\binterest\b",
            r"\bper hour\b",
            r"\bminutes?\b",
            r"\btotal cost\b",
        )
    ),
}


def derive_category(question: str) -> str:
    """Return a conservative report-only category; ambiguous cases remain unknown."""

    value = unicodedata.normalize("NFKC", question).casefold()
    matches = [
        category
        for category, patterns in _CATEGORY_PATTERNS.items()
        if any(pattern.search(value) for pattern in patterns)
    ]
    return matches[0] if len(matches) == 1 else "unknown"


def _numeric_distribution(rows: Sequence[CleanRow]) -> dict[str, Any]:
    answers = [row.answer for row in rows]
    lengths = np.asarray([len(row.question) for row in rows], dtype=np.float64)
    total = len(rows)
    magnitude = Counter()
    for answer in answers:
        absolute = abs(answer)
        if absolute == 0:
            magnitude["zero"] += 1
        elif absolute < 10:
            magnitude["1_to_9"] += 1
        elif absolute < 100:
            magnitude["10_to_99"] += 1
        elif absolute < 1_000:
            magnitude["100_to_999"] += 1
        elif absolute < 1_000_000:
            magnitude["1k_to_999999"] += 1
        else:
            magnitude["1m_or_more"] += 1
    signs = {
        "negative": sum(answer < 0 for answer in answers),
        "zero": sum(answer == 0 for answer in answers),
        "positive": sum(answer > 0 for answer in answers),
    }
    return {
        "rows": total,
        "answer_sign": {
            name: {"count": count, "ratio": round(count / total, 8)}
            for name, count in signs.items()
        },
        "answer_magnitude": {
            name: {"count": magnitude[name], "ratio": round(magnitude[name] / total, 8)}
            for name in ("zero", "1_to_9", "10_to_99", "100_to_999", "1k_to_999999", "1m_or_more")
        },
        "question_character_length": {
            "min": int(lengths.min()),
            "median": round(float(np.median(lengths)), 6),
            "mean": round(float(lengths.mean()), 6),
            "p95": round(float(np.percentile(lengths, 95)), 6),
            "max": int(lengths.max()),
        },
    }


def _distribution_delta(train: dict[str, Any], val: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer_sign_ratio_absolute_delta": {
            name: round(
                abs(train["answer_sign"][name]["ratio"] - val["answer_sign"][name]["ratio"]),
                8,
            )
            for name in ("negative", "zero", "positive")
        },
        "answer_magnitude_ratio_absolute_delta": {
            name: round(
                abs(
                    train["answer_magnitude"][name]["ratio"]
                    - val["answer_magnitude"][name]["ratio"]
                ),
                8,
            )
            for name in train["answer_magnitude"]
        },
        "question_length": {
            metric: round(
                val["question_character_length"][metric]
                - train["question_character_length"][metric],
                6,
            )
            for metric in ("median", "mean", "p95")
        },
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: object) -> None:
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded + b"\n")


def _atomic_write_csv(path: Path, columns: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    _atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _category_report(rows: Sequence[CleanRow], categories: Sequence[str]) -> dict[str, Any]:
    counts = Counter(categories[row.index] for row in rows)
    total = len(rows)
    return {
        category: {"count": counts[category], "ratio": round(counts[category] / total, 8)}
        for category in sorted(counts)
    }


def _leaderboard_overlap_audit(
    settings: SplitSettings,
    normalization: NormalizationSettings,
    train_rows: Sequence[CleanRow],
    val_rows: Sequence[CleanRow],
) -> dict[str, Any]:
    if not settings.leaderboard_audit_enabled:
        return {"enabled": False}
    actual_hash = (
        sha256_file(settings.leaderboard_path) if settings.leaderboard_path.is_file() else None
    )
    if actual_hash != settings.leaderboard_sha256:
        raise SplitValidationError(
            "leaderboard audit source SHA-256 mismatch: "
            f"expected {settings.leaderboard_sha256}, got {actual_hash}."
        )
    leaderboard = _read_csv(
        settings.leaderboard_path, settings.leaderboard_columns, "leaderboard audit source"
    )
    if len(leaderboard) != settings.leaderboard_row_count:
        raise SplitValidationError(
            f"leaderboard audit expected {settings.leaderboard_row_count} rows, "
            f"got {len(leaderboard)}."
        )
    if "answer" in settings.leaderboard_columns:
        raise SplitValidationError("leaderboard audit schema must never contain answer.")
    leaderboard_ids = {row["id"] for row in leaderboard}
    leaderboard_exact = {row["question"] for row in leaderboard}
    leaderboard_normalized = {
        normalize_question(row["question"], normalization) for row in leaderboard
    }

    def overlap(rows: Sequence[CleanRow]) -> dict[str, int]:
        return {
            "id": len({row.sample_id for row in rows} & leaderboard_ids),
            "exact_question": len({row.question for row in rows} & leaderboard_exact),
            "normalized_question": len(
                {normalize_question(row.question, normalization) for row in rows}
                & leaderboard_normalized
            ),
        }

    return {
        "enabled": True,
        "purpose": "exact_and_normalized_overlap_audit_only",
        "leaderboard_rows": len(leaderboard),
        "leaderboard_sha256": actual_hash,
        "train_overlap": overlap(train_rows),
        "val_overlap": overlap(val_rows),
        "answer_accessed": False,
        "near_duplicate_answer_lookup": False,
    }


def run_split_pipeline(config: LoadedConfig, *, project_root: str | Path) -> Phase2Result:
    """Run the complete Phase 2 leakage-safe split pipeline."""

    root = Path(project_root).resolve()
    settings = load_split_settings(config, root)
    rows = read_clean_source(settings)
    source_hash_before = sha256_file(settings.source_path)
    git_info = collect_git_info(root)
    grouping = create_duplicate_groups(rows, settings.normalization, settings.near_duplicate)
    val_group_ids = deterministic_group_split(
        grouping.group_members,
        total_rows=len(rows),
        target_val_ratio=settings.target_val_ratio,
        seed=settings.seed,
    )
    val_indices = {
        index for group_id in val_group_ids for index in grouping.group_members[group_id]
    }
    train_rows = tuple(row for row in rows if row.index not in val_indices)
    val_rows = tuple(row for row in rows if row.index in val_indices)
    actual_val_ratio = len(val_rows) / len(rows)
    if not settings.allowed_val_ratio[0] <= actual_val_ratio <= settings.allowed_val_ratio[1]:
        raise SplitValidationError(
            f"actual validation ratio {actual_val_ratio:.6f} is outside "
            f"{settings.allowed_val_ratio}."
        )

    train_ids = {row.sample_id for row in train_rows}
    val_ids = {row.sample_id for row in val_rows}
    train_groups = {grouping.group_by_index[row.index] for row in train_rows}
    val_groups = {grouping.group_by_index[row.index] for row in val_rows}
    id_overlap = sorted(train_ids & val_ids)
    group_overlap = sorted(train_groups & val_groups)
    if id_overlap or group_overlap:
        raise SplitValidationError(
            f"leakage invariant failed: ID overlap={id_overlap[:10]}, "
            f"group overlap={group_overlap[:10]}."
        )
    if len(train_rows) + len(val_rows) != len(rows):
        raise SplitValidationError("split lost or duplicated source rows.")

    categories = tuple(
        derive_category(row.question) if settings.category_enabled else "unknown" for row in rows
    )
    output_dir = settings.output_dir
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    groups_path = output_dir / "groups.csv"
    candidates_path = output_dir / "near_duplicate_candidates.csv"
    report_path = output_dir / "split_report.json"
    manifest_path = output_dir / "split_manifest.json"

    source_columns = list(settings.source_columns)
    _atomic_write_csv(
        train_path,
        source_columns,
        [[row.sample_id, row.question, row.answer_text] for row in train_rows],
    )
    _atomic_write_csv(
        val_path,
        source_columns,
        [[row.sample_id, row.question, row.answer_text] for row in val_rows],
    )
    _atomic_write_csv(
        groups_path,
        ["id", "group_id", "group_size", "split", "derived_category"],
        [
            [
                row.sample_id,
                grouping.group_by_index[row.index],
                len(grouping.group_members[grouping.group_by_index[row.index]]),
                "validation" if row.index in val_indices else "train",
                categories[row.index],
            ]
            for row in rows
        ],
    )
    _atomic_write_csv(
        candidates_path,
        [
            "left_id",
            "right_id",
            "score",
            "decision",
            "candidate_reasons",
            "base_character_jaccard",
            "template_character_jaccard",
            "template_token_bigram_jaccard",
            "template_sequence_ratio",
            "left_question",
            "right_question",
        ],
        [
            [
                rows[pair.left].sample_id,
                rows[pair.right].sample_id,
                f"{pair.score:.6f}",
                "group"
                if pair.score + 1e-12 >= settings.near_duplicate.grouping_threshold
                else "review",
                "|".join(pair.candidate_reasons),
                f"{pair.base_character_jaccard:.6f}",
                f"{pair.template_character_jaccard:.6f}",
                f"{pair.template_token_bigram_jaccard:.6f}",
                f"{pair.template_sequence_ratio:.6f}",
                rows[pair.left].question,
                rows[pair.right].question,
            ]
            for pair in grouping.candidates
        ],
    )

    artifact_hashes = {
        "train.csv": sha256_file(train_path),
        "val.csv": sha256_file(val_path),
        "groups.csv": sha256_file(groups_path),
        "near_duplicate_candidates.csv": sha256_file(candidates_path),
    }
    group_stats = _group_size_summary(grouping.group_members)
    train_distribution = _numeric_distribution(train_rows)
    val_distribution = _numeric_distribution(val_rows)
    report = {
        "schema_version": 1,
        "split_version": settings.split_version,
        "pipeline_version": PIPELINE_VERSION,
        "source": {
            "dataset_version": settings.source_dataset_version,
            "path": _display_path(settings.source_path, root),
            "sha256": settings.source_sha256,
            "rows": len(rows),
            "heuristic_suspects_removed": False,
        },
        "duplicate_audit": grouping.duplicate_audit,
        "near_duplicate": {
            "candidate_generation": grouping.candidate_generation,
            "similarity_method": {
                "base": "character_ngram_jaccard",
                "template": "weighted character Jaccard + token-bigram Jaccard + SequenceMatcher",
                "number_template": True,
                "character_ngram_size": settings.near_duplicate.character_ngram_size,
            },
            "review_threshold": settings.near_duplicate.review_threshold,
            "grouping_threshold": settings.near_duplicate.grouping_threshold,
            "review_or_group_candidate_pairs": len(grouping.candidates),
            "auto_group_pairs": sum(
                pair.score + 1e-12 >= settings.near_duplicate.grouping_threshold
                for pair in grouping.candidates
            ),
            "review_only_pairs": sum(
                pair.score + 1e-12 < settings.near_duplicate.grouping_threshold
                for pair in grouping.candidates
            ),
            "threshold_calibration": grouping.threshold_calibration,
            "selection_rationale": settings.near_duplicate.threshold_rationale,
        },
        "groups": group_stats,
        "split": {
            "seed": settings.seed,
            "algorithm": "seeded_greedy_whole_group_target_v1",
            "target_val_ratio": settings.target_val_ratio,
            "actual_val_ratio": round(actual_val_ratio, 10),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_val_id_overlap": len(id_overlap),
            "train_val_group_overlap": len(group_overlap),
        },
        "category": {
            "enabled": settings.category_enabled,
            "version": settings.category_version,
            "derived_not_gold": True,
            "used_for_split": settings.category_use_for_split,
            "unknown_allowed": True,
            "rationale": (
                "No gold category exists; conservative keywords are report-only so uncertain "
                "labels cannot weaken group leakage prevention."
            ),
            "train": _category_report(train_rows, categories),
            "validation": _category_report(val_rows, categories),
        },
        "distribution_sanity": {
            "train": train_distribution,
            "validation": val_distribution,
            "train_validation_delta": _distribution_delta(train_distribution, val_distribution),
            "answer_used_as_stratification_key": False,
        },
        "leaderboard_audit": _leaderboard_overlap_audit(
            settings, settings.normalization, train_rows, val_rows
        ),
        "artifact_sha256": artifact_hashes,
    }
    _atomic_write_json(report_path, report)
    report_hash = sha256_file(report_path)

    source_hash_after = sha256_file(settings.source_path)
    if source_hash_after != source_hash_before:
        raise SplitValidationError("clean source changed while Phase 2 pipeline was running.")
    module_path = Path(__file__).resolve()
    identity = {
        "source_sha256": settings.source_sha256,
        "config_sha256": config.source_sha256,
        "module_sha256": sha256_file(module_path),
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "artifact_sha256": artifact_hashes,
        "split_report_sha256": report_hash,
    }
    created_at = datetime.now(UTC).isoformat()
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            previous = None
        if isinstance(previous, Mapping) and previous.get("reproducibility_identity") == identity:
            previous_created_at = previous.get("created_at")
            if isinstance(previous_created_at, str):
                created_at = previous_created_at
    manifest = {
        "schema_version": 1,
        "split_version": settings.split_version,
        "source_dataset_version": settings.source_dataset_version,
        "source_clean_path": _display_path(settings.source_path, root),
        "source_clean_sha256": settings.source_sha256,
        "source_row_count": len(rows),
        "source_hash_verified_before_and_after": True,
        "seed": settings.seed,
        "target_val_ratio": settings.target_val_ratio,
        "actual_val_ratio": round(actual_val_ratio, 10),
        "normalization_version": settings.normalization.version,
        "grouping_algorithm": settings.near_duplicate.algorithm,
        "grouping_version": settings.near_duplicate.version,
        "similarity_method": "char_ngram_jaccard_and_number_template_weighted_text_similarity",
        "similarity_threshold": settings.near_duplicate.grouping_threshold,
        **group_stats,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_sha256": artifact_hashes["train.csv"],
        "val_sha256": artifact_hashes["val.csv"],
        "groups_sha256": artifact_hashes["groups.csv"],
        "near_duplicate_candidates_sha256": artifact_hashes["near_duplicate_candidates.csv"],
        "split_report_sha256": report_hash,
        "train_val_id_overlap": len(id_overlap),
        "train_val_group_overlap": len(group_overlap),
        "config_sha256": config.source_sha256,
        "module_path": _display_path(module_path, root),
        "module_sha256": identity["module_sha256"],
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "created_at": created_at,
        "artifact_paths": {
            "train": _display_path(train_path, root),
            "validation": _display_path(val_path, root),
            "groups": _display_path(groups_path, root),
            "near_duplicate_candidates": _display_path(candidates_path, root),
            "split_report": _display_path(report_path, root),
        },
        "reproducibility_identity": identity,
    }
    _atomic_write_json(manifest_path, manifest)
    return Phase2Result(
        split_version=settings.split_version,
        train_rows=len(train_rows),
        val_rows=len(val_rows),
        actual_val_ratio=actual_val_ratio,
        total_groups=group_stats["total_groups"],
        largest_group_size=group_stats["largest_group_size"],
        train_sha256=artifact_hashes["train.csv"],
        val_sha256=artifact_hashes["val.csv"],
        groups_sha256=artifact_hashes["groups.csv"],
        candidates_sha256=artifact_hashes["near_duplicate_candidates.csv"],
        report_sha256=report_hash,
        manifest_sha256=sha256_file(manifest_path),
        output_dir=_display_path(output_dir, root),
    )
