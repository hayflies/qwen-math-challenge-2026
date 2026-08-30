"""E001 adapter evaluation settings and paired canonical E000 comparison."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from qwen_math_challenge.config import LoadedConfig
from qwen_math_challenge.evaluation.zero_shot import (
    ANSWER_PARSER_VERSION,
    EvaluationError,
    PredictionRecord,
    ZeroShotSettings,
    integer_exact_match,
    load_zero_shot_settings,
    parse_integer_answer,
)
from qwen_math_challenge.training.sft import SFTSettings, load_sft_settings

E001_EVALUATION_PIPELINE_VERSION = "phase4_e001_eval_v1"
CANONICAL_E000_TOTAL = 1_637
CANONICAL_E000_CORRECT = 1_075
CANONICAL_E000_ACCURACY = 0.6566890653634697


def load_e001_evaluation_settings(
    config: LoadedConfig, project_root: str | Path
) -> tuple[SFTSettings, ZeroShotSettings]:
    """Require E001 training identity plus the unchanged E000 evaluation protocol."""

    sft = load_sft_settings(config, project_root)
    evaluation = load_zero_shot_settings(
        config,
        project_root,
        expected_experiment_id="E001",
        expected_phase=4,
    )
    if evaluation.parser_version != ANSWER_PARSER_VERSION:
        raise EvaluationError("E001 must retain integer_v001.")
    generation = asdict(evaluation.generation)
    if generation != {
        "version": "greedy_v001",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 1024,
        "use_cache": True,
    }:
        raise EvaluationError("E001 must retain the complete canonical E000 generation config.")
    if evaluation.prompt != sft.prompt:
        raise EvaluationError("E001 training/evaluation prompt identity mismatch.")
    return sft, evaluation


def _validate_prediction_integrity(records: Sequence[PredictionRecord], label: str) -> None:
    seen: set[str] = set()
    for record in records:
        if record.sample_id in seen:
            raise EvaluationError(f"{label} contains duplicate ID {record.sample_id!r}.")
        seen.add(record.sample_id)
        parsed = parse_integer_answer(record.raw_output)
        if (
            parsed.value != record.parsed_answer
            or parsed.status != record.parse_status
            or integer_exact_match(parsed.value, record.gold_answer) != record.correct
        ):
            raise EvaluationError(f"{label} prediction integrity failed for {record.sample_id!r}.")


def _summary(values: Sequence[float | int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 8),
        "median": round(float(np.median(array)), 8),
        "p95": round(float(np.percentile(array, 95)), 8),
        "p99": round(float(np.percentile(array, 99)), 8),
        "max": round(float(array.max()), 8),
    }


def _group_comparison(
    e000: Sequence[PredictionRecord],
    e001: Sequence[PredictionRecord],
    key: Callable[[PredictionRecord], str],
) -> dict[str, Any]:
    grouped_e000: defaultdict[str, list[PredictionRecord]] = defaultdict(list)
    grouped_e001: defaultdict[str, list[PredictionRecord]] = defaultdict(list)
    for record in e000:
        grouped_e000[key(record)].append(record)
    for record in e001:
        grouped_e001[key(record)].append(record)
    if set(grouped_e000) != set(grouped_e001):
        raise EvaluationError("E000/E001 grouped comparison labels differ.")
    output: dict[str, Any] = {}
    for group in sorted(grouped_e000):
        before = grouped_e000[group]
        after = grouped_e001[group]
        if len(before) != len(after):
            raise EvaluationError(f"E000/E001 group size differs for {group!r}.")
        before_correct = sum(record.correct for record in before)
        after_correct = sum(record.correct for record in after)
        before_accuracy = before_correct / len(before)
        after_accuracy = after_correct / len(after)
        output[group] = {
            "count": len(before),
            "e000_correct": before_correct,
            "e001_correct": after_correct,
            "e000_accuracy": round(before_accuracy, 10),
            "e001_accuracy": round(after_accuracy, 10),
            "absolute_delta_percentage_points": round(
                100.0 * (after_accuracy - before_accuracy), 8
            ),
        }
    return output


def _max_hit_metrics(records: Sequence[PredictionRecord]) -> dict[str, Any]:
    hits = [record for record in records if record.finish_reason == "max_new_tokens"]
    correct = sum(record.correct for record in hits)
    return {
        "count": len(hits),
        "correct": correct,
        "accuracy": round(correct / len(hits), 10) if hits else None,
    }


def compare_e000_e001(
    e000: Sequence[PredictionRecord], e001: Sequence[PredictionRecord]
) -> dict[str, Any]:
    """Pair predictions by frozen validation ID and quantify gains and regressions."""

    _validate_prediction_integrity(e000, "E000")
    _validate_prediction_integrity(e001, "E001")
    if len(e000) != CANONICAL_E000_TOTAL or len(e001) != CANONICAL_E000_TOTAL:
        raise EvaluationError("Canonical comparison requires 1637 E000 and 1637 E001 rows.")
    if sum(record.correct for record in e000) != CANONICAL_E000_CORRECT:
        raise EvaluationError("E000 predictions do not match the canonical 1075-correct baseline.")
    if [record.sample_id for record in e000] != [record.sample_id for record in e001]:
        raise EvaluationError("E000/E001 validation ordering or coverage differs.")
    for before, after in zip(e000, e001, strict=True):
        if (
            before.question != after.question
            or before.gold_answer != after.gold_answer
            or before.derived_category != after.derived_category
        ):
            raise EvaluationError(f"E000/E001 row identity differs for {before.sample_id!r}.")

    transitions = {
        "e000_correct_to_e001_correct": 0,
        "e000_correct_to_e001_wrong": 0,
        "e000_wrong_to_e001_correct": 0,
        "e000_wrong_to_e001_wrong": 0,
    }
    for before, after in zip(e000, e001, strict=True):
        if before.correct and after.correct:
            key = "e000_correct_to_e001_correct"
        elif before.correct and not after.correct:
            key = "e000_correct_to_e001_wrong"
        elif not before.correct and after.correct:
            key = "e000_wrong_to_e001_correct"
        else:
            key = "e000_wrong_to_e001_wrong"
        transitions[key] += 1
    gained = transitions["e000_wrong_to_e001_correct"]
    regressed = transitions["e000_correct_to_e001_wrong"]
    e001_correct = sum(record.correct for record in e001)
    e001_accuracy = e001_correct / len(e001)
    e000_errors = len(e000) - CANONICAL_E000_CORRECT
    return {
        "schema_version": 1,
        "comparison_version": "e000_e001_paired_v001",
        "total": len(e001),
        "e000_correct": CANONICAL_E000_CORRECT,
        "e000_accuracy": CANONICAL_E000_ACCURACY,
        "e001_correct": e001_correct,
        "e001_accuracy": e001_accuracy,
        "absolute_delta_percentage_points": 100.0 * (e001_accuracy - CANONICAL_E000_ACCURACY),
        "relative_error_reduction": (gained - regressed) / e000_errors,
        "transitions": transitions,
        "gained": gained,
        "regressed": regressed,
        "net_gain": gained - regressed,
        "parse_failures": {
            "e000": sum(r.parse_status.startswith("parse_failure") for r in e000),
            "e001": sum(r.parse_status.startswith("parse_failure") for r in e001),
        },
        "max_token_hits": {
            "e000": _max_hit_metrics(e000),
            "e001": _max_hit_metrics(e001),
        },
        "answer_sign": _group_comparison(
            e000,
            e001,
            lambda record: (
                "negative"
                if record.gold_answer < 0
                else "zero"
                if record.gold_answer == 0
                else "positive"
            ),
        ),
        "heuristic_category": {
            "derived_not_gold": True,
            "metrics": _group_comparison(e000, e001, lambda record: record.derived_category),
        },
        "output_tokens": {
            "e000": _summary([record.output_tokens for record in e000]),
            "e001": _summary([record.output_tokens for record in e001]),
        },
        "latency_sec": {
            "e000": _summary([record.latency_sec for record in e000]),
            "e001": _summary([record.latency_sec for record in e001]),
        },
        "same_validation_set": True,
        "same_parser_version": ANSWER_PARSER_VERSION,
        "category_labels_derived_not_gold": True,
    }
