from pathlib import Path

from qwen_math_challenge.config import load_config
from qwen_math_challenge.evaluation.sft import (
    compare_e000_e001,
    load_e001_evaluation_settings,
)
from qwen_math_challenge.evaluation.zero_shot import PredictionRecord


def _record(index: int, correct: bool) -> PredictionRecord:
    gold = 1
    parsed = 1 if correct else 2
    return PredictionRecord(
        sample_id=f"train-{index:06d}",
        question=f"Question {index}",
        gold_answer=gold,
        raw_output=str(parsed),
        parsed_answer=parsed,
        correct=correct,
        parse_status="parsed_plain",
        input_tokens=10,
        output_tokens=index % 20 + 1,
        latency_sec=0.01,
        finish_reason="eos_token",
        truncated=False,
        derived_category="algebra" if index % 2 else "unknown",
    )


def test_e001_evaluation_retains_e000_protocol() -> None:
    config = load_config("configs/sft/e001_official_direct_answer.yaml")
    sft, evaluation = load_e001_evaluation_settings(config, Path.cwd())

    assert evaluation.prompt == sft.prompt
    assert evaluation.generation.version == "greedy_v001"
    assert evaluation.generation.max_new_tokens == 1024
    assert evaluation.parser_version == "integer_v001"


def test_paired_e000_e001_comparison_counts_gains_and_regressions() -> None:
    e000 = [_record(index, index < 1075) for index in range(1637)]
    e001 = []
    for index, before in enumerate(e000):
        if 0 <= index < 5:
            correct = False
        elif 1075 <= index < 1085:
            correct = True
        else:
            correct = before.correct
        e001.append(_record(index, correct))

    result = compare_e000_e001(e000, e001)

    assert result["gained"] == 10
    assert result["regressed"] == 5
    assert result["net_gain"] == 5
    assert result["e001_correct"] == 1080
    assert result["transitions"] == {
        "e000_correct_to_e001_correct": 1070,
        "e000_correct_to_e001_wrong": 5,
        "e000_wrong_to_e001_correct": 10,
        "e000_wrong_to_e001_wrong": 552,
    }
    assert result["same_parser_version"] == "integer_v001"
