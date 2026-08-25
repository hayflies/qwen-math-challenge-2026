import csv
import json
from pathlib import Path

import pytest

from qwen_math_challenge.config import load_config
from qwen_math_challenge.data.official import sha256_file
from qwen_math_challenge.evaluation.zero_shot import (
    ANSWER_PARSER_VERSION,
    PREDICTION_COLUMNS,
    DeviceSpec,
    EvaluationError,
    GenerationResult,
    PromptSettings,
    aggregate_metrics,
    build_chat_messages,
    build_run_manifest_fields,
    integer_exact_match,
    load_validation_rows,
    load_zero_shot_settings,
    parse_integer_answer,
    render_chat_prompt,
    run_zero_shot_evaluation,
    select_device,
)


def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _build_fixture(
    tmp_path: Path,
    *,
    prompt_version: str = "zero_shot_v001",
    batch_size: int = 2,
) -> Path:
    val_path = tmp_path / "data" / "splits" / "test_v001" / "val.csv"
    rows = [
        ["val-001", "Question alpha", "1"],
        ["val-002", "Question beta", "-2"],
        ["val-003", "Question gamma", "0"],
        ["val-004", "Question delta", "4"],
        ["val-005", "Question epsilon", "5"],
        ["val-006", "Question zeta", "6"],
    ]
    _write_csv(val_path, ["id", "question", "answer"], rows)
    groups_path = tmp_path / "data" / "splits" / "test_v001" / "groups.csv"
    _write_csv(
        groups_path,
        ["id", "group_id", "group_size", "split", "derived_category"],
        [
            [row[0], f"group-{index}", "1", "validation", "unknown"]
            for index, row in enumerate(rows)
        ],
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("fixture rules\n", encoding="utf-8")
    config_path = tmp_path / "configs" / "inference" / "e000.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""schema_version: 1
experiment:
  experiment_id: E000
  phase: 3
  model: Qwen/Qwen2.5-3B-Instruct
  dataset_version: test_v001
  validation_split: test_split_v001
  seed: 2026
runtime:
  output_root: outputs/e000
  log_level: INFO
  deterministic: true
  device: auto
  dtype: auto
  batch_size: {batch_size}
model:
  name_or_path: Qwen/Qwen2.5-3B-Instruct
  revision: main
  tokenizer_revision: main
  local_files_only: true
  trust_remote_code: false
data:
  split_version: test_split_v001
  validation_path: data/splits/test_v001/val.csv
  validation_sha256: "{sha256_file(val_path)}"
  validation_rows: 6
  groups_path: data/splits/test_v001/groups.csv
  groups_sha256: "{sha256_file(groups_path)}"
prompt:
  version: {prompt_version}
  system_text: You solve math problems and return an integer.
  user_template: |-
    {{question}}

    End with: Final answer: <integer>
generation:
  version: greedy_v001
  do_sample: false
  num_beams: 1
  max_new_tokens: 1024
  use_cache: true
parser:
  version: integer_v001
output:
  predictions_filename: predictions.csv
  failures_filename: failures.csv
  metrics_filename: metrics.json
  resume_identity_filename: resume_identity.json
""",
        encoding="utf-8",
    )
    return config_path


class FakeTokenizer:
    chat_template = "fake official template"

    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.calls.append((messages, tokenize, add_generation_prompt))
        return f"SYSTEM:{messages[0]['content']}\nUSER:{messages[1]['content']}\nASSISTANT:"


class FakeGenerator:
    model_commit = "model-commit-123"
    tokenizer_commit = "tokenizer-commit-123"
    chat_template = "fake official template"
    device_spec = DeviceSpec(device="cpu", dtype="float32", name="Fake CPU")
    model_load_time_sec = 0.25

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.answers = {
            "Question alpha": "Final answer: 1",
            "Question beta": "Final answer: -2",
            "Question gamma": "0",
            "Question delta": "Reasoning used 2 and 3. Final answer: 4",
            "Question epsilon": "Final answer: 999",
            "Question zeta": "No numerical result was produced.",
        }

    def render_prompt(self, question: str) -> str:
        return f"PROMPT:{question}"

    def generate(self, questions):
        self.calls.append(list(questions))
        return [
            GenerationResult(
                raw_output=self.answers[question],
                input_tokens=10 + index,
                output_tokens=5 + index,
                latency_sec=0.01 + index * 0.001,
                finish_reason="eos_token",
                truncated=False,
            )
            for index, question in enumerate(questions)
        ]

    def runtime_metadata(self):
        return {
            "device": "cpu",
            "device_name": "Fake CPU",
            "dtype": "float32",
            "model_load_time_sec": self.model_load_time_sec,
        }


def _load_fixture(tmp_path: Path, **kwargs):
    config = load_config(_build_fixture(tmp_path, **kwargs))
    settings = load_zero_shot_settings(config, tmp_path)
    return config, settings


def test_prompt_construction() -> None:
    prompt = PromptSettings("v1", "System rule", "Problem:\n{question}\nFinal integer.")
    messages = build_chat_messages("What is 1+1?", prompt)

    assert messages == [
        {"role": "system", "content": "System rule"},
        {"role": "user", "content": "Problem:\nWhat is 1+1?\nFinal integer."},
    ]


def test_chat_template_interface_is_used() -> None:
    tokenizer = FakeTokenizer()
    prompt = PromptSettings("v1", "System", "{question}\nFinal answer.")
    rendered = render_chat_prompt(tokenizer, "Question", prompt)

    assert rendered.endswith("ASSISTANT:")
    assert tokenizer.calls[0][1:] == (False, True)


def test_plain_integer_extraction() -> None:
    result = parse_integer_answer("42")
    assert result.value == 42
    assert result.status == "parsed_plain"


def test_negative_integer_extraction() -> None:
    assert parse_integer_answer("-17").value == -17


def test_zero_extraction() -> None:
    assert parse_integer_answer("0").value == 0


def test_boxed_integer_extraction() -> None:
    result = parse_integer_answer(r"After reasoning, \boxed{-1,234}")
    assert result.value == -1234
    assert result.status == "parsed_boxed"


def test_explicit_final_answer_extraction() -> None:
    result = parse_integer_answer("Therefore, the final answer is 42.")
    assert result.value == 42
    assert result.status == "parsed_explicit"


def test_multiple_number_reasoning_prefers_explicit_answer() -> None:
    result = parse_integer_answer("We used 3, 7, and 21. Final answer: 42")
    assert result.value == 42
    assert result.status == "parsed_explicit"


@pytest.mark.parametrize("raw", ["No integer here.", "42.0", "6*7", ""])
def test_no_acceptable_integer_is_parse_failure(raw: str) -> None:
    assert parse_integer_answer(raw).status == "parse_failure_no_integer"


def test_conflicting_answer_patterns_fail() -> None:
    result = parse_integer_answer(r"Final answer: 41, but \boxed{42}")
    assert result.value is None
    assert result.status == "parse_failure_conflict"


def test_integer_exact_match_scoring() -> None:
    assert integer_exact_match(42, 42)
    assert not integer_exact_match(-42, 42)
    assert not integer_exact_match(None, 42)


def test_validation_sha_mismatch_is_detected(tmp_path: Path) -> None:
    _, settings = _load_fixture(tmp_path)
    settings.data.validation_path.write_text("changed\n", encoding="utf-8")

    with pytest.raises(EvaluationError, match="validation SHA-256 mismatch"):
        load_validation_rows(settings.data)


def test_resume_config_mismatch_is_rejected(tmp_path: Path) -> None:
    first_config, _ = _load_fixture(tmp_path)
    generator = FakeGenerator()
    first_run = tmp_path / "run-first"
    run_zero_shot_evaluation(
        first_config,
        project_root=tmp_path,
        run_dir=first_run,
        generator=generator,
        limit=4,
    )
    second_root = tmp_path / "changed"
    second_config, _ = _load_fixture(second_root, prompt_version="zero_shot_changed")

    with pytest.raises(EvaluationError, match="identity mismatch"):
        run_zero_shot_evaluation(
            second_config,
            project_root=second_root,
            run_dir=second_root / "run-second",
            generator=FakeGenerator(),
            limit=4,
            resume_from=first_run,
        )


def test_completed_resume_ids_are_skipped(tmp_path: Path) -> None:
    config, _ = _load_fixture(tmp_path)
    first_run = tmp_path / "run-first"
    run_zero_shot_evaluation(
        config,
        project_root=tmp_path,
        run_dir=first_run,
        generator=FakeGenerator(),
        limit=4,
    )
    resumed_generator = FakeGenerator()
    result = run_zero_shot_evaluation(
        config,
        project_root=tmp_path,
        run_dir=tmp_path / "run-resumed",
        generator=resumed_generator,
        limit=4,
        resume_from=first_run,
    )

    assert resumed_generator.calls == []
    assert result.metrics["resumed_predictions"] == 4


def test_batch_result_ordering_is_validation_order(tmp_path: Path) -> None:
    config, _ = _load_fixture(tmp_path, batch_size=2)
    generator = FakeGenerator()
    result = run_zero_shot_evaluation(
        config,
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        generator=generator,
        limit=5,
    )
    with result.predictions_path.open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))

    assert [row["id"] for row in predictions] == [f"val-{index:03d}" for index in range(1, 6)]
    assert [len(batch) for batch in generator.calls] == [2, 2, 1]


def test_prediction_output_schema(tmp_path: Path) -> None:
    config, _ = _load_fixture(tmp_path)
    result = run_zero_shot_evaluation(
        config,
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        generator=FakeGenerator(),
        limit=2,
    )
    with result.predictions_path.open(encoding="utf-8", newline="") as handle:
        assert tuple(next(csv.reader(handle))) == PREDICTION_COLUMNS


def test_run_manifest_schema_fields(tmp_path: Path) -> None:
    config, settings = _load_fixture(tmp_path)
    generator = FakeGenerator()
    result = run_zero_shot_evaluation(
        config,
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        generator=generator,
        limit=2,
    )
    fields = build_run_manifest_fields(settings, generator, result, limit=2, resumed_from=None)

    required = {
        "model_name",
        "model_revision",
        "tokenizer_revision",
        "split_version",
        "val_sha256",
        "prompt_version",
        "prompt_text",
        "chat_template",
        "generation_config",
        "answer_parser_version",
        "device",
        "dtype",
        "python_version",
        "torch_version",
        "transformers_version",
        "total_samples",
        "correct",
        "incorrect",
        "parse_failures",
        "accuracy",
    }
    assert required <= fields.keys()
    assert fields["answer_parser_version"] == ANSWER_PARSER_VERSION
    assert fields["tool_use"] is False


def test_limit_smoke_mode_uses_same_pipeline(tmp_path: Path) -> None:
    config, _ = _load_fixture(tmp_path)
    result = run_zero_shot_evaluation(
        config,
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        generator=FakeGenerator(),
        limit=3,
    )

    assert result.metrics["total"] == 3
    assert result.metrics["limit"] == 3
    assert result.metrics["correct"] == 3


class _FakeCuda:
    def __init__(self, available: bool, bf16: bool = False) -> None:
        self._available = available
        self._bf16 = bf16

    def is_available(self):
        return self._available

    def is_bf16_supported(self):
        return self._bf16

    def current_device(self):
        return 0

    def get_device_name(self, index):
        return f"Fake CUDA {index}"


class _FakeMps:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self):
        return self._available


class _FakeTorch:
    def __init__(self, *, cuda: bool, mps: bool, bf16: bool = False) -> None:
        self.cuda = _FakeCuda(cuda, bf16)
        self.backends = type("Backends", (), {"mps": _FakeMps(mps)})()


@pytest.mark.parametrize(
    ("torch_module", "device", "dtype"),
    [
        (_FakeTorch(cuda=True, mps=True, bf16=True), "cuda", "bfloat16"),
        (_FakeTorch(cuda=False, mps=True), "mps", "float16"),
        (_FakeTorch(cuda=False, mps=False), "cpu", "float32"),
    ],
)
def test_device_auto_selection_priority(torch_module, device: str, dtype: str) -> None:
    selected = select_device("auto", "auto", torch_module)
    assert selected.device == device
    assert selected.dtype == dtype


def test_raw_output_is_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "SHOULD_NOT_EXIST"
    raw = f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')"
    result = parse_integer_answer(raw)

    assert not marker.exists()
    assert result.status == "parse_failure_no_integer"


def test_metrics_include_token_latency_and_truncation_statistics() -> None:
    records = []
    metrics = aggregate_metrics(records, total_wall_clock_sec=1.25)

    assert "input_token_statistics" in metrics
    assert "output_token_statistics" in metrics
    assert "latency_sec_statistics" in metrics
    assert "p99" in metrics["latency_sec_statistics"]
    assert set(metrics["per_answer_sign"]) == {"negative", "zero", "positive"}
    assert metrics["truncated"] == 0


def test_metrics_and_resume_identity_are_json_artifacts(tmp_path: Path) -> None:
    config, _ = _load_fixture(tmp_path)
    result = run_zero_shot_evaluation(
        config,
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        generator=FakeGenerator(),
        limit=2,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    identity = json.loads(result.resume_identity_path.read_text(encoding="utf-8"))
    assert metrics["total"] == 2
    assert identity["identity"]["validation_sha256"] == result.metrics["validation_sha256"]
