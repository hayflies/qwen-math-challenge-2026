"""Phase 3 E000 zero-shot inference, parsing, evaluation, and resume support."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from qwen_math_challenge.config import LoadedConfig
from qwen_math_challenge.data.official import sha256_file

ALLOWED_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ANSWER_PARSER_VERSION = "integer_v001"
PIPELINE_VERSION = "phase3_e000_v2"
PREDICTION_COLUMNS = (
    "id",
    "question",
    "gold_answer",
    "raw_output",
    "parsed_answer",
    "correct",
    "parse_status",
    "input_tokens",
    "output_tokens",
    "latency_sec",
    "finish_reason",
    "truncated",
    "derived_category",
)
_INTEGER_TEXT = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"
_PLAIN_INTEGER_PATTERN = re.compile(rf"^\s*(?P<value>{_INTEGER_TEXT})\s*[.!]?\s*$")
_BOXED_INTEGER_PATTERN = re.compile(
    rf"\\boxed\s*\{{\s*(?P<value>{_INTEGER_TEXT})\s*\}}", re.IGNORECASE
)
_EXPLICIT_INTEGER_PATTERN = re.compile(
    rf"(?:final\s+answer|the\s+answer|answer)\s*(?:is|:|=)\s*"
    rf"(?:\\boxed\s*\{{\s*)?(?P<value>{_INTEGER_TEXT})(?:\s*\}})?",
    re.IGNORECASE,
)
_STANDALONE_INTEGER_PATTERN = re.compile(rf"(?<![\w.])(?P<value>{_INTEGER_TEXT})(?![\w.]|\.\d)")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EvaluationError(ValueError):
    """Raised when E000 config, data, resume state, or outputs are invalid."""


class ModelUnavailableError(RuntimeError):
    """Raised when the one allowed model is not complete in the local HF cache."""


@dataclass(frozen=True)
class ModelSettings:
    name_or_path: str
    revision: str
    tokenizer_revision: str
    local_files_only: bool
    trust_remote_code: bool


@dataclass(frozen=True)
class DataSettings:
    split_version: str
    validation_path: Path
    validation_sha256: str
    validation_rows: int
    groups_path: Path
    groups_sha256: str


@dataclass(frozen=True)
class PromptSettings:
    version: str
    system_text: str
    user_template: str


@dataclass(frozen=True)
class GenerationSettings:
    version: str
    do_sample: bool
    num_beams: int
    max_new_tokens: int
    use_cache: bool


@dataclass(frozen=True)
class RuntimeSettings:
    device: str
    dtype: str
    batch_size: int


@dataclass(frozen=True)
class OutputSettings:
    predictions_filename: str
    failures_filename: str
    metrics_filename: str
    resume_identity_filename: str


@dataclass(frozen=True)
class ZeroShotSettings:
    model: ModelSettings
    data: DataSettings
    prompt: PromptSettings
    generation: GenerationSettings
    runtime: RuntimeSettings
    output: OutputSettings
    parser_version: str
    seed: int


@dataclass(frozen=True)
class ValidationRow:
    index: int
    sample_id: str
    question: str
    gold_answer: int
    gold_answer_text: str
    derived_category: str


@dataclass(frozen=True)
class ParseResult:
    value: int | None
    status: str
    matched_text: str | None


@dataclass(frozen=True)
class GenerationResult:
    raw_output: str
    input_tokens: int
    output_tokens: int
    latency_sec: float
    finish_reason: str
    truncated: bool


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    question: str
    gold_answer: int
    raw_output: str
    parsed_answer: int | None
    correct: bool
    parse_status: str
    input_tokens: int
    output_tokens: int
    latency_sec: float
    finish_reason: str
    truncated: bool
    derived_category: str

    def as_csv_row(self) -> list[str]:
        return [
            self.sample_id,
            self.question,
            str(self.gold_answer),
            self.raw_output,
            "" if self.parsed_answer is None else str(self.parsed_answer),
            str(self.correct).lower(),
            self.parse_status,
            str(self.input_tokens),
            str(self.output_tokens),
            f"{self.latency_sec:.9f}",
            self.finish_reason,
            str(self.truncated).lower(),
            self.derived_category,
        ]


@dataclass(frozen=True)
class DeviceSpec:
    device: str
    dtype: str
    name: str


@dataclass(frozen=True)
class ModelSnapshot:
    path: Path
    commit_hash: str


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    resume_identity: dict[str, Any]
    predictions_path: Path
    failures_path: Path
    metrics_path: Path
    resume_identity_path: Path
    artifact_sha256: dict[str, str]


class BatchGenerator(Protocol):
    model_commit: str
    tokenizer_commit: str
    chat_template: str
    device_spec: DeviceSpec
    model_load_time_sec: float

    def render_prompt(self, question: str) -> str: ...

    def generate(self, questions: Sequence[str]) -> Sequence[GenerationResult]: ...

    def runtime_metadata(self) -> Mapping[str, Any]: ...


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"'{label}' must be a mapping.")
    return dict(value)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"'{label}' must be a non-empty string.")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationError(f"'{label}' must be a boolean.")
    return value


def _require_int(value: object, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvaluationError(f"'{label}' must be an integer >= {minimum}.")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise EvaluationError(f"'{label}' must be a lowercase SHA-256 digest.")
    return value


def _resolve_path(root: Path, value: object, label: str) -> Path:
    path = Path(_require_string(value, label)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_filename(value: object, label: str, suffix: str) -> str:
    filename = _require_string(value, label)
    path = Path(filename)
    if path.name != filename or path.suffix.lower() != suffix:
        raise EvaluationError(f"'{label}' must be a plain basename ending in {suffix!r}.")
    return filename


def load_zero_shot_settings(config: LoadedConfig, project_root: str | Path) -> ZeroShotSettings:
    """Validate the single-model deterministic E000 configuration."""

    root = Path(project_root).resolve()
    if config.phase != 3 or config.experiment_id != "E000":
        raise EvaluationError("E000 config must set experiment_id E000 and phase 3.")

    model = _require_mapping(config.raw.get("model"), "model")
    name_or_path = _require_string(model.get("name_or_path"), "model.name_or_path")
    if name_or_path != ALLOWED_MODEL_ID:
        raise EvaluationError(f"E000 permits only {ALLOWED_MODEL_ID!r}; got {name_or_path!r}.")
    if config.experiment.get("model") != ALLOWED_MODEL_ID:
        raise EvaluationError("experiment.model must match the one permitted model ID.")
    local_files_only = _require_bool(model.get("local_files_only"), "model.local_files_only")
    if not local_files_only:
        raise EvaluationError(
            "E000 runtime must use local_files_only=true; download weights separately."
        )
    trust_remote_code = _require_bool(model.get("trust_remote_code"), "model.trust_remote_code")
    if trust_remote_code:
        raise EvaluationError("E000 must use built-in Transformers code, not remote code.")

    data = _require_mapping(config.raw.get("data"), "data")
    validation_path = _resolve_path(root, data.get("validation_path"), "data.validation_path")
    groups_path = _resolve_path(root, data.get("groups_path"), "data.groups_path")
    raw_root = (root / "data" / "raw").resolve()
    if validation_path == raw_root or raw_root in validation_path.parents:
        raise EvaluationError("E000 validation may not read from data/raw.")

    prompt = _require_mapping(config.raw.get("prompt"), "prompt")
    user_template = _require_string(prompt.get("user_template"), "prompt.user_template")
    if user_template.count("{question}") != 1:
        raise EvaluationError("prompt.user_template must contain exactly one '{question}'.")
    try:
        user_template.format(question="probe")
    except (KeyError, ValueError) as exc:
        raise EvaluationError("prompt.user_template contains unsupported format fields.") from exc

    generation = _require_mapping(config.raw.get("generation"), "generation")
    do_sample = _require_bool(generation.get("do_sample"), "generation.do_sample")
    if do_sample:
        raise EvaluationError("E000 generation must be deterministic with do_sample=false.")
    num_beams = _require_int(generation.get("num_beams"), "generation.num_beams")
    if num_beams != 1:
        raise EvaluationError("E000 generation must use num_beams=1.")
    forbidden_generation_keys = {"temperature", "top_p", "top_k", "num_return_sequences"}
    configured_forbidden = sorted(forbidden_generation_keys & generation.keys())
    if configured_forbidden:
        raise EvaluationError(
            f"Deterministic E000 config must omit sampling/scaling keys: {configured_forbidden}."
        )

    parser = _require_mapping(config.raw.get("parser"), "parser")
    parser_version = _require_string(parser.get("version"), "parser.version")
    if parser_version != ANSWER_PARSER_VERSION:
        raise EvaluationError(
            f"Unsupported E000 parser {parser_version!r}; expected {ANSWER_PARSER_VERSION!r}."
        )

    runtime_raw = _require_mapping(config.raw.get("runtime"), "runtime")
    device = _require_string(runtime_raw.get("device"), "runtime.device").lower()
    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise EvaluationError("runtime.device must be auto, cuda, mps, or cpu.")
    dtype = _require_string(runtime_raw.get("dtype"), "runtime.dtype").lower()
    if dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise EvaluationError("runtime.dtype must be auto, float32, float16, or bfloat16.")

    output = _require_mapping(config.raw.get("output"), "output")
    revision = _require_string(model.get("revision"), "model.revision")
    tokenizer_revision = _require_string(
        model.get("tokenizer_revision"), "model.tokenizer_revision"
    )
    if tokenizer_revision != revision:
        raise EvaluationError(
            "E000 model and tokenizer revisions must match to freeze one coherent snapshot."
        )

    return ZeroShotSettings(
        model=ModelSettings(
            name_or_path=name_or_path,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        ),
        data=DataSettings(
            split_version=_require_string(data.get("split_version"), "data.split_version"),
            validation_path=validation_path,
            validation_sha256=_require_sha256(
                data.get("validation_sha256"), "data.validation_sha256"
            ),
            validation_rows=_require_int(data.get("validation_rows"), "data.validation_rows"),
            groups_path=groups_path,
            groups_sha256=_require_sha256(data.get("groups_sha256"), "data.groups_sha256"),
        ),
        prompt=PromptSettings(
            version=_require_string(prompt.get("version"), "prompt.version"),
            system_text=_require_string(prompt.get("system_text"), "prompt.system_text"),
            user_template=user_template,
        ),
        generation=GenerationSettings(
            version=_require_string(generation.get("version"), "generation.version"),
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=_require_int(
                generation.get("max_new_tokens"), "generation.max_new_tokens"
            ),
            use_cache=_require_bool(generation.get("use_cache"), "generation.use_cache"),
        ),
        runtime=RuntimeSettings(
            device=device,
            dtype=dtype,
            batch_size=_require_int(runtime_raw.get("batch_size"), "runtime.batch_size"),
        ),
        output=OutputSettings(
            predictions_filename=_require_filename(
                output.get("predictions_filename"), "output.predictions_filename", ".csv"
            ),
            failures_filename=_require_filename(
                output.get("failures_filename"), "output.failures_filename", ".csv"
            ),
            metrics_filename=_require_filename(
                output.get("metrics_filename"), "output.metrics_filename", ".json"
            ),
            resume_identity_filename=_require_filename(
                output.get("resume_identity_filename"),
                "output.resume_identity_filename",
                ".json",
            ),
        ),
        parser_version=parser_version,
        seed=config.seed,
    )


def build_chat_messages(question: str, prompt: PromptSettings) -> list[dict[str, str]]:
    """Build the one canonical system/user prompt without hand-written ChatML."""

    if not isinstance(question, str) or not question:
        raise EvaluationError("Question must be a non-empty string.")
    return [
        {"role": "system", "content": prompt.system_text},
        {"role": "user", "content": prompt.user_template.format(question=question)},
    ]


def render_chat_prompt(tokenizer: Any, question: str, prompt: PromptSettings) -> str:
    """Render messages only through the tokenizer's official chat-template interface."""

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise EvaluationError("Tokenizer does not provide apply_chat_template().")
    rendered = apply_template(
        build_chat_messages(question, prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise EvaluationError("Tokenizer chat template returned an empty/non-string prompt.")
    return rendered


def _canonical_integer(value: str) -> int:
    return int(value.replace(",", ""))


def _consistent_values(matches: Sequence[re.Match[str]]) -> tuple[int | None, bool]:
    values = {_canonical_integer(match.group("value")) for match in matches}
    if not values:
        return None, False
    if len(values) > 1:
        return None, True
    return next(iter(values)), False


def parse_integer_answer(raw_output: str) -> ParseResult:
    """Extract one integer deterministically without evaluating expressions or code."""

    if not isinstance(raw_output, str) or not raw_output.strip():
        return ParseResult(None, "parse_failure_no_integer", None)

    plain = _PLAIN_INTEGER_PATTERN.fullmatch(raw_output)
    if plain:
        return ParseResult(
            _canonical_integer(plain.group("value")),
            "parsed_plain",
            plain.group(0),
        )

    boxed_matches = list(_BOXED_INTEGER_PATTERN.finditer(raw_output))
    explicit_matches = list(_EXPLICIT_INTEGER_PATTERN.finditer(raw_output))
    boxed_value, boxed_conflict = _consistent_values(boxed_matches)
    explicit_value, explicit_conflict = _consistent_values(explicit_matches)
    if boxed_conflict or explicit_conflict:
        return ParseResult(None, "parse_failure_conflict", None)
    if boxed_value is not None and explicit_value is not None and boxed_value != explicit_value:
        return ParseResult(None, "parse_failure_conflict", None)
    if boxed_value is not None:
        return ParseResult(boxed_value, "parsed_boxed", boxed_matches[-1].group(0))
    if explicit_value is not None:
        return ParseResult(
            explicit_value,
            "parsed_explicit",
            explicit_matches[-1].group(0),
        )

    fallback_matches: list[re.Match[str]] = []
    for match in _STANDALONE_INTEGER_PATTERN.finditer(raw_output):
        before = raw_output[: match.start()].rstrip()
        after = raw_output[match.end() :].lstrip()
        if before[-1:] in {"*", "/", "^", "+"} or after[:1] in {"*", "/", "^", "+"}:
            continue
        fallback_matches.append(match)
    if fallback_matches:
        match = fallback_matches[-1]
        return ParseResult(
            _canonical_integer(match.group("value")),
            "parsed_fallback",
            match.group(0),
        )
    return ParseResult(None, "parse_failure_no_integer", None)


def integer_exact_match(prediction: int | None, gold: int) -> bool:
    """Apply the competition's integer equality metric only."""

    return prediction is not None and prediction == gold


def _read_csv(path: Path, columns: Sequence[str], role: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise EvaluationError(f"{role} does not exist: {path}")
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationError(f"{role} must be valid UTF-8.") from exc
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        actual_columns = tuple(next(reader))
    except StopIteration as exc:
        raise EvaluationError(f"{role} CSV is empty.") from exc
    if actual_columns != tuple(columns):
        raise EvaluationError(
            f"{role} columns must be {list(columns)}, got {list(actual_columns)}."
        )
    rows: list[dict[str, str]] = []
    for logical_row, values in enumerate(reader, start=2):
        if len(values) != len(actual_columns):
            raise EvaluationError(
                f"{role} logical row {logical_row} has {len(values)} fields; "
                f"expected {len(actual_columns)}."
            )
        rows.append(dict(zip(actual_columns, values, strict=True)))
    return rows


def load_validation_rows(settings: DataSettings) -> tuple[ValidationRow, ...]:
    """Verify the frozen Phase 2 split and attach report-only derived categories."""

    actual_val_hash = (
        sha256_file(settings.validation_path) if settings.validation_path.is_file() else None
    )
    if actual_val_hash != settings.validation_sha256:
        raise EvaluationError(
            "validation SHA-256 mismatch: "
            f"expected {settings.validation_sha256}, got {actual_val_hash}."
        )
    actual_groups_hash = (
        sha256_file(settings.groups_path) if settings.groups_path.is_file() else None
    )
    if actual_groups_hash != settings.groups_sha256:
        raise EvaluationError(
            f"groups SHA-256 mismatch: expected {settings.groups_sha256}, got {actual_groups_hash}."
        )

    val_rows = _read_csv(
        settings.validation_path, ("id", "question", "answer"), "validation source"
    )
    if len(val_rows) != settings.validation_rows:
        raise EvaluationError(
            f"validation expected {settings.validation_rows} rows, got {len(val_rows)}."
        )
    groups = _read_csv(
        settings.groups_path,
        ("id", "group_id", "group_size", "split", "derived_category"),
        "groups source",
    )
    category_by_id: dict[str, str] = {}
    for row in groups:
        if row["id"] in category_by_id:
            raise EvaluationError(f"groups source has duplicate ID {row['id']!r}.")
        if row["split"] == "validation":
            category_by_id[row["id"]] = row["derived_category"]

    output: list[ValidationRow] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(val_rows):
        if any(row[column] == "" for column in ("id", "question", "answer")):
            raise EvaluationError(f"validation row {index + 2} has an empty required value.")
        if row["id"] in seen_ids:
            raise EvaluationError(f"validation source has duplicate ID {row['id']!r}.")
        seen_ids.add(row["id"])
        try:
            gold = int(row["answer"])
        except ValueError as exc:
            raise EvaluationError(
                f"validation answer for {row['id']!r} is not an integer."
            ) from exc
        category = category_by_id.get(row["id"])
        if category is None:
            raise EvaluationError(
                f"validation ID {row['id']!r} is not marked validation in groups.csv."
            )
        output.append(
            ValidationRow(
                index=index,
                sample_id=row["id"],
                question=row["question"],
                gold_answer=gold,
                gold_answer_text=row["answer"],
                derived_category=category,
            )
        )
    return tuple(output)


def select_device(requested: str, dtype: str, torch_module: Any | None = None) -> DeviceSpec:
    """Select CUDA, then MPS, then CPU; explicit unavailable devices are errors."""

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:  # pragma: no cover - environment boundary
            raise EvaluationError("PyTorch is required for model inference.") from exc

    cuda_available = bool(torch_module.cuda.is_available())
    mps_backend = getattr(torch_module.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    if requested == "auto":
        device = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    else:
        device = requested
    if device == "cuda" and not cuda_available:
        raise EvaluationError("CUDA was requested but is unavailable.")
    if device == "mps" and not mps_available:
        raise EvaluationError("MPS was requested but is unavailable.")

    if dtype == "auto":
        if device == "cuda":
            supports_bf16 = getattr(torch_module.cuda, "is_bf16_supported", lambda: False)
            selected_dtype = "bfloat16" if supports_bf16() else "float16"
        elif device == "mps":
            selected_dtype = "float16"
        else:
            selected_dtype = "float32"
    else:
        selected_dtype = dtype
    if device == "cpu" and selected_dtype in {"float16", "bfloat16"}:
        raise EvaluationError("CPU E000 must use float32 for reliable operator support.")
    if device == "mps" and selected_dtype == "bfloat16":
        raise EvaluationError("MPS E000 does not use bfloat16; choose auto or float16.")

    if device == "cuda":
        index = torch_module.cuda.current_device()
        name = str(torch_module.cuda.get_device_name(index))
    elif device == "mps":
        name = "Apple Metal Performance Shaders"
    else:
        name = platform.processor() or platform.machine() or "CPU"
    return DeviceSpec(device=device, dtype=selected_dtype, name=name)


def resolve_local_model_snapshot(model: ModelSettings) -> ModelSnapshot:
    """Resolve a complete cached snapshot without making any network request."""

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise ModelUnavailableError(
            "huggingface-hub is not installed; sync the locked Phase 3 environment."
        ) from exc
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=model.name_or_path,
                revision=model.revision,
                local_files_only=True,
            )
        ).resolve()
    except (LocalEntryNotFoundError, OSError) as exc:
        raise ModelUnavailableError(
            f"{model.name_or_path}@{model.revision} is not present in the local Hugging "
            "Face cache. Download that exact model separately; no fallback is permitted."
        ) from exc
    commit_hash = snapshot.name if snapshot.parent.name == "snapshots" else model.revision
    return ModelSnapshot(path=snapshot, commit_hash=commit_hash)


def _torch_dtype(torch_module: Any, name: str) -> Any:
    value = getattr(torch_module, name, None)
    if value is None:
        raise EvaluationError(f"Installed PyTorch does not provide dtype {name!r}.")
    return value


class TransformersBatchGenerator:
    """Thin local-only Transformers generation boundary for the one allowed model."""

    def __init__(
        self,
        *,
        settings: ZeroShotSettings,
        snapshot: ModelSnapshot,
        device_spec: DeviceSpec,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment boundary
            raise ModelUnavailableError(
                "torch and transformers must be installed from the locked environment."
            ) from exc

        self._torch = torch
        self._settings = settings
        self.device_spec = device_spec
        if device_spec.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                snapshot.path,
                local_files_only=True,
                trust_remote_code=False,
            )
        except OSError as exc:
            raise ModelUnavailableError(
                "The cached Qwen tokenizer snapshot is incomplete or unreadable."
            ) from exc
        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token_id is None:
                raise EvaluationError("Tokenizer has neither pad_token_id nor eos_token_id.")
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._tokenizer.padding_side = "left"
        self.chat_template = str(self._tokenizer.chat_template or "")
        if not self.chat_template:
            raise EvaluationError("Qwen tokenizer is missing its official chat template.")
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                snapshot.path,
                local_files_only=True,
                trust_remote_code=False,
                dtype=_torch_dtype(torch, device_spec.dtype),
                low_cpu_mem_usage=True,
            )
        except OSError as exc:
            raise ModelUnavailableError(
                "The cached Qwen model snapshot is incomplete or unreadable."
            ) from exc
        try:
            self._model.to(device_spec.device)
        except RuntimeError as exc:
            raise EvaluationError(
                f"Could not place Qwen on {device_spec.device}/{device_spec.dtype}; "
                "select CPU explicitly if needed. No automatic device fallback was applied."
            ) from exc
        self._model.train(False)
        self.model_commit = str(
            getattr(self._model.config, "_commit_hash", None) or snapshot.commit_hash
        )
        self.tokenizer_commit = str(
            getattr(self._tokenizer, "_commit_hash", None) or snapshot.commit_hash
        )
        self.model_load_time_sec = time.perf_counter() - load_started

    def runtime_metadata(self) -> dict[str, Any]:
        """Return hardware and peak-memory facts without changing generation behavior."""

        metadata: dict[str, Any] = {
            "device": self.device_spec.device,
            "device_name": self.device_spec.name,
            "dtype": self.device_spec.dtype,
            "model_load_time_sec": round(self.model_load_time_sec, 8),
        }
        if self.device_spec.device != "cuda":
            return metadata

        index = self._torch.cuda.current_device()
        properties = self._torch.cuda.get_device_properties(index)
        metadata["cuda"] = {
            "torch_cuda_version": self._torch.version.cuda,
            "gpu_name": self._torch.cuda.get_device_name(index),
            "total_vram_bytes": int(properties.total_memory),
            "capability": list(self._torch.cuda.get_device_capability(index)),
            "bf16_supported": bool(self._torch.cuda.is_bf16_supported()),
            "peak_allocated_bytes": int(self._torch.cuda.max_memory_allocated(index)),
            "peak_reserved_bytes": int(self._torch.cuda.max_memory_reserved(index)),
        }
        return metadata

    def render_prompt(self, question: str) -> str:
        return render_chat_prompt(self._tokenizer, question, self._settings.prompt)

    def _synchronize(self) -> None:
        if self.device_spec.device == "cuda":
            self._torch.cuda.synchronize()
        elif self.device_spec.device == "mps":
            synchronize = getattr(self._torch.mps, "synchronize", None)
            if callable(synchronize):
                synchronize()

    def generate(self, questions: Sequence[str]) -> Sequence[GenerationResult]:
        if not questions:
            return []
        prompts = [self.render_prompt(question) for question in questions]
        encoded = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        input_width = int(encoded["input_ids"].shape[1])
        model_limit = int(getattr(self._model.config, "max_position_embeddings", 0) or 0)
        if model_limit and input_width + self._settings.generation.max_new_tokens > model_limit:
            raise EvaluationError(
                f"Prompt width {input_width} + max_new_tokens "
                f"{self._settings.generation.max_new_tokens} exceeds model context {model_limit}."
            )
        encoded = {key: value.to(self.device_spec.device) for key, value in encoded.items()}
        self._synchronize()
        started = time.perf_counter()
        try:
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=self._settings.generation.max_new_tokens,
                    use_cache=self._settings.generation.use_cache,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
        except RuntimeError as exc:
            raise EvaluationError(
                f"Generation failed on {self.device_spec.device}/{self.device_spec.dtype}; "
                "no automatic CPU or alternate-model fallback was applied."
            ) from exc
        self._synchronize()
        batch_latency = time.perf_counter() - started
        sequences = generated.sequences[:, input_width:]
        eos_ids_value = self._tokenizer.eos_token_id
        eos_ids = (
            {int(value) for value in eos_ids_value}
            if isinstance(eos_ids_value, list)
            else {int(eos_ids_value)}
            if eos_ids_value is not None
            else set()
        )
        results: list[GenerationResult] = []
        per_problem_latency = batch_latency / len(questions)
        for index in range(len(questions)):
            token_ids = [int(value) for value in sequences[index].tolist()]
            eos_positions = [
                position for position, token_id in enumerate(token_ids) if token_id in eos_ids
            ]
            if eos_positions:
                token_ids = token_ids[: eos_positions[0] + 1]
                finish_reason = "eos_token"
            else:
                while token_ids and token_ids[-1] == self._tokenizer.pad_token_id:
                    token_ids.pop()
                finish_reason = (
                    "max_new_tokens"
                    if len(token_ids) >= self._settings.generation.max_new_tokens
                    else "stopped"
                )
            results.append(
                GenerationResult(
                    raw_output=self._tokenizer.decode(token_ids, skip_special_tokens=True),
                    input_tokens=int(encoded["attention_mask"][index].sum().item()),
                    output_tokens=len(token_ids),
                    latency_sec=per_problem_latency,
                    finish_reason=finish_reason,
                    truncated=finish_reason == "max_new_tokens",
                )
            )
        return results


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: object) -> None:
    serialized = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    _atomic_write_bytes(path, serialized + b"\n")


def _write_prediction_rows(path: Path, records: Sequence[PredictionRecord]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(PREDICTION_COLUMNS)
    writer.writerows(record.as_csv_row() for record in records)
    _atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def _append_prediction_rows(path: Path, records: Sequence[PredictionRecord]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(record.as_csv_row() for record in records)
        handle.flush()


def _prediction_from_csv(row: Mapping[str, str]) -> PredictionRecord:
    if set(row) != set(PREDICTION_COLUMNS):
        raise EvaluationError("Prediction row schema does not match E000 schema.")
    parsed = None if row["parsed_answer"] == "" else int(row["parsed_answer"])
    if row["correct"] not in {"true", "false"} or row["truncated"] not in {
        "true",
        "false",
    }:
        raise EvaluationError("Prediction booleans must be lowercase true/false.")
    return PredictionRecord(
        sample_id=row["id"],
        question=row["question"],
        gold_answer=int(row["gold_answer"]),
        raw_output=row["raw_output"],
        parsed_answer=parsed,
        correct=row["correct"] == "true",
        parse_status=row["parse_status"],
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        latency_sec=float(row["latency_sec"]),
        finish_reason=row["finish_reason"],
        truncated=row["truncated"] == "true",
        derived_category=row["derived_category"],
    )


def load_prediction_records(path: Path) -> tuple[PredictionRecord, ...]:
    rows = _read_csv(path, PREDICTION_COLUMNS, "predictions")
    try:
        return tuple(_prediction_from_csv(row) for row in rows)
    except (TypeError, ValueError) as exc:
        raise EvaluationError("Predictions contain malformed typed fields.") from exc


def build_resume_identity(
    config: LoadedConfig,
    settings: ZeroShotSettings,
    generator: BatchGenerator,
    *,
    limit: int | None,
) -> dict[str, Any]:
    """Build the exact compatibility identity required before reusing predictions."""

    identity = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "model_name": settings.model.name_or_path,
        "model_revision": settings.model.revision,
        "model_commit": generator.model_commit,
        "tokenizer_revision": settings.model.tokenizer_revision,
        "tokenizer_commit": generator.tokenizer_commit,
        "split_version": settings.data.split_version,
        "validation_sha256": settings.data.validation_sha256,
        "prompt_version": settings.prompt.version,
        "prompt_text": asdict(settings.prompt),
        "chat_template_sha256": hashlib.sha256(generator.chat_template.encode()).hexdigest(),
        "generation_config": asdict(settings.generation),
        "answer_parser_version": settings.parser_version,
        "limit": limit,
        "config_sha256": config.source_sha256,
    }
    canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {"identity": identity, "identity_sha256": hashlib.sha256(canonical).hexdigest()}


def _load_resume_records(
    resume_dir: Path,
    expected_identity: Mapping[str, Any],
    selected_rows: Sequence[ValidationRow],
    settings: OutputSettings,
) -> tuple[PredictionRecord, ...]:
    identity_path = resume_dir / settings.resume_identity_filename
    if not identity_path.is_file():
        raise EvaluationError(f"Resume identity does not exist: {identity_path}")
    try:
        actual_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError("Resume identity must be valid UTF-8 JSON.") from exc
    if actual_identity != expected_identity:
        raise EvaluationError("Resume config/model/split identity mismatch; refusing to mix runs.")

    records = load_prediction_records(resume_dir / settings.predictions_filename)
    expected_prefix = selected_rows[: len(records)]
    if [record.sample_id for record in records] != [row.sample_id for row in expected_prefix]:
        raise EvaluationError("Resume predictions must be an ordered prefix of validation IDs.")
    for record, source in zip(records, expected_prefix, strict=True):
        parsed = parse_integer_answer(record.raw_output)
        if (
            record.question != source.question
            or record.gold_answer != source.gold_answer
            or record.derived_category != source.derived_category
            or record.parsed_answer != parsed.value
            or record.parse_status != parsed.status
            or record.correct != integer_exact_match(parsed.value, source.gold_answer)
        ):
            raise EvaluationError(
                f"Resume prediction for {source.sample_id!r} failed integrity validation."
            )
    return records


def _summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 8),
        "median": round(float(np.median(array)), 8),
        "p95": round(float(np.percentile(array, 95)), 8),
        "p99": round(float(np.percentile(array, 99)), 8),
        "max": round(float(array.max()), 8),
    }


def aggregate_metrics(
    records: Sequence[PredictionRecord], *, total_wall_clock_sec: float
) -> dict[str, Any]:
    """Aggregate integer Exact Match, parse, category, token, and latency metrics."""

    total = len(records)
    correct = sum(record.correct for record in records)
    parse_failures = sum(record.parse_status.startswith("parse_failure") for record in records)
    category_records: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        category_records[record.derived_category].append(record)
    per_category = {}
    for category, members in sorted(category_records.items()):
        category_correct = sum(record.correct for record in members)
        category_parse_failures = sum(
            record.parse_status.startswith("parse_failure") for record in members
        )
        per_category[category] = {
            "derived_not_gold": True,
            "total": len(members),
            "correct": category_correct,
            "accuracy": round(category_correct / len(members), 10),
            "parse_failures": category_parse_failures,
        }
    answer_sign_records = {
        "negative": [record for record in records if record.gold_answer < 0],
        "zero": [record for record in records if record.gold_answer == 0],
        "positive": [record for record in records if record.gold_answer > 0],
    }
    per_answer_sign = {}
    for sign, members in answer_sign_records.items():
        sign_correct = sum(record.correct for record in members)
        sign_parse_failures = sum(
            record.parse_status.startswith("parse_failure") for record in members
        )
        per_answer_sign[sign] = {
            "total": len(members),
            "correct": sign_correct,
            "accuracy": round(sign_correct / len(members), 10) if members else None,
            "parse_failures": sign_parse_failures,
        }
    return {
        "metric": "integer_exact_match",
        "answer_parser_version": ANSWER_PARSER_VERSION,
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "parse_failures": parse_failures,
        "accuracy": round(correct / total, 10) if total else 0.0,
        "parse_failure_rate": round(parse_failures / total, 10) if total else 0.0,
        "per_category": per_category,
        "per_answer_sign": per_answer_sign,
        "category_labels_derived_not_gold": True,
        "input_token_statistics": _summary([record.input_tokens for record in records]),
        "output_token_statistics": _summary([record.output_tokens for record in records]),
        "latency_sec_statistics": _summary([record.latency_sec for record in records]),
        "total_wall_clock_sec": round(total_wall_clock_sec, 8),
        "finish_reason_counts": dict(sorted(Counter(r.finish_reason for r in records).items())),
        "truncated": sum(record.truncated for record in records),
        "max_new_tokens_hits": sum(record.finish_reason == "max_new_tokens" for record in records),
        "empty_outputs": sum(not record.raw_output.strip() for record in records),
        "generation_failures": 0,
        "truncation_rate": round(sum(record.truncated for record in records) / total, 10)
        if total
        else 0.0,
    }


def run_zero_shot_evaluation(
    config: LoadedConfig,
    *,
    project_root: str | Path,
    run_dir: str | Path,
    generator: BatchGenerator,
    limit: int | None = None,
    resume_from: str | Path | None = None,
) -> EvaluationResult:
    """Evaluate the frozen validation prefix with incremental, compatible resume output."""

    settings = load_zero_shot_settings(config, project_root)
    rows = load_validation_rows(settings.data)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise EvaluationError("limit must be a positive integer.")
        if limit > len(rows):
            raise EvaluationError(f"limit {limit} exceeds validation rows {len(rows)}.")
        selected_rows = rows[:limit]
    else:
        selected_rows = rows

    output_dir = Path(run_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = build_resume_identity(config, settings, generator, limit=limit)
    identity_path = output_dir / settings.output.resume_identity_filename
    _atomic_write_json(identity_path, identity)

    existing: tuple[PredictionRecord, ...] = tuple()
    if resume_from is not None:
        existing = _load_resume_records(
            Path(resume_from).resolve(), identity, selected_rows, settings.output
        )
    predictions_path = output_dir / settings.output.predictions_filename
    _write_prediction_rows(predictions_path, existing)
    completed = len(existing)
    started = time.perf_counter()
    for start in range(completed, len(selected_rows), settings.runtime.batch_size):
        batch = selected_rows[start : start + settings.runtime.batch_size]
        generated = list(generator.generate([row.question for row in batch]))
        if len(generated) != len(batch):
            raise EvaluationError(
                f"Generator returned {len(generated)} results for batch of {len(batch)}."
            )
        records: list[PredictionRecord] = []
        for source, result in zip(batch, generated, strict=True):
            parsed = parse_integer_answer(result.raw_output)
            records.append(
                PredictionRecord(
                    sample_id=source.sample_id,
                    question=source.question,
                    gold_answer=source.gold_answer,
                    raw_output=result.raw_output,
                    parsed_answer=parsed.value,
                    correct=integer_exact_match(parsed.value, source.gold_answer),
                    parse_status=parsed.status,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_sec=result.latency_sec,
                    finish_reason=result.finish_reason,
                    truncated=result.truncated,
                    derived_category=source.derived_category,
                )
            )
        _append_prediction_rows(predictions_path, records)
    total_wall_clock = time.perf_counter() - started

    final_records = load_prediction_records(predictions_path)
    if [record.sample_id for record in final_records] != [row.sample_id for row in selected_rows]:
        raise EvaluationError("Final prediction ordering/coverage does not match validation.")
    failures = [record for record in final_records if not record.correct]
    failures_path = output_dir / settings.output.failures_filename
    _write_prediction_rows(failures_path, failures)
    metrics = aggregate_metrics(final_records, total_wall_clock_sec=total_wall_clock)
    metrics["runtime"] = dict(generator.runtime_metadata())
    metrics.update(
        {
            "schema_version": 1,
            "pipeline_version": PIPELINE_VERSION,
            "experiment_id": "E000",
            "split_version": settings.data.split_version,
            "validation_sha256": settings.data.validation_sha256,
            "limit": limit,
            "resumed_predictions": len(existing),
        }
    )
    metrics_path = output_dir / settings.output.metrics_filename
    _atomic_write_json(metrics_path, metrics)
    artifact_sha256 = {
        settings.output.predictions_filename: sha256_file(predictions_path),
        settings.output.failures_filename: sha256_file(failures_path),
        settings.output.metrics_filename: sha256_file(metrics_path),
        settings.output.resume_identity_filename: sha256_file(identity_path),
    }
    return EvaluationResult(
        metrics=metrics,
        resume_identity=identity,
        predictions_path=predictions_path,
        failures_path=failures_path,
        metrics_path=metrics_path,
        resume_identity_path=identity_path,
        artifact_sha256=artifact_sha256,
    )


def build_run_manifest_fields(
    settings: ZeroShotSettings,
    generator: BatchGenerator,
    result: EvaluationResult,
    *,
    limit: int | None,
    resumed_from: str | None,
) -> dict[str, Any]:
    """Build the E000-specific run-manifest fields required for provenance."""

    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = None
    return {
        "pipeline_version": PIPELINE_VERSION,
        "model_name": settings.model.name_or_path,
        "model_revision": settings.model.revision,
        "model_commit": generator.model_commit,
        "tokenizer_revision": settings.model.tokenizer_revision,
        "tokenizer_commit": generator.tokenizer_commit,
        "split_version": settings.data.split_version,
        "val_sha256": settings.data.validation_sha256,
        "prompt_version": settings.prompt.version,
        "prompt_text": asdict(settings.prompt),
        "chat_template": generator.chat_template,
        "chat_template_sha256": hashlib.sha256(generator.chat_template.encode()).hexdigest(),
        "generation_config": asdict(settings.generation),
        "answer_parser_version": settings.parser_version,
        "device": generator.device_spec.device,
        "device_name": generator.device_spec.name,
        "dtype": generator.device_spec.dtype,
        "runtime": dict(generator.runtime_metadata()),
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "limit": limit,
        "resumed_from": resumed_from,
        "total_samples": result.metrics["total"],
        "correct": result.metrics["correct"],
        "incorrect": result.metrics["incorrect"],
        "parse_failures": result.metrics["parse_failures"],
        "accuracy": result.metrics["accuracy"],
        "artifact_sha256": result.artifact_sha256,
        "no_training": True,
        "external_datasets": [],
        "tool_use": False,
    }
