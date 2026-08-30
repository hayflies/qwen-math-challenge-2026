"""Phase 4 E001 official-only direct-answer QLoRA training utilities."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qwen_math_challenge.config import LoadedConfig
from qwen_math_challenge.data.official import sha256_file
from qwen_math_challenge.evaluation.zero_shot import (
    ALLOWED_MODEL_ID,
    ModelSettings,
    PromptSettings,
    resolve_local_model_snapshot,
)

PIPELINE_VERSION = "phase4_e001_qlora_v1"
TARGET_FORMAT_VERSION = "direct_integer_v001"
LOSS_MASK_VERSION = "assistant_only_v001"
EXPECTED_EXPERIMENT_ID = "E001"
EXPECTED_SPLIT_VERSION = "official_v001_split_v001"
EXPECTED_DATASET_VERSION = "official_v001"
EXPECTED_TRAIN_ROWS = 14_736
EXPECTED_VAL_ROWS = 1_637
EXPECTED_TRAINABLE_PARAMETERS = 29_933_568
_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


class SFTError(ValueError):
    """Raised when E001 config, data, masking, model, or resume state is invalid."""


@dataclass(frozen=True)
class SFTDataSettings:
    dataset_version: str
    split_version: str
    train_path: Path
    train_sha256: str
    train_rows: int
    validation_path: Path
    validation_sha256: str
    validation_rows: int
    groups_path: Path
    groups_sha256: str
    split_manifest_path: Path
    split_manifest_sha256: str
    length_audit_path: Path
    length_audit_sha256: str | None


@dataclass(frozen=True)
class TargetSettings:
    format_version: str
    loss_mask_version: str
    supervise_eos: bool


@dataclass(frozen=True)
class QuantizationSettings:
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str
    bnb_4bit_use_double_quant: bool


@dataclass(frozen=True)
class LoRASettings:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    bias: str
    task_type: str


@dataclass(frozen=True)
class TrainingSettings:
    method: str
    max_seq_length: int
    truncation_policy: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    num_train_epochs: int
    expected_steps_per_epoch: int
    expected_total_optimizer_steps: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    warmup_ratio: float
    expected_warmup_steps: int
    lr_scheduler_type: str
    max_grad_norm: float
    gradient_checkpointing: bool
    logging_steps: int
    eval_strategy: str
    save_strategy: str
    save_steps: int
    save_total_limit: int
    fp16: bool
    bf16: bool
    dataloader_num_workers: int
    seed: int
    data_seed: int


@dataclass(frozen=True)
class OutputSettings:
    adapter_directory: str
    training_metrics_filename: str
    telemetry_filename: str
    identity_filename: str


@dataclass(frozen=True)
class SFTSettings:
    model: ModelSettings
    data: SFTDataSettings
    prompt: PromptSettings
    target: TargetSettings
    quantization: QuantizationSettings
    lora: LoRASettings
    training: TrainingSettings
    output: OutputSettings
    seed: int
    require_clean_git_for_full_run: bool


@dataclass(frozen=True)
class SFTSourceRow:
    index: int
    sample_id: str
    question: str
    answer: int


@dataclass(frozen=True)
class EncodedExample:
    sample_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    supervised_tokens: int


@dataclass(frozen=True)
class SplitValidationResult:
    train: tuple[SFTSourceRow, ...]
    validation: tuple[SFTSourceRow, ...]
    report: dict[str, Any]


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SFTError(f"'{label}' must be a mapping.")
    return dict(value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SFTError(f"'{label}' must be a non-empty string.")
    return value


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise SFTError(f"'{label}' must be a boolean.")
    return value


def _int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SFTError(f"'{label}' must be an integer >= {minimum}.")
    return value


def _float(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SFTError(f"'{label}' must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise SFTError(f"'{label}' must be finite and >= {minimum}.")
    return result


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise SFTError(f"'{label}' must be a lowercase SHA-256 digest.")
    return value


def _path(root: Path, value: object, label: str) -> Path:
    candidate = Path(_string(value, label)).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _filename(value: object, label: str, suffix: str | None = None) -> str:
    filename = _string(value, label)
    path = Path(filename)
    if path.name != filename or (suffix is not None and path.suffix != suffix):
        raise SFTError(
            f"'{label}' must be a plain filename{f' ending in {suffix}' if suffix else ''}."
        )
    return filename


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SFTError(f"Could not read YAML mapping: {path}") from exc
    return _mapping(value, str(path))


def load_sft_settings(config: LoadedConfig, project_root: str | Path) -> SFTSettings:
    """Validate the single, canonical Phase 4 E001 configuration."""

    root = Path(project_root).resolve()
    if config.phase != 4 or config.experiment_id != EXPECTED_EXPERIMENT_ID:
        raise SFTError("E001 config must set experiment_id E001 and phase 4.")
    if config.experiment.get("model") != ALLOWED_MODEL_ID:
        raise SFTError(f"E001 permits only {ALLOWED_MODEL_ID!r}.")
    if config.experiment.get("dataset_version") != EXPECTED_DATASET_VERSION:
        raise SFTError("E001 must use dataset_version official_v001.")
    if config.experiment.get("validation_split") != EXPECTED_SPLIT_VERSION:
        raise SFTError("E001 must use the frozen official_v001_split_v001 split.")
    if config.experiment.get("external_datasets") != []:
        raise SFTError("E001 external_datasets must be an empty list.")

    model_raw = _mapping(config.raw.get("model"), "model")
    name = _string(model_raw.get("name_or_path"), "model.name_or_path")
    if name != ALLOWED_MODEL_ID:
        raise SFTError(f"E001 permits only {ALLOWED_MODEL_ID!r}; got {name!r}.")
    revision = _string(model_raw.get("revision"), "model.revision")
    tokenizer_revision = _string(model_raw.get("tokenizer_revision"), "model.tokenizer_revision")
    if tokenizer_revision != revision:
        raise SFTError("Model and tokenizer revisions must match.")
    local_files_only = _bool(model_raw.get("local_files_only"), "model.local_files_only")
    if not local_files_only:
        raise SFTError(
            "Training must use a pre-cached exact snapshot; local_files_only must be true."
        )
    trust_remote_code = _bool(model_raw.get("trust_remote_code"), "model.trust_remote_code")
    if trust_remote_code:
        raise SFTError(
            "E001 uses built-in Qwen2 Transformers code; trust_remote_code must be false."
        )

    data_raw = _mapping(config.raw.get("data"), "data")
    split_version = _string(data_raw.get("split_version"), "data.split_version")
    if split_version != EXPECTED_SPLIT_VERSION:
        raise SFTError("data.split_version must be official_v001_split_v001.")
    train_path = _path(root, data_raw.get("train_path"), "data.train_path")
    validation_path = _path(root, data_raw.get("validation_path"), "data.validation_path")
    groups_path = _path(root, data_raw.get("groups_path"), "data.groups_path")
    split_manifest_path = _path(
        root, data_raw.get("split_manifest_path"), "data.split_manifest_path"
    )
    split_root = (root / "data" / "splits").resolve()
    for label, candidate in {
        "train": train_path,
        "validation": validation_path,
        "groups": groups_path,
        "split manifest": split_manifest_path,
    }.items():
        if split_root not in candidate.parents:
            raise SFTError(f"E001 {label} must be a versioned file below data/splits.")
    length_audit_path = _path(root, data_raw.get("length_audit_path"), "data.length_audit_path")
    audit_sha_value = data_raw.get("length_audit_sha256")
    length_audit_sha = (
        None if audit_sha_value is None else _sha(audit_sha_value, "data.length_audit_sha256")
    )

    prompt_raw = _mapping(config.raw.get("prompt"), "prompt")
    prompt = PromptSettings(
        version=_string(prompt_raw.get("version"), "prompt.version"),
        system_text=_string(prompt_raw.get("system_text"), "prompt.system_text"),
        user_template=_string(prompt_raw.get("user_template"), "prompt.user_template"),
    )
    if prompt.version != "zero_shot_v001" or prompt.user_template.count("{question}") != 1:
        raise SFTError("E001 must use zero_shot_v001 with exactly one {question} field.")
    try:
        prompt.user_template.format(question="probe")
    except (KeyError, ValueError) as exc:
        raise SFTError("prompt.user_template contains unsupported format fields.") from exc
    reference_path = _path(
        root, prompt_raw.get("reference_config_path"), "prompt.reference_config_path"
    )
    reference_sha = _sha(
        prompt_raw.get("reference_config_sha256"), "prompt.reference_config_sha256"
    )
    if not reference_path.is_file() or sha256_file(reference_path) != reference_sha:
        raise SFTError("Canonical E000 prompt reference hash mismatch.")
    reference = _load_yaml_mapping(reference_path)
    if reference.get("prompt") != {
        "version": prompt.version,
        "system_text": prompt.system_text,
        "user_template": prompt.user_template,
    }:
        raise SFTError("E001 prompt must exactly match the canonical E000 prompt mapping.")

    target_raw = _mapping(config.raw.get("target"), "target")
    target = TargetSettings(
        format_version=_string(target_raw.get("format_version"), "target.format_version"),
        loss_mask_version=_string(target_raw.get("loss_mask_version"), "target.loss_mask_version"),
        supervise_eos=_bool(target_raw.get("supervise_eos"), "target.supervise_eos"),
    )
    if target != TargetSettings(TARGET_FORMAT_VERSION, LOSS_MASK_VERSION, True):
        raise SFTError(
            "E001 target must be direct_integer_v001 with assistant_only_v001 and EOS supervision."
        )

    quant_raw = _mapping(config.raw.get("quantization"), "quantization")
    quantization = QuantizationSettings(
        load_in_4bit=_bool(quant_raw.get("load_in_4bit"), "quantization.load_in_4bit"),
        bnb_4bit_quant_type=_string(
            quant_raw.get("bnb_4bit_quant_type"), "quantization.bnb_4bit_quant_type"
        ),
        bnb_4bit_compute_dtype=_string(
            quant_raw.get("bnb_4bit_compute_dtype"),
            "quantization.bnb_4bit_compute_dtype",
        ),
        bnb_4bit_use_double_quant=_bool(
            quant_raw.get("bnb_4bit_use_double_quant"),
            "quantization.bnb_4bit_use_double_quant",
        ),
    )
    if quantization != QuantizationSettings(True, "nf4", "float16", True):
        raise SFTError("Canonical E001 quantization must be 4-bit NF4, FP16 compute, double quant.")

    lora_raw = _mapping(config.raw.get("lora"), "lora")
    target_modules_raw = lora_raw.get("target_modules")
    if not isinstance(target_modules_raw, list) or not all(
        isinstance(value, str) and value for value in target_modules_raw
    ):
        raise SFTError("lora.target_modules must be a non-empty string list.")
    target_modules = tuple(target_modules_raw)
    if len(set(target_modules)) != len(target_modules):
        raise SFTError("lora.target_modules must not contain duplicates.")
    if set(target_modules) != _ALLOWED_TARGET_MODULES:
        raise SFTError(
            "Canonical E001 must target Qwen2 q/k/v/o and gate/up/down projection modules."
        )
    lora = LoRASettings(
        r=_int(lora_raw.get("r"), "lora.r", minimum=1),
        lora_alpha=_int(lora_raw.get("lora_alpha"), "lora.lora_alpha", minimum=1),
        lora_dropout=_float(lora_raw.get("lora_dropout"), "lora.lora_dropout"),
        target_modules=target_modules,
        bias=_string(lora_raw.get("bias"), "lora.bias"),
        task_type=_string(lora_raw.get("task_type"), "lora.task_type"),
    )
    if lora.bias != "none" or lora.task_type != "CAUSAL_LM":
        raise SFTError("E001 LoRA bias/task_type must be none/CAUSAL_LM.")

    training_raw = _mapping(config.raw.get("training"), "training")
    training = TrainingSettings(
        method=_string(training_raw.get("method"), "training.method"),
        max_seq_length=_int(
            training_raw.get("max_seq_length"), "training.max_seq_length", minimum=1
        ),
        truncation_policy=_string(
            training_raw.get("truncation_policy"), "training.truncation_policy"
        ),
        per_device_train_batch_size=_int(
            training_raw.get("per_device_train_batch_size"),
            "training.per_device_train_batch_size",
            minimum=1,
        ),
        per_device_eval_batch_size=_int(
            training_raw.get("per_device_eval_batch_size"),
            "training.per_device_eval_batch_size",
            minimum=1,
        ),
        gradient_accumulation_steps=_int(
            training_raw.get("gradient_accumulation_steps"),
            "training.gradient_accumulation_steps",
            minimum=1,
        ),
        effective_batch_size=_int(
            training_raw.get("effective_batch_size"), "training.effective_batch_size", minimum=1
        ),
        num_train_epochs=_int(
            training_raw.get("num_train_epochs"), "training.num_train_epochs", minimum=1
        ),
        expected_steps_per_epoch=_int(
            training_raw.get("expected_steps_per_epoch"),
            "training.expected_steps_per_epoch",
            minimum=1,
        ),
        expected_total_optimizer_steps=_int(
            training_raw.get("expected_total_optimizer_steps"),
            "training.expected_total_optimizer_steps",
            minimum=1,
        ),
        learning_rate=_float(
            training_raw.get("learning_rate"), "training.learning_rate", minimum=1e-12
        ),
        optimizer=_string(training_raw.get("optimizer"), "training.optimizer"),
        weight_decay=_float(training_raw.get("weight_decay"), "training.weight_decay"),
        warmup_ratio=_float(training_raw.get("warmup_ratio"), "training.warmup_ratio"),
        expected_warmup_steps=_int(
            training_raw.get("expected_warmup_steps"), "training.expected_warmup_steps"
        ),
        lr_scheduler_type=_string(
            training_raw.get("lr_scheduler_type"), "training.lr_scheduler_type"
        ),
        max_grad_norm=_float(
            training_raw.get("max_grad_norm"), "training.max_grad_norm", minimum=1e-12
        ),
        gradient_checkpointing=_bool(
            training_raw.get("gradient_checkpointing"), "training.gradient_checkpointing"
        ),
        logging_steps=_int(training_raw.get("logging_steps"), "training.logging_steps", minimum=1),
        eval_strategy=_string(training_raw.get("eval_strategy"), "training.eval_strategy"),
        save_strategy=_string(training_raw.get("save_strategy"), "training.save_strategy"),
        save_steps=_int(training_raw.get("save_steps"), "training.save_steps", minimum=1),
        save_total_limit=_int(
            training_raw.get("save_total_limit"), "training.save_total_limit", minimum=1
        ),
        fp16=_bool(training_raw.get("fp16"), "training.fp16"),
        bf16=_bool(training_raw.get("bf16"), "training.bf16"),
        dataloader_num_workers=_int(
            training_raw.get("dataloader_num_workers"), "training.dataloader_num_workers"
        ),
        seed=_int(training_raw.get("seed"), "training.seed"),
        data_seed=_int(training_raw.get("data_seed"), "training.data_seed"),
    )
    if training.method != "qlora" or training.truncation_policy != "error":
        raise SFTError("Canonical E001 uses QLoRA and refuses sequence truncation.")
    if training.fp16 is not True or training.bf16 is not False:
        raise SFTError("Canonical T4 E001 uses FP16, not BF16.")
    if training.seed != config.seed or training.data_seed != config.seed:
        raise SFTError("Config, Trainer, and data seeds must all match.")
    effective = training.per_device_train_batch_size * training.gradient_accumulation_steps
    steps_per_epoch = math.ceil(EXPECTED_TRAIN_ROWS / effective)
    total_steps = steps_per_epoch * training.num_train_epochs
    warmup_steps = math.ceil(total_steps * training.warmup_ratio)
    if (
        training.effective_batch_size != effective
        or training.expected_steps_per_epoch != steps_per_epoch
        or training.expected_total_optimizer_steps != total_steps
        or training.expected_warmup_steps != warmup_steps
    ):
        raise SFTError("Recorded effective batch/step/warmup calculations are inconsistent.")

    output_raw = _mapping(config.raw.get("output"), "output")
    output = OutputSettings(
        adapter_directory=_filename(
            output_raw.get("adapter_directory"), "output.adapter_directory"
        ),
        training_metrics_filename=_filename(
            output_raw.get("training_metrics_filename"),
            "output.training_metrics_filename",
            ".json",
        ),
        telemetry_filename=_filename(
            output_raw.get("telemetry_filename"), "output.telemetry_filename", ".jsonl"
        ),
        identity_filename=_filename(
            output_raw.get("identity_filename"), "output.identity_filename", ".json"
        ),
    )
    provenance = _mapping(config.raw.get("provenance"), "provenance")
    return SFTSettings(
        model=ModelSettings(
            name, revision, tokenizer_revision, local_files_only, trust_remote_code
        ),
        data=SFTDataSettings(
            dataset_version=EXPECTED_DATASET_VERSION,
            split_version=split_version,
            train_path=train_path,
            train_sha256=_sha(data_raw.get("train_sha256"), "data.train_sha256"),
            train_rows=_int(data_raw.get("train_rows"), "data.train_rows", minimum=1),
            validation_path=validation_path,
            validation_sha256=_sha(data_raw.get("validation_sha256"), "data.validation_sha256"),
            validation_rows=_int(
                data_raw.get("validation_rows"), "data.validation_rows", minimum=1
            ),
            groups_path=groups_path,
            groups_sha256=_sha(data_raw.get("groups_sha256"), "data.groups_sha256"),
            split_manifest_path=split_manifest_path,
            split_manifest_sha256=_sha(
                data_raw.get("split_manifest_sha256"), "data.split_manifest_sha256"
            ),
            length_audit_path=length_audit_path,
            length_audit_sha256=length_audit_sha,
        ),
        prompt=prompt,
        target=target,
        quantization=quantization,
        lora=lora,
        training=training,
        output=output,
        seed=config.seed,
        require_clean_git_for_full_run=_bool(
            provenance.get("require_clean_git_for_full_run"),
            "provenance.require_clean_git_for_full_run",
        ),
    )


def canonical_integer_target(value: int | str) -> str:
    """Return the only semantic supervision allowed in E001: one canonical integer."""

    if isinstance(value, bool):
        raise SFTError("Boolean values are not integer answer targets.")
    text = str(value)
    if not _INTEGER_PATTERN.fullmatch(text):
        raise SFTError(f"Direct-answer target is not an integer: {text!r}.")
    return str(int(text))


def _read_csv(path: Path, columns: Sequence[str], label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise SFTError(f"{label} does not exist: {path}")
    try:
        handle = path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise SFTError(f"Could not open {label}: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise SFTError(f"{label} columns must be {list(columns)}, got {reader.fieldnames!r}.")
        return list(reader)


def _verify_file_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path) if path.is_file() else None
    if actual != expected:
        raise SFTError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}.")


def _source_rows(rows: Sequence[Mapping[str, str]], label: str) -> tuple[SFTSourceRow, ...]:
    output: list[SFTSourceRow] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if any(not row[field] for field in ("id", "question", "answer")):
            raise SFTError(f"{label} row {index + 2} contains an empty required value.")
        sample_id = row["id"]
        if sample_id in seen:
            raise SFTError(f"{label} has duplicate ID {sample_id!r}.")
        seen.add(sample_id)
        try:
            answer = int(canonical_integer_target(row["answer"]))
        except SFTError as exc:
            raise SFTError(f"{label} answer for {sample_id!r} is invalid.") from exc
        output.append(SFTSourceRow(index, sample_id, row["question"], answer))
    return tuple(output)


def validate_official_split(settings: SFTDataSettings) -> SplitValidationResult:
    """Re-assert split hashes, row coverage, ID isolation, and group isolation."""

    for path, expected, label in (
        (settings.train_path, settings.train_sha256, "training split"),
        (settings.validation_path, settings.validation_sha256, "validation split"),
        (settings.groups_path, settings.groups_sha256, "groups artifact"),
        (
            settings.split_manifest_path,
            settings.split_manifest_sha256,
            "split manifest",
        ),
    ):
        _verify_file_hash(path, expected, label)
    if settings.train_rows != EXPECTED_TRAIN_ROWS or settings.validation_rows != EXPECTED_VAL_ROWS:
        raise SFTError("E001 row invariants must be exactly 14736 train and 1637 validation.")

    train = _source_rows(
        _read_csv(settings.train_path, ("id", "question", "answer"), "training split"),
        "training split",
    )
    validation = _source_rows(
        _read_csv(
            settings.validation_path,
            ("id", "question", "answer"),
            "validation split",
        ),
        "validation split",
    )
    if len(train) != settings.train_rows or len(validation) != settings.validation_rows:
        raise SFTError("Actual E001 split row counts do not match the frozen config.")
    train_ids = {row.sample_id for row in train}
    validation_ids = {row.sample_id for row in validation}
    id_overlap = train_ids & validation_ids
    if id_overlap:
        raise SFTError(f"Training/validation ID leakage detected ({len(id_overlap)} IDs).")

    group_rows = _read_csv(
        settings.groups_path,
        ("id", "group_id", "group_size", "split", "derived_category"),
        "groups artifact",
    )
    group_by_id: dict[str, str] = {}
    train_groups: set[str] = set()
    validation_groups: set[str] = set()
    for row in group_rows:
        sample_id = row["id"]
        if sample_id in group_by_id:
            raise SFTError(f"groups artifact has duplicate ID {sample_id!r}.")
        if row["split"] not in {"train", "validation"}:
            raise SFTError(f"groups artifact has invalid split {row['split']!r}.")
        group_by_id[sample_id] = row["group_id"]
        if row["split"] == "train":
            if sample_id not in train_ids:
                raise SFTError(
                    f"Group row {sample_id!r} is marked train but absent from train.csv."
                )
            train_groups.add(row["group_id"])
        else:
            if sample_id not in validation_ids:
                raise SFTError(
                    f"Group row {sample_id!r} is marked validation but absent from val.csv."
                )
            validation_groups.add(row["group_id"])
    if set(group_by_id) != train_ids | validation_ids:
        raise SFTError("groups.csv ID coverage does not exactly match train+validation IDs.")
    group_overlap = train_groups & validation_groups
    if group_overlap:
        raise SFTError(f"Training/validation group leakage detected ({len(group_overlap)} groups).")

    try:
        manifest = json.loads(settings.split_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SFTError("Split manifest must be valid UTF-8 JSON.") from exc
    expected_manifest = {
        "split_version": settings.split_version,
        "train_rows": settings.train_rows,
        "val_rows": settings.validation_rows,
        "train_sha256": settings.train_sha256,
        "val_sha256": settings.validation_sha256,
        "groups_sha256": settings.groups_sha256,
        "train_val_id_overlap": 0,
        "train_val_group_overlap": 0,
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise SFTError(
            f"Split manifest identity mismatch: {json.dumps(mismatches, sort_keys=True)}"
        )
    return SplitValidationResult(
        train=train,
        validation=validation,
        report={
            "status": "PASS",
            "dataset_version": settings.dataset_version,
            "split_version": settings.split_version,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_sha256": settings.train_sha256,
            "validation_sha256": settings.validation_sha256,
            "groups_sha256": settings.groups_sha256,
            "split_manifest_sha256": settings.split_manifest_sha256,
            "train_validation_id_overlap": 0,
            "train_validation_group_overlap": 0,
            "validation_used_for_training": False,
        },
    )


def load_training_tokenizer(settings: SFTSettings) -> tuple[Any, str]:
    """Load only the exact cached Qwen tokenizer and return its resolved commit."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise SFTError("Transformers is required to load the E001 tokenizer.") from exc
    snapshot = resolve_local_model_snapshot(settings.model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot.path,
            local_files_only=True,
            trust_remote_code=False,
        )
    except OSError as exc:  # pragma: no cover - environment boundary
        raise SFTError("The exact Qwen tokenizer snapshot is incomplete.") from exc
    if not tokenizer.chat_template:
        raise SFTError("Qwen tokenizer does not contain its official chat template.")
    if tokenizer.eos_token_id is None:
        raise SFTError("Qwen tokenizer must define eos_token_id.")
    if tokenizer.pad_token_id is None:
        raise SFTError("Qwen tokenizer must define a distinct padding policy.")
    tokenizer.padding_side = "right"
    return tokenizer, snapshot.commit_hash


def build_training_messages(
    question: str, answer: int | str, prompt: PromptSettings
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build E000-aligned prompt messages and append one direct integer answer."""

    if not isinstance(question, str) or not question:
        raise SFTError("Question must be a non-empty string.")
    prompt_messages = [
        {"role": "system", "content": prompt.system_text},
        {"role": "user", "content": prompt.user_template.format(question=question)},
    ]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": canonical_integer_target(answer)},
    ]
    return prompt_messages, full_messages


def _token_ids(value: object, label: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(token, int) for token in value):
        raise SFTError(f"Tokenizer {label} output must be a sequence of integer token IDs.")
    return list(value)


def encode_sft_example(
    tokenizer: Any,
    row: SFTSourceRow,
    prompt: PromptSettings,
    *,
    max_seq_length: int,
) -> EncodedExample:
    """Apply Qwen's chat template and mask every non-assistant-answer token."""

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise SFTError("Tokenizer does not provide apply_chat_template().")
    prompt_messages, full_messages = build_training_messages(row.question, row.answer, prompt)
    prompt_ids = _token_ids(
        apply_template(prompt_messages, tokenize=True, add_generation_prompt=True),
        "prompt",
    )
    full_ids = _token_ids(
        apply_template(full_messages, tokenize=True, add_generation_prompt=False),
        "full conversation",
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise SFTError(
            f"Chat template assistant boundary is not prefix-aligned for {row.sample_id!r}."
        )
    tail = full_ids[len(prompt_ids) :]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None or eos_token_id not in tail:
        raise SFTError(f"Assistant target for {row.sample_id!r} does not contain EOS.")
    eos_index = tail.index(eos_token_id)
    if eos_index == 0:
        raise SFTError(f"Assistant target for {row.sample_id!r} has no integer answer token.")
    supervised_tail = tail[: eos_index + 1]
    input_ids = prompt_ids + supervised_tail
    if len(input_ids) > max_seq_length:
        raise SFTError(
            f"Sequence {row.sample_id!r} has {len(input_ids)} tokens, exceeding "
            f"max_seq_length={max_seq_length}; E001 refuses silent truncation."
        )
    labels = [-100] * len(prompt_ids) + supervised_tail
    if labels[: len(prompt_ids)] != [-100] * len(prompt_ids):
        raise AssertionError("Prompt loss mask invariant failed.")
    if labels[-1] != eos_token_id:
        raise AssertionError("EOS must be the final supervised token.")
    return EncodedExample(
        sample_id=row.sample_id,
        input_ids=tuple(input_ids),
        attention_mask=(1,) * len(input_ids),
        labels=tuple(labels),
        prompt_tokens=len(prompt_ids),
        supervised_tokens=len(supervised_tail),
    )


def _numeric_summary(values: Sequence[int]) -> dict[str, int | float]:
    array = np.asarray(values, dtype=np.int64)
    return {
        "min": int(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def audit_token_lengths(
    rows: Sequence[SFTSourceRow],
    tokenizer: Any,
    settings: SFTSettings,
    *,
    tokenizer_commit: str,
) -> dict[str, Any]:
    """Measure complete direct-answer examples with no truncation."""

    encoded = [
        encode_sft_example(
            tokenizer,
            row,
            settings.prompt,
            max_seq_length=settings.training.max_seq_length,
        )
        for row in rows
    ]
    lengths = [len(item.input_ids) for item in encoded]
    prompt_lengths = [item.prompt_tokens for item in encoded]
    target_lengths = [item.supervised_tokens for item in encoded]
    thresholds = (512, 768, 1024, settings.training.max_seq_length)
    return {
        "schema_version": 1,
        "audit_version": "e001_token_length_v001",
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "model_name": settings.model.name_or_path,
        "model_revision": settings.model.revision,
        "tokenizer_revision": settings.model.tokenizer_revision,
        "tokenizer_commit": tokenizer_commit,
        "dataset_version": settings.data.dataset_version,
        "split_version": settings.data.split_version,
        "train_sha256": settings.data.train_sha256,
        "rows": len(encoded),
        "prompt_version": settings.prompt.version,
        "target_format_version": settings.target.format_version,
        "loss_mask_version": settings.target.loss_mask_version,
        "assistant_eos_supervised": True,
        "post_eos_template_tokens_excluded": True,
        "total_token_length": _numeric_summary(lengths),
        "prompt_token_length": _numeric_summary(prompt_lengths),
        "supervised_token_length": _numeric_summary(target_lengths),
        "counts_above_threshold": {
            str(threshold): sum(length > threshold for length in lengths)
            for threshold in thresholds
        },
        "max_seq_length": settings.training.max_seq_length,
        "truncation_policy": settings.training.truncation_policy,
        "truncated_rows": 0,
        "truncated_percentage": 0.0,
        "decision": (
            "max_seq_length=1152 preserves every official training example; dynamic padding "
            "avoids padding shorter examples to the maximum."
        ),
    }


def validate_length_audit(settings: SFTSettings) -> dict[str, Any]:
    """Verify the versioned audit before any model training begins."""

    if settings.data.length_audit_sha256 is None:
        raise SFTError("Canonical training requires a frozen data.length_audit_sha256.")
    _verify_file_hash(
        settings.data.length_audit_path,
        settings.data.length_audit_sha256,
        "token length audit",
    )
    try:
        audit = json.loads(settings.data.length_audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SFTError("Token length audit must be valid UTF-8 JSON.") from exc
    expected = {
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "model_revision": settings.model.revision,
        "tokenizer_revision": settings.model.tokenizer_revision,
        "train_sha256": settings.data.train_sha256,
        "rows": settings.data.train_rows,
        "prompt_version": settings.prompt.version,
        "target_format_version": settings.target.format_version,
        "loss_mask_version": settings.target.loss_mask_version,
        "max_seq_length": settings.training.max_seq_length,
        "truncated_rows": 0,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise SFTError("Token length audit identity does not match the E001 config.")
    return audit


class EncodedSFTDataset:
    """Small torch-compatible dataset without adding the datasets dependency."""

    def __init__(self, examples: Sequence[EncodedExample]) -> None:
        self.examples = tuple(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        item = self.examples[index]
        return {
            "input_ids": list(item.input_ids),
            "attention_mask": list(item.attention_mask),
            "labels": list(item.labels),
        }


class AssistantOnlyDataCollator:
    """Right-pad inputs while preserving -100 prompt and padding labels."""

    def __init__(self, pad_token_id: int, *, pad_to_multiple_of: int = 8) -> None:
        if pad_token_id < 0 or pad_to_multiple_of < 1:
            raise SFTError("Invalid collator padding configuration.")
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, Any]:
        if not features:
            raise SFTError("Cannot collate an empty batch.")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment boundary
            raise SFTError("PyTorch is required for SFT collation.") from exc
        maximum = max(len(feature["input_ids"]) for feature in features)
        width = math.ceil(maximum / self.pad_to_multiple_of) * self.pad_to_multiple_of
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            input_ids = list(feature["input_ids"])
            attention_mask = list(feature["attention_mask"])
            labels = list(feature["labels"])
            if not (len(input_ids) == len(attention_mask) == len(labels)):
                raise SFTError("input_ids, attention_mask, and labels must have equal lengths.")
            padding = width - len(input_ids)
            batch["input_ids"].append(input_ids + [self.pad_token_id] * padding)
            batch["attention_mask"].append(attention_mask + [0] * padding)
            batch["labels"].append(labels + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def validate_lora_target_modules(model: Any, targets: Sequence[str]) -> dict[str, int]:
    """Confirm configured targets against the instantiated Qwen2 architecture."""

    counts = {target: 0 for target in targets}
    for name, _module in model.named_modules():
        terminal_name = name.rsplit(".", maxsplit=1)[-1]
        if terminal_name in counts:
            counts[terminal_name] += 1
    missing = sorted(target for target, count in counts.items() if count == 0)
    if missing:
        raise SFTError(f"LoRA target modules are absent from this model: {missing}")
    if set(targets) != _ALLOWED_TARGET_MODULES:
        raise SFTError("LoRA targets differ from the frozen Qwen2 E001 target set.")
    return counts


def derive_lora_parameter_breakdown(
    model: Any, targets: Sequence[str], rank: int
) -> dict[str, Any]:
    """Derive LoRA A/B sizes from exact target-module dimensions."""

    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise SFTError("LoRA rank must be a positive integer.")
    target_counts = validate_lora_target_modules(model, targets)
    dimensions: dict[str, dict[tuple[int, int], int]] = {target: {} for target in targets}
    for name, module in model.named_modules():
        terminal_name = name.rsplit(".", maxsplit=1)[-1]
        if terminal_name not in dimensions:
            continue
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if (
            isinstance(in_features, bool)
            or not isinstance(in_features, int)
            or in_features < 1
            or isinstance(out_features, bool)
            or not isinstance(out_features, int)
            or out_features < 1
        ):
            raise SFTError(
                f"LoRA target module lacks positive integer dimensions: {name} "
                f"(in_features={in_features}, out_features={out_features})."
            )
        shape = (in_features, out_features)
        dimensions[terminal_name][shape] = dimensions[terminal_name].get(shape, 0) + 1

    by_target: dict[str, Any] = {}
    expected_trainable = 0
    for target in targets:
        dimension_counts = []
        target_a = 0
        target_b = 0
        for (in_features, out_features), module_count in sorted(dimensions[target].items()):
            lora_a = rank * in_features * module_count
            lora_b = rank * out_features * module_count
            target_a += lora_a
            target_b += lora_b
            dimension_counts.append(
                {
                    "in_features": in_features,
                    "out_features": out_features,
                    "module_count": module_count,
                    "lora_a_parameters": lora_a,
                    "lora_b_parameters": lora_b,
                    "total_lora_parameters": lora_a + lora_b,
                }
            )
        if sum(item["module_count"] for item in dimension_counts) != target_counts[target]:
            raise SFTError(f"LoRA dimension count mismatch for target {target}.")
        target_total = target_a + target_b
        expected_trainable += target_total
        by_target[target] = {
            "module_count": target_counts[target],
            "dimensions": dimension_counts,
            "lora_a_parameters": target_a,
            "lora_b_parameters": target_b,
            "total_lora_parameters": target_total,
        }
    return {
        "rank": rank,
        "target_module_counts": target_counts,
        "by_target": by_target,
        "expected_trainable_parameters": expected_trainable,
    }


def _logical_parameter_count(parameter: Any) -> int:
    """Return logical elements, undoing BitsAndBytes' packed 4-bit storage count."""

    count = int(parameter.numel())
    if count == 0 and hasattr(parameter, "ds_numel"):
        count = int(parameter.ds_numel)
    if parameter.__class__.__name__ == "Params4bit":
        num_bytes = int(parameter.element_size()) if hasattr(parameter, "element_size") else 1
        count *= 2 * num_bytes
    return count


def logical_model_parameter_count(model: Any) -> int:
    """Count logical model parameters consistently with PEFT's 4-bit reporting."""

    return sum(_logical_parameter_count(parameter) for parameter in model.parameters())


def inspect_base_model_architecture(settings: SFTSettings) -> dict[str, Any]:
    """Inspect Qwen module names on the meta device without loading model weights."""

    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise SFTError(
            "Accelerate and Transformers are required for architecture inspection."
        ) from exc
    snapshot = resolve_local_model_snapshot(settings.model)
    config = AutoConfig.from_pretrained(
        snapshot.path, local_files_only=True, trust_remote_code=False
    )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.tie_weights()
    breakdown = derive_lora_parameter_breakdown(
        model, settings.lora.target_modules, settings.lora.r
    )
    return {
        "model_type": config.model_type,
        "architectures": list(config.architectures or []),
        "num_hidden_layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "max_position_embeddings": int(config.max_position_embeddings),
        "base_parameters": logical_model_parameter_count(model),
        "target_module_counts": breakdown["target_module_counts"],
        "lora_parameter_breakdown": breakdown,
        "model_commit": snapshot.commit_hash,
    }


def trainable_parameter_report(
    model: Any,
    *,
    expected_trainable_parameters: int | None = None,
    expected_total_parameters: int | None = None,
    target_breakdown: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail if a QLoRA run accidentally unfreezes non-LoRA base parameters."""

    total = 0
    trainable = 0
    unexpected: list[str] = []
    target_names = tuple((target_breakdown or {}).get("by_target", {}))
    trainable_by_target = {target: 0 for target in target_names}
    unattributed_lora: list[str] = []
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = _logical_parameter_count(parameter)
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
            if "lora_" not in name:
                unexpected.append(name)
            elif target_names:
                name_parts = set(name.split("."))
                matched_targets = [target for target in target_names if target in name_parts]
                if len(matched_targets) == 1:
                    trainable_by_target[matched_targets[0]] += count
                else:
                    unattributed_lora.append(name)
    report = {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_percentage": None if total == 0 else round(100.0 * trainable / total, 8),
        "trainable_tensor_count": len(trainable_names),
        "actual_trainable_parameters_by_target": trainable_by_target,
        "unexpected_trainable_parameters": unexpected[:10],
        "unexpected_trainable_parameter_count": len(unexpected),
        "unattributed_trainable_lora_parameters": unattributed_lora[:10],
    }
    if expected_trainable_parameters is not None:
        report["expected_trainable_parameters"] = expected_trainable_parameters
    if expected_total_parameters is not None:
        report["expected_total_parameters"] = expected_total_parameters
    if target_breakdown is not None:
        report["target_module_breakdown"] = dict(target_breakdown)

    failures = []
    if total == 0 or trainable == 0:
        failures.append("QLoRA model must have nonzero total and trainable parameters")
    if unexpected:
        failures.append("Unexpected trainable base-model parameters")
    if unattributed_lora:
        failures.append("Trainable LoRA parameters could not be attributed to a configured target")
    if expected_trainable_parameters is not None and trainable != expected_trainable_parameters:
        failures.append("Actual trainable parameter count differs from analytical expectation")
    if expected_total_parameters is not None and total != expected_total_parameters:
        failures.append("Actual total parameter count differs from analytical expectation")
    if target_breakdown is not None:
        expected_by_target = {
            target: int(details["total_lora_parameters"])
            for target, details in target_breakdown["by_target"].items()
        }
        if trainable_by_target != expected_by_target:
            failures.append("Actual LoRA parameter counts differ by target")
    if failures:
        diagnostic = {"failures": failures, **report}
        raise SFTError(
            "QLoRA parameter validation failed: "
            + json.dumps(diagnostic, sort_keys=True, ensure_ascii=True)
        )
    return report


def build_training_identity(
    config: LoadedConfig,
    settings: SFTSettings,
    *,
    tokenizer_commit: str,
    git_commit: str,
    git_dirty: bool,
    limit: int | None,
    max_steps: int | None,
) -> dict[str, Any]:
    """Build the exact identity that checkpoints and adapters must preserve."""

    identity = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "config_sha256": config.source_sha256,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "base_model_name": settings.model.name_or_path,
        "base_model_revision": settings.model.revision,
        "tokenizer_revision": settings.model.tokenizer_revision,
        "tokenizer_commit": tokenizer_commit,
        "dataset_version": settings.data.dataset_version,
        "split_version": settings.data.split_version,
        "train_sha256": settings.data.train_sha256,
        "train_rows": settings.data.train_rows,
        "validation_sha256": settings.data.validation_sha256,
        "validation_rows": settings.data.validation_rows,
        "groups_sha256": settings.data.groups_sha256,
        "prompt": asdict(settings.prompt),
        "target": asdict(settings.target),
        "quantization": asdict(settings.quantization),
        "lora": asdict(settings.lora),
        "training": asdict(settings.training),
        "seed": settings.seed,
        "limit": limit,
        "max_steps_override": max_steps,
        "canonical_full_run": limit is None and max_steps is None,
        "external_datasets": [],
        "reasoning_augmentation": False,
    }
    canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "identity": identity,
        "identity_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_resume_checkpoint(
    checkpoint: str | Path, expected_identity: Mapping[str, Any]
) -> Path:
    """Reject checkpoints created from any other model/data/config/override identity."""

    path = Path(checkpoint).expanduser().resolve()
    metadata_path = path / "checkpoint_metadata.json"
    if not metadata_path.is_file():
        raise SFTError(f"Resume checkpoint metadata does not exist: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SFTError("Resume checkpoint metadata must be valid UTF-8 JSON.") from exc
    if metadata.get("training_identity") != expected_identity:
        raise SFTError("Resume checkpoint identity mismatch; refusing to mix E001 runs.")
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
    missing = [filename for filename in required if not (path / filename).is_file()]
    if missing:
        raise SFTError(f"Resume checkpoint is incomplete; missing {missing}.")
    return path


def _directory_sha256(path: Path, *, excluded: set[str] | None = None) -> str:
    excluded = excluded or set()
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.relative_to(path).as_posix() not in excluded
    )
    if not files:
        raise SFTError(f"Cannot hash empty artifact directory: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = candidate.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_adapter_compatibility(
    adapter_path: str | Path,
    settings: SFTSettings,
    *,
    expected_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify adapter metadata so base-only E000 cannot be mistaken for E001."""

    path = Path(adapter_path).expanduser().resolve()
    config_path = path / "adapter_config.json"
    metadata_path = path / "e001_metadata.json"
    weights = [path / "adapter_model.safetensors", path / "adapter_model.bin"]
    if (
        not config_path.is_file()
        or not metadata_path.is_file()
        or not any(candidate.is_file() for candidate in weights)
    ):
        raise SFTError("Adapter directory is missing config, E001 metadata, or adapter weights.")
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SFTError("Adapter metadata must be valid UTF-8 JSON.") from exc
    adapter_base = adapter_config.get("base_model_name_or_path")
    if adapter_base != settings.model.name_or_path:
        cached_snapshot = str(resolve_local_model_snapshot(settings.model).path)
        if adapter_base != cached_snapshot:
            raise SFTError("Adapter base_model_name_or_path is not the designated Qwen base.")
    expected_metadata = {
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "base_model_name": settings.model.name_or_path,
        "base_model_revision": settings.model.revision,
        "train_sha256": settings.data.train_sha256,
        "validation_sha256": settings.data.validation_sha256,
        "prompt_version": settings.prompt.version,
        "target_format_version": settings.target.format_version,
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise SFTError("Adapter E001 metadata does not match the canonical config.")
    if (
        expected_identity_sha256 is not None
        and metadata.get("training_identity_sha256") != expected_identity_sha256
    ):
        raise SFTError("Adapter training identity hash mismatch.")
    return {
        "status": "PASS",
        "adapter_path": str(path),
        "adapter_sha256": _directory_sha256(path),
        "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
        "training_identity_sha256": metadata.get("training_identity_sha256"),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _normalize_telemetry_value(value: Any, *, path: str = "") -> tuple[Any, list[dict[str, str]]]:
    """Replace non-finite telemetry floats with null and record their stable paths."""

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric if isinstance(value, np.floating) else value, []
        label = "NaN" if math.isnan(numeric) else "+Infinity" if numeric > 0 else "-Infinity"
        return None, [{"path": path or "$", "value": label}]
    if isinstance(value, Mapping):
        normalized = {}
        issues = []
        for key, nested in value.items():
            nested_path = f"{path}.{key}" if path else str(key)
            safe_value, nested_issues = _normalize_telemetry_value(nested, path=nested_path)
            normalized[key] = safe_value
            issues.extend(nested_issues)
        return normalized, issues
    if isinstance(value, (list, tuple)):
        normalized = []
        issues = []
        for index, nested in enumerate(value):
            nested_path = f"{path}[{index}]" if path else f"[{index}]"
            safe_value, nested_issues = _normalize_telemetry_value(nested, path=nested_path)
            normalized.append(safe_value)
            issues.extend(nested_issues)
        return normalized, issues
    return value, []


def _validate_finite_training_metrics(values: Mapping[str, Any]) -> None:
    """Reject non-finite objectives while allowing recoverable AMP gradient overflows."""

    for key in ("loss", "eval_loss", "train_loss"):
        if key in values and not math.isfinite(float(values[key])):
            raise FloatingPointError(f"Non-finite {key}: {values[key]}")


class _TrainingTelemetryCallback:
    """Transformers callback implementation created lazily to keep imports lightweight."""

    def __init__(self, path: Path, training_identity: Mapping[str, Any], torch_module: Any) -> None:
        self.path = path
        self.training_identity = dict(training_identity)
        self.torch = torch_module
        self.started = time.perf_counter()
        self.callback = self._build_callback()

    def _build_callback(self) -> Any:
        from transformers import TrainerCallback

        outer = self

        class Callback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                values = dict(logs or {})
                _validate_finite_training_metrics(values)
                record: dict[str, Any] = {
                    "step": int(state.global_step),
                    "epoch": None if state.epoch is None else float(state.epoch),
                    "elapsed_sec": round(time.perf_counter() - outer.started, 8),
                    **values,
                }
                if outer.torch.cuda.is_available():
                    device = outer.torch.cuda.current_device()
                    record.update(
                        {
                            "gpu_allocated_bytes": int(outer.torch.cuda.memory_allocated(device)),
                            "gpu_reserved_bytes": int(outer.torch.cuda.memory_reserved(device)),
                        }
                    )
                safe_record, non_finite_values = _normalize_telemetry_value(record)
                if non_finite_values:
                    safe_record["_non_finite_values"] = non_finite_values
                with outer.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(safe_record, sort_keys=True, allow_nan=False) + "\n")

            def on_save(self, args, state, control, **kwargs):
                checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                _atomic_json(
                    checkpoint / "checkpoint_metadata.json",
                    {
                        "schema_version": 1,
                        "pipeline_version": PIPELINE_VERSION,
                        "global_step": int(state.global_step),
                        "training_identity": outer.training_identity,
                    },
                )

        return Callback()


def _prepare_model(settings: SFTSettings) -> tuple[Any, dict[str, int], dict[str, Any]]:
    """Load the one designated base in 4-bit and insert the frozen QLoRA adapter."""

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise SFTError(
            "Locked torch/transformers/peft/bitsandbytes dependencies are required."
        ) from exc
    if not torch.cuda.is_available():
        raise SFTError("Canonical E001 QLoRA training requires CUDA; MPS/CPU training is disabled.")
    if torch.cuda.device_count() != 1:
        raise SFTError(
            "Canonical E001 requires exactly one visible CUDA device so effective batch and "
            "optimizer-step counts remain fixed; set CUDA_VISIBLE_DEVICES to one T4."
        )
    if settings.training.bf16 and not torch.cuda.is_bf16_supported():
        raise SFTError("BF16 was requested but this CUDA device does not report BF16 support.")
    snapshot = resolve_local_model_snapshot(settings.model)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=settings.quantization.load_in_4bit,
        bnb_4bit_quant_type=settings.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=settings.quantization.bnb_4bit_use_double_quant,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            settings.model.name_or_path,
            revision=settings.model.revision,
            local_files_only=True,
            trust_remote_code=False,
            quantization_config=quantization_config,
            dtype=torch.float16,
            device_map={"": torch.cuda.current_device()},
            low_cpu_mem_usage=True,
        )
    except (OSError, RuntimeError) as exc:  # pragma: no cover - CUDA boundary
        raise SFTError("Could not load the exact Qwen base with the frozen QLoRA config.") from exc
    model.config.use_cache = False
    lora_breakdown = derive_lora_parameter_breakdown(
        model, settings.lora.target_modules, settings.lora.r
    )
    target_counts = lora_breakdown["target_module_counts"]
    expected_trainable_parameters = lora_breakdown["expected_trainable_parameters"]
    if expected_trainable_parameters != EXPECTED_TRAINABLE_PARAMETERS:
        raise SFTError(
            "Analytical LoRA count differs from the frozen E001 invariant: "
            + json.dumps(lora_breakdown, sort_keys=True, ensure_ascii=True)
        )
    base_parameters = logical_model_parameter_count(model)
    expected_total_parameters = base_parameters + expected_trainable_parameters
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=settings.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=settings.lora.r,
            lora_alpha=settings.lora.lora_alpha,
            lora_dropout=settings.lora.lora_dropout,
            target_modules=list(settings.lora.target_modules),
            bias=settings.lora.bias,
            task_type=settings.lora.task_type,
        ),
    )
    report = trainable_parameter_report(
        model,
        expected_trainable_parameters=expected_trainable_parameters,
        expected_total_parameters=expected_total_parameters,
        target_breakdown=lora_breakdown,
    )
    report.update(
        {
            "base_model_commit": snapshot.commit_hash,
            "base_parameters": base_parameters,
            "target_module_counts": target_counts,
        }
    )
    return model, target_counts, report


def run_sft_training(
    config: LoadedConfig,
    settings: SFTSettings,
    split: SplitValidationResult,
    tokenizer: Any,
    *,
    tokenizer_commit: str,
    run_dir: str | Path,
    limit: int | None = None,
    max_steps: int | None = None,
    resume_from: str | Path | None = None,
    git_commit: str,
    git_dirty: bool,
) -> dict[str, Any]:
    """Execute canonical or explicitly non-canonical smoke E001 training on CUDA."""

    try:
        import torch
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise SFTError("Locked PyTorch and Transformers are required for E001 training.") from exc
    if limit is not None and (isinstance(limit, bool) or limit < 1 or limit > len(split.train)):
        raise SFTError(f"limit must be between 1 and {len(split.train)}.")
    if max_steps is not None and (isinstance(max_steps, bool) or max_steps < 1):
        raise SFTError("max_steps must be a positive integer.")
    if (limit is None) != (max_steps is None):
        raise SFTError("Smoke overrides require both --limit and --max-steps together.")
    identity = build_training_identity(
        config,
        settings,
        tokenizer_commit=tokenizer_commit,
        git_commit=git_commit,
        git_dirty=git_dirty,
        limit=limit,
        max_steps=max_steps,
    )
    resume_path = None if resume_from is None else validate_resume_checkpoint(resume_from, identity)
    selected_train = split.train if limit is None else split.train[:limit]
    selected_validation = (
        split.validation if limit is None else split.validation[: min(limit, len(split.validation))]
    )
    train_examples = [
        encode_sft_example(
            tokenizer,
            row,
            settings.prompt,
            max_seq_length=settings.training.max_seq_length,
        )
        for row in selected_train
    ]
    validation_examples = [
        encode_sft_example(
            tokenizer,
            row,
            settings.prompt,
            max_seq_length=settings.training.max_seq_length,
        )
        for row in selected_validation
    ]
    model, target_counts, parameter_report = _prepare_model(settings)
    output_dir = Path(run_dir).resolve()
    checkpoints_dir = output_dir / "checkpoints"
    telemetry_path = output_dir / settings.output.telemetry_filename
    telemetry_path.touch(exist_ok=False)
    callback = _TrainingTelemetryCallback(telemetry_path, identity, torch)
    save_steps = max_steps if max_steps is not None else settings.training.save_steps
    arguments = TrainingArguments(
        output_dir=str(checkpoints_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        eval_strategy=settings.training.eval_strategy,
        save_strategy=settings.training.save_strategy,
        save_steps=save_steps,
        save_total_limit=settings.training.save_total_limit,
        logging_strategy="steps",
        logging_steps=1 if max_steps is not None else settings.training.logging_steps,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        learning_rate=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
        optim=settings.training.optimizer,
        lr_scheduler_type=settings.training.lr_scheduler_type,
        warmup_ratio=settings.training.warmup_ratio,
        max_grad_norm=settings.training.max_grad_norm,
        num_train_epochs=settings.training.num_train_epochs,
        max_steps=-1 if max_steps is None else max_steps,
        per_device_train_batch_size=settings.training.per_device_train_batch_size,
        per_device_eval_batch_size=settings.training.per_device_eval_batch_size,
        gradient_accumulation_steps=settings.training.gradient_accumulation_steps,
        gradient_checkpointing=settings.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=settings.training.fp16,
        bf16=settings.training.bf16,
        dataloader_num_workers=settings.training.dataloader_num_workers,
        dataloader_pin_memory=True,
        seed=settings.training.seed,
        data_seed=settings.training.data_seed,
        report_to="none",
        remove_unused_columns=False,
        prediction_loss_only=True,
        save_only_model=False,
        restore_callback_states_from_checkpoint=True,
        include_num_input_tokens_seen=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=EncodedSFTDataset(train_examples),
        eval_dataset=EncodedSFTDataset(validation_examples),
        data_collator=AssistantOnlyDataCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[callback.callback],
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_result = trainer.train(
        resume_from_checkpoint=None if resume_path is None else str(resume_path)
    )
    elapsed = time.perf_counter() - started
    final_eval = trainer.evaluate()
    trainer.save_state()
    adapter_dir = output_dir / settings.output.adapter_directory
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    metadata = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "base_model_name": settings.model.name_or_path,
        "base_model_revision": settings.model.revision,
        "base_model_commit": parameter_report["base_model_commit"],
        "tokenizer_revision": settings.model.tokenizer_revision,
        "tokenizer_commit": tokenizer_commit,
        "dataset_version": settings.data.dataset_version,
        "split_version": settings.data.split_version,
        "train_sha256": settings.data.train_sha256,
        "validation_sha256": settings.data.validation_sha256,
        "prompt_version": settings.prompt.version,
        "target_format_version": settings.target.format_version,
        "training_identity_sha256": identity["identity_sha256"],
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "canonical_full_run": identity["identity"]["canonical_full_run"],
    }
    _atomic_json(adapter_dir / "e001_metadata.json", metadata)
    adapter_validation = validate_adapter_compatibility(
        adapter_dir,
        settings,
        expected_identity_sha256=identity["identity_sha256"],
    )
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(torch.cuda.current_device()))
        if torch.cuda.is_available()
        else None
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(torch.cuda.current_device()))
        if torch.cuda.is_available()
        else None
    )
    device_index = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_index)
    runtime = {
        "device": "cuda",
        "device_index": int(device_index),
        "gpu_name": torch.cuda.get_device_name(device_index),
        "total_vram_bytes": int(device_properties.total_memory),
        "compute_capability": list(torch.cuda.get_device_capability(device_index)),
        "torch_cuda_version": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "training_compute_dtype": settings.quantization.bnb_4bit_compute_dtype,
        "visible_cuda_device_count": int(torch.cuda.device_count()),
    }
    metrics = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "canonical_full_run": identity["identity"]["canonical_full_run"],
        "training_identity_sha256": identity["identity_sha256"],
        "train_rows_used": len(selected_train),
        "validation_rows_used_for_loss": len(selected_validation),
        "validation_rows_used_for_training": 0,
        "elapsed_sec": round(elapsed, 8),
        "train_metrics": train_result.metrics,
        "final_validation_metrics": final_eval,
        "global_step": int(trainer.state.global_step),
        "parameter_report": parameter_report,
        "target_module_counts": target_counts,
        "peak_gpu_allocated_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "runtime": runtime,
        "adapter": adapter_validation,
        "resume_from": None if resume_path is None else str(resume_path),
        "package_versions": {
            name: _package_version(name)
            for name in ("torch", "transformers", "accelerate", "peft", "bitsandbytes")
        },
    }
    _atomic_json(output_dir / settings.output.training_metrics_filename, metrics)
    _atomic_json(output_dir / settings.output.identity_filename, identity)
    return metrics


def training_manifest_fields(
    config: LoadedConfig,
    settings: SFTSettings,
    split_report: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    tokenizer_commit: str,
    limit: int | None,
    max_steps: int | None,
    resume_from: str | None,
    project_root: str | Path,
) -> dict[str, Any]:
    """Build complete E001 provenance for the common run manifest."""

    canonical = limit is None and max_steps is None
    root = Path(project_root).resolve()
    return {
        "pipeline_version": PIPELINE_VERSION,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "phase": 4,
        "base_model_name": settings.model.name_or_path,
        "base_model_revision": settings.model.revision,
        "tokenizer_revision": settings.model.tokenizer_revision,
        "tokenizer_commit": tokenizer_commit,
        "dataset_version": settings.data.dataset_version,
        "split_version": settings.data.split_version,
        "train_path": settings.data.train_path.relative_to(root).as_posix(),
        "train_sha256": settings.data.train_sha256,
        "train_rows": settings.data.train_rows,
        "val_path": settings.data.validation_path.relative_to(root).as_posix(),
        "val_sha256": settings.data.validation_sha256,
        "val_rows": settings.data.validation_rows,
        "leakage_validation": dict(split_report),
        "prompt_version": settings.prompt.version,
        "target_format_version": settings.target.format_version,
        "loss_mask_version": settings.target.loss_mask_version,
        "training_method": settings.training.method,
        "quantization_config": asdict(settings.quantization),
        "lora_config": asdict(settings.lora),
        "max_seq_length": settings.training.max_seq_length,
        "truncation_policy": settings.training.truncation_policy,
        "optimizer": settings.training.optimizer,
        "learning_rate": settings.training.learning_rate,
        "weight_decay": settings.training.weight_decay,
        "lr_scheduler": settings.training.lr_scheduler_type,
        "warmup_ratio": settings.training.warmup_ratio,
        "warmup_steps": settings.training.expected_warmup_steps,
        "batch_size": settings.training.per_device_train_batch_size,
        "gradient_accumulation": settings.training.gradient_accumulation_steps,
        "effective_batch_size": settings.training.effective_batch_size,
        "epochs": settings.training.num_train_epochs,
        "steps_per_epoch": settings.training.expected_steps_per_epoch,
        "total_optimizer_steps": settings.training.expected_total_optimizer_steps,
        "seed": settings.seed,
        "limit": limit,
        "max_steps_override": max_steps,
        "canonical_full_run": canonical,
        "resume_from": resume_from,
        "adapter_loaded": False,
        "final_checkpoint": metrics["adapter"]["adapter_path"],
        "checkpoint": metrics["adapter"]["adapter_path"],
        "adapter_sha256": metrics["adapter"]["adapter_sha256"],
        "device": metrics["runtime"]["device"],
        "gpu": metrics["runtime"]["gpu_name"],
        "vram_bytes": metrics["runtime"]["total_vram_bytes"],
        "cuda_version": metrics["runtime"]["torch_cuda_version"],
        "dtype": metrics["runtime"]["training_compute_dtype"],
        "bf16_supported": metrics["runtime"]["bf16_supported"],
        "trainable_parameters": metrics["parameter_report"]["trainable_parameters"],
        "total_parameters": metrics["parameter_report"]["total_parameters"],
        "trainable_percentage": metrics["parameter_report"]["trainable_percentage"],
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "peft_version": _package_version("peft"),
        "bitsandbytes_version": _package_version("bitsandbytes"),
        "external_datasets": [],
        "reasoning_augmentation": False,
        "validation_used_for_training": False,
        "tool_use": False,
        "config_sha256": config.source_sha256,
    }
