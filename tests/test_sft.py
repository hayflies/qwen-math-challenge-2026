import csv
import json
from pathlib import Path

import pytest
import torch

from qwen_math_challenge.config import load_config
from qwen_math_challenge.data.official import sha256_file
from qwen_math_challenge.training import sft
from qwen_math_challenge.training.sft import (
    AssistantOnlyDataCollator,
    EncodedExample,
    SFTDataSettings,
    SFTError,
    SFTSourceRow,
    build_training_identity,
    build_training_messages,
    canonical_integer_target,
    encode_sft_example,
    load_sft_settings,
    trainable_parameter_report,
    validate_adapter_compatibility,
    validate_lora_target_modules,
    validate_official_split,
    validate_resume_checkpoint,
)


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 0
    chat_template = "fake"

    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.calls.append((messages, tokenize, add_generation_prompt))
        if add_generation_prompt:
            return [1, 2, 3]
        return [1, 2, 3, 42, 9, 77]


def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _split_settings(
    tmp_path: Path, *, id_overlap: bool = False, group_overlap: bool = False
) -> SFTDataSettings:
    train_rows = [["train-a", "Question A", "42"], ["train-b", "Question B", "-7"]]
    val_id = "train-a" if id_overlap else "val-a"
    val_rows = [[val_id, "Question V", "0"]]
    train_path = tmp_path / "data" / "splits" / "v" / "train.csv"
    val_path = tmp_path / "data" / "splits" / "v" / "val.csv"
    groups_path = tmp_path / "data" / "splits" / "v" / "groups.csv"
    manifest_path = tmp_path / "data" / "splits" / "v" / "split_manifest.json"
    _write_csv(train_path, ["id", "question", "answer"], train_rows)
    _write_csv(val_path, ["id", "question", "answer"], val_rows)
    validation_group = "group-a" if group_overlap else "group-v"
    _write_csv(
        groups_path,
        ["id", "group_id", "group_size", "split", "derived_category"],
        [
            ["train-a", "group-a", "1", "train", "unknown"],
            ["train-b", "group-b", "1", "train", "unknown"],
            [val_id, validation_group, "1", "validation", "unknown"],
        ],
    )
    manifest = {
        "split_version": "official_v001_split_v001",
        "train_rows": 2,
        "val_rows": 1,
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
        "groups_sha256": sha256_file(groups_path),
        "train_val_id_overlap": 0,
        "train_val_group_overlap": 0,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return SFTDataSettings(
        dataset_version="official_v001",
        split_version="official_v001_split_v001",
        train_path=train_path,
        train_sha256=sha256_file(train_path),
        train_rows=2,
        validation_path=val_path,
        validation_sha256=sha256_file(val_path),
        validation_rows=1,
        groups_path=groups_path,
        groups_sha256=sha256_file(groups_path),
        split_manifest_path=manifest_path,
        split_manifest_sha256=sha256_file(manifest_path),
        length_audit_path=tmp_path / "audit.json",
        length_audit_sha256=None,
    )


def test_canonical_integer_target_is_only_integer_content() -> None:
    assert canonical_integer_target("00042") == "42"
    assert canonical_integer_target(3431577212128939) == "3431577212128939"
    with pytest.raises(SFTError, match="not an integer"):
        canonical_integer_target("The answer is 42.")


def test_negative_and_zero_direct_targets() -> None:
    assert canonical_integer_target("-17") == "-17"
    assert canonical_integer_target("-0") == "0"


def test_chat_template_and_assistant_only_masking() -> None:
    tokenizer = FakeTokenizer()
    row = SFTSourceRow(0, "sample", "What is 6*7?", 42)
    encoded = encode_sft_example(
        tokenizer,
        row,
        sft.PromptSettings("v", "System", "{question}"),
        max_seq_length=8,
    )

    assert tokenizer.calls[0][1:] == (True, True)
    assert tokenizer.calls[1][1:] == (True, False)
    assert encoded.input_ids == (1, 2, 3, 42, 9)
    assert encoded.labels == (-100, -100, -100, 42, 9)
    assert encoded.supervised_tokens == 2


def test_training_messages_contain_no_synthetic_reasoning() -> None:
    prompt, full = build_training_messages(
        "Question", -9, sft.PromptSettings("v", "System", "Problem: {question}")
    )
    assert prompt[-1]["content"] == "Problem: Question"
    assert full[-1] == {"role": "assistant", "content": "-9"}


def test_padding_labels_are_masked() -> None:
    collator = AssistantOnlyDataCollator(0, pad_to_multiple_of=4)
    short = EncodedExample("a", (1, 2), (1, 1), (-100, 2), 1, 1)
    long = EncodedExample("b", (1, 2, 3), (1, 1, 1), (-100, 2, 3), 1, 2)
    batch = collator(
        [
            {
                "input_ids": short.input_ids,
                "attention_mask": short.attention_mask,
                "labels": short.labels,
            },
            {
                "input_ids": long.input_ids,
                "attention_mask": long.attention_mask,
                "labels": long.labels,
            },
        ]
    )

    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"][0].tolist() == [-100, 2, -100, -100]
    assert batch["attention_mask"][0].tolist() == [1, 1, 0, 0]


def test_eos_is_supervised_and_post_eos_template_token_is_removed() -> None:
    encoded = encode_sft_example(
        FakeTokenizer(),
        SFTSourceRow(0, "sample", "Question", 42),
        sft.PromptSettings("v", "System", "{question}"),
        max_seq_length=8,
    )
    assert encoded.input_ids[-1] == FakeTokenizer.eos_token_id
    assert encoded.labels[-1] == FakeTokenizer.eos_token_id
    assert 77 not in encoded.input_ids


def test_sequence_truncation_is_rejected() -> None:
    with pytest.raises(SFTError, match="refuses silent truncation"):
        encode_sft_example(
            FakeTokenizer(),
            SFTSourceRow(0, "sample", "Question", 42),
            sft.PromptSettings("v", "System", "{question}"),
            max_seq_length=4,
        )


def test_split_hash_and_no_validation_in_training(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sft, "EXPECTED_TRAIN_ROWS", 2)
    monkeypatch.setattr(sft, "EXPECTED_VAL_ROWS", 1)
    result = validate_official_split(_split_settings(tmp_path))

    assert result.report["train_validation_id_overlap"] == 0
    assert result.report["train_validation_group_overlap"] == 0
    assert result.report["validation_used_for_training"] is False
    assert not (
        {row.sample_id for row in result.train} & {row.sample_id for row in result.validation}
    )


def test_split_sha_mismatch_is_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sft, "EXPECTED_TRAIN_ROWS", 2)
    monkeypatch.setattr(sft, "EXPECTED_VAL_ROWS", 1)
    settings = _split_settings(tmp_path)
    settings.train_path.write_text("changed\n", encoding="utf-8")

    with pytest.raises(SFTError, match="SHA-256 mismatch"):
        validate_official_split(settings)


def test_train_validation_id_overlap_is_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sft, "EXPECTED_TRAIN_ROWS", 2)
    monkeypatch.setattr(sft, "EXPECTED_VAL_ROWS", 1)

    with pytest.raises(SFTError, match="ID leakage"):
        validate_official_split(_split_settings(tmp_path, id_overlap=True))


def test_train_validation_group_overlap_is_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sft, "EXPECTED_TRAIN_ROWS", 2)
    monkeypatch.setattr(sft, "EXPECTED_VAL_ROWS", 1)

    with pytest.raises(SFTError, match="group leakage"):
        validate_official_split(_split_settings(tmp_path, group_overlap=True))


class Linear4bit(torch.nn.Module):
    """GPU-free stand-in for the BitsAndBytes class used after 4-bit loading."""

    def __init__(self) -> None:
        super().__init__()


class FakeQwenAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, Linear4bit())


class FakeQwenMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, Linear4bit())


class FakeQwenLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = FakeQwenAttention()
        self.mlp = FakeQwenMLP()


class FakeQwen(torch.nn.Module):
    def __init__(self, *, layers: int = 2) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(FakeQwenLayer() for _ in range(layers))


def test_lora_target_modules_match_fully_qualified_qwen_names_and_counts() -> None:
    model = FakeQwen()
    counts = validate_lora_target_modules(model, sorted(sft._ALLOWED_TARGET_MODULES))

    module_names = {name for name, _module in model.named_modules()}
    assert "model.layers.0.self_attn.q_proj" in module_names
    assert "model.layers.0.mlp.down_proj" in module_names
    assert set(counts) == sft._ALLOWED_TARGET_MODULES
    assert counts == {target: 2 for target in sorted(sft._ALLOWED_TARGET_MODULES)}


def test_lora_target_module_missing_is_rejected() -> None:
    model = FakeQwen()
    for layer in model.model.layers:
        delattr(layer.self_attn, "q_proj")
    with pytest.raises(SFTError, match="absent"):
        validate_lora_target_modules(model, sorted(sft._ALLOWED_TARGET_MODULES))


def test_lora_target_module_partial_names_do_not_match() -> None:
    model = FakeQwen(layers=1)
    model.q_proj_extra = Linear4bit()
    model.foo_q_proj = Linear4bit()
    model.wrapper = torch.nn.Module()
    model.wrapper.down_proj_extra = Linear4bit()

    counts = validate_lora_target_modules(model, sorted(sft._ALLOWED_TARGET_MODULES))

    assert counts == {target: 1 for target in sorted(sft._ALLOWED_TARGET_MODULES)}


class TinyAdapter(torch.nn.Module):
    def __init__(self, *, unfreeze_base: bool = False) -> None:
        super().__init__()
        self.base = torch.nn.Parameter(torch.zeros(3), requires_grad=unfreeze_base)
        self.lora_A = torch.nn.Parameter(torch.zeros(2), requires_grad=True)


def test_only_lora_parameters_may_be_trainable() -> None:
    report = trainable_parameter_report(TinyAdapter())
    assert report["trainable_parameters"] == 2
    assert report["unexpected_trainable_parameters"] == []
    with pytest.raises(SFTError, match="Unexpected trainable"):
        trainable_parameter_report(TinyAdapter(unfreeze_base=True))


def test_canonical_config_validation_and_schedule() -> None:
    config = load_config("configs/sft/e001_official_direct_answer.yaml")
    settings = load_sft_settings(config, Path.cwd())

    assert settings.training.effective_batch_size == 16
    assert settings.training.expected_steps_per_epoch == 921
    assert settings.training.expected_total_optimizer_steps == 921
    assert settings.training.expected_warmup_steps == 28
    assert settings.data.train_rows == 14736
    assert settings.data.validation_rows == 1637


def test_resume_identity_includes_overrides() -> None:
    config = load_config("configs/sft/e001_official_direct_answer.yaml")
    settings = load_sft_settings(config, Path.cwd())
    full = build_training_identity(
        config,
        settings,
        tokenizer_commit=settings.model.revision,
        git_commit="abc123",
        git_dirty=False,
        limit=None,
        max_steps=None,
    )
    smoke = build_training_identity(
        config,
        settings,
        tokenizer_commit=settings.model.revision,
        git_commit="abc123",
        git_dirty=True,
        limit=32,
        max_steps=2,
    )

    assert full["identity"]["canonical_full_run"] is True
    assert full["identity"]["git_commit"] == "abc123"
    assert full["identity"]["git_dirty"] is False
    assert smoke["identity"]["canonical_full_run"] is False
    assert full["identity_sha256"] != smoke["identity_sha256"]


def test_checkpoint_metadata_identity_and_state_are_required(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    identity = {"identity": {"experiment_id": "E001"}, "identity_sha256": "abc"}
    (checkpoint / "checkpoint_metadata.json").write_text(
        json.dumps({"training_identity": identity}), encoding="utf-8"
    )
    for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
        (checkpoint / filename).write_bytes(b"state")

    assert validate_resume_checkpoint(checkpoint, identity) == checkpoint
    with pytest.raises(SFTError, match="identity mismatch"):
        validate_resume_checkpoint(checkpoint, {"identity_sha256": "different"})


def test_adapter_base_and_e001_metadata_are_required(tmp_path: Path) -> None:
    config = load_config("configs/sft/e001_official_direct_answer.yaml")
    settings = load_sft_settings(config, Path.cwd())
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": settings.model.name_or_path}), encoding="utf-8"
    )
    (adapter / "e001_metadata.json").write_text(
        json.dumps(
            {
                "experiment_id": "E001",
                "base_model_name": settings.model.name_or_path,
                "base_model_revision": settings.model.revision,
                "train_sha256": settings.data.train_sha256,
                "validation_sha256": settings.data.validation_sha256,
                "prompt_version": settings.prompt.version,
                "target_format_version": settings.target.format_version,
                "training_identity_sha256": "identity",
            }
        ),
        encoding="utf-8",
    )

    result = validate_adapter_compatibility(adapter, settings, expected_identity_sha256="identity")
    assert result["status"] == "PASS"
    assert result["adapter_sha256"]
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Other/model"}), encoding="utf-8"
    )
    with pytest.raises(SFTError, match="designated Qwen"):
        validate_adapter_compatibility(adapter, settings)
