# Qwen Math Challenge 2026

Reproducible training, evaluation, and final-inference code for the 2026 아주 소중한 딥러닝
챌린지. The organizer-mandated base model is
[`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), fixed at
revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Repository rules, data policy, and
phase gates are defined in [`AGENTS.md`](AGENTS.md).

## Final submission

The submitted system was the **E000 base-model inference pipeline**, not the E001 fine-tuned
adapter. It used the frozen `zero_shot_v001` chat prompt, greedy decoding with
`max_new_tokens=1024`, and parser
`integer_v002_last_explicit_on_conflict`. No external answer lookup, API, solver, calculator,
or inference-time code execution was used.

Under the final-day deadline, the submitted CSV combined:

- 989 predictions from the original interrupted E000 run;
- 382 completed predictions from emergency shard 0;
- 332 completed predictions from emergency shard 1;
- 297 missing answers filled with the unique mode (`2`) of the 1,703 parsed model predictions.

All 2,000 official rows remained in their original order, including all 120 organizer-flagged
rows. The final submitted file has SHA-256:

```text
1ff78693423e011464f3bcb6f334eb8e181b192be92919a718831e9fc292d538
```

E001 was an official-only direct-answer QLoRA experiment. It reduced internal validation
accuracy from E000's `65.67%` to `22.91%` and was **not** used for the final submission. E001
weights are not needed to reproduce the submitted answers.

## Environment and setup

- Python: `>=3.11,<3.12` (development and Kaggle runs used Python 3.11)
- Package manager: `uv`; exact transitive versions are locked in `uv.lock`
- Main libraries: PyTorch, Transformers, Accelerate, PEFT, bitsandbytes, NumPy, PyYAML
- Final inference hardware: Kaggle NVIDIA T4; emergency completion used two independent T4
  processes, each seeing one GPU
- Inference is offline: the exact model snapshot must already exist in the Hugging Face cache
  because `local_files_only: true` is frozen in the config

```bash
git clone <PUBLIC_GITHUB_REPOSITORY_URL>
cd qwen-math-challenge-2026
python -m pip install uv==0.12.6
uv sync --frozen --group dev
uv run --frozen --no-sync pytest -q
```

Do not commit Hugging Face tokens, Kaggle credentials, official raw data, model caches, or model
weights. The base-model license and access terms are provided by its Hugging Face repository.

## Mode A: reproduce the model-inference pipeline

Place the organizer-provided `test_submission.csv` and `test_flag.csv` outside Git, make the
exact model revision available in the local Hugging Face cache, then run one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen --no-sync python scripts/generate_submission.py \
  --config configs/inference/e000_final_submission.yaml \
  --test-file /path/to/test_submission.csv \
  --flag-file /path/to/test_flag.csv
```

This runs E000 on all 2,000 questions and reproduces the **pipeline**. It is not expected to
produce the deadline submission byte-for-byte because that artifact contains 297 deterministic
fallback rows.

### Crash-safe resume

The runner fsyncs every prediction and resumes only after config, model, tokenizer, prompt,
generation, parser, input, and code identities match:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen --no-sync python scripts/generate_submission.py \
  --config configs/inference/e000_final_submission.yaml \
  --test-file /path/to/test_submission.csv \
  --flag-file /path/to/test_flag.csv \
  --resume /path/to/prior/run
```

### Emergency two-GPU sharding

Plan by completed-ID membership, never by an assumed row prefix:

```bash
uv run --frozen --no-sync python scripts/manage_final_shards.py plan \
  --config configs/inference/e000_final_submission.yaml \
  --test-file /path/to/test_submission.csv \
  --flag-file /path/to/test_flag.csv \
  --existing-run /path/to/original/partial-run \
  --num-shards 2
```

Launch two independent background processes (no DataParallel or distributed training):

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run --frozen --no-sync python scripts/generate_submission.py \
  --config configs/inference/e000_final_submission.yaml \
  --test-file /path/to/test_submission.csv \
  --flag-file /path/to/test_flag.csv \
  --exclude-run /path/to/original/partial-run --shard-index 0 --num-shards 2 \
  > shard0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup uv run --frozen --no-sync python scripts/generate_submission.py \
  --config configs/inference/e000_final_submission.yaml \
  --test-file /path/to/test_submission.csv \
  --flag-file /path/to/test_flag.csv \
  --exclude-run /path/to/original/partial-run --shard-index 1 --num-shards 2 \
  > shard1.log 2>&1 &
```

`scripts/manage_final_shards.py merge` accepts only complete, disjoint planned shards. The exact
deadline artifact instead uses the separate Mode B path below because both emergency shards were
still partial at submission time.

## Mode B: reproduce the exact submitted CSV

The 12 small frozen Kaggle files required for exact reconstruction are versioned under
`artifacts/final_submission/` as described in its
[`README.md`](artifacts/final_submission/README.md). The script fails closed if any file, source
identity, parser result, shard prefix, row count, flag set, or final checksum differs.

```bash
uv run --frozen --no-sync python scripts/reproduce_final_submission.py \
  --config configs/inference/e000_final_submission.yaml \
  --test-file /path/to/test_submission.csv \
  --flag-file /path/to/test_flag.csv \
  --artifact-dir artifacts/final_submission \
  --output-dir outputs/final_submission_reproduced
```

This CPU-only command writes `submission.csv`, `submission.csv.sha256`, and
`reproduction_manifest.json`. Answers are parsed with Python arbitrary-precision integers; no
`int64` conversion is used. Success requires the fixed SHA-256 shown above.

## Integrity and reproducibility identities

- Kaggle emergency sharding code commit:
  `d7adc8961a65e0354e4089ef525bf63646f8c665`
- Final config: `configs/inference/e000_final_submission.yaml`
- Final config SHA-256:
  `60517b11d3bfc69a069b833f7d2ea934f97513ed19979300e9be481895da6b4d`
- Canonical E000 reference config SHA-256:
  `f2d6d851d263466ee32fbdeabf70be326a0ff5a63f1e63e0376bf7bc10daaaea`
- Base model/tokenizer revision:
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Submitted CSV SHA-256:
  `1ff78693423e011464f3bcb6f334eb8e181b192be92919a718831e9fc292d538`

Run manifests record config, Git, environment, input, parser, and model identities. Derived data
and canonical experiment artifacts remain versioned separately; raw organizer data is immutable
and excluded from Git.

## Repository structure

```text
configs/       Frozen data, training, evaluation, and inference YAML configs
src/           Reusable data, training, evaluation, inference, and reproducibility logic
scripts/       CLI entry points
tests/         CPU/unit tests and guarded GPU integration paths
experiments/   Canonical experiment records
analysis/      Versioned E000 post-analysis summaries
docs/          Kaggle execution documentation
artifacts/     Frozen final predictions/provenance; large model/experiment payloads stay local
outputs/       Ignored run directories and generated submissions
```

## Model and weights statement

The final submitted inference path uses the organizer-specified
`Qwen/Qwen2.5-3B-Instruct` base weights at the exact revision above and loads no LoRA or other
fine-tuned adapter. Therefore there are no fine-tuned weights required for final-submission
reproduction; model-inference reproduction requires only the official base-model snapshot.
