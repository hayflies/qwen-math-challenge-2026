# Exact final-submission artifacts

This directory contains the exact frozen inputs for `scripts/reproduce_final_submission.py`.
They are the Kaggle prediction prefixes actually used for the submitted CSV; do not replace them
with later shard versions or edit them. The reproducer fails rather than inventing or regenerating
missing content.

## Required layout

```text
artifacts/final_submission/
├── partial_989/
│   ├── predictions.csv
│   ├── input_identity.json
│   ├── resume_identity.json
│   └── run_manifest.json
├── shard_0/
│   ├── predictions.csv
│   ├── input_identity.json
│   ├── resume_identity.json
│   └── run_manifest.json
└── shard_1/
    ├── predictions.csv
    ├── input_identity.json
    ├── resume_identity.json
    └── run_manifest.json
```

Copy, without editing, the four named files from each exact Kaggle run:

- `partial_989`: `outputs/final_e000/FINAL_E000/20260831T064339154894Z_60517b11`
- `shard_0`: the emergency shard run whose `resume_identity.json` has `shard.index = 0`; its
  `predictions.csv` must contain exactly 382 completed rows
- `shard_1`: the emergency shard run whose `resume_identity.json` has `shard.index = 1`; its
  `predictions.csv` must contain exactly 332 completed rows

The organizer-provided `test_submission.csv` and `test_flag.csv` are also required, but pass them
by CLI path rather than copying them here. They remain the canonical input source.

Example local preparation after downloading the three run directories:

```bash
mkdir -p artifacts/final_submission/partial_989 \
  artifacts/final_submission/shard_0 artifacts/final_submission/shard_1

cp /path/to/partial-run/{predictions.csv,input_identity.json,resume_identity.json,run_manifest.json} \
  artifacts/final_submission/partial_989/
cp /path/to/shard-0-run/{predictions.csv,input_identity.json,resume_identity.json,run_manifest.json} \
  artifacts/final_submission/shard_0/
cp /path/to/shard-1-run/{predictions.csv,input_identity.json,resume_identity.json,run_manifest.json} \
  artifacts/final_submission/shard_1/
```

Then run from the repository root:

```bash
uv run --frozen --no-sync python scripts/reproduce_final_submission.py \
  --config configs/inference/e000_final_submission.yaml \
  --test-file /path/to/test_submission.csv \
  --flag-file /path/to/test_flag.csv \
  --artifact-dir artifacts/final_submission \
  --output-dir outputs/final_submission_reproduced
```

The command validates all source identities, exact 989/382/332 counts, deterministic shard
prefixes, parser v2 outputs, the unique mode fallback (`2`) on exactly 297 IDs, all 120 flags, and
final SHA-256 `1ff78693423e011464f3bcb6f334eb8e181b192be92919a718831e9fc292d538`.

The three `predictions.csv` files and their input/resume/run identity JSON files are intentionally
included in the public reproducibility commit. The organizer-provided raw test and flag CSVs are
still excluded and must be supplied separately.
