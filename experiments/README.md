# Experiment records

각 실행은 `outputs/<experiment_id>/<run_id>/run_manifest.json`에 최소 다음 필드를 기록한다.

```text
experiment_id, phase, git_commit, model, checkpoint, dataset_version,
train_samples, validation_split, external_datasets, seed, learning_rate,
epochs, batch_size, gradient_accumulation, max_length, lora_config,
prompt_template, inference_config, parser_version, local_validation_score,
per_category_scores, leaderboard_score, notes
```

실행 시점의 config snapshot과 environment 정보는 같은 run directory에 보존한다. 데이터셋,
checkpoint, parser 또는 주요 실험 변수가 바뀌면 동일한 기록으로 덮어쓰지 않고 새 run을 만든다.
