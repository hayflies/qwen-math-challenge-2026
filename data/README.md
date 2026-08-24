# Data directory policy

이 디렉터리는 원본과 파생 데이터의 경계를 명확히 유지한다.

- `raw/`: 주최 측이 제공한 원본. 내용을 직접 수정하거나 덮어쓰지 않는다.
- `external/`: 출처, 버전, 라이선스가 확인된 외부 데이터 원본.
- `processed/`: versioned 정제/변환 데이터와 작은 manifest.
- `splits/`: versioned train/validation split과 작은 manifest.

대용량 데이터 payload는 Git에서 제외한다. 재현에 필요한 manifest와 config에는 원본 파일의
논리적 역할, 해시, 행 수, schema, 생성 규칙을 기록한다. 실제 감사와 정제 로직은 Phase 1의
범위이며 Phase 0에는 포함하지 않는다.

공식 CSV 3개는 Phase 1 시작 시 기준 hash를 재확인한 뒤 다음 immutable version으로
배치했다.

```text
data/raw/official_v001/
├── deep_chal_math_train.csv
├── deep_chal_math_leaderboard_filtered.csv
└── train_filtered_ids.csv
```

dataset version의 raw hash 기준은 `data/manifests/official_v001_raw_manifest.json`에 기록한다.
공식 627개 ID는 mandatory exclusion이며, 아직 확보하지 않은 커뮤니티 후보는 자동 생성하거나
적용하지 않는다.

`data/splits/official_v001/`은 `train_clean.csv`의 고정 hash를 입력으로 하는 Phase 2
group-safe split이다. CSV payload는 재생성 가능하므로 Git에서 제외하고, grouping/split
설정·통계·artifact hash는 `split_manifest.json`과 `split_report.json`에 보존한다. 동일
group은 train과 validation에 동시에 나타날 수 없다.
