# Qwen Math Challenge 2026

`Qwen/Qwen2.5-3B-Instruct`를 출발점으로 수학 문제 해결 모델을 개발하기 위한
재현 가능한 실험 저장소다. 대회 규칙, 데이터 정책, 금지사항, Phase Gate의 최상위
기준은 루트의 `AGENTS.md`다.

현재 구현 범위는 **Phase 0 — 프로젝트 기반 및 재현성**, **Phase 1 — 공식 데이터 감사 및
정제**, **Phase 2 — leakage-safe 내부 validation 구축**, **Phase 3 — E000 zero-shot
평가**다. 학습과 제출 생성 로직은 아직 구현하지 않는다.

## 환경 준비

Python 3.11과 `uv`를 사용한다. 직접 의존성의 허용 범위는 `pyproject.toml`, 실제 설치
버전과 전이 의존성은 `uv.lock`이 고정한다.

```bash
uv sync --frozen --group dev
```

Phase 3은 추론에 필요한 PyTorch, Transformers, Accelerate만 추가한다. PEFT, TRL,
bitsandbytes, datasets, DeepSpeed는 아직 설치하지 않는다.

## 검증

```bash
uv run --frozen ruff check src scripts tests
uv run --frozen pytest -q
uv run --frozen python scripts/phase0_smoke.py \
  --config configs/phase0/smoke.yaml
```

smoke 실행은 `outputs/<experiment_id>/<run_id>/`에 다음을 저장한다.

- 실행 config 원본 snapshot과 SHA-256
- Git branch/commit/dirty 상태
- Python, 주요 패키지, PyTorch, CUDA, MPS, GPU 환경 정보
- run manifest, log, deterministic probe 결과

Git에 아직 커밋이 없으면 오류를 내지 않고 `git_commit: null`,
`git_head_state: unborn`으로 기록한다.

## 데이터 안전 정책

- 공식 raw 파일은 직접 수정하지 않는다.
- raw와 외부 데이터는 Git에 commit하지 않는다.
- 파생 데이터는 `data/processed/` 또는 `data/splits/`에 versioning한다.
- 현재 루트에 제공된 공식 CSV는 Phase 0에서 이동하거나 변경하지 않는다.
- leaderboard/test 데이터는 어떤 training 입력에도 연결하지 않는다.

자세한 디렉터리 정책은 `data/README.md`를 참고한다.

## Phase 1 공식 데이터 준비

공식 파일은 `data/raw/official_v001/`에 원본 바이트 그대로 보관한다. 다음 명령은 raw
hash와 schema를 검증하고, 공식 오류 ID 627개를 오직 `id` 기준으로 제거한다.

```bash
uv run --frozen python scripts/inspect_data.py \
  --config configs/data/official_v001.yaml
```

산출물은 `data/processed/official_v001/`에 생성된다.

- `train_clean.csv`: 공식 627개 ID만 제거한 clean train
- `audit_report.json`: schema, 통계, 교차검사, suspect reporting
- `dataset_manifest.json`: raw-to-clean provenance와 hash

커뮤니티 candidate exclusion은 현재 파일이 없으므로 적용하지 않는다. 향후 별도 dataset
variant에서 검증하고 ablation한다.

## Phase 2 내부 validation 생성

다음 명령은 `official_v001` clean train의 hash와 16,373행 invariant를 확인한 뒤 보수적
정규화, text-only near-duplicate blocking/유사도 평가, 연결요소 grouping, seed 2026의
group-safe 약 10% split을 순서대로 실행한다.

```bash
uv run --frozen python scripts/create_split.py \
  --config configs/data/split_official_v001.yaml
```

산출물은 `data/splits/official_v001/`에 생성된다. `train.csv`, `val.csv`, `groups.csv`,
`near_duplicate_candidates.csv`는 대용량 payload로 Git에서 제외하고, 재현에 필요한
`split_manifest.json`과 `split_report.json`은 추적한다. leaderboard는 exact/normalized
overlap 감사에만 읽으며 train/validation에는 포함하지 않는다. derived category는 gold
label이 아니므로 보고에만 사용하고 split 기준으로 사용하지 않는다.

## Phase 3 E000 zero-shot 평가

E000은 `Qwen/Qwen2.5-3B-Instruct`의 로컬 Hugging Face cache만 사용한다. 모델을 다른
checkpoint로 대체하거나 실행 중 자동 다운로드하지 않는다. 다음 명령으로 5문항 smoke를 먼저
실행한다.

```bash
uv run --frozen --no-sync python scripts/evaluate_zero_shot.py \
  --config configs/inference/e000_zero_shot.yaml \
  --limit 5
```

전체 validation은 `--limit`을 제거한다. 중단된 호환 run의 완료 prefix를 재사용하려면 새
run을 만들면서 `--resume <prior-run-directory>`를 지정한다. model commit, split hash,
prompt, generation config, parser, limit 또는 config hash가 다르면 resume을 거부한다.

기본값은 공식 tokenizer chat template, greedy decoding, `max_new_tokens=1024`, batch size
1이다. 출력에는 token/latency, finish reason, truncation, raw generation과 parser 결과를 함께
보존한다. 문제 풀이용 Python/SymPy/calculator/tool은 사용하지 않는다.

### Phase 3 동결 상태

Phase 3 Gate는 Kaggle CUDA에서 완료한 canonical full E000을 근거로 **PASS**다. 고정
validation 1,637문항에서 1,075문항을 맞혀 Accuracy `0.6566890654`를 기록했고 parse
failure는 6건이다. 최종 run은 호환성 검사를 통과한 completed prefix에서 resume했으며,
78건이 `max_new_tokens=1024`에 도달했다. 설정을 소급 변경하거나 재측정하지 않는다.

정확한 identity, metric, artifact 보존 위치, 로컬 검증 범위와 후속 분석 TODO는
[`experiments/e000_zero_shot.yaml`](experiments/e000_zero_shot.yaml)에 동결한다. canonical
archive는 Kaggle Notebook Version 1 Output에 보존되어 있고, 현재 repository root에도 import
후보가 있으나 별도 integrity 검증 전에는 신뢰하지 않는다. 기존 MPS 1문항/5문항 결과는
pre-CUDA smoke evidence로 `outputs/`에 그대로 유지하며 canonical full metric으로 사용하지
않는다.
