# Qwen Math Challenge 2026

`Qwen/Qwen2.5-3B-Instruct`를 출발점으로 수학 문제 해결 모델을 개발하기 위한
재현 가능한 실험 저장소다. 대회 규칙, 데이터 정책, 금지사항, Phase Gate의 최상위
기준은 루트의 `AGENTS.md`다.

현재 구현 범위는 **Phase 0 — 프로젝트 기반 및 재현성**과
**Phase 1 — 공식 데이터 감사 및 정제**다. validation split, 모델 로드, 학습, 평가,
제출 생성 로직은 아직 구현하지 않는다.

## 환경 준비

Python 3.11과 `uv`를 사용한다. 직접 의존성의 허용 범위는 `pyproject.toml`, 실제 설치
버전과 전이 의존성은 `uv.lock`이 고정한다.

```bash
uv sync --frozen --group dev
```

Phase 0은 대형 학습 패키지를 설치하지 않는다. PyTorch, Transformers, PEFT 등은 실제로
필요한 후속 Phase에서 실행 환경에 맞는 버전을 검증한 뒤 추가한다.

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
