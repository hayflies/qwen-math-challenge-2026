# Kaggle E001 실행 절차

이 절차는 Tesla T4 한 장에서 `E001` official-only direct-answer QLoRA를 smoke한 뒤 전체
학습·고정 validation 평가·artifact packaging 순서로 실행한다. 핵심 로직은 모두 repository
script에 있으며 notebook에는 복사하지 않는다.

## 1. 입력 준비

Kaggle working directory의 repository가 Phase 4 구현 commit을 가리키고 clean 상태인지
확인한다. 다음 ignored payload도 정확한 상대 경로에 있어야 한다.

```text
data/splits/official_v001/train.csv
data/splits/official_v001/val.csv
data/splits/official_v001/groups.csv
E000_20260828_canonical_artifacts.tar.gz
```

`Qwen/Qwen2.5-3B-Instruct` revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`도 Hugging Face cache에 미리 준비한다. 학습 및
평가 script는 `local_files_only=true`라서 자동 다운로드나 다른 revision fallback을 하지 않는다.

## 2. 환경 동기화 및 preflight

```bash
cd /kaggle/working/qwen-math-challenge-2026
python -m pip install --quiet uv==0.12.6
uv sync --frozen --group dev
uv run --frozen --no-sync python scripts/train_sft.py \
  --config configs/sft/e001_official_direct_answer.yaml \
  --validate-only
```

preflight는 split 4개 hash, 14,736/1,637행, ID/group overlap 0, token audit hash, Qwen2 module
name, chat boundary, assistant-only target 및 EOS를 검사한다.

## 3. CUDA smoke와 checkpoint reload

```bash
export CUDA_VISIBLE_DEVICES=0
uv run --frozen --no-sync python scripts/train_sft.py \
  --config configs/sft/e001_official_direct_answer.yaml \
  --limit 32 \
  --max-steps 2
```

출력된 smoke run directory에서 `training_metrics.json`의 loss가 finite인지, adapter와
`checkpoints/checkpoint-2/checkpoint_metadata.json`이 생성됐는지 확인한다. 같은 override
identity로 checkpoint load를 검증한다.

```bash
export CUDA_VISIBLE_DEVICES=0
uv run --frozen --no-sync python scripts/train_sft.py \
  --config configs/sft/e001_official_direct_answer.yaml \
  --limit 32 \
  --max-steps 2 \
  --resume <SMOKE_RUN>/checkpoints/checkpoint-2
```

## 4. Canonical full training

full run은 worktree가 clean commit 상태가 아니면 실패한다.

```bash
export CUDA_VISIBLE_DEVICES=0
git status --short
uv run --frozen --no-sync python scripts/train_sft.py \
  --config configs/sft/e001_official_direct_answer.yaml
```

출력된 directory를 `<TRAINING_RUN>`으로 사용한다. 이 run은 1 epoch, effective batch 16,
921 optimizer steps, warmup 28 steps를 실행하며 step 250마다 저장하되 최신 checkpoint 하나만
보존한다.

## 5. 전체 validation adapter 평가

```bash
export CUDA_VISIBLE_DEVICES=0
uv run --frozen --no-sync python scripts/evaluate_sft.py \
  --config configs/sft/e001_official_direct_answer.yaml \
  --adapter <TRAINING_RUN>/adapter \
  --e000-archive E000_20260828_canonical_artifacts.tar.gz
```

`--limit`을 주지 않아야 canonical 1,637행 paired comparison이 생성된다. 출력된 directory를
`<EVALUATION_RUN>`으로 사용한다.

## 6. Artifact 동결

```bash
uv run --frozen --no-sync python scripts/package_e001_artifacts.py \
  --training-run <TRAINING_RUN> \
  --evaluation-run <EVALUATION_RUN> \
  --output E001_20260830_canonical_artifacts.tar.gz
```

resume state까지 다운로드하려면 마지막 명령에 `--include-latest-checkpoint`를 추가한다.
archive에는 final adapter를 포함하지만 base Qwen weight는 포함하지 않는다. 나란히 생성되는
manifest에 archive SHA-256, 모든 member hash, training/evaluation run ID와 adapter hash가
기록된다.

## 중단 후 재개

full run의 `<TRAINING_RUN>/checkpoints/checkpoint-N`을 보존한 경우 동일한 clean commit과 동일
config에서 다음처럼 새 run으로 재개한다.

```bash
export CUDA_VISIBLE_DEVICES=0
uv run --frozen --no-sync python scripts/train_sft.py \
  --config configs/sft/e001_official_direct_answer.yaml \
  --resume <TRAINING_RUN>/checkpoints/checkpoint-N
```

config/model/split/seed/CLI override identity 또는 optimizer/scheduler state가 다르면 재개를
거부한다.
