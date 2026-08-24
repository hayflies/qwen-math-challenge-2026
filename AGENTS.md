# Codex Implementation Guide — 아주 소중한 딥러닝 챌린지 2026

> **Codex 운영 지침:** 이 문서는 이 저장소에서 구현 작업을 수행할 때의 상시 기준 문서다. Codex는 새로운 Phase를 구현하기 전에 이 문서를 읽고, 현재 Phase의 목적·금지사항·완료조건을 확인한다. 한 번에 여러 Phase를 임의로 섞지 말고, 각 Phase의 산출물과 검증을 완료한 뒤 다음 Phase로 이동한다.

## 0. 프로젝트 개요

### 대회
- 대회명: 제5회 대학 연합 **아주 소중한 딥러닝 챌린지 2026**
- 주제: **LLM 모델을 활용한 수학 문제 해결**
- 공식 일정(대학 공지 기준): 2026-07-31 ~ 2026-08-30
- 출발 모델: **Qwen2.5-3B-Instruct**
- 목표: 수학 특화 모델이 아닌 범용 3B 모델을 출발점으로 하여, 처음 보는 수학 문제를 정확히 해결하는 일반화된 추론 능력을 최대화한다.
- 핵심 관점: 단순 정답 암기보다 데이터 품질, reasoning trajectory 품질, 일반화, 평가 신뢰성, inference scaling을 중시한다.

### 공식 대회 규칙 및 데이터 정책 — PDF 확인 반영

아래 내용은 사용자가 제공한 대회 Overview / Data / Rules PDF를 기준으로 확정한다.

#### 모델
1. 유일한 출발점은 `Qwen/Qwen2.5-3B-Instruct`이다.
2. Qwen2.5-Math, DeepSeek-R1, Llama, GPT 등 다른 모델을 베이스 모델로 사용할 수 없다.
3. 다른 모델의 **가중치를 로드하거나 병합(merge)** 하는 것도 금지된다.
4. 처음부터 Pre-training하는 것은 금지되고 **Fine-tuning만 허용**된다.
5. Full Fine-Tuning, LoRA, QLoRA/PEFT, SFT, GRPO/DPO/PPO/KTO 등 RL 기반 학습, 데이터 증강, Curriculum Learning, Quantization은 허용된다.

#### 공식 데이터
- `deep_chal_math_dataset_train.csv`
  - 학습에 직접 사용하는 공식 train.
  - 필드: `id`, `question`, `answer`.
  - `answer`는 정수.
  - **공식 train에는 별도 reasoning/solution 컬럼이 명시되어 있지 않다.**
- `deep_chal_math_dataset_leaderboard_filtered.csv`
  - 2026-08-03 14:00부터 기존 leaderboard 파일을 대체한 오류 제거 버전.
  - 실시간 리더보드 평가용이며 answer는 제공되지 않는다.
  - 제출은 반드시 이 정제 버전의 ID를 기준으로 작성한다.
- `train_filtered_ids.csv`
  - train에서 오류가 확인된 항목의 ID 목록.
  - 학습 시 해당 ID를 자체 필터링한다.
- `deep_chal_math_dataset_test.csv`
  - 최종 순위 평가용.
  - 2026-08-31 00:00 공개 예정.
  - 제출 기간: 2026-08-31 00:00 ~ 23:59.
  - Test 문제를 학습 데이터로 사용하는 것은 금지된다.

> 문서 일부 Rules에는 `test.parquet`이라는 표현이 있으나 Data 섹션의 실제 제공 파일명은 CSV로 명시되어 있다. 구현에서는 실제 배포 파일을 source of truth로 삼고, 파일명 차이를 hard-code하지 않도록 config화한다.

#### 외부 데이터
1. 공개 데이터셋 추가 사용은 자유다.
2. 모든 참가자가 **무료로 동등하게 접근 가능한 공개 데이터**여야 한다.
3. 유료 구독, 특수 라이선스 또는 비공개 협약이 필요한 데이터는 금지된다.
4. 사용한 외부 데이터셋은 최종 제출 시 목록을 명시해야 한다.
5. 학습 데이터 구축 목적의 **상용 API 사용은 허용**된다. 예: GPT-4를 이용한 풀이 생성/데이터 증강.
6. 단, 상용 API로 Test 문제의 답을 직접 생성하는 것은 금지된다.
7. Test 문제를 검색 엔진/외부 서비스에 입력해 답을 찾는 것도 금지된다.

#### 추론
1. 추론 시 인터넷 접속은 차단되며 외부 API/웹 검색을 사용할 수 없다.
2. 모든 추론은 제공 환경 내에서 로컬로 수행한다.
3. Majority Voting, Self-Consistency, Best-of-N 등의 test-time 기법은 허용된다.
4. 다른 외부 모델을 추론 시 호출해 ensemble하는 것은 금지된다.

#### 평가 및 제출
- 평가 지표: **Accuracy (Exact Match)**.
- 모든 정답은 **정수**다.
- Public Leaderboard: Test의 30%, 최종 순위에 영향 없음.
- Private Leaderboard: Test의 70%, 최종 순위 결정.
- 모델 출력에 reasoning/수식/설명이 있어도 되지만 `submission.csv`의 `answer`에는 **최종 정수만** 들어가야 한다.
- 제출 파일은 `ID`, `answer` 두 컬럼이며 모든 문제의 답을 포함해야 한다. 빈 값은 오답 처리된다.
- leaderboard probing으로 정답을 역추적하는 행위는 금지된다.

#### 재현성 의무
수상 후보자는 다음을 제출할 수 있어야 한다.
- training code
- inference code
- trained model weights
- 사용 데이터셋 목록 및 접근 방법
- 하드웨어/라이브러리 버전 등 환경 설명
- 아키텍처, 전처리, 학습 세부사항, hyperparameter를 포함한 방법론 문서

주최 측 재현 검증에 실패하면 수상이 취소될 수 있으므로 **재현성은 선택사항이 아니라 프로젝트 요구사항**으로 취급한다.

#### 대회 일정/평가
- 챌린지 및 이론 학습: 2026-07-31 ~ 2026-08-30.
- 최종 Test 공개/제출: 2026-08-31.
- 평가 및 검증: 2026-08-01 ~ 2026-09-20 예정.
- 수상자 발표: 2026-09-28 예정.
- 최종 12팀은 발표 예정이며, 문서상 `모델 성능 50% + 발표 평가(모델 우수성) 50%`로 검증 후 최종 9팀(명)이 수상 대상이 된다.
- 따라서 최고 Accuracy뿐 아니라 **방법론의 타당성, 실험 근거, 재현 가능한 구현**도 프로젝트 산출물로 관리한다.

### 설계 원칙
- **Leakage prevention first.**
- **Reproducibility first.**
- 한 실험에서 주요 변수를 가능한 한 하나만 변경한다.
- Public/leaderboard 점수만 보고 반복 최적화하지 않는다.
- 내부 validation을 주 실험 기준으로 사용하고 leaderboard는 제한적으로 사용한다.
- 데이터 추가량보다 **정확한 reasoning supervision의 품질**을 우선한다.
- 모든 전처리/학습/평가/추론 단계는 CLI로 재실행 가능해야 한다.
- raw data는 수정하지 않는다. 파생 데이터는 별도 디렉터리에 저장한다.
- seed, config, dataset version, model/checkpoint, metrics를 항상 기록한다.


### 실제 제공 CSV 감사 결과 — 2026-08-24 확인

업로드된 실제 3개 CSV를 직접 검사한 결과를 구현의 source of truth로 사용한다.

| 파일 | shape | columns | null | ID 중복 |
|---|---:|---|---:|---:|
| `deep_chal_math_train.csv` | 17,000 × 3 | `id`, `question`, `answer` | 0 | 0 |
| `deep_chal_math_leaderboard_filtered.csv` | 831 × 2 | `id`, `question` | 0 | 0 |
| `train_filtered_ids.csv` | 627 × 3 | `id`, `answer`, `question` | 0 | 0 |

#### Train 관찰
- `id`: 17,000개 모두 unique.
- `question`: 17,000개 모두 exact unique.
- `answer`: 실제 pandas 로드 시 `int64`; 정수 정답임이 데이터에서도 확인된다.
- answer 범위는 음수/0/매우 큰 양수를 모두 포함한다. 따라서 parser/submission 코드는 **양의 정수만 가정하면 안 된다.**
- 전체 train answer 범위(현재 파일): `-5,765,435` ~ `3,431,577,212,128,939`.
- 전체 train에는 URL 152건, Markdown image syntax 127건이 탐지되었다.

#### `train_filtered_ids.csv`의 실제 성격
파일명은 IDs 목록처럼 보이지만 실제로는 `id`뿐 아니라 `answer`, `question`도 포함한 627행짜리 3-column CSV다.

중요한 사실:
- 627개 ID는 **모두** train에 존재한다.
- 627개 answer는 train의 동일 ID answer와 모두 일치한다.
- 그러나 `question` 문자열은 train과 exact match가 275/627에 불과하다. 나머지는 줄바꿈/공백 등 표현 차이가 존재한다.
- 따라서 오류 제거는 **question 문자열 join이 아니라 `id` membership만으로 수행**해야 한다.
- 627개를 제거하면 official clean train은 **16,373 rows**가 된다.
- filtered ID 데이터에는 URL 150건, Markdown image syntax 127건이 집중되어 있다. 이미지 의존/깨진 번역·스크래핑 문제 등이 오류 목록에 포함된 실제 사례가 확인된다.
- 그러나 오류 원인을 URL/이미지 패턴만으로 일반화해서 추가 삭제하면 안 된다. 공식 627 IDs를 먼저 authoritative exclusion list로 취급한다.

#### Clean Train 관찰
공식 627 IDs 제거 후:
- rows: **16,373**
- null: 없음
- exact question duplicate: 없음
- Markdown image syntax: 0건
- URL-like 문자열은 2건 남는다. 이 2건은 텍스트 문제 내부의 URL/LaTeX URL 표현이며, URL이 있다는 이유만으로 자동 삭제하지 않는다.
- answer 범위: `-2,025,078` ~ `3,431,577,212,128,939`.
- negative answer 486건, zero answer 210건이 존재한다.

#### Filtered Leaderboard 관찰
- rows: **831**
- columns: `id`, `question`; `answer` 컬럼 자체가 없다.
- null 없음.
- ID/question exact duplicate 없음.
- ID prefix는 실제 샘플에서 `val-...` 형태다.
- train과 leaderboard 사이 ID exact overlap = 0.
- train과 leaderboard 사이 question exact overlap = 0.
- clean train과 leaderboard 사이 question exact overlap = 0.
- 이는 exact-match 수준의 결과일 뿐이며 **near-duplicate leakage가 없음을 증명하지는 않는다.** Phase 2/8에서 fuzzy/semantic 검사를 별도로 수행한다.

#### 길이 통계
문자 수 기준 대략:
- raw train: median 203, p95 492, max 4517
- clean train: median 201, p95 477, max 3560
- leaderboard filtered: median 205, p95 약 476, max 4391
- filtered/error rows: median 284, p95 약 885, max 4486

따라서 token length 분석 없이 임의의 작은 `max_length`를 고정하지 않는다. 실제 tokenizer 기반 token 분포를 Phase 1에서 추가 측정한다.

#### 구현에 즉시 반영할 invariant
```text
RAW_TRAIN_ROWS = 17000
OFFICIAL_FILTERED_ID_ROWS = 627
EXPECTED_CLEAN_TRAIN_ROWS = 16373
LEADERBOARD_FILTERED_ROWS = 831
```

위 숫자는 **현재 업로드된 데이터 버전에 대한 검증 invariant**다. 향후 주최측 파일이 갱신되면 hard failure만 내지 말고 파일 hash/shape 변화와 함께 명시적으로 dataset version을 갱신한다.


### 운영팀 공식 질의응답 반영 — 2026-08-24

Discord 질의응답의 운영팀 답변은 PDF 규칙의 구체적 해석으로 취급하며, 더 최신 공지가 나오면 최신 공지가 우선한다.

#### Final Test
- 약 **2,000문항**.
- 2026-08-31 00:00 공개, 23:59 마감.
- Google Form은 마감 전 수정/재제출 가능.
- 최종 inference config는 약 2,000문항 전체의 wall-clock/VRAM/token 비용을 사전 benchmark한다.

#### 추가 Train 오류
- 공식 `train_filtered_ids.csv` 이후에도 오류가 있을 수 있으며 운영팀은 참가자의 자체 추가 필터링을 허용했다.
- 커뮤니티 제보에는 11개 명백 오류 + 3개 모호 후보, 이후 `mislabel_442.csv` 442건과 `illposed_623.csv` 623건이 있다.
- 운영팀은 후자의 예시가 타당하고 원본 소스 오류로 보인다고 답했으나 목록 전체를 공식 확정하지는 않았다.
- 공식 627 IDs=`mandatory_exclusion`; 커뮤니티 후보=`candidate_exclusion`.
- 후보는 검증 후 별도 dataset variant로 ablation하며 자동 전량 삭제하지 않는다.

#### Test-time 외부 정답 lookup 금지
사전 다운로드한 GSM8K/MATH/NuminaMath 등 공개 데이터라도 Test 문제를 exact/fuzzy matching해 기존 정답을 반환하는 것은 금지다. 정답은 참가자의 **모델 추론 결과**여야 한다.
- 외부 데이터는 training/fine-tuning/prompt 구성에 활용 가능.
- Test/leaderboard inference에는 external-answer lookup DB/index를 포함하지 않는다.
- leakage matching은 학습 데이터 정제용으로만 사용한다.
- 최종 제출 답안은 제출 코드/모델 inference로 재현 가능해야 한다.

#### Inference-time Python/tool 금지
문제 풀이 과정에서 다음은 금지다.
- Program-of-Thought / Tool-Integrated Reasoning.
- 모델이 생성한 Python 코드 실행 후 결과 재입력.
- 방정식 solver, 소인수분해, GCD, SymPy/calculator 등 deterministic 계산 tool 호출.

Majority Voting, Self-Consistency, Best-of-N 등 모델 generation 기반 test-time 기법은 허용된다. CSV I/O, 모델 orchestration, 최종 정수 추출 같은 일반 후처리는 별개다.

#### 동일 베이스 복수 LoRA / verifier
- 동일 `Qwen2.5-3B-Instruct` 베이스의 복수 LoRA adapter ensemble은 허용.
- 동일 베이스 verifier adapter로 Best-of-N 후보 선별도 허용.
- 단, 과도한 유형별 전문 adapter routing으로 사실상 다수 특화 모델 체계를 만들지 않는다.

#### 구조/파라미터 변경
운영팀은 모두 허용한다고 답했다.
1. LoRA 외 adapter/파라미터 추가.
2. layer 추가/복제 등 capacity 변경.
3. prompt tuning/prefix tuning 등 trainable embedding 추가.
4. attention head 제거/sparsity mask 등 파라미터 삭제.

다른 베이스 모델의 가중치 load/merge 금지는 유지된다.

#### 규칙 우선순위
```text
최신 운영팀 공식 공지/답변
> Rules/Overview/Data
> AGENTS.md 해석
> 참가자 제보/커뮤니티 추정
```

---

# 1. 권장 저장소 구조

```text
qwen-math-challenge-2026/
├── AGENTS.md                         # Codex가 항상 읽을 핵심 지침(이 파일 권장명)
├── README.md
├── pyproject.toml / requirements.txt
├── configs/
│   ├── data/
│   ├── sft/
│   ├── grpo/
│   └── inference/
├── data/
│   ├── raw/                          # 원본, 절대 직접 수정 금지
│   ├── external/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── utils/
├── scripts/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── train_sft.py
│   ├── evaluate.py
│   └── generate_submission.py
├── tests/
├── experiments/
├── outputs/
└── notebooks/
```

Codex는 기능을 notebook에만 구현하지 않는다. notebook은 EDA/시각화용이며, 재현에 필요한 로직은 `src/` 또는 `scripts/`에 둔다.

---

# 2. 공통 실험 기록 규격

모든 학습/평가 run은 최소 다음 정보를 저장한다.

```yaml
experiment_id:
phase:
git_commit:
model:
checkpoint:
dataset_version:
train_samples:
validation_split:
external_datasets:
seed:
learning_rate:
epochs:
batch_size:
gradient_accumulation:
max_length:
lora_config:
prompt_template:
inference_config:
local_validation_score:
per_category_scores:
leaderboard_score:
notes:
```

권장 실험 순서:

| ID | 실험 | 목적 |
|---|---|---|
| E000 | Qwen2.5-3B zero-shot | 기준 성능 |
| E001 | clean official SFT | 공식 데이터 효과 |
| E002 | prompt / answer format | 출력 안정화 |
| E003 | LoRA hyperparameter | 학습 최적화 |
| E004 | external math SFT | 추론/문제 범위 확대 |
| E005 | data mixing | 외부 데이터 비율 |
| E006 | curriculum | 난이도 전략 |
| E007 | rejection sampling | reasoning trajectory 개선 |
| E008 | self-training | 모델 자체 생성 데이터 활용 |
| E009 | GRPO | reward 기반 추론 강화 |
| E010 | self-consistency | inference scaling |
| E011 | best-of-N / verifier | 최종 추론 성능 |

---

# Phase 0 — 프로젝트 기반 및 재현성

## 목적
이후 모든 실험이 동일한 명령과 config로 재현되도록 기반을 구축한다.

## 구현
- Python 환경 및 dependency 고정.
- `src/`, `scripts/`, `configs/`, `data/`, `outputs/`, `experiments/`, `tests/` 생성.
- config loader 구현(YAML 권장).
- deterministic seed helper 구현.
- logging 구현.
- run별 output directory 생성.
- 실행 당시 config snapshot 저장.
- 가능하면 git commit hash 기록.
- GPU/CUDA/PyTorch/Transformers 버전 기록.

## 주의사항
- 개인 로컬 절대경로를 코드에 hard-code하지 않는다.
- API key/token을 저장소에 commit하지 않는다.
- 대용량 dataset/checkpoint를 Git에 commit하지 않는다.
- `.gitignore`를 먼저 구성한다.

## 완료 조건
- clean environment에서 설치 가능.
- 최소 smoke test 실행 가능.
- 동일 config로 동일 pipeline을 재호출 가능.

---

# Phase 1 — 공식 데이터 감사(EDA) 및 정제

## 목적
공식 train을 신뢰 가능한 학습 데이터로 변환한다.

## 현재 파일 기준 schema
```text
deep_chal_math_train.csv:
  id: string
  question: string
  answer: integer

train_filtered_ids.csv:
  id: string
  answer: integer
  question: string

deep_chal_math_leaderboard_filtered.csv:
  id: string
  question: string
```

`train_filtered_ids.csv`는 이름과 달리 3개 컬럼을 가진다. **필터링 key는 오직 `id`**로 사용한다. question 문자열 일치를 요구하지 않는다.

## 필수 순서
1. raw 파일 schema 확인.
2. row count / null / dtype / ID uniqueness 검사.
3. 주최측 오류 ID 목록 검증.
4. 오류 ID를 official train에서 `id` 기준으로 제거.
   - 현재 버전에서는 627개가 모두 train에 존재해야 한다.
   - 제거 후 16,373 rows가 예상된다.
   - `question` 문자열 equality로 필터링하지 않는다.
5. 추가 데이터 품질 검사.
6. 정제 결과와 통계를 저장.

## 구현 사항
`inspect_data.py` 또는 동등 기능:
- 파일명, row 수, column 목록
- ID 중복
- 문제/풀이/정답 null
- 문자열 길이 통계
- exact duplicate
- normalized duplicate
- 비정상적으로 짧거나 긴 문제/풀이
- LaTeX/특수문자 이상 탐지 후보
- 오류 ID 중 train에 존재하지 않는 ID 보고
- 제거된 ID 수 및 비율 보고

정제 데이터는 예를 들어:
```text
data/processed/train_clean_v001.*
data/processed/train_clean_v001_manifest.json
```
처럼 versioning한다.

## 절대 금지
- `data/raw/` 원본 수정.
- 오류 ID 샘플을 실수로 validation/train에 재포함.
- 원인을 기록하지 않은 임의 row 삭제.

## 추가 품질 검사 후보
- 풀이와 최종 정답 불일치
- answer field format anomaly
- 중복/거의 동일 문제
- 깨진 LaTeX
- 메타 텍스트 혼입
- 비정상 CoT
- 동일 문제에 상충하는 정답

자동으로 확정하기 어려운 오류는 삭제하지 말고 `suspect` report로 분리한다.

## 필터 정책 구현
최소 `official_raw`, `official_clean_627`, `official_clean_plus_verified_candidates` variant를 지원한다.
추가 후보는 `source`, `reason`, `verification_status`를 기록하며 공식 627과 하나의 blacklist로 합치지 않는다.

## 완료 조건
- raw → clean 변환이 한 명령으로 재현됨.
- 공식 오류 IDs 제거 검증 test 통과.
- 정제 전/후 통계 report 존재.

---

# Phase 2 — Leakage-safe 내부 Validation 구축

## 목적
leaderboard에 반복 적응하지 않고 실험을 평가할 독립적인 local validation을 확보한다.

## 기본 정책
clean official train의 약 5~10%를 internal validation으로 보존한다.

가능하면 단순 random split보다 다음 기준의 stratification/grouping을 사용한다.
- algebra
- geometry
- number theory
- combinatorics
- probability
- calculus
- equation solving
- word problem
- 기타 실제 데이터에서 발견되는 유형

가능하면 난이도(easy/medium/hard)도 태깅한다.

## 중요
거의 동일한 문제의 변형이 train과 validation 양쪽에 존재하면 leakage다.
따라서 duplicate/near-duplicate group 단위 split을 우선한다.

## 산출물
```text
data/splits/train_v001.*
data/splits/val_v001.*
data/splits/split_manifest_v001.json
```

manifest에는 seed, split rule, source dataset hash/count를 기록한다.

## 완료 조건
- 같은 seed/config에서 동일 split 생성.
- train/val ID intersection = 0.
- duplicate group leakage 검사 통과.
- category별 샘플 수 report 존재.

---

# Phase 3 — Qwen2.5-3B-Instruct Zero-shot Baseline (E000)

## 목적
fine-tuning 전 기준 성능을 확정한다.

## 구현
- Qwen2.5-3B-Instruct 로드.
- 대회 입력 형식에 맞는 prompt template 구현.
- 우선 deterministic greedy decoding.
- internal validation 전체 평가.
- raw generation과 parsed answer 모두 저장.
- overall + category별 accuracy 저장.
- inference latency/token 통계 가능하면 기록.

## 주의
이 Phase에서 training하지 않는다.
prompt variation을 여러 개 시험한다면 각각 별도 experiment ID를 부여한다.

## 완료 조건
E000 결과 파일과 실패 사례가 재현 가능하게 저장됨.

---

# Phase 4 — 공식 Clean Train SFT Baseline (E001~E003)

## 목적
공식 clean train(`question`, integer `answer`)만으로 가능한 가장 단순하고 재현 가능한 fine-tuning baseline을 먼저 확립한다.

## 핵심 데이터 제약
공식 Data 설명상 train의 필드는 `id`, `question`, `answer`이며 **gold reasoning/solution은 제공되지 않는다.**
따라서 이전 설계처럼 공식 train에 이미 존재하는 CoT를 그대로 SFT한다고 가정해서는 안 된다.

공식 데이터만 사용하는 첫 baseline은 다음 두 계열을 구분해 실험한다.

### A. Direct-answer SFT
```text
User:
<question>

Assistant:
<integer answer>
```

가장 순수한 official-only baseline이다. 평가가 integer Exact Match이므로 반드시 확보한다.

### B. Reasoning-augmented SFT
별도 Phase에서 허용된 공개 데이터 또는 학습 데이터 구축용 상용 API 등을 이용해 reasoning trajectory를 생성/정제한 뒤 사용한다.
이 경우 더 이상 `official-only raw SFT`와 같은 실험으로 취급하지 않고 데이터 provenance를 명확히 분리한다.

## 우선 방식
초기 탐색은 LoRA/QLoRA SFT를 우선한다. 자원이 충분한 경우 이후 full fine-tuning을 별도 ablation으로 비교한다.

## 구현
- Qwen 공식 tokenizer/chat template 검증.
- direct-answer target을 우선 구현.
- answer는 정수 canonical string으로 변환.
- max sequence length 통계 기반 설정.
- truncation 발생률 기록.
- LoRA target modules 명시.
- checkpoint 저장/재개.
- train/eval loss logging.
- best checkpoint selection은 internal validation 기준.
- prompt/answer formatting config화.
- 생성 모델이 설명을 출력할 가능성을 고려해 inference parser는 별도로 유지한다.

## 주의
- leaderboard/test 데이터를 학습에 사용하지 않는다.
- 공식 train에 reasoning이 있다고 가정하지 않는다.
- synthetic reasoning을 만들 경우 생성 도구/모델/API, prompt, 생성일, filtering 방법을 provenance에 기록한다.
- 다른 모델의 **가중치**를 현재 모델에 load/merge하지 않는다.
- 학습 데이터 구축을 위한 상용 API 사용과 다른 모델 가중치 병합은 규칙상 서로 다른 행위이므로 구분한다.
- E001 이후 E002/E003에서는 주요 변수 하나씩 변경한다.

## 완료 조건
- `clean official question → integer answer`만으로 E001 baseline 확보.
- E000 대비 증감 보고.
- reasoning augmentation을 적용한 실험은 별도 experiment/dataset version으로 분리.

# Phase 5 — Integer Answer Extractor 및 공식 지표 Evaluator

## 목적
대회의 실제 scoring과 동일하게 **최종 정수 Exact Match**를 신뢰성 있게 평가하고, 모델의 자유 형식 출력에서 제출 가능한 정수를 추출한다.

## 공식 scoring
```text
prediction_integer == reference_integer
```
이면 정답, 아니면 오답이다.

분수/소수/상징식의 수학적 동치성을 공식 평가 지표로 확장해서는 안 된다.
예를 들어 모델이 `42.0` 또는 `6*7`을 출력했다면 최종 submission에는 반드시 정수 `42`로 후처리되어야 하며, parser가 이를 어떻게 처리할지는 명시적 정책과 test로 관리한다.

## 구현 분리
### 1. `extract_final_answer(raw_output)`
- 모델 출력에서 최종 답 후보 추출.
- boxed answer, 명시적 final answer 문구, 마지막 정수 등 우선순위를 정책화.
- 음수 정수 처리.
- 천 단위 쉼표 등 허용 여부 명시.
- 복수 정수가 있을 때 임의 선택하지 않도록 deterministic rule 적용.

### 2. `normalize_integer(candidate)`
- 최종적으로 Python integer 또는 canonical integer string으로 변환.
- 변환 불가능하면 parse failure.

### 3. `exact_match(pred, gold)`
- integer equality만 사용.

## 반드시 테스트할 사례
- `42`
- `The answer is 42.`
- `\\boxed{42}`
- `-17`
- `0`
- 매우 큰 정수(예: `3431577212128939`)
- 풀이에 여러 숫자가 있고 마지막 답만 42인 경우
- 빈 출력
- 정수가 없는 출력
- 여러 final-answer 후보가 충돌하는 출력
- `42.0`
- `6*7`

마지막 두 사례의 허용/거부 정책은 명시적으로 고정하고 실험 중 조용히 변경하지 않는다.

## 금지
- 공식 metric 자체에 symbolic equivalence를 섞기.
- 모델 출력에 `eval()` 사용.
- parser 변경 후 과거 실험과 같은 metric version으로 기록.

## 완료 조건
- 공식 integer Exact Match evaluator unit test 통과.
- parser version 기록 가능.
- submission 생성과 local evaluation이 동일 extractor를 공유.

# Phase 6 — 오류 분석 시스템

## 목적
점수 하나가 아니라 모델이 왜 틀리는지 데이터화한다.

## 최소 오류 taxonomy
- reasoning_error
- calculation_error
- problem_understanding
- answer_format_error
- premature_answer
- hallucination
- parse_failure
- context/truncation 관련 오류

수학 category와 결합해 분석한다.

예:
```text
geometry + reasoning_error
algebra + calculation_error
number_theory + reasoning_error
```

## 구현
각 validation sample에 다음을 저장:
- id
- category
- problem
- reference answer
- model raw output
- parsed answer
- correctness
- error label(가능한 경우)
- generation metadata

자동 분류가 불확실하면 `unknown`을 허용한다. 억지로 라벨을 만들지 않는다.

## 완료 조건
최소 category별 accuracy와 대표 실패 사례 report 생성.

---

# Phase 7 — 외부 공개 수학 데이터 도입 (E004~E005)

## 목적
공식 데이터만으로 부족한 수학 영역/추론 패턴을 보완한다.

## 후보 예시
실제 사용 전 **2026년 현재 라이선스/공개 상태/대회 규칙 허용 여부를 반드시 재확인**한다.
- GSM8K
- MATH
- MetaMathQA
- OpenMathInstruct 계열
- NuminaMath 계열
- DeepScaleR 계열 공개 데이터
- 기타 공개 mathematical reasoning dataset

후보라는 이유만으로 자동 사용하지 않는다.

## Synthetic/API-generated 학습 데이터
대회 규칙상 학습 데이터 구축 목적의 상용 API 사용은 허용된다. 이를 활용해 official train 또는 공개 문제에 reasoning/풀이를 생성할 수 있다.

단:
- Test/leaderboard 문제를 API나 검색 엔진에 보내지 않는다.
- 생성에 사용한 API/모델명, prompt template, 생성일, source problem ID, filtering rule을 기록한다.
- API 생성 reasoning은 gold reasoning으로 간주하지 않고 품질 검증 대상으로 취급한다.
- 유료 API 사용 자체는 허용되지만, **외부 데이터셋**은 무료·동등 접근 가능한 공개 데이터여야 한다는 별도 규칙과 혼동하지 않는다.

## 데이터 registry 필수 필드
```yaml
name:
version:
source:
license:
download_date:
original_count:
filtered_count:
language:
reasoning_format:
answer_format:
quality_notes:
leakage_check:
```

## Mixing
외부 데이터를 전부 단순 concatenate하지 않는다.
예시 탐색값일 뿐 고정값이 아님:
- official clean: 40%
- high-quality external: 40%
- augmented/synthetic: 20%

공식 데이터의 비중을 의도적으로 유지하고 mixing ratio를 실험 변수로 둔다.

## 데이터 선택 기준
Phase 6 오류 분석을 사용한다.
예: geometry/combinatorics가 약하면 해당 영역의 고품질 데이터를 우선한다.

## 완료 조건
각 외부 dataset의 출처/라이선스/필터링/샘플 수/누수 검사 기록이 존재.

---

# Phase 8 — External Data Leakage / Duplicate 검사

## 목적
외부 데이터로 인해 internal validation 또는 공식 평가 문제를 암기하는 현상을 방지한다.

## 검사 단계
- exact normalized text match
- hash match
- n-gram/fuzzy similarity
- 필요 시 embedding similarity

## 정책
평가/validation과 동일 또는 지나치게 유사한 외부 문제는 학습 후보에서 격리한다.
threshold는 config로 관리하고, 자동 제거와 manual-review 후보를 구분한다.

## 주의
유사한 “수학 개념” 자체는 leakage가 아니다. 특정 문제의 문구/수치/구조가 사실상 동일한지를 본다.

## 완료 조건
외부 데이터마다 leakage report 생성 및 제외 샘플 추적 가능.

---

# Phase 9 — Curriculum SFT (E006)

## 목적
3B 모델이 학습하기 적합한 순서로 reasoning complexity를 증가시킨다.

## 후보 curriculum
1. 쉬운 문제 + 명확하고 짧은 풀이
2. 중간 난이도
3. 고난도 multi-step reasoning
4. 마지막에 official clean 중심 재정렬/재학습

## 난이도 정의
가능하면 객관적인 proxy를 사용:
- source difficulty label
- solution length
- required steps
- baseline solve rate
- verifier confidence 등

solution length만을 난이도로 간주하지 않는다.

## 완료 조건
non-curriculum baseline과 동일 조건 비교.

---

# Phase 10 — Rejection Sampling / Self-Training (E007~E008)

## 목적
현재 모델이 생성한 여러 reasoning trajectory 중 정답에 도달한 고품질 경로를 선별해 다시 학습한다.

## 기본 흐름
```text
problem
  → N candidate generations
  → answer parser/evaluator
  → correct candidates
  → quality filtering
  → deduplication
  → SFT dataset
```

초기 N 예: 4/8. 자원 제한에 따라 config화한다.

## 반드시 저장할 것
- source problem ID
- generating checkpoint
- seed
- generation parameters
- candidate index
- raw reasoning
- parsed final answer
- correctness
- selection reason

## 주의
정답만 맞고 reasoning이 무의미한 trajectory를 무조건 좋은 데이터로 간주하지 않는다.
중복 trajectory와 비정상적으로 장황한 출력 필터링을 검토한다.
validation/evaluation 문제로 self-training 데이터를 만들지 않는다.

## 완료 조건
self-generated dataset provenance가 완전히 추적 가능하며 이전 SFT baseline 대비 ablation 결과 존재.

---

# Phase 11 — GRPO / Reward-based Reasoning Optimization (E009)

## 진입 조건
SFT + evaluator + rejection sampling 기반이 안정화된 뒤에만 진행한다.

## 목적
final answer correctness 중심 reward로 reasoning policy를 추가 개선한다.

## Reward 우선순위
1. final answer correctness
2. required output format correctness
3. reasoning quality reward — 신뢰할 수 있는 verifier가 있을 때만 추가

## 주의사항
다음 현상을 반드시 모니터링:
- reward hacking
- format collapse
- reasoning degeneration
- 지나치게 긴 CoT
- reward는 증가하지만 validation accuracy는 감소
- 특정 문제 유형에 편향

KL/length 관련 제어가 필요한지 실험한다.

## 중단 기준
reward만 오르고 독립 validation이 악화되면 해당 run을 승격시키지 않는다.

---

# Phase 12 — Inference Scaling / 최종 추론 최적화 (E010~E011)

## 허용
- greedy / sampling / Self-Consistency / Majority Voting / Best-of-N
- 동일 지정 베이스 verifier adapter selection
- 합리적 범위의 동일 베이스 LoRA ensemble

## 금지
- Python/SymPy/calculator/solver/tool로 문제 풀이
- Program-of-Thought 코드 실행
- 외부 공개 dataset Test-answer lookup
- 인터넷/API/검색 엔진
- 다른 베이스 모델 inference ensemble

## 기록
N, temperature, top-p, max_new_tokens, adapter/checkpoint IDs, verifier 여부, latency, VRAM, generated tokens, accuracy를 기록한다.

## 약 2,000문항 capacity benchmark
`estimated_total_wall_clock_for_2000`, `peak_vram`, `tokens_per_problem`, `samples_per_problem`, `failure/retry_rate`를 측정해 8/31 제출 window 내 안전 margin을 확보한다.

# Phase 13 — Leaderboard 및 최종 모델 선택

## 원칙
개발 loop:
```text
internal validation
→ 다수의 통제된 실험
→ 상위 후보 소수 선정
→ leaderboard 확인
→ 최종 후보 결정
```

leaderboard를 매 실험의 validation set처럼 사용하지 않는다.
Public Leaderboard는 Test 30% 기반 참고용이며 최종 순위에는 영향이 없다. Private Test 70%가 최종 순위를 결정한다. leaderboard probing은 규칙상 금지된다.

## 최종 후보 선정 기준
- internal overall accuracy
- category robustness
- seed 안정성
- inference 비용
- output parse failure
- leaderboard 결과
- leakage audit 결과

Public score 하나만으로 최종 모델을 결정하지 않는다.

---

# 3. Codex 작업 프로토콜

Codex는 각 Phase 요청을 받을 때 다음 절차를 따른다.

## 작업 전
1. 이 문서를 읽는다.
2. 현재 Phase와 선행 Phase 완료 여부를 확인한다.
3. 기존 repository 구조/코드를 먼저 검사한다.
4. 기존 구현을 불필요하게 재작성하지 않는다.
5. 공식 데이터 schema가 필요한데 확인할 수 없으면 추측으로 고정 구현하지 않는다. schema inspection 코드를 먼저 만든다.

## 구현 중
1. 재사용 로직은 `src/`.
2. 실행 entrypoint는 `scripts/`.
3. 하이퍼파라미터/경로는 config.
4. 중요한 로직에는 unit test.
5. raw data immutable.
6. 생성 산출물 versioning.
7. random seed 고정 가능.
8. 실패 시 명확한 error message.
9. dataset/model provenance 보존.

## 작업 후
Codex는 반드시 다음을 보고한다.
- 변경한 파일
- 구현한 기능
- 실행 명령
- 생성되는 산출물
- 수행한 test
- 아직 확인하지 못한 사항
- 다음 Phase 진입 전 필요한 조건

## 금지
- 사용자의 명시적 요청 없이 다음 Phase까지 대규모로 선행 구현.
- 평가 데이터를 training에 사용.
- 공식 오류 ID 무시.
- 외부 데이터 출처/라이선스 미기록.
- metric 개선 근거 없이 복잡한 기법 추가.
- secret/token commit.
- 테스트 없이 evaluator 핵심 로직 변경.
- leaderboard 결과를 local validation처럼 반복 최적화.

---

# 4. Phase Gate

각 Phase는 다음 gate를 만족한 뒤 다음 단계로 이동한다.

```text
P0 reproducible project
 ↓
P1 clean official dataset
 ↓
P2 leakage-safe local validation
 ↓
P3 E000 zero-shot baseline
 ↓
P4 official-only SFT baseline
 ↓
P5 trusted evaluator
 ↓
P6 error analysis
 ↓
P7 external data
 ↓
P8 leakage audit
 ↓
P9 curriculum
 ↓
P10 rejection/self-training
 ↓
P11 GRPO (optional; evidence-based)
 ↓
P12 inference scaling
 ↓
P13 final selection
```

GRPO는 필수 단계가 아니다. 앞 단계 대비 유의미한 성능 향상이 없거나 자원 대비 효율이 낮으면 생략할 수 있다.

---

# 5. 최우선 마일스톤

현재 최초 구현 범위는 **Phase 0~4**를 우선한다.

1. 프로젝트/실험 기반 구축
2. 공식 파일 schema inspection
3. 오류 ID 제거 및 clean train 생성
4. leakage-safe internal validation 생성
5. Qwen2.5-3B-Instruct E000 zero-shot
6. official clean train 기반 E001 SFT

이후 실제 측정 결과를 근거로 Phase 5 이상을 진행한다.

---

# 6. 2026-08-31 Final Test 운영 절차

Test 공개 이후에는 개발 단계와 분리된 **submission-only mode**로 전환한다.

1. `deep_chal_math_dataset_test.csv`를 읽되 training pipeline 입력 경로에는 절대 연결하지 않는다.
2. 인터넷/API/웹 검색 없이 로컬 모델만 사용한다. 문제 풀이용 Python/SymPy/calculator/tool 및 외부 dataset answer lookup도 금지한다.
3. 사전에 확정한 checkpoint + inference config를 우선 사용한다.
4. 허용된 Self-Consistency/Majority Voting/Best-of-N을 사용할 경우 시간/자원 제한 내에서 실행한다.
5. 공통 integer extractor로 최종 답을 추출한다.
6. 모든 ID에 answer가 존재하는지 검사한다.
7. `answer`가 전부 정수인지 검사한다.
8. 원본 test ID 순서/집합과 submission ID 집합이 정확히 일치하는지 검사한다.
9. `submission.csv`에는 공식 요구 형식의 ID/answer 두 컬럼만 기록한다.
   - 실제 입력 CSV의 식별자 컬럼명은 소문자 `id`다.
   - PDF의 submission 예시는 `ID`로 표기되어 있으므로, 최종 제출 직전 sample submission/실제 Kaggle 요구 header를 확인해 출력 컬럼명을 config로 확정한다.
   - 내부 코드에서는 canonical key를 `id`로 유지하고 submission writer에서만 rename한다.
10. 제출에 사용한 git commit, checkpoint hash/path, inference config, parser version을 freeze하여 기록한다.
11. 약 2,000문항 전체 예상 추론 시간을 확인한다.
12. 재제출 시 각 제출본의 artifact/config를 별도 보존한다.

Test 문제 자체 또는 생성 답안을 학습 데이터로 되돌려 넣지 않는다.

---

# 7. 규칙 업데이트 절차

대회 Rules/Data/Overview의 새 정보가 확인되면:
1. 이 문서의 `현재 확인된 데이터 정책`을 먼저 갱신한다.
2. 기존 구현과 충돌하는 규칙을 검색한다.
3. 충돌 시 공식 규칙을 기준으로 코드를 수정한다.
4. 변경 이유를 git commit/experiment notes에 기록한다.
5. 평가 데이터 사용 범위, 외부 데이터 허용 범위, 제출 자원 제한은 특히 재검증한다.

이 문서는 구현 계획이지만 **공식 Rules를 대체하지 않는다.**


---

# 8. 문서 근거 및 버전 메모

- 본 문서의 대회 규칙/데이터/평가 관련 확정사항은 사용자가 제공한 `아주소중한딥러닝챌린지_2026.pdf`의 Overview, Data, Rules를 반영해 업데이트했다.
- 주요 반영 사항:
  - 공식 train schema가 `id/question/answer`이며 gold reasoning 컬럼이 명시되지 않음을 반영.
  - Public 30% / Private 70%, integer Exact Match 반영.
  - leaderboard filtered 파일 및 train error-ID 파일의 정확한 역할 반영.
  - 2026-08-31 Final Test 공개/당일 제출 절차 반영.
  - 공개 외부 데이터의 무료·동등 접근 조건 반영.
  - 학습 데이터 구축용 상용 API 허용 및 Test 직접 질의 금지 반영.
  - 다른 모델 가중치 load/merge, inference ensemble, from-scratch pre-training 금지 반영.
  - 수상 후보자의 코드/가중치/데이터 목록/환경/방법론 제출 및 재현 검증 의무 반영.
  - 모델 성능뿐 아니라 발표 평가를 고려한 실험 근거/방법론 기록 강화.


## 데이터 감사 버전 메모 — 2026-08-24
본 문서의 shape/schema/invariant는 사용자가 업로드한 다음 파일을 직접 읽어 확인한 결과다.
- `deep_chal_math_train.csv`
- `deep_chal_math_leaderboard_filtered.csv`
- `train_filtered_ids.csv`

향후 파일 교체 시 Phase 1 검사 결과와 본 수치를 비교하고, 변경이 공식 업데이트인지 확인한 뒤 문서를 갱신한다.


## 운영팀 Q&A 반영 메모 — 2026-08-24
Final Test 규모/재제출, 추가 Train 오류 자체 필터링, Test-answer lookup 금지, inference tool 금지, 동일 베이스 LoRA/verifier 허용, 구조/파라미터 변경 허용에 관한 운영팀 답변을 반영했다.
