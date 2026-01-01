# Checkpoint 저장 및 재개 가이드

## 📁 Checkpoint 저장 설정

### 1. Step 기반 자동 저장

`configs/trainer/llada8b.yaml` 파일에서 설정:

```yaml
# N step마다 checkpoint 저장
save_on_n_steps: 500              # 500 step마다 저장
save_top_k_checkpoints: 5         # 최근 5개만 유지 (-1 = 모두 저장)

# Best model 저장
save_top_k_best: 3                # 상위 3개 best 모델 유지

# Epoch 기반 추가 저장 (선택)
save_every_n_epochs: 1            # 1 epoch마다 추가 저장 (0 = 비활성화)
```

### 2. 저장되는 Checkpoint 파일

학습 중 다음과 같은 checkpoint들이 자동 저장됩니다:

```
checkpoint/Custom_LLaDA/stage1_llm_pretraining/
├── epoch=00-step=000500-train.ckpt    # 500 step
├── epoch=00-step=001000-train.ckpt    # 1000 step
├── epoch=01-step=001500-train.ckpt    # 1500 step
├── last.ckpt                          # 가장 최근 checkpoint (자동 업데이트)
├── best_20231231_epoch=02_step=003500_loss=0.1234.ckpt  # Best #1
├── best_20231231_epoch=03_step=004200_loss=0.1567.ckpt  # Best #2
└── best_20231231_epoch=04_step=005100_loss=0.1892.ckpt  # Best #3
```

---

## 🔄 학습 재개 (Resume Training)

### 방법 1: Config 파일에서 설정

`configs/train_llada.yaml` 파일 수정:

```yaml
# 특정 step checkpoint에서 재개
ckpt_path: "/home/jovyan/CHJ/Mol-LLM_Custom/checkpoint/Custom_LLaDA/stage1_llm_pretraining/epoch=03-step=001500-train.ckpt"

# 또는 가장 최근 checkpoint에서 재개
ckpt_path: "/home/jovyan/CHJ/Mol-LLM_Custom/checkpoint/Custom_LLaDA/stage1_llm_pretraining/last.ckpt"
```

그 후 학습 실행:
```bash
python stage3.py
```

### 방법 2: 커맨드라인에서 지정

```bash
python stage3.py ckpt_path="/path/to/checkpoint/epoch=03-step=001500-train.ckpt"
```

### Resume 시 복원되는 항목

- ✅ **모델 가중치** (Model weights)
- ✅ **Optimizer 상태** (학습률, momentum 등)
- ✅ **Learning rate scheduler 상태**
- ✅ **현재 epoch/step 번호**
- ✅ **Best validation loss**
- ✅ **난수 생성기 상태** (재현성 보장)

---

## 🎯 사전 학습 모델 로드 (Fine-tuning)

처음부터 학습을 시작하지만 특정 모델의 가중치만 로드하는 경우:

### Config 파일 설정

```yaml
# pretrained model 가중치만 로드 (optimizer는 초기화)
pretrained_ckpt_path: "/home/jovyan/CHJ/Mol-LLM_Custom/checkpoint/Custom_LLaDA/stage1_llm_pretraining/epoch=07-step=051600-train.ckpt"

# ckpt_path는 null로 유지
ckpt_path: null
```

### ⚠️ 주의사항

- `ckpt_path`와 `pretrained_ckpt_path`는 **동시에 사용할 수 없습니다**
- `pretrained_ckpt_path` 사용 시:
  - 모델 가중치만 로드
  - Optimizer, scheduler는 초기화
  - Epoch/Step은 0부터 시작

---

## 📊 Checkpoint 파일명 형식

### Step-based Checkpoint
```
epoch={epoch:02d}-step={step:06d}-train.ckpt
예: epoch=03-step=001500-train.ckpt
```
- `epoch`: 현재 epoch (2자리)
- `step`: 전역 step 번호 (6자리)
- `train`: 학습 중 저장된 checkpoint

### Best Checkpoint
```
best_{날짜}_epoch={epoch:02d}_step={step:06d}_loss={val_loss:.4f}.ckpt
예: best_20231231_epoch=02_step=003500_loss=0.1234.ckpt
```
- 날짜: checkpoint 저장 날짜
- `val_loss`: Validation loss 값

---

## 💡 활용 예시

### 1. 학습 중단 후 재개

학습이 중단되었을 때:

```bash
# 가장 최근 checkpoint에서 재개
python stage3.py ckpt_path="checkpoint/Custom_LLaDA/stage1_llm_pretraining/last.ckpt"
```

### 2. 특정 Step에서 재개

특정 step부터 다시 실험하고 싶을 때:

```bash
python stage3.py ckpt_path="checkpoint/Custom_LLaDA/stage1_llm_pretraining/epoch=03-step=001500-train.ckpt"
```

### 3. Best 모델로 추가 학습

Best validation loss를 기록한 모델에서 계속 학습:

```bash
python stage3.py ckpt_path="checkpoint/Custom_LLaDA/stage1_llm_pretraining/best_20231231_epoch=02_step=003500_loss=0.1234.ckpt"
```

### 4. Checkpoint 정리 (디스크 공간 절약)

필요 없는 중간 checkpoint 삭제:

```bash
cd checkpoint/Custom_LLaDA/stage1_llm_pretraining/

# Step checkpoint만 삭제 (best, last는 유지)
rm epoch=*-step=*-train.ckpt

# 또는 특정 step 이전 checkpoint만 삭제
rm epoch=00-step=00*.ckpt
```

---

## 🔍 Checkpoint 정보 확인

저장된 checkpoint의 정보를 확인하려면:

```python
import torch

ckpt = torch.load("checkpoint/path/to/file.ckpt", map_location="cpu")
print(f"Epoch: {ckpt['epoch']}")
print(f"Global step: {ckpt['global_step']}")
print(f"Best validation loss: {ckpt.get('callbacks', {}).get('ModelCheckpoint', {}).get('best_model_score', 'N/A')}")
```

---

## ⚙️ 고급 설정

### 디스크 공간 절약

자주 저장하되 오래된 checkpoint는 자동 삭제:

```yaml
save_on_n_steps: 100              # 자주 저장
save_top_k_checkpoints: 10        # 최근 10개만 유지
```

### 모든 Checkpoint 보관

실험 재현을 위해 모든 checkpoint 저장:

```yaml
save_on_n_steps: 500
save_top_k_checkpoints: -1        # 모두 저장 (주의: 디스크 공간 많이 사용)
```

### Validation 기반 저장만 사용

Step 저장 비활성화하고 best model만 저장:

```yaml
save_on_n_steps: 0                # Step 저장 비활성화
save_top_k_best: 5                # Best 5개만 유지
```

---

## 📝 권장 설정

일반적인 학습:
```yaml
save_on_n_steps: 500              # 500 step마다
save_top_k_checkpoints: 5         # 최근 5개 유지
save_top_k_best: 3                # Best 3개 유지
```

긴 학습 (며칠):
```yaml
save_on_n_steps: 1000             # 덜 자주 저장
save_top_k_checkpoints: 3         # 디스크 공간 절약
save_top_k_best: 5                # 더 많은 best 모델 유지
```

짧은 실험:
```yaml
save_on_n_steps: 100              # 자주 저장
save_top_k_checkpoints: -1        # 모두 저장
save_top_k_best: 1                # Best 1개만
```
