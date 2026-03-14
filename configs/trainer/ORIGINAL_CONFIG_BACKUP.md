# LLaDA Stage3 Original Config Backup

**백업 날짜:** 2026-03-02
**파일:** `llada8b_stage3.yaml`
**목적:** Flash Attention 및 속도 최적화 적용 전 원본 값 보존

---

## 📋 원본 설정값

### 1. Batch Size 설정 (라인 78-82)

```yaml
# Batch Size 설정
# effective_batch_size = batch_size * num_devices * accumulate_grad_batches
batch_size: 1                 # ⬅️ Stage 3: 2 (메모리 고려)
total_batch_size: 80        # ⬅️ Stage 3: 1024 (논문 권장)
accumulate_grad_batches: 10   # ⬅️ Stage 3: 64 (1024 / (2 * 8) = 64)
```

**Effective Batch Size 계산:**
- `batch_size × num_devices × accumulate_grad_batches = 1 × 8 × 10 = 80`

---

### 2. Hardware 설정 (라인 24-30)

```yaml
strategy_name: null           # null: auto | "ddp" | "fsdp" | "deepspeed_stage_2"
accelerator: gpu              # "gpu" | "cpu" | "tpu"
devices: "0,1,2,3,4,5,6,7"    # 사용할 GPU 인덱스
precision: bf16-mixed         # "bf16-mixed" (권장) | "16-mixed" | "32"
num_workers: 0                # DataLoader workers (0: main process에서 로드)
find_unused_parameters: true  # ⬅️ Stage 3: true (MolPO용 DDP)
```

---

### 3. 모델 아키텍처 설정 (라인 34-39)

```yaml
llm_model: "GSAI-ML/LLaDA-8B-Instruct"  # LLaDA-8B-Base | LLaDA-8B-Instruct
tune_llm: lora                # "lora": LoRA 학습 | "full": Full FT | "freeze": 동결
tune_gnn: true                # ⬅️ Stage 3: true (Graph Encoder trainable)
mol_representation: string+graph  # ⬅️ Stage 3: "string+graph" (Graph 사용)
load_in_8bit: false           # 8bit 양자화 로드
```

**Flash Attention 설정:**
- ❌ **없음** (추가 필요)

---

### 4. Optimizer 및 Learning Rate (라인 98-112)

```yaml
optimizer: adamw
weight_decay: 0.1
gradient_clip_val: 1.0
log_every_n_steps: 10

scheduler: warmup_stable_decay_lr
warmup_steps: 100             # ⬅️ Stage 3: 100 steps warmup
warmup_lr: 0.0
decay_ratio: 0.1
min_lr_ratio: 0.1

init_lr: 0.00004              # ⬅️ Stage 3: 4e-5 (논문 권장)
min_lr: 0.00002               # ⬅️ Stage 3: 2e-5
```

---

### 5. 학습 설정 (라인 70-76)

```yaml
max_steps: -1
max_epochs: 6                 # ⬅️ Stage 3: 6 epochs (논문 권장)
every_n_epochs: 1
task: null
num_beams: 1
skip_sanity_check: true       # ⬅️ true 권장
num_sanity_val_steps: 0
```

---

### 6. Debug Logging (라인 178-187)

```yaml
log_embedding_status: true
embedding_log_interval: 100
log_model_init_details: true
log_nan_details: true
nan_log_dir: './nan_logs'

log_stepwise_denoising: false
stepwise_log_dir: '/app/stepwise_logs'
stepwise_max_samples: '8'
```

---

## 🎯 속도 최적화를 위한 권장 변경사항

### 변경 1: Flash Attention 활성화

**추가할 위치:** 라인 39 다음 (모델 아키텍처 섹션)

```yaml
load_in_8bit: false           # 8bit 양자화 로드
use_flash_attention: true     # ⭐ Flash Attention 2 활성화
```

---

### 변경 2: Batch Size 증가

**기존:**
```yaml
batch_size: 1
accumulate_grad_batches: 10
```

**권장 (옵션 1 - 안정적):**
```yaml
batch_size: 4                 # 4배 증가
accumulate_grad_batches: 2    # 조정 (effective batch = 4 × 8 × 2 = 64)
```

**권장 (옵션 2 - 최대 속도):**
```yaml
batch_size: 8                 # 8배 증가
accumulate_grad_batches: 1    # 조정 (effective batch = 8 × 8 × 1 = 64)
```

---

### 변경 3: num_workers 증가 (선택)

**기존:**
```yaml
num_workers: 0
```

**권장:**
```yaml
num_workers: 4                # 데이터 로딩 병렬화
```

---

## 📊 예상 성능 개선

| 설정 | 현재 | Flash Attn만 | Flash + Batch 4배 | Flash + Batch 8배 |
|------|------|--------------|-------------------|-------------------|
| 스텝당 시간 | 7.3초 | 5.1-5.8초 | 1.8-2.4초 | 1.0-1.5초 |
| 5,629 스텝 | 11.4시간 | 8-9시간 | 3-4시간 | 1.5-2.3시간 |
| 6 에포크 전체 | 68시간 | 48-54시간 | 17-23시간 | 10-14시간 |

---

## ⚠️ 주의사항

1. **Flash Attention 사용 전 확인:**
   ```bash
   conda activate MolDA_CHJ
   python -c "import flash_attn; print(flash_attn.__version__)"
   ```

2. **OOM 발생 시:**
   - `batch_size: 4` → `batch_size: 2`로 감소
   - `accumulate_grad_batches`를 비례해서 증가

3. **Effective Batch Size 유지:**
   - 원본: `1 × 8 × 10 = 80`
   - 변경 후에도 동일하게 유지 권장

---

## 📝 변경 이력

- **2026-03-02:** 원본 설정 백업 생성
