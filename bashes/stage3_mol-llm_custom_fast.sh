#!/bin/bash
# ============================================
# LLaDA Stage 3 Training Script (FAST VERSION)
#
# 🚀 속도 최적화 적용:
#   - Flash Attention 2 활성화
#   - Batch Size 4배 증가 (1 → 4)
#   - num_workers 증가 (0 → 4)
#   - 로깅 빈도 감소 (10 → 50)
#
# 예상 성능:
#   - 속도 향상: 3-4배
#   - 5,629 스텝: 11시간 23분 → 3-4시간
#   - 6 에포크: 68시간 → 17-23시간
# ============================================

export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
file_name=stage3_512_Truncation_molpo-replace-0.3_v3_ccc_replacement_all_FAST

python Mol-LLM_Custom/stage3.py \
trainer.devices=$gpus \
filename="$file_name" \
data=multi_task_stage3 \
gnn=gine_tokengt \
trainer=llada8b_stage3_fast \
pretrained_ckpt_path="Mol-LLM_Custom/checkpoint/LLaDA_Stage2/stage1_HPC_total_240steps_stage2/last.ckpt"
