#!/bin/bash
export TOKENIZERS_PARALLELISM=false
gpus="'0,1,2,3,4,5,6,7'"
file_name="stage1_llm_pretraining_test"

# [수정 2] 각 줄 끝의 공백 제거 및 마지막 줄 백슬래시 제거 확인
python Mol-LLM_Custom/stage3.py \
    --config-name test_llada \
    trainer.devices="${gpus}" \
    filename="${file_name}" \
    trainer.mol_representation=string_only \
    trainer.skip_sanity_check=true