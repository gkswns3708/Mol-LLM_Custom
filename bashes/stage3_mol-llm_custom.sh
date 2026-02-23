export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
file_name=stage3_HPC_total_240steps_lowconfidence_random_semi_ar

python Mol-LLM_Custom/stage3.py \
trainer.devices=$gpus \
filename="$file_name" \
data=multi_task_stage3 \
gnn=gine_tokengt \
trainer=llada8b_stage3 \
pretrained_ckpt_path="Mol-LLM_Custom/checkpoint/LLaDA_Stage2/stage1_HPC_total_240steps_stage2/last.ckpt"

