export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
file_name=stage3_512_Truncation_molpo-replace-0.3_v3_ccc_replacement_all

python Mol-LLM_Custom/stage3.py \
trainer.devices=$gpus \
filename="$file_name" \
data=multi_task_stage3 \
gnn=gine_tokengt \
trainer=llada8b_stage3 \
ckpt_path="/home/jovyan/CHJ/Mol-LLM_Custom/checkpoint/LLaDA_Stage3/stage3_512_Truncation_molpo-replace-0.3_v3_ccc_replacement_all/last.ckpt"

