export TOKENIZERS_PARALLELISM=false;
file_name='stage1_HPC_512_Truncation_total_merged_24step'
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM_Custom/stage3.py \
trainer.devices=${gpus} \
filename=${file_name} \
trainer=llada8b_stage1 \
trainer.mol_representation=string_only \
wandb_id=50m9n3jc \
trainer.skip_sanity_check=False