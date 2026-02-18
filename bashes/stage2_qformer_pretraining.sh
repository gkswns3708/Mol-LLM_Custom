export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
gnn="'gine_tokengt'"
file_name='stage1_HPC_total_240steps_stage2'

python Mol-LLM_Custom/stage3.py \
trainer.devices=$gpus \
trainer=llada8b_stage2 \
filename=$file_name \
data=multi_task_stage2 \
gnn=$gnn \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=true \
pretrained_ckpt_path="'/home/jovyan/CHJ/Mol-LLM_Custom/checkpoint/LLaDA_Stage1/stage1_HPC_total_240steps/epoch=12-end.ckpt'"
