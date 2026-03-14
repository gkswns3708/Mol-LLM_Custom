export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
file_name='512_Truncation_chebi-20-mol2text'

python Mol-LLM_Custom/stage3.py \
trainer.devices=${gpus} \
filename=${file_name} \
data=multi_task_stage3 \
gnn=gine_tokengt \
trainer=llada8b_stage3 \
pretrained_ckpt_path="'/home/jovyan/CHJ/Mol-LLM_Custom/checkpoint/LLaDA_Stage3/stage3_512_Truncation_molpo-replace-0.3_v3_ccc_replacement_all/epoch=00-step=005000-train.ckpt'" \
trainer.sampling_steps=32 \
trainer.semi_ar_block_size=32 \
trainer.semi_ar_steps_per_block=4
