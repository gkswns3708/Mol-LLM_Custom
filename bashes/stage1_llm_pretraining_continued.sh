export TOKENIZERS_PARALLELISM=false;
file_name='stage1_llm_pretraining'
gpus="'0,1,2,3,4,5,6,7'"
model='llada8b'
mol_representation='string_only'


python Mol-LLM_Custom/stage3.py \
trainer.devices=${gpus} \
filename=${file_name} \
trainer=${model} \
trainer.mol_representation=${mol_representation} \
trainer.skip_sanity_check=true \
wandb_id=xehuqcy3 \
validate_first=false \
ckpt_path="'/home/jovyan/CHJ/Mol-LLM_Custom/checkpoint/Custom_LLaDA/stage1_llm_pretraining/last.ckpt'"