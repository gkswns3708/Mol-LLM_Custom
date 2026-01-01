export TOKENIZERS_PARALLELISM=false;
file_name='stage1_llm_pretraining'
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM_Custom/stage3.py \
trainer.devices=${gpus} \
filename=${file_name} \
trainer=llada8b \
trainer.mol_representation=string_only \
trainer.skip_sanity_check=false \
pretrained_ckpt_path="'/home/jovyan/CHJ/Mol-LLM_Custom/checkpoint/Custom_LLaDA/stage1_llm_pretraining/epoch=07-step=51600-train.ckpt'"