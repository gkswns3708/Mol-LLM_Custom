export TOKENIZERS_PARALLELISM=false;
file_name='stage1_llm_pretraining_Tpod_bace_only_eos'
gpus="'0,1,2,3'"

python Mol-LLM_Custom/stage3.py \
trainer.devices=${gpus} \
filename=${file_name} \
trainer=llada8b \
trainer.mol_representation=string_only \
trainer.skip_sanity_check=true \