export TOKENIZERS_PARALLELISM=false;
file_name='stage1_llm_pretraining'
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=${gpus} \
filename=${file_name} \
trainer=mistral7b_80gb \
trainer.mol_representation=string_only \
trainer.skip_sanity_check=false \


