export TOKENIZERS_PARALLELISM=false;
gpus=$1
data_tag=3.3M_0415_molpo-replace-0.3
filename="HJChoi" # Replace with your actual filename
ckpt_path="/app/Mol-LLM/checkpoint/mol-llm.ckpt" # Replace with your actual checkpoint path'"

echo "==============Executing task: ablation==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
mode=test \
filename=${filename} \
data.data_tag=${data_tag} \
trainer=mistral7b_80gb \
trainer.skip_sanity_check=false \
ckpt_path=${ckpt_path}

