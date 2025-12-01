export TOKENIZERS_PARALLELISM=false;
export HF_DATASETS_CACHE="/tmp/jovyan/hf_cache"
export HF_HOME="/tmp/jovyan/hf_home"

gpus="'0,1,2,3,4,5,6,7'"
data_tag=3.3M_0415
filename="HJChoi"
ckpt_path="/home/jovyan/CHJ/Mol-LLM_Custom/checkpoint/mol-llm.ckpt"

echo "==============Executing task: ablation==============="

# --config-name "test" 를 추가하여 default.yaml 대신 test.yaml을 읽도록 함
python Mol-LLM_Custom/stage3.py \
--config-name "test" \
trainer.devices=$gpus \
mode=test \
filename=${filename} \
data.data_tag=${data_tag} \
trainer=mistral7b_80gb \
trainer.skip_sanity_check=false \
ckpt_path=${ckpt_path}