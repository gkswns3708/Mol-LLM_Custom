export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'" # 이게 실제 GPU 갯수 책임지는 곳,
# gpus="'0,1'" # 이게 실제 GPU 갯수 책임지는 곳,
# data_tag=512_Truncation
data_tag=512_Truncation_100_sampled
filename="val_epoch1_test" # Replace with your actual filename
ckpt_path="/home/jovyan/CHJ/Mol-LLM_Custom/checkpoint/Custom_LLaDA/stage1_llm_pretraining/last.ckpt" # Replace with your actual checkpoint path'"

echo "==============Executing task: Specific Task==============="
python Mol-LLM_Custom/stage3.py \
trainer.devices=$gpus \
mode=val \
filename=${filename} \
data.data_tag=${data_tag} \
trainer=llada8b \
trainer.mol_representation=string_only \
trainer.skip_sanity_check=false \
ckpt_path=${ckpt_path}

