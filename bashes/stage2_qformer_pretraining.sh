export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
gnn=$1
file_name=stage2_qformer_pretraining

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=multi_task \
gnn=$gnn \
trainer=mistral7b_80gb_llava_pretraining \
trainer.skip_sanity_check=true \
pretrained_ckpt_path="'/data/all_checkpoints/MT_mistral7b_string_only_12ep_0415/epoch=11-step=40991.ckpt'"
