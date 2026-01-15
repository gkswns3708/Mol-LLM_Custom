export TOKENIZERS_PARALLELISM=false;
file_name='stage1_Tpod_chebi_mol2text_HighLR_continued'
gpus="'0,1,2,3'"

python Mol-LLM_Custom/stage3.py \
trainer.devices=${gpus} \
filename=${file_name} \
trainer=llada8b \
trainer.mol_representation=string_only \
wandb_id=3she8m78
trainer.skip_sanity_check=false \
pretrained_ckpt_path="'/app/Mol-LLM_Custom/checkpoint/LLaDA_Stage1/stage1_Tpod_chebi_mol2text_HighLR/last.ckpt'"w