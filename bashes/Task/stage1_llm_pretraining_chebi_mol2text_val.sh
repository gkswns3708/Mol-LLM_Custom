export TOKENIZERS_PARALLELISM=false;
file_name='stage1_Tpod_chebi_mol2text_10xHighLR_val_500iter_32steps'
gpus="'0,1,2,3'"

python Mol-LLM_Custom/stage3.py \
trainer.devices=${gpus} \
filename=${file_name} \
trainer=llada8b_stage1 \
trainer.mol_representation=string_only \
trainer.skip_sanity_check=true \
ckpt_path="'/app/Mol-LLM_Custom/checkpoint/LLaDA_Stage1/stage1_Tpod_chebi_mol2text_10xHighLR/epoch=04-step=000500-train.ckpt'" \
trainer.sampling_steps=32 \
trainer.semi_ar_block_size=32 \
trainer.semi_ar_steps_per_block=4