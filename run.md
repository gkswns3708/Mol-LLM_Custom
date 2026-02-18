# mol2text Test
bash /app/Mol-LLM_Custom/bashes/stage1_llm_pretraining_test.sh > ./log/test/temp_256steps_mol2text.txt
# Mol2text + homo + bace+ Text2mol 학습
bash /app/Mol-LLM_Custom/bashes/Task/stage1_llm_pretraining_merged_bace_chebi_mol2text_chebi_text2mol_qm9_homo.sh > log/merged_bace_chebi_mol2text_chebi_text2mol_qm9_homo/stage1_EOSLoss_126349vocab_$(TZ='UTC-9' date +%Y%m%d_%H%M%S).txt

# Run HPC
bash /home/jovyan/CHJ/Mol-LLM_Custom/bashes/Task/stage1_llm_pret-LLM_Custom/bashes/Task/stage1_llm_pretraining_chebi_mol2text_text2mol.sh > log/chebi_mol2text_text2mol/adding_10HighLR_53epoch_Continued_12634stage1__FullEOSLoss_10validation_RightPadding_10HighLR_53epoch_Continued_126349vocab_$(TZ='UTC-9' date +%Y%m%d_%H%M%S).txt

# Run HPC total_merged
bash /home/jovyan/CHJ/Mol-LLM_Custom/bashes/Task/stage1_llm_pretraining_total_merged.sh > log/total_merged/FullEOSLoss_10total_RightPadding_126349vocab_$(TZ='UTC-9' date +%Y%m%d_%H%M%S).txt

bash /home/jovyan/CHJ/Mol-LLM_Custom/bashes/Task/stage1_llm_pretraining_total_merged.sh > log/512_Truncation_total_merged_1percent_sampled/FullEOSLoss_1pct_total_RightPadding_126349vocab_$(TZ='UTC-9' date +%Y%m%d_%H%M%S).txt

bash /home/jovyan/CHJ/Mol-LLM_Custom/bashes/Task/stage1_llm_pretraining_total_merged_continued.sh > log/512_Truncation_10percent_sampled/FullEOSLoss_From_09epoch_13500step_RightPadding_126349vocab_$(TZ='UTC-9' date +%Y%m%d_%H%M%S).txt

bash /home/jovyan/CHJ/Mol-LLM_Custom/bashes/stage1_llm_pretraining.sh > /home/jovyan/CHJ/log/512_Truncation_chebi_mol2text/train_$(TZ='UTC-9' date +%Y%m%d_%H%M%S).txt

bash /home/jovyan/CHJ/Mol-LLM_Custom/bashes/stage2_qformer_pretraining.sh > /home/jovyan/CHJ/log/stage2_512_Trucation_240steps/step2_train_240steps_$(TZ='UTC-9' date +%Y%m%d_%H%M%S).txt