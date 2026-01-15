# mol2text Test
bash /app/Mol-LLM_Custom/bashes/stage1_llm_pretraining_test.sh > ./log/test/temp_256steps_mol2text.txt
# Mol2text + homo + bace+ Text2mol 학습
bash /app/Mol-LLM_Custom/bashes/Task/stage1_llm_pretraining_merged_bace_chebi_mol2text_chebi_text2mol_qm9_homo.sh > log/merged_bace_chebi_mol2text_chebi_text2mol_qm9_homo/stage1_EOSLoss_126349vocab_$(TZ='UTC-9' date +%Y%m%d_%H%M%S).txt