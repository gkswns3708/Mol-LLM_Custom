import os
import re
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
import selfies as sf
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from datasets import load_from_disk, Dataset, enable_progress_bar
from tabulate import tabulate
import multiprocessing

# HF Progress Bar 활성화
enable_progress_bar()

# =============================================================================
# [설정: Configuration]
# =============================================================================

# 사용자가 명시한 "같은 태스크" 목록만 그룹화합니다.
# 이 목록에 없는 태스크(예: bace, bbbp 등)는 자기 자신의 이름을 그룹 ID로 사용합니다.
MERGED_TASK_GROUPS = {
    # 1. Retrosynthesis
    "retrosynthesis": "GROUP_RETROSYNTHESIS",
    "smol-retrosynthesis": "GROUP_RETROSYNTHESIS",

    # 2. Molecule Captioning
    "chebi-20-mol2text": "GROUP_CAPTIONING",
    "smol-molecule_captioning": "GROUP_CAPTIONING",

    # 3. Molecule Generation
    "chebi-20-text2mol": "GROUP_GENERATION",
    "smol-molecule_generation": "GROUP_GENERATION",

    # 4. Forward Synthesis
    "forward_reaction_prediction": "GROUP_FORWARD",
    "smol-forward_synthesis": "GROUP_FORWARD",
}

# Scaffold Split을 적용해야 하는 태스크 목록 (원래 코드 기준)
# 주의: 여기 있는 태스크들은 Test 셋의 Scaffold와 겹치면 Train에서 제거됩니다.
# 단, 그룹 ID가 같을 때만 적용됩니다.
SCAFFOLD_SPLIT_TARGETS = {
    "bace", "bbbp", "clintox", "tox21", "toxcast", "sider", 
    "hiv", "muv", "esol", "freesolv", "lipo", "hopv",
    # smol 시리즈가 있다면 아래와 같이 매핑될 것입니다.
    "smol-property_prediction-bbbp", "smol-property_prediction-clintox",
    "smol-property_prediction-hiv", "smol-property_prediction-sider",
    "smol-property_prediction-tox21", "smol-property_prediction-toxcast",
    "smol-property_prediction-muv", "smol-property_prediction-esol",
    "smol-property_prediction-lipo", "smol-property_prediction-freesolv"
}

# =============================================================================
# [Helper Functions]
# =============================================================================

def get_strict_task_group(task_name):
    """
    사용자가 지정한 쌍(Pair)은 공통 그룹 ID를 반환하고,
    그 외(나머지 Property Prediction 등)는 태스크 이름 그대로를 반환하여
    서로 섞이지 않도록 합니다.
    """
    # 명시된 그룹이면 해당 그룹 ID 반환
    if task_name in MERGED_TASK_GROUPS:
        return MERGED_TASK_GROUPS[task_name]
    
    # 명시되지 않은 태스크(bace, hiv, bbbp 등)는 자기 자신이 곧 그룹 ID
    return task_name

from rdkit import RDLogger

# [중요] RDKit의 C++ 레벨 에러 로그 끄기 (SMILES Parse Error 등 출력 방지)
RDLogger.DisableLog('rdApp.*') 

def decode_and_get_info(batch):
    """
    SELFIES 전용 디코딩 함수.
    SELFIES 디코딩 실패 시 RDKit으로 넘기지 않고 즉시 Invalid 처리합니다.
    """
    input_mols = batch["input_mol_string"]
    canon_smiles_list, scaffold_list, valid_list = [], [], []
    
    for input_mol in input_mols:
        res_smiles, res_scaffold, is_valid = "", "", False
        
        if input_mol:
            try:
                # 1. 태그 제거 및 공백 제거
                clean_str = re.sub(r"<[^>]+>", "", str(input_mol)).strip()
                
                # 2. SELFIES -> SMILES 디코딩
                # 에러 발생 시 except로 점프하여 RDKit 호출을 방지함
                smiles = sf.decoder(clean_str)
                
                # 3. 디코딩된 SMILES가 존재할 때만 RDKit으로 변환
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        res_smiles = Chem.MolToSmiles(mol, canonical=True)
                        res_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or ""
                        is_valid = True
                        
            except Exception:
                # sf.decoder에서 에러가 나거나, RDKit 변환 중 에러가 나면
                # 조용히 넘어가고 valid=False 처리
                pass
        
        canon_smiles_list.append(res_smiles)
        scaffold_list.append(res_scaffold)
        valid_list.append(is_valid)
        
    return {"canon_smiles": canon_smiles_list, "scaffold": scaffold_list, "valid": valid_list}

# =============================================================================
# [Main Pipeline]
# =============================================================================
def main_cleaning_pipeline(train_path, val_path, test_path, base_save_dir, num_cores=24):
    print(f"=== [Step 1] 데이터 로드 및 SMILES 파싱 (Multiprocessing: {num_cores} cores) ===")
    
    splits = {"train": train_path, "val": val_path, "test": test_path}
    dfs = {}
    drop_stats = defaultdict(int) # 통계 저장용

    for name, path in splits.items():
        print(f" -> Loading {name.upper()}...")
        ds = load_from_disk(path)
        
        # 1. 필요한 컬럼만 선택하여 파싱 (속도 최적화)
        # task 정보와 input_mol_string은 필수
        cols_to_map = ['task', 'input_mol_string']
        
        parsed_ds = ds.select_columns(cols_to_map).map(
            decode_and_get_info, 
            batched=True, 
            batch_size=1000, 
            num_proc=num_cores, 
            desc=f"Parsing {name} Molecules"
        )
        
        # 2. Pandas DataFrame으로 변환 및 병합
        df_raw = ds.to_pandas()
        df_info = parsed_ds.to_pandas()[['canon_smiles', 'scaffold', 'valid']]
        
        # 인덱스 리셋 후 병합
        combined_df = pd.concat([df_raw.reset_index(drop=True), df_info.reset_index(drop=True)], axis=1)
        
        # 3. Task Group ID 부여 (핵심 로직)
        combined_df['task_group'] = combined_df['task'].apply(get_strict_task_group)
        
        dfs[name] = combined_df
        print(f"    - {name}: {len(combined_df):,} rows loaded.")

    print(f"\n=== [Step 2] Test Set 기준 블랙리스트 생성 (Data Leakage 방지) ===")
    # Test 셋에 있는 (Task Group, Canonical SMILES) 조합을 수집
    # Test 셋에 있는 (Task Group, Scaffold) 조합을 수집 (Scaffold Split 대상인 경우만)
    
    test_df = dfs["test"]
    valid_test = test_df[test_df['valid']]
    
    # 딕셔너리 구조: { task_group_id: { set of smiles } }
    test_black_smiles = valid_test.groupby('task_group')['canon_smiles'].apply(set).to_dict()
    test_black_scaf = valid_test.groupby('task_group')['scaffold'].apply(set).to_dict()

    print(f" -> Test Set 블랙리스트 생성 완료 (Task Group 별로 격리됨)")

    print(f"\n=== [Step 3] Train/Val 정제 (오염 제거 및 중복 제거) ===")
    
    final_dfs = {"test": dfs["test"]} # Test셋은 수정하지 않음 (단, invalid 마킹 등은 필요없으니 원본 유지)

    for split_name in ["train", "val"]:
        df = dfs[split_name]
        print(f" -> Processing {split_name.upper()} ({len(df):,} rows)...")
        
        # 출력 컬럼 찾기 (중복 제거 시 Input + Output이 완전히 같으면 제거하기 위해)
        out_col = next((c for c in ['label', 'output_string', 'output', 'target'] if c in df.columns), "output")
        
        # 삭제 여부를 판단하는 함수 (Pandas Apply용 아님, 로직 설명용 -> 벡터화/Apply로 구현)
        # 로직:
        # 1. SMILES가 Valid하지 않으면 -> Keep (파싱 실패는 안전하게 둠 or Drop? 보통 둠)
        # 2. (Task Group, SMILES)가 Test 블랙리스트에 있으면 -> Drop (Exact Match Leakage)
        # 3. (Task Group, Scaffold)가 Test 블랙리스트에 있고, 해당 Task가 Scaffold Split 대상이면 -> Drop
        # 4. (Task Group, Input, Output)이 이전 행과 중복되면 -> Drop (Dedup)
        
        def check_leakage(row):
            t_group = row['task_group']
            smiles = row['canon_smiles']
            scaffold = row['scaffold']
            is_valid = row['valid']
            original_task = row['task']
            
            if not is_valid:
                return "Keep"
            
            # 1. Exact Match Check (같은 그룹 내에서만)
            if smiles in test_black_smiles.get(t_group, set()):
                return "Drop: Test Leakage (Exact)"
            
            # 2. Scaffold Match Check (Scaffold Split 대상 태스크인 경우만)
            # original_task 이름이 scaffold split 목록에 있거나, 그룹 자체가 그래야 하는 경우
            if original_task in SCAFFOLD_SPLIT_TARGETS:
                if scaffold in test_black_scaf.get(t_group, set()):
                    return "Drop: Test Leakage (Scaffold)"
            
            return "Keep"

        # Apply로 Leakage 체크 (Progress Bar 표시)
        tqdm.pandas(desc=f"Checking Leakage in {split_name}")
        df['leakage_status'] = df.progress_apply(check_leakage, axis=1)
        
        # 통계 집계
        leakage_counts = df['leakage_status'].value_counts()
        for status, count in leakage_counts.items():
            if status != "Keep":
                # 대표 태스크 이름으로 기록 (그룹핑된 경우 하나만 잡힐 수 있음)
                drop_stats[f"{split_name}_{status}"] += count

        # 1차 필터링 (Leakage 제거)
        df_no_leak = df[df['leakage_status'] == "Keep"].copy()
        
        # 2차 필터링 (중복 제거)
        # 조건: 'task_group', 'input_mol_string', 'output' 이 모두 같으면 중복
        # 즉, bace는 bace끼리만, retrosynthesis는 smol-retro와 retro가 섞여서 중복 검사됨
        before_dedup = len(df_no_leak)
        
        # keep='first': 첫 번째 등장만 남김
        df_clean = df_no_leak.drop_duplicates(
            subset=['task_group', 'input_mol_string', out_col], 
            keep='first'
        )
        
        after_dedup = len(df_clean)
        dedup_count = before_dedup - after_dedup
        drop_stats[f"{split_name}_Drop: Internal Duplicate"] += dedup_count
        
        print(f"    - Original: {len(df):,}")
        print(f"    - After Leakage Filter: {before_dedup:,}")
        print(f"    - After Dedup Filter: {after_dedup:,} (Final)")
        
        final_dfs[split_name] = df_clean

    print(f"\n=== [Step 4] 결과 저장 및 리포트 ===")
    
    # 간단한 통계 출력
    print("\n[Drop Statistics Summary]")
    for k, v in drop_stats.items():
        print(f"  {k}: {v:,}")

    # 데이터셋 저장
    for name, df in final_dfs.items():
        # 임시 컬럼 제거
        cols_to_drop = ['canon_smiles', 'scaffold', 'valid', 'task_group', 'leakage_status']
        save_df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # 저장 경로 생성
        file_name = f"GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_{name}_FINAL_CLEANED"
        save_full_path = os.path.join(base_save_dir, file_name)
        
        print(f" -> Saving {name} to {save_full_path} ...")
        Dataset.from_pandas(save_df, preserve_index=False).save_to_disk(save_full_path)

if __name__ == "__main__":
    # Multiprocessing 시작 방식 설정 (Linux 환경 권장)
    try: multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass

    # 경로 설정 (사용자 환경에 맞게 수정)
    train_in = "Mol-LLM_Custom/dataset/train_official/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_train_3.3M_0415_raw"
    val_in = "Mol-LLM_Custom/dataset/train_official/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_val_3.3M_0415_raw"
    test_in = "Mol-LLM_Custom/dataset/train_official/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_test_3.3M_0415_raw"
    save_dir = "Mol-LLM_Custom/dataset/train_official/"
    
    main_cleaning_pipeline(train_in, val_in, test_in, save_dir, num_cores=24)