import os
import re
import time
import argparse
import multiprocessing
from collections import defaultdict
import pandas as pd
import numpy as np
import selfies as sf
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from datasets import load_from_disk
from tqdm import tqdm

# =============================================================================
# [Configuration] Task Categories based on Mol-LLM Paper Methodology
# =============================================================================

# 논문 및 관행상 Scaffold Split을 엄격히 지켜야 하는 Task (MoleculeNet 등)
SCAFFOLD_SPLIT_TASKS = {
    "bace", "bbbp", "clintox", "tox21", "toxcast", "sider", 
    "hiv", "muv", "esol", "freesolv", "lipo", "hopv"
}

# 논문에서 Random Split을 사용했거나(암묵적), 구조적 다양성이 워낙 커서 허용되는 Task
# (Mol-Instructions, CheBI-20, QM9 등)
RANDOM_SPLIT_TASKS = {
    "forward_reaction_prediction", "retrosynthesis", "reagent_prediction",
    "chebi-20-mol2text", "chebi-20-text2mol",
    "smol-molecule_captioning", "smol-molecule_generation",
    "smol-forward_synthesis", "smol-retrosynthesis",
    "qm9_homo", "qm9_lumo", "qm9_homo_lumo_gap", "qm9_additional_label"
}

# =============================================================================
# [Helper Functions]
# =============================================================================

def decode_selfies_to_smiles(selfies_str):
    """SELFIES 문자열을 SMILES로 디코딩"""
    try:
        # 태그 제거 (<SELFIES>, </SELFIES> 등)
        clean_str = re.sub(r"<[^>]+>", "", selfies_str).strip()
        smiles = sf.decoder(clean_str)
        return smiles
    except:
        return None

def get_scaffold(smiles):
    """SMILES에서 Murcko Scaffold 추출"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except:
        return None

def process_batch(batch_data):
    """멀티프로세싱을 위한 배치 처리 함수"""
    results = []
    for item in batch_data:
        task = item.get("task", "unknown")
        # 'input_mol_string'은 SELFIES 형태라고 가정
        input_mol = item.get("input_mol_string", "")
        
        smiles = decode_selfies_to_smiles(input_mol)
        
        if smiles:
            scaffold = get_scaffold(smiles)
            valid = True
            canon_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        else:
            scaffold = None
            valid = False
            canon_smiles = None
            
        results.append({
            "task": task,
            "valid": valid,
            "smiles": canon_smiles,
            "scaffold": scaffold
        })
    return results

def parallel_process(dataset, num_cores=64):
    """데이터셋을 병렬로 처리하여 SMILES/Scaffold 추출"""
    data_list = list(dataset)
    chunk_size = len(data_list) // num_cores + 1
    chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
    
    results = []
    with multiprocessing.Pool(num_cores) as pool:
        for chunk_res in tqdm(pool.imap(process_batch, chunks), total=len(chunks)):
            results.extend(chunk_res)
            
    return results

# =============================================================================
# [Main Verification Logic]
# =============================================================================

def verify_dataset(train_path, test_path, num_cores=32):
    print(f"=== Mol-LLM Dataset Verification (Reproduction Mode) ===")
    print(f"[Info] Loading TRAIN set from: {train_path}")
    train_ds = load_from_disk(train_path)
    print(f"[Info] TRAIN size: {len(train_ds)}")
    
    print(f"[Info] Loading TEST set from: {test_path}")
    test_ds = load_from_disk(test_path)
    print(f"[Info] TEST size: {len(test_ds)}")

    print(f"[Info] Processing molecules with {num_cores} cores...")
    
    # 1. Process Data
    train_processed = parallel_process(train_ds, num_cores)
    test_processed = parallel_process(test_ds, num_cores)
    
    # 2. Organize by Task
    task_stats = defaultdict(lambda: {
        "train_smiles": set(), "test_smiles": set(),
        "train_scaffolds": set(), "test_scaffolds": set(),
        "train_invalid": 0, "test_invalid": 0,
        "train_total": 0, "test_total": 0
    })
    
    # Fill Train Stats
    for res in train_processed:
        t = res["task"]
        task_stats[t]["train_total"] += 1
        if res["valid"]:
            task_stats[t]["train_smiles"].add(res["smiles"])
            if res["scaffold"]: task_stats[t]["train_scaffolds"].add(res["scaffold"])
        else:
            task_stats[t]["train_invalid"] += 1

    # Fill Test Stats
    for res in test_processed:
        t = res["task"]
        task_stats[t]["test_total"] += 1
        if res["valid"]:
            task_stats[t]["test_smiles"].add(res["smiles"])
            if res["scaffold"]: task_stats[t]["test_scaffolds"].add(res["scaffold"])
        else:
            task_stats[t]["test_invalid"] += 1

    # 3. Generate Report
    print("\n" + "="*80)
    print(f"{'TASK NAME':<40} | {'TYPE':<10} | {'LEAK(Exact)':<12} | {'LEAK(Scaf)':<12} | {'STATUS'}")
    print("="*80)

    for task, stats in sorted(task_stats.items()):
        # Determine Split Type Expectation
        if any(x in task for x in SCAFFOLD_SPLIT_TASKS):
            split_type = "SCAFFOLD"
        else:
            split_type = "RANDOM"
            
        # Check Exact Leakage
        intersection_smiles = stats["train_smiles"].intersection(stats["test_smiles"])
        leak_exact_count = len(intersection_smiles)
        
        # Check Scaffold Leakage
        intersection_scaffolds = stats["train_scaffolds"].intersection(stats["test_scaffolds"])
        leak_scaf_count = len(intersection_scaffolds)
        
        # Determine Status
        status = "[PASS]"
        
        # Rule 1: Exact Leakage is usually suspicious, but we flag it as WARNING for reproduction
        if leak_exact_count > 0:
            status = "[WARN:Leak]"

        # Rule 2: Scaffold Leakage logic
        if split_type == "SCAFFOLD":
            if leak_scaf_count > 0:
                status = "[FAIL:Scaf]" # Scaffold task must have 0 scaffold leakage
        else: # RANDOM
            # Random split tasks implicitly allow scaffold leakage for reproduction
            if leak_scaf_count > 0:
                pass # It matches "Random Split" expectation
            
        # Formatting Output
        scaf_display = f"{leak_scaf_count}"
        if split_type == "RANDOM" and leak_scaf_count > 0:
            scaf_display += " (OK)"
            
        print(f"{task:<40} | {split_type:<10} | {leak_exact_count:<12} | {scaf_display:<12} | {status}")

    print("="*80)
    print("Note: 'RANDOM' tasks allow scaffold leakage as per standard reproduction settings.")
    print("      'SCAFFOLD' tasks must have 0 scaffold leakage.")
    print("="*80)

# 위쪽의 함수 정의들(verify_dataset 등)은 그대로 실행해두세요.
# 맨 아래 if __name__ == "__main__": 블록만 이걸로 교체하시면 됩니다.

if __name__ == "__main__":
    # argparse 부분 제거하고 직접 변수에 할당
    
    # 1. 검증할 데이터셋 경로 (사용자 로그 기반 설정)
    train_path = "/home/jovyan/CHJ/Mol-LLM_Custom/dataset/train_official/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_train_3.3M_0415"
    test_path = "/home/jovyan/CHJ/Mol-LLM_Custom/dataset/train_official/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_test_3.3M_0415"
    
    # 2. CPU 코어 수 (Notebook 환경에서는 너무 높으면 불안정할 수 있으니 32~64 권장)
    num_cores = 64

    # 3. 검증 함수 실행
    try:
        verify_dataset(train_path, test_path, num_cores)
    except Exception as e:
        print(f"오류 발생: {e}")
        # 멀티프로세싱 오류 시 코어 수를 1로 줄여서 디버깅 해보세요.
        # verify_dataset(train_path, test_path, num_cores=1)