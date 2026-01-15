import os
import glob
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# =============================================================================
# 0. 라이브러리 및 모델 토크나이저 설정 (가장 중요)
# =============================================================================

# --- Transformers (Tokenizer 로드용) ---
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[Critical Warning] 'transformers' 라이브러리가 없습니다. 토큰 길이를 정확히 잴 수 없습니다.")

# --- RDKit & Selfies ---
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    import selfies as sf
    HAS_SELFIES = True
except ImportError:
    HAS_SELFIES = False

# -----------------------------------------------------------------------------
# [USER TODO] 아래 경로를 실제 모델 경로(로컬 폴더 or HuggingFace ID)로 수정하세요.
# 경로가 없거나 로드에 실패하면, 자동으로 'SMILES Regex' 방식(Fallback)으로 계산합니다.
# -----------------------------------------------------------------------------
TOKENIZER_PATHS = {
    # 예: 'facebook/galactica-1.3b'
    'Galactica': 'facebook/galactica-1.3b',

    # 예: '/home/jovyan/models/ChemDFM-13B' 혹은 HuggingFace ID
    'ChemDFM': 'OpenDFM/ChemDFM-v1.0-13B',

    # 예: '/home/jovyan/models/LlaSMol-7b'
    'LlaSMol': 'osunlp/LlaSMol-7b',

    # Mol-LLM: 커스텀 토크나이저 (SELFIES dict 포함)
    # 'Mol-LLM': 'facebook/opt-1.3b',  # Base tokenizer (actual: custom with SELFIES)
}

# Mol-LLM 커스텀 토크나이저 설정
MOL_LLM_CONFIG = {
    'base_model': 'facebook/opt-1.3b',  # 또는 'meta-llama/Llama-2-7b-hf', 'mistralai/Mistral-7B-v0.1'
    'selfies_dict_path': 'Mol-LLM_Custom/model/selfies_dict.txt',
    'added_tokens_path': 'Mol-LLM_Custom/model/added_tokens.py',
    'add_selfies_tokens': True,
}

# 토크나이저 로드 (전역 변수)
LOADED_TOKENIZERS = {}

def create_mol_llm_tokenizer(base_model_path, selfies_dict_path, added_tokens_path):
    """
    Mol-LLM의 실제 inference 시 사용되는 커스텀 토크나이저 생성
    - SELFIES dictionary (2944 tokens) 추가
    - 특수 화학 토큰들 추가 (BOOLEAN, FLOAT, DESCRIPTION, etc.)
    """
    import sys
    import importlib.util

    # 1. Base tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,  # Mol-LLM은 use_fast=False 사용
        padding_side="left",
        trust_remote_code=True
    )

    # 2. SELFIES tokens 추가 (2944개)
    if os.path.exists(selfies_dict_path):
        with open(selfies_dict_path, "r") as f:
            selfies_tokens = [line.strip() for line in f.readlines() if line.strip()]
        tokenizer.add_tokens(selfies_tokens)
        print(f"   └─ Added {len(selfies_tokens)} SELFIES tokens")

    # 3. added_tokens.py에서 특수 토큰들 추가
    if os.path.exists(added_tokens_path):
        spec = importlib.util.spec_from_file_location("added_tokens", added_tokens_path)
        added_tokens = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(added_tokens)

        # 모든 특수 토큰 수집
        additional_tokens = []
        for attr_name in dir(added_tokens):
            if not attr_name.startswith("__"):
                attr_value = getattr(added_tokens, attr_name)
                if isinstance(attr_value, list):
                    additional_tokens.extend(attr_value)

        tokenizer.add_tokens(additional_tokens)
        print(f"   └─ Added {len(additional_tokens)} special chemical tokens")

        # MOL_EMBEDDING token을 special token으로 추가
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [added_tokens.MOL_EMBEDDING[0]]}
        )

    return tokenizer

if HAS_TRANSFORMERS:
    print("\n=== Loading Tokenizers ===")
    for name, path in TOKENIZER_PATHS.items():
        try:
            # Mol-LLM은 커스텀 토크나이저 생성
            if name == 'Mol-LLM':
                print(f"[{name}] Creating custom tokenizer with SELFIES dictionary...")
                tokenizer = create_mol_llm_tokenizer(
                    MOL_LLM_CONFIG['base_model'],
                    MOL_LLM_CONFIG['selfies_dict_path'],
                    MOL_LLM_CONFIG['added_tokens_path']
                )
                LOADED_TOKENIZERS[name] = tokenizer
                print(f"✅ [{name}] Custom tokenizer loaded successfully.")
            else:
                # 다른 모델들은 기존 방식대로
                tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
                LOADED_TOKENIZERS[name] = tokenizer
                print(f"✅ [{name}] Tokenizer loaded successfully.")
        except Exception as e:
            print(f"⚠️ [{name}] Failed to load tokenizer from '{path}'. Error: {e}")
            print(f"   (Fallback to Regex)")
            LOADED_TOKENIZERS[name] = None
    print("==========================\n")

# --- Fallback용 SMILES Regex (토크나이저 로드 실패 시 사용) ---
SMILES_REGEX = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")

# =============================================================================
# 1. 길이 계산 함수 (핵심 로직)
# =============================================================================
def get_model_specific_length(text, model_identifier):
    """
    1. 해당 모델의 Tokenizer가 로드되어 있으면 -> tokenizer.encode() 길이 반환
    2. 로드 안 되어 있으면 -> 데이터 타입(SMILES/Text)에 따른 Fallback 계산
    """
    if not text: return 0
    text = str(text)

    # 1. 모델명 매핑 (CSV의 모델명 -> TOKENIZER_PATHS의 키)
    # 예: 'ChemDFM (Cls)' -> 'ChemDFM'
    tokenizer_key = next((k for k in LOADED_TOKENIZERS.keys() if k in model_identifier), None)
    tokenizer = LOADED_TOKENIZERS.get(tokenizer_key)

    # 2. Tokenizer 사용 (우선순위 1)
    if tokenizer:
        try:
            # add_special_tokens=False: 순수 텍스트 길이만 측정 (BOS/EOS 제외)
            return len(tokenizer.encode(text, add_special_tokens=False))
        except:
            pass # 인코딩 에러 시 Fallback으로 이동

    # 3. Fallback (우선순위 2 - 토크나이저 없을 때)
    # 간단히 공백 기준(Text) 혹은 Regex(SMILES)
    if '[' in text and ']' in text and HAS_SELFIES: # SELFIES 추정
        try: return sf.len_selfies(text)
        except: pass
        
    if " " in text: # 일반 텍스트 문장
        return len(text.split())
    
    # SMILES Regex
    return len(SMILES_REGEX.findall(text))

# =============================================================================
# 2. 메트릭 및 헬퍼 함수
# =============================================================================
def levenshtein_dist(s1, s2):
    if len(s1) < len(s2): return levenshtein_dist(s2, s1)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def get_mol(smiles):
    if not HAS_RDKIT or not smiles: return None
    try: return Chem.MolFromSmiles(smiles)
    except: return None

def get_canonical_smiles(mol):
    if mol: return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return None

def get_fingerprint(mol, fp_type):
    if not mol: return None
    try:
        if fp_type == 'morgan':
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        elif fp_type == 'maccs':
            return MACCSkeys.GenMACCSKeys(mol)
        elif fp_type == 'rdk':
            return Chem.RDKFingerprint(mol)
    except: return None
    return None

def parse_text(text):
    if pd.isna(text): return ""
    text = str(text).strip()
    for tag in ['SELFIES', 'SMILES', 'DESCRIPTION']:
        match = re.search(f'<{tag}>\s*(.*?)\s*</{tag}>', text, re.DOTALL)
        if match: return match.group(1).strip()
    for t in ['<|end_of_text|>', '<|eot_id|>', '</s>']:
        text = text.replace(t, '')
    return text.strip()

def calculate_row_metrics(row, group_name, model_name):
    metrics = {}
    
    raw_label = parse_text(row['label'])
    raw_pred = parse_text(row['pred'])
    
    # [핵심] 모델별 토크나이저를 이용한 길이 계산
    # Label(정답)의 길이를 기준으로 Binning 합니다.
    metrics['seq_len'] = get_model_specific_length(raw_label, model_name)

    # 화학적 변환 (SELFIES -> SMILES)
    label_chem, pred_chem = raw_label, raw_pred
    if HAS_SELFIES and '[' in raw_label:
        try: label_chem = sf.decoder(raw_label)
        except: pass
    if HAS_SELFIES and '[' in raw_pred:
        try: pred_chem = sf.decoder(raw_pred)
        except: pass

    if group_name == "Captioning":
        ref = raw_label.split()
        hyp = raw_pred.split()
        sf_smooth = SmoothingFunction().method1
        metrics['BLEU-4'] = sentence_bleu([ref], hyp, weights=(0.25,0.25,0.25,0.25), smoothing_function=sf_smooth) * 100
        metrics['METEOR'] = 0.0 # NLTK dependency 제외
    else:
        # Structure Based
        mol_label = get_mol(label_chem)
        mol_pred = get_mol(pred_chem)
        
        metrics['Validity'] = 1.0 if mol_pred else 0.0
        
        canon_label = get_canonical_smiles(mol_label)
        canon_pred = get_canonical_smiles(mol_pred)
        metrics['Exact Match'] = 1.0 if (canon_label and canon_pred and canon_label == canon_pred) else 0.0
        
        fp_l = get_fingerprint(mol_label, 'maccs')
        fp_p = get_fingerprint(mol_pred, 'maccs')
        metrics['MACCS FTS'] = DataStructs.TanimotoSimilarity(fp_l, fp_p) if (fp_l and fp_p) else 0.0
        
        if group_name == "Generation":
            metrics['BLEU-4'] = sentence_bleu([SMILES_REGEX.findall(raw_label)], SMILES_REGEX.findall(raw_pred), weights=(0.25,0.25,0.25,0.25), smoothing_function=SmoothingFunction().method1) * 100
            metrics['Levenshtein'] = levenshtein_dist(raw_label, raw_pred)

    return metrics

# =============================================================================
# 3. 설정 (Configuration)
# =============================================================================
BASE_DIR = "Mol-LLM_Custom/Inference_log/Benchmark_inference_csv"
SAVE_DIR = "results_model_tokenized" # 저장 폴더

os.makedirs(SAVE_DIR, exist_ok=True)

model_configs = {
    'ChemDFM (Cls)': {'path_pattern': os.path.join(BASE_DIR, "classification_chemdfm_13B_20251223/csv_results/20251223/125813_chemdfm_test_rank*.csv")},
    'Galactica': {'path_pattern': os.path.join(BASE_DIR, "galactica_20251223/csv_results/20251223/134454_galactica_test_rank*.csv")},
    'LlaSMol': {'path_pattern': os.path.join(BASE_DIR, "llasmol_20251223/csv_results/20251223/011220_llasmol_test_rank*.csv")},
    'ChemDFM (Gen/Cap)': {'path_pattern': os.path.join(BASE_DIR, "non_classification_chemdfm_13B_20251223/csv_results/20251223/131017_chemdfm_test_rank*.csv")}
}

TASK_GROUPS = {
    "Captioning": {
        "tasks": ['chebi-20-mol2text', 'smol-molecule_captioning'],
        "metrics": ['BLEU-4', 'METEOR'],
        "bin_size": 10
    },
    "Generation": {
        "tasks": ['smol-molecule_generation', 'chebi-20-text2mol'], 
        "metrics": ['BLEU-4', 'Exact Match', 'Levenshtein', 'MACCS FTS', 'Validity'], 
        "bin_size": 10
    },
    "Synthesis": {
        "tasks": ['forward_reaction_prediction', 'retrosynthesis', 'reagent_prediction', 
                  'smol-forward_synthesis', 'smol-retrosynthesis'],
        "metrics": ['Exact Match', 'MACCS FTS'],
        "bin_size": 10
    }
}

VIS_CONFIG = {
    "min_samples_per_bin": 1,
    "palette": "tab10",
    "fig_size": (12, 7),
    "font_scale": 1.2
}

# =============================================================================
# 4. 실행 및 저장
# =============================================================================
def main():
    all_data = []
    print("데이터 처리 시작... (토크나이징으로 인해 시간이 조금 더 소요될 수 있습니다)")
    
    for model_name, config in model_configs.items():
        files = glob.glob(config['path_pattern'])
        # 모델 이름 단순화 (파일 저장 및 범례용)
        simple_model_name = 'ChemDFM' if 'ChemDFM' in model_name else model_name
        
        for f in files:
            try:
                df = pd.read_csv(f)
                all_target_tasks = [t for g in TASK_GROUPS.values() for t in g['tasks']]
                df = df[df['task'].isin(all_target_tasks)]
                
                for _, row in df.iterrows():
                    task = row['task']
                    group_name = next(name for name, g in TASK_GROUPS.items() if task in g['tasks'])
                    
                    # model_name을 넘겨주어 해당 토크나이저를 쓰도록 함
                    metrics = calculate_row_metrics(row, group_name, simple_model_name)
                    
                    metrics['model'] = simple_model_name
                    metrics['task'] = task
                    metrics['group'] = group_name
                    all_data.append(metrics)
            except Exception as e:
                # print(f"Error processing file {f}: {e}")
                pass

    df_results = pd.DataFrame(all_data)
    if df_results.empty:
        print("데이터 로드 실패. 경로를 확인하세요.")
        return

    print(f"[{SAVE_DIR}] 폴더에 시각화 저장 중...")
    
    for group_name, group_info in TASK_GROUPS.items():
        group_df = df_results[df_results['group'] == group_name]
        if group_df.empty: continue
        
        for task in group_info['tasks']:
            task_df = group_df[group_df['task'] == task]
            if task_df.empty: continue
            
            # Binning (Bin 10)
            bin_size = group_info['bin_size']
            task_df['len_bin'] = (task_df['seq_len'] // bin_size) * bin_size
            
            valid_groups = task_df.groupby(['model', 'len_bin']).filter(lambda x: len(x) >= VIS_CONFIG['min_samples_per_bin'])
            if valid_groups.empty: continue
            
            for metric in group_info['metrics']:
                if metric not in valid_groups.columns: continue
                
                fig, ax1 = plt.subplots(figsize=VIS_CONFIG['fig_size'])
                sns.set_theme(style="white", font_scale=VIS_CONFIG['font_scale'])
                
                # --- Right Axis: Data Distribution (Line Plot) ---
                ax2 = ax1.twinx()
                num_models = valid_groups['model'].nunique()
                dist_counts = valid_groups.groupby('len_bin').size() / num_models
                
                sns.lineplot(
                    x=dist_counts.index, y=dist_counts.values,
                    color='gray', linestyle='--', linewidth=2, marker='X', markersize=6,
                    label='Avg Data Distribution', ax=ax2
                )
                ax2.set_ylabel('Number of Samples', color='gray')
                ax2.tick_params(axis='y', colors='gray')
                ax2.legend(loc='upper right') 
                ax2.grid(False)

                # --- Left Axis: Metric vs Model Tokens ---
                sns.lineplot(
                    data=valid_groups, x='len_bin', y=metric, hue='model', style='model',
                    markers=True, dashes=False, palette=VIS_CONFIG['palette'],
                    linewidth=2.5, markersize=8, errorbar='sd', ax=ax1
                )
                
                # X축 라벨을 명확히 변경
                ax1.set_title(f'[{task}] {metric} vs Length', fontsize=16)
                ax1.set_xlabel(f"Sequence Length (Model Tokens) (Bin: {bin_size})", fontsize=14)
                ax1.set_ylabel(metric, fontsize=14)
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                # Z-order
                ax1.set_zorder(ax2.get_zorder() + 1)
                ax1.patch.set_visible(False)
                ax1.legend(title='Model', loc='upper left')
                
                plt.tight_layout()
                filename = os.path.join(SAVE_DIR, f"{task}_{metric.replace(' ', '_')}.png")
                plt.savefig(filename, dpi=150)
                plt.close()
                print(f"  Saved: {filename}")

    print("\n완료되었습니다.")

if __name__ == "__main__":
    main()