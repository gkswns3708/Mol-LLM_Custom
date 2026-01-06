#!/usr/bin/env python3
"""
MolDA Benchmark Results Visualization Script
=============================================
- Processes CSV files from csv_with_prompt directory
- Computes metrics for different task groups (Captioning, Generation, Synthesis, Classification, Regression)
- Generates visualizations: probability distributions, sample counts, metric plots
- Caches processed dataset for fast subsequent runs
"""

import os
import glob
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 0. Libraries & Setup
# =============================================================================
try:
    from transformers import AutoTokenizer
    from datasets import Dataset, load_from_disk
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("[Critical Warning] 'transformers' or 'datasets' library not found.")

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[Warning] 'rdkit' library not found.")

try:
    import selfies as sf
    HAS_SELFIES = True
except ImportError:
    HAS_SELFIES = False
    print("[Warning] 'selfies' library not found.")

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("[Warning] 'rouge_score' library not found.")

# =============================================================================
# 1. Configuration
# =============================================================================
BASE_DIR = "/home/jovyan/iclr_genmol/backup_aica/MolDA_result/csv_with_prompt"
SAVE_DIR = "/home/jovyan/iclr_genmol/backup_aica/MolDA_result/visualization_results"
CACHE_DIR = os.path.join(SAVE_DIR, "processed_dataset_cache")
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_PROC = max(1, multiprocessing.cpu_count() - 2)

# Tokenizer paths for length calculation
TOKENIZER_PATHS = {
    'Galactica': 'facebook/galactica-1.3b',
    'ChemDFM': 'OpenDFM/ChemDFM-v1.0-13B',
    '3D-MolM': 'meta-llama/Llama-2-7b-hf',
}

VIS_CONFIG = {
    "min_samples_per_bin": 1,
    "palette": "tab10",
    "fig_size": (14, 8),
    "font_scale": 1.2,
    "dpi": 150
}

# Manual thresholds for filtering outliers (97th percentile fallback)
MANUAL_TASK_THRESHOLDS = {
    'chebi-20-mol2text': 120,
    'smol-molecule_captioning': 120,
    'smol-molecule_generation': 140,
    'chebi-20-text2mol': 140,
    'forward_reaction_prediction': 300,
    'retrosynthesis': 300,
    'reagent_prediction': 200,
    'smol-forward_synthesis': 300,
    'smol-retrosynthesis': 400,
}

# Task grouping configuration
TASK_GROUPS = {
    "Captioning": {
        "tasks": ['chebi-20-mol2text', 'smol-molecule_captioning'],
        "metrics": ['BLEU-2', 'BLEU-4', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
        "bin_size": 10,
        "unit": "Words"
    },
    "Generation": {
        "tasks": ['smol-molecule_generation', 'chebi-20-text2mol'],
        "metrics": ['BLEU-4', 'Exact Match', 'Levenshtein', 'MACCS FTS', 'Validity'],
        "bin_size": 10,
        "unit": "Tokens"
    },
    "Synthesis": {
        "tasks": ['forward_reaction_prediction', 'retrosynthesis', 'reagent_prediction',
                  'smol-forward_synthesis', 'smol-retrosynthesis'],
        "metrics": ['Exact Match', 'MACCS FTS', 'Validity'],
        "bin_size": 10,
        "unit": "Tokens"
    },
    "Classification": {
        "tasks": ['bace', 'smol-property_prediction-bbbp', 'smol-property_prediction-clintox',
                  'smol-property_prediction-hiv', 'smol-property_prediction-sider'],
        "metrics": ['Accuracy'],
        "bin_size": 10,
        "unit": "Tokens"
    },
    "Regression": {
        "tasks": ['qm9_homo', 'qm9_lumo', 'qm9_homo_lumo_gap',
                  'smol-property_prediction-esol', 'smol-property_prediction-lipo'],
        "metrics": ['MAE', 'RMSE'],
        "bin_size": 10,
        "unit": "Tokens"
    }
}

# All possible metric keys
ALL_POSSIBLE_METRICS = [
    'BLEU-2', 'BLEU-4', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L',
    'Exact Match', 'Levenshtein', 'MACCS FTS', 'Validity',
    'Accuracy', 'MAE', 'RMSE'
]

# Model configurations - mapping model names to file patterns
MODEL_CONFIGS = {
    'ChemDFM (Cls)': {
        'path': os.path.join(BASE_DIR, "chemdfm_classification"),
        'pattern': "*.csv",
        'display_name': 'ChemDFM'
    },
    'ChemDFM (Gen)': {
        'path': os.path.join(BASE_DIR, "chemdfm_non_classification"),
        'pattern': "*.csv",
        'display_name': 'ChemDFM'
    },
    'Galactica': {
        'path': os.path.join(BASE_DIR, "galactica"),
        'pattern': "*.csv",
        'display_name': 'Galactica'
    },
    '3D-MolM': {
        'path': os.path.join(BASE_DIR, "3d_molm"),
        'pattern': "*test_rank*.csv",  # Exclude property prediction specific files
        'display_name': '3D-MolM'
    }
}

SMILES_REGEX = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")

# =============================================================================
# 2. Global Tokenizers Loading
# =============================================================================
GLOBAL_TOKENIZERS = {}
if HAS_HF:
    print("\n=== Loading Tokenizers ===")
    for name, path in tqdm(TOKENIZER_PATHS.items(), desc="Loading Tokenizers"):
        try:
            GLOBAL_TOKENIZERS[name] = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
        except Exception as e:
            print(f"  [Warning] Failed to load {name}: {e}")
            GLOBAL_TOKENIZERS[name] = None

# =============================================================================
# 3. Helper Functions
# =============================================================================
def parse_text(text):
    """Extract clean text from various tag formats."""
    if text is None:
        return ""
    text = str(text).strip()

    # Try to extract from tags
    for tag in ['SELFIES', 'SMILES', 'DESCRIPTION']:
        match = re.search(f'<{tag}>\\s*(.*?)\\s*</{tag}>', text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Remove common end tokens
    for t in ['<|end_of_text|>', '<|eot_id|>', '</s>', '<eos>']:
        text = text.replace(t, '')

    return text.strip()


def get_mol(smiles):
    """Safely create RDKit molecule from SMILES."""
    if not HAS_RDKIT or not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None


def get_canonical_smiles(mol):
    """Get canonical SMILES from molecule."""
    if mol:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return None


def get_fingerprint(mol):
    """Get MACCS fingerprint from molecule."""
    if not mol:
        return None
    try:
        return MACCSkeys.GenMACCSKeys(mol)
    except:
        return None


def levenshtein_dist(s1, s2):
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_dist(s2, s1)
    if len(s2) == 0:
        return len(s1)

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


def parse_numeric(value):
    """Parse numeric value from string, handling various formats."""
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except:
        # Try to extract number from string
        match = re.search(r'[-+]?\d*\.?\d+', str(value))
        if match:
            return float(match.group())
        return np.nan


# =============================================================================
# 4. Core Metric Computation
# =============================================================================
def compute_metrics_row(example):
    """Compute all metrics for a single row."""
    results = {k: np.nan for k in ALL_POSSIBLE_METRICS}

    task = example['task']
    model_name = example['model']
    raw_label = example['label']
    raw_pred = example['pred']

    # Determine task group
    group_name = None
    for g_name, g_info in TASK_GROUPS.items():
        if task in g_info['tasks']:
            group_name = g_name
            break

    if group_name is None:
        group_name = "Unknown"

    label_text = parse_text(raw_label)
    pred_text = parse_text(raw_pred)

    results['clean_label'] = label_text
    results['clean_pred'] = pred_text
    results['group'] = group_name
    results['seq_len'] = 0

    # --- Length Calculation ---
    length_val = 0
    if group_name == "Captioning":
        length_val = len(label_text.split())
    else:
        # Try tokenizer first
        tokenizer_key = None
        for k in GLOBAL_TOKENIZERS.keys():
            if k in model_name:
                tokenizer_key = k
                break

        tokenizer = GLOBAL_TOKENIZERS.get(tokenizer_key)
        length_found = False

        if tokenizer:
            try:
                length_val = len(tokenizer.encode(label_text, add_special_tokens=False))
                length_found = True
            except:
                pass

        if not length_found and HAS_SELFIES and '[' in label_text:
            try:
                length_val = sf.len_selfies(label_text)
                length_found = True
            except:
                pass

        if not length_found:
            length_val = len(SMILES_REGEX.findall(label_text))

    results['seq_len'] = int(length_val)

    # --- Metrics Calculation ---
    # Convert SELFIES to SMILES if needed
    label_chem, pred_chem = label_text, pred_text
    if HAS_SELFIES and '[' in label_text:
        try:
            label_chem = sf.decoder(label_text)
        except:
            pass
    if HAS_SELFIES and '[' in pred_text:
        try:
            pred_chem = sf.decoder(pred_text)
        except:
            pass

    if group_name == "Captioning":
        ref = label_text.split()
        hyp = pred_text.split()
        sf_smooth = SmoothingFunction().method1

        if ref and hyp:
            results['BLEU-2'] = float(sentence_bleu([ref], hyp, weights=(0.5, 0.5), smoothing_function=sf_smooth) * 100)
            results['BLEU-4'] = float(sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf_smooth) * 100)
        else:
            results['BLEU-2'] = 0.0
            results['BLEU-4'] = 0.0

        if HAS_ROUGE and label_text and pred_text:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(label_text, pred_text)
            results['ROUGE-1'] = float(scores['rouge1'].fmeasure * 100)
            results['ROUGE-2'] = float(scores['rouge2'].fmeasure * 100)
            results['ROUGE-L'] = float(scores['rougeL'].fmeasure * 100)

    elif group_name == "Classification":
        # Binary classification accuracy
        label_lower = label_text.lower().strip()
        pred_lower = pred_text.lower().strip()

        # Normalize common responses
        positive_words = ['yes', 'true', '1', 'positive', 'active']
        negative_words = ['no', 'false', '0', 'negative', 'inactive']

        label_is_pos = any(w in label_lower for w in positive_words)
        label_is_neg = any(w in label_lower for w in negative_words)
        pred_is_pos = any(w in pred_lower for w in positive_words)
        pred_is_neg = any(w in pred_lower for w in negative_words)

        if (label_is_pos and pred_is_pos) or (label_is_neg and pred_is_neg):
            results['Accuracy'] = 1.0
        elif label_is_pos or label_is_neg:
            results['Accuracy'] = 0.0
        else:
            results['Accuracy'] = 1.0 if label_text == pred_text else 0.0

    elif group_name == "Regression":
        label_val = parse_numeric(label_text)
        pred_val = parse_numeric(pred_text)

        if not np.isnan(label_val) and not np.isnan(pred_val):
            error = abs(label_val - pred_val)
            results['MAE'] = float(error)
            results['RMSE'] = float(error ** 2)  # Will be sqrt'd during aggregation

    else:  # Generation, Synthesis, Unknown
        mol_label = get_mol(label_chem)
        mol_pred = get_mol(pred_chem)

        results['Validity'] = 1.0 if mol_pred else 0.0

        canon_label = get_canonical_smiles(mol_label)
        canon_pred = get_canonical_smiles(mol_pred)
        results['Exact Match'] = 1.0 if (canon_label and canon_pred and canon_label == canon_pred) else 0.0

        fp_l = get_fingerprint(mol_label)
        fp_p = get_fingerprint(mol_pred)
        if fp_l and fp_p:
            results['MACCS FTS'] = float(DataStructs.TanimotoSimilarity(fp_l, fp_p))
        else:
            results['MACCS FTS'] = 0.0

        if group_name == "Generation":
            ref_tokens = SMILES_REGEX.findall(label_text)
            hyp_tokens = SMILES_REGEX.findall(pred_text)
            if ref_tokens and hyp_tokens:
                results['BLEU-4'] = float(sentence_bleu([ref_tokens], hyp_tokens,
                                                        weights=(0.25, 0.25, 0.25, 0.25),
                                                        smoothing_function=SmoothingFunction().method1) * 100)
            results['Levenshtein'] = float(levenshtein_dist(label_text, pred_text))

    return results


# =============================================================================
# 5. Plotting Functions
# =============================================================================
def plot_probability_distribution(task_df, task_name, save_dir, cutoff_val, unit):
    """Plot probability distribution of sequence lengths per model."""
    plt.figure(figsize=VIS_CONFIG['fig_size'], dpi=VIS_CONFIG['dpi'])
    sns.set_theme(style="whitegrid", font_scale=VIS_CONFIG['font_scale'])

    models = task_df['model'].unique()

    sns.histplot(
        data=task_df,
        x='seq_len',
        hue='model',
        stat='probability',
        kde=True,
        bins=30,
        palette=VIS_CONFIG['palette'],
        common_norm=False,
        element="step",
        alpha=0.3,
        line_kws={'linewidth': 2}
    )

    plt.axvline(x=cutoff_val, color='red', linestyle='--', linewidth=2, label=f'Cutoff ({cutoff_val})')
    plt.title(f'Length Distribution: {task_name}', fontweight='bold', fontsize=14)
    plt.xlabel(f'Length ({unit})')
    plt.ylabel('Probability')
    plt.legend(title='Model', loc='upper right')

    plt.tight_layout()
    filename = os.path.join(save_dir, f"{task_name.replace('/', '_')}_prob_distribution.png")
    plt.savefig(filename)
    plt.close()
    return filename


def plot_binned_counts(task_df, task_name, save_dir, x_label_text):
    """Plot sample counts per length bin using bar plot."""
    bin_counts = task_df.groupby(['model', 'len_bin']).size().reset_index(name='count')

    plt.figure(figsize=VIS_CONFIG['fig_size'], dpi=VIS_CONFIG['dpi'])
    sns.set_theme(style="whitegrid", font_scale=VIS_CONFIG['font_scale'])

    sns.barplot(
        data=bin_counts,
        x='len_bin',
        y='count',
        hue='model',
        palette=VIS_CONFIG['palette'],
        edgecolor='black',
        alpha=0.8
    )

    plt.title(f'Sample Counts: {task_name}', fontweight='bold', fontsize=14)
    plt.xlabel(x_label_text)
    plt.ylabel('Number of Samples')
    plt.legend(title='Model', loc='upper right', frameon=True)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)

    plt.tight_layout()
    filename = os.path.join(save_dir, f"{task_name.replace('/', '_')}_sample_counts.png")
    plt.savefig(filename)
    plt.close()
    return filename


def plot_metric_by_length(task_df, task_name, metric, save_dir, unit):
    """Plot metric vs length with error bands."""
    plt.figure(figsize=VIS_CONFIG['fig_size'], dpi=VIS_CONFIG['dpi'])
    sns.set_theme(style="white", font_scale=VIS_CONFIG['font_scale'])

    # Filter groups with minimum samples
    valid_groups = task_df.groupby(['model', 'len_bin']).filter(
        lambda x: len(x) >= VIS_CONFIG['min_samples_per_bin']
    )

    if valid_groups.empty or metric not in valid_groups.columns:
        plt.close()
        return None

    if valid_groups[metric].isna().all():
        plt.close()
        return None

    sns.lineplot(
        data=valid_groups,
        x='len_bin',
        y=metric,
        hue='model',
        style='model',
        markers=True,
        dashes=False,
        palette=VIS_CONFIG['palette'],
        linewidth=2.5,
        markersize=8,
        errorbar='sd'
    )

    plt.title(f'{task_name}: {metric}', fontweight='bold', fontsize=14)
    plt.xlabel(f"Length ({unit})")
    plt.ylabel(metric)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Model', loc='best')

    plt.tight_layout()
    filename = os.path.join(save_dir, f"{task_name.replace('/', '_')}_{metric.replace(' ', '_')}.png")
    plt.savefig(filename)
    plt.close()
    return filename


# =============================================================================
# 6. Summary Statistics
# =============================================================================
def compute_summary_statistics(df_results, save_dir):
    """Compute and save summary statistics per task and model."""
    summary_data = []

    for group_name, group_info in TASK_GROUPS.items():
        group_df = df_results[df_results['group'] == group_name]

        for task in group_info['tasks']:
            task_df = group_df[group_df['task'] == task]
            if task_df.empty:
                continue

            for model in task_df['model'].unique():
                model_df = task_df[task_df['model'] == model]

                row = {
                    'Group': group_name,
                    'Task': task,
                    'Model': model,
                    'N_samples': len(model_df),
                    'Avg_length': model_df['seq_len'].mean(),
                }

                for metric in group_info['metrics']:
                    if metric in model_df.columns:
                        values = model_df[metric].dropna()
                        if len(values) > 0:
                            if metric == 'RMSE':
                                row[metric] = np.sqrt(values.mean())
                            else:
                                row[metric] = values.mean()

                summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(save_dir, "summary_statistics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📊 Summary statistics saved to: {summary_path}")

    return summary_df


# =============================================================================
# 7. Main Execution
# =============================================================================
def main():
    processed_dataset = None

    # 1. Try to load from cache
    if os.path.exists(CACHE_DIR):
        try:
            print(f"📂 Found cached dataset at: {CACHE_DIR}")
            processed_dataset = load_from_disk(CACHE_DIR)
            print(f"✅ Loaded {len(processed_dataset)} samples from cache.")
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}. Re-calculating...")
            processed_dataset = None

    # 2. If no cache, load CSVs and process
    if processed_dataset is None:
        print("\n=== Reading CSV Files ===")
        df_list = []

        for model_name, config in tqdm(MODEL_CONFIGS.items(), desc="Reading Models"):
            search_path = os.path.join(config['path'], config['pattern'])
            files = glob.glob(search_path)

            display_name = config['display_name']

            temp_dfs = []
            for f in files:
                try:
                    sub_df = pd.read_csv(f)

                    # Filter to target tasks
                    all_target_tasks = [t for g in TASK_GROUPS.values() for t in g['tasks']]
                    sub_df = sub_df[sub_df['task'].isin(all_target_tasks)].copy()

                    if not sub_df.empty:
                        sub_df['model'] = display_name
                        sub_df = sub_df[['task', 'model', 'label', 'pred']].astype(str)
                        temp_dfs.append(sub_df)
                except Exception as e:
                    print(f"  Error reading {f}: {e}")

            if temp_dfs:
                df_model = pd.concat(temp_dfs, ignore_index=True)
                # Remove duplicates within model
                df_model = df_model.drop_duplicates(subset=['task', 'label', 'pred'])
                df_list.append(df_model)
                print(f"  {model_name}: {len(df_model)} samples")

        if not df_list:
            print("❌ No data loaded. Check file paths.")
            return

        full_df = pd.concat(df_list, ignore_index=True)
        print(f"\n📈 Total samples: {len(full_df)}")
        print(f"   Tasks: {full_df['task'].nunique()}")
        print(f"   Models: {full_df['model'].unique().tolist()}")

        # Convert to HF Dataset
        raw_dataset = Dataset.from_pandas(full_df)

        print(f"\n=== Computing Metrics (Processes: {NUM_PROC}) ===")
        processed_dataset = raw_dataset.map(
            compute_metrics_row,
            num_proc=NUM_PROC,
            desc="Computing Metrics",
            load_from_cache_file=False
        )

        # Save cache
        print(f"\n💾 Saving cache to: {CACHE_DIR}")
        processed_dataset.save_to_disk(CACHE_DIR)

    # 3. Generate visualizations
    print("\n=== Generating Visualizations ===")
    df_results = processed_dataset.to_pandas()

    # Create subdirectories for each group
    for group_name in TASK_GROUPS.keys():
        os.makedirs(os.path.join(SAVE_DIR, group_name), exist_ok=True)

    total_plots = sum(len(g['tasks']) * (len(g['metrics']) + 2) for g in TASK_GROUPS.values())
    pbar = tqdm(total=total_plots, desc="Generating Plots")

    generated_files = []

    for group_name, group_info in TASK_GROUPS.items():
        group_df = df_results[df_results['group'] == group_name]
        unit = group_info['unit']
        group_save_dir = os.path.join(SAVE_DIR, group_name)

        for task in group_info['tasks']:
            task_df = group_df[group_df['task'] == task].copy()

            if task_df.empty:
                pbar.update(len(group_info['metrics']) + 2)
                continue

            # Determine cutoff
            if task in MANUAL_TASK_THRESHOLDS:
                cutoff_val = MANUAL_TASK_THRESHOLDS[task]
            else:
                cutoff_val = int(task_df['seq_len'].quantile(0.97))

            # Plot 1: Probability Distribution
            fname = plot_probability_distribution(task_df, task, group_save_dir, cutoff_val, unit)
            if fname:
                generated_files.append(fname)
            pbar.update(1)

            # Filter data for remaining plots
            task_df_filtered = task_df[task_df['seq_len'] <= cutoff_val].copy()
            task_df_filtered['len_bin'] = (task_df_filtered['seq_len'] // group_info['bin_size']) * group_info['bin_size']

            # Plot 2: Sample Counts
            fname = plot_binned_counts(task_df_filtered, task, group_save_dir, f"Length ({unit})")
            if fname:
                generated_files.append(fname)
            pbar.update(1)

            # Plot 3+: Metrics
            for metric in group_info['metrics']:
                fname = plot_metric_by_length(task_df_filtered, task, metric, group_save_dir, unit)
                if fname:
                    generated_files.append(fname)
                pbar.update(1)

    pbar.close()

    # 4. Compute and save summary statistics
    summary_df = compute_summary_statistics(df_results, SAVE_DIR)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY STATISTICS")
    print("=" * 60)

    for group_name in TASK_GROUPS.keys():
        group_summary = summary_df[summary_df['Group'] == group_name]
        if not group_summary.empty:
            print(f"\n### {group_name} ###")
            metrics = TASK_GROUPS[group_name]['metrics']
            display_cols = ['Task', 'Model', 'N_samples'] + [m for m in metrics if m in group_summary.columns]
            print(group_summary[display_cols].to_string(index=False))

    print("\n" + "=" * 60)
    print(f"✅ Generated {len(generated_files)} visualization files")
    print(f"📁 Results saved to: {SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
