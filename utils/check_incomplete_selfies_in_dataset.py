"""
원본 데이터셋에서 incomplete SELFIES label이 있는지 확인하는 스크립트.

체크 대상 태스크:
- chebi-20-text2mol
- smol-molecule_generation
- smol-retrosynthesis
- reagent_prediction
- forward_reaction_prediction
- retrosynthesis
- smol-forward_synthesis
등 SELFIES/SMILES 형태의 label을 가지는 태스크들
"""

import pandas as pd
from datasets import load_from_disk
from collections import defaultdict

# Configuration
DATASET_PATH = '/home/jovyan/CHJ/Mol-LLM_Custom/dataset/train_official/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_test_512_Truncation'
OUTPUT_PATH = '/home/jovyan/CHJ/Mol-LLM_Custom/utils/dataset_incomplete_selfies_check.csv'

# 체크 대상 태스크들 (SELFIES/SMILES label을 가지는 태스크)
TARGET_TASKS = [
    'chebi-20-text2mol',
    'smol-molecule_generation',
    'smol-retrosynthesis',
    'reagent_prediction',
    'forward_reaction_prediction',
    'retrosynthesis',
    'smol-forward_synthesis',
]

def check_incomplete_label(label_text, target_text):
    """
    label 또는 target_text에서 incomplete한 패턴을 체크

    체크 항목:
    1. <SELFIES>가 있는데 </SELFIES>가 없는 경우
    2. <SMILES>가 있는데 </SMILES>가 없는 경우
    3. <eot_id> 또는 <|eot_id|>가 없는 경우
    """
    issues = []

    # Check label column
    label = str(label_text) if label_text else ""
    target = str(target_text) if target_text else ""

    # SELFIES 체크
    if '<SELFIES>' in label and '</SELFIES>' not in label:
        issues.append('label: missing </SELFIES>')
    if '<SELFIES>' in target and '</SELFIES>' not in target:
        issues.append('target_text: missing </SELFIES>')

    # SMILES 체크
    if '<SMILES>' in label and '</SMILES>' not in label:
        issues.append('label: missing </SMILES>')
    if '<SMILES>' in target and '</SMILES>' not in target:
        issues.append('target_text: missing </SMILES>')

    # eot_id 체크 (target_text에서만 체크)
    if target and '<eot_id>' not in target and '<|eot_id|>' not in target:
        issues.append('target_text: missing eot_id')

    return issues


def main():
    print("=" * 80)
    print("Checking for Incomplete SELFIES/SMILES in Original Dataset")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading dataset from: {DATASET_PATH}")
    ds = load_from_disk(DATASET_PATH)
    print(f"Total samples: {len(ds)}")

    # Filter target tasks
    print(f"\nTarget tasks: {TARGET_TASKS}")

    # Collect problems
    all_problems = []
    task_stats = defaultdict(lambda: {'total': 0, 'incomplete': 0})

    for idx, example in enumerate(ds):
        task = example['task']

        if task not in TARGET_TASKS:
            continue

        task_stats[task]['total'] += 1

        label = example.get('label', '')
        target_text = example.get('target_text', '')

        issues = check_incomplete_label(label, target_text)

        if issues:
            task_stats[task]['incomplete'] += 1
            all_problems.append({
                'dataset_idx': idx,
                'task': task,
                'issues': '; '.join(issues),
                'label': label,
                'target_text': target_text
            })

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY BY TASK")
    print("=" * 80)

    for task in TARGET_TASKS:
        stats = task_stats[task]
        if stats['total'] > 0:
            pct = (stats['incomplete'] / stats['total']) * 100
            status = "❌" if stats['incomplete'] > 0 else "✅"
            print(f"{status} {task}: {stats['incomplete']}/{stats['total']} incomplete ({pct:.2f}%)")

    # Total
    total_samples = sum(s['total'] for s in task_stats.values())
    total_incomplete = sum(s['incomplete'] for s in task_stats.values())

    print("\n" + "-" * 40)
    print(f"TOTAL: {total_incomplete}/{total_samples} incomplete samples")

    # Save to CSV if there are problems
    if all_problems:
        result_df = pd.DataFrame(all_problems)
        result_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n💾 Saved {len(result_df)} problematic samples to:")
        print(f"   {OUTPUT_PATH}")

        # Show sample issues
        print("\n" + "=" * 80)
        print("SAMPLE ISSUES (first 5)")
        print("=" * 80)
        for i, prob in enumerate(all_problems[:5]):
            print(f"\n[{i+1}] IDX: {prob['dataset_idx']}, Task: {prob['task']}")
            print(f"    Issues: {prob['issues']}")
            print(f"    Label (first 200 chars): {prob['label'][:200]}...")
    else:
        print("\n✅ No incomplete samples found in the original dataset!")
        print("   The issue likely originates from the inference/generation process.")


if __name__ == "__main__":
    main()
