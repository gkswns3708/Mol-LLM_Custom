"""
chebi-20과 smol 데이터셋 간의 molecule 중복 확인 스크립트

chebi-20-text2mol / chebi-20-mol2text와
smol-molecule_generation / smol-molecule_captioning이
동일한 source에서 온 데이터인지 확인
"""

import os
from datasets import load_from_disk
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# 데이터셋 경로
DATASET_PATH = "/app/Mol-LLM_Custom/dataset/train_official/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_train_512_Truncation"

# 병렬 처리 설정
BATCH_SIZE = 100000
NUM_PROC = 50


def process_batch(batch, indices):
    """배치 단위로 task별 데이터 추출"""
    results = []
    for i, idx in enumerate(indices):
        task = batch['task_subtask_pair'][i] if batch['task_subtask_pair'][i] else batch['task'][i] if 'task' in batch else 'unknown'
        results.append({
            'idx': idx,
            'task': task,
            'mol_string': batch['input_mol_string'][i] if batch['input_mol_string'][i] else '',
            'label': batch['label'][i] if batch['label'][i] else '',
            'target_text': batch['target_text'][i] if batch['target_text'][i] else '',
        })
    return results


def main():
    print("=" * 80)
    print("chebi-20 vs smol 데이터셋 중복 확인")
    print("=" * 80)

    # 데이터셋 로드
    print("\n[1] 데이터셋 로드 중...")
    dataset = load_from_disk(DATASET_PATH)
    print(f"총 샘플 수: {len(dataset)}")

    # chebi/smol task만 필터링하여 처리 (전체 데이터를 변환하지 않음)
    print(f"\n[2] chebi/smol 관련 데이터만 필터링 중... (batch_size={BATCH_SIZE}, num_proc={NUM_PROC})")

    # 먼저 task 목록 확인 (task_subtask_pair 컬럼만 빠르게 추출)
    print("  task 목록 추출 중...")

    def extract_tasks_only(examples):
        return {'task': [t if t else 'unknown' for t in examples['task_subtask_pair']]}

    tasks_only = dataset.map(
        extract_tasks_only,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=dataset.column_names,  # 다른 컬럼 제거하여 속도 향상
        desc="Extracting tasks"
    )

    # unique task 목록
    unique_tasks = set(tasks_only['task'])
    print(f"\n[3] 전체 task 목록 ({len(unique_tasks)}개):")

    # task별 count
    from collections import Counter
    task_counts = Counter(tasks_only['task'])
    for task, count in sorted(task_counts.items()):
        print(f"  - {task}: {count}개")

    # 관심 있는 task 필터링
    # chebi-20: text2mol (text->molecule), mol2text (molecule->caption)
    # smol: molecule_captioning (mol2text), molecule_generation (text2mol)
    chebi_tasks = [t for t in unique_tasks if 'chebi' in t.lower() and ('mol2text' in t.lower() or 'text2mol' in t.lower())]
    smol_tasks = [t for t in unique_tasks if 'smol' in t.lower() and ('captioning' in t.lower() or 'generation' in t.lower())]

    print(f"\n[4] 관련 task 식별:")
    print(f"  chebi 관련 (mol2text/text2mol): {chebi_tasks}")
    print(f"  smol 관련 (captioning/generation): {smol_tasks}")

    if not chebi_tasks or not smol_tasks:
        print("\n[!] chebi 또는 smol task를 찾을 수 없습니다.")
        print("  전체 task 목록에서 관련 task를 확인하세요.")
        return

    # chebi/smol 데이터만 필터링
    print("\n[5] chebi/smol 데이터만 필터링 중...")
    target_tasks = set(chebi_tasks + smol_tasks)

    def filter_target_tasks(examples, indices):
        # 타겟 task인 샘플만 유지
        mask = [t in target_tasks for t in examples['task_subtask_pair']]
        return {
            'idx': [idx for idx, m in zip(indices, mask) if m],
            'task': [t if t else 'unknown' for t, m in zip(examples['task_subtask_pair'], mask) if m],
            'mol_string': [s if s else '' for s, m in zip(examples['input_mol_string'], mask) if m],
            'label': [l if l else '' for l, m in zip(examples['label'], mask) if m],
            'target_text': [t if t else '' for t, m in zip(examples['target_text'], mask) if m],
        }

    filtered = dataset.map(
        filter_target_tasks,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc="Filtering chebi/smol"
    )

    print(f"  필터링된 샘플 수: {len(filtered)}")

    # 각 task에서 molecule 추출 (pandas로 벡터화 처리)
    print("\n[6] 각 task에서 molecule 추출 (pandas 변환)...")

    # 필터링된 데이터는 양이 적으므로 to_pandas() 빠름
    print("  DataFrame 변환 중...")
    df_filtered = filtered.to_pandas()

    # chebi/smol 분리
    chebi_tasks_set = set(chebi_tasks)
    smol_tasks_set = set(smol_tasks)

    df_chebi = df_filtered[df_filtered['task'].isin(chebi_tasks_set)].copy()
    df_smol = df_filtered[df_filtered['task'].isin(smol_tasks_set)].copy()

    # 빈 mol_string 제거
    df_chebi = df_chebi[df_chebi['mol_string'] != '']
    df_smol = df_smol[df_smol['mol_string'] != '']

    print(f"  chebi 샘플 수: {len(df_chebi)}")
    print(f"  smol 샘플 수: {len(df_smol)}")

    # mol_string 기준 groupby로 딕셔너리 생성
    print("  molecule별 그룹화 중...")

    chebi_molecules = {}
    for mol, group in tqdm(df_chebi.groupby('mol_string'), desc="Grouping chebi"):
        chebi_molecules[mol] = group[['task', 'idx', 'label', 'target_text']].rename(
            columns={'target_text': 'target'}
        ).to_dict('records')

    smol_molecules = {}
    for mol, group in tqdm(df_smol.groupby('mol_string'), desc="Grouping smol"):
        smol_molecules[mol] = group[['task', 'idx', 'label', 'target_text']].rename(
            columns={'target_text': 'target'}
        ).to_dict('records')

    print(f"\n  chebi unique molecules: {len(chebi_molecules)}")
    print(f"  smol unique molecules: {len(smol_molecules)}")

    # 중복 확인
    print("\n[6] 중복 molecule 확인...")
    overlap_molecules = set(chebi_molecules.keys()) & set(smol_molecules.keys())

    print(f"\n  중복 molecule 수: {len(overlap_molecules)}")
    print(f"  chebi 대비 중복 비율: {len(overlap_molecules) / len(chebi_molecules) * 100:.2f}%")
    print(f"  smol 대비 중복 비율: {len(overlap_molecules) / len(smol_molecules) * 100:.2f}%")

    # 중복 샘플 상세 분석
    if overlap_molecules:
        print("\n[7] 중복 샘플 상세 분석 (처음 5개):")
        print("-" * 80)

        for i, mol in enumerate(list(overlap_molecules)[:5]):
            print(f"\n=== 중복 Molecule #{i+1} ===")
            print(f"SMILES: {mol[:100]}..." if len(mol) > 100 else f"SMILES: {mol}")

            print("\n  [chebi 데이터]:")
            for info in chebi_molecules[mol][:2]:
                print(f"    - task: {info['task']}")
                label_preview = info['label'][:200] if info['label'] else info['target'][:200]
                print(f"    - label/target: {label_preview}...")

            print("\n  [smol 데이터]:")
            for info in smol_molecules[mol][:2]:
                print(f"    - task: {info['task']}")
                label_preview = info['label'][:200] if info['label'] else info['target'][:200]
                print(f"    - label/target: {label_preview}...")

            print("-" * 80)

    # label/caption 비교 (같은 molecule에 대해 다른 caption이 있는지)
    print("\n[8] 동일 molecule에 대한 caption 비교:")

    different_captions = 0
    same_captions = 0

    caption_comparison = []

    for mol in overlap_molecules:
        chebi_captions = set()
        smol_captions = set()

        for info in chebi_molecules[mol]:
            caption = info['label'] or info['target']
            if caption:
                chebi_captions.add(caption.strip())

        for info in smol_molecules[mol]:
            caption = info['label'] or info['target']
            if caption:
                smol_captions.add(caption.strip())

        if chebi_captions & smol_captions:
            same_captions += 1
        else:
            different_captions += 1
            if len(caption_comparison) < 3:
                caption_comparison.append({
                    'mol': mol,
                    'chebi_caption': list(chebi_captions)[:1],
                    'smol_caption': list(smol_captions)[:1]
                })

    print(f"\n  동일 caption: {same_captions}개")
    print(f"  다른 caption: {different_captions}개")

    if caption_comparison:
        print("\n  [다른 caption 예시]:")
        for comp in caption_comparison:
            print(f"\n  Molecule: {comp['mol'][:80]}...")
            print(f"  chebi caption: {comp['chebi_caption'][0][:150] if comp['chebi_caption'] else 'N/A'}...")
            print(f"  smol caption: {comp['smol_caption'][0][:150] if comp['smol_caption'] else 'N/A'}...")

    # 결과 요약
    print("\n" + "=" * 80)
    print("결과 요약")
    print("=" * 80)
    print(f"""
1. chebi-20 관련 task: {chebi_tasks}
   - unique molecules: {len(chebi_molecules)}개

2. smol 관련 task: {smol_tasks}
   - unique molecules: {len(smol_molecules)}개

3. 중복 분석:
   - 공통 molecule 수: {len(overlap_molecules)}개
   - chebi 대비 중복률: {len(overlap_molecules) / len(chebi_molecules) * 100:.2f}%
   - smol 대비 중복률: {len(overlap_molecules) / len(smol_molecules) * 100:.2f}%

4. Caption 비교:
   - 동일 caption: {same_captions}개
   - 다른 caption: {different_captions}개

결론: {"동일한 source에서 온 데이터일 가능성이 높음 (중복률 높음)" if len(overlap_molecules) > min(len(chebi_molecules), len(smol_molecules)) * 0.5 else "별도 source에서 온 데이터일 가능성이 있음"}
""")

    # 중복 데이터 상세 정보 CSV로 저장
    print("\n[9] 중복 데이터 상세 정보 저장 중...")

    overlap_details = []
    for mol in tqdm(overlap_molecules, desc="Preparing CSV"):
        # chebi descriptions 수집
        chebi_descriptions = []
        chebi_tasks_list = []
        for info in chebi_molecules[mol]:
            desc = info['label'] if info['label'] else info['target']
            if desc:
                chebi_descriptions.append(desc)
                chebi_tasks_list.append(info['task'])

        # smol descriptions 수집
        smol_descriptions = []
        smol_tasks_list = []
        for info in smol_molecules[mol]:
            desc = info['label'] if info['label'] else info['target']
            if desc:
                smol_descriptions.append(desc)
                smol_tasks_list.append(info['task'])

        # 중복 제거한 unique descriptions
        chebi_unique_descs = list(set(chebi_descriptions))
        smol_unique_descs = list(set(smol_descriptions))

        # description이 동일한지 체크
        is_same_desc = bool(set(chebi_descriptions) & set(smol_descriptions))

        overlap_details.append({
            'molecule': mol,
            'chebi_tasks': ', '.join(set(chebi_tasks_list)),
            'chebi_description_count': len(chebi_unique_descs),
            'chebi_descriptions': ' ||| '.join(chebi_unique_descs),  # 구분자로 ||| 사용
            'smol_tasks': ', '.join(set(smol_tasks_list)),
            'smol_description_count': len(smol_unique_descs),
            'smol_descriptions': ' ||| '.join(smol_unique_descs),
            'is_same_description': is_same_desc,
        })

    if overlap_details:
        df = pd.DataFrame(overlap_details)
        output_path = "/app/Mol-LLM_Custom/utils/chebi_smol_overlap_analysis.csv"
        df.to_csv(output_path, index=False)
        print(f"  저장 완료: {output_path}")
        print(f"  총 {len(df)}개의 중복 쌍")


if __name__ == "__main__":
    main()
