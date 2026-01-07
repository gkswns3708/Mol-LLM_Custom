"""
특정 Task만 추출하여 새로운 HuggingFace Dataset으로 저장하는 스크립트
pandas를 사용하여 빠르게 필터링
"""

import os
from datasets import load_from_disk, Dataset
from collections import Counter
import argparse
import pandas as pd


# 사용 가능한 모든 Task 목록
ALL_TASKS = [
    'smol-forward_synthesis',
    'smol-retrosynthesis',
    'smol-name_conversion-i2s',
    'smol-name_conversion-s2i',
    'reagent_prediction',
    'forward_reaction_prediction',
    'retrosynthesis',
    'qm9_lumo',
    'qm9_homo',
    'qm9_homo_lumo_gap',
    'smol-molecule_captioning',
    'smol-property_prediction-hiv',
    'chebi-20-mol2text',
    'chebi-20-text2mol',
    'smol-molecule_generation',
    'smol-property_prediction-sider',
    'smol-property_prediction-lipo',
    'smol-property_prediction-bbbp',
    'bace',
    'smol-property_prediction-clintox',
    'smol-property_prediction-esol',
]


def get_dataset_paths(base_dir: str) -> dict:
    """base_dir 내의 모든 HF 데이터셋 폴더를 찾음"""
    dataset_paths = {}
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            if os.path.exists(os.path.join(folder_path, 'dataset_info.json')) or \
               os.path.exists(os.path.join(folder_path, 'state.json')):
                dataset_paths[folder] = folder_path
    return dataset_paths


def extract_tasks(
    base_dir: str,
    target_tasks: list[str],
    output_base_dir: str,
):
    """
    특정 Task만 추출하여 새로운 Dataset으로 저장 (pandas 사용)

    Args:
        base_dir: 원본 데이터셋이 있는 기본 디렉토리
        target_tasks: 추출할 Task 이름 리스트
        output_base_dir: 새로운 Dataset을 저장할 기본 디렉토리
    """

    dataset_paths = get_dataset_paths(base_dir)

    # 유효한 Task인지 확인
    invalid_tasks = [t for t in target_tasks if t not in ALL_TASKS]
    if invalid_tasks:
        print(f"경고: 알 수 없는 Task가 포함되어 있습니다: {invalid_tasks}")
        print(f"유효한 Task 목록: {ALL_TASKS}")

    # target_tasks를 set으로 변환 (빠른 조회)
    target_tasks_set = set(target_tasks)

    for split_name, path in dataset_paths.items():
        print(f"\n{'='*50}")
        print(f"Processing {split_name} split...")
        print(f"Path: {path}")

        if not os.path.exists(path):
            print(f"경고: {path}가 존재하지 않습니다. 건너뜁니다.")
            continue

        # 데이터셋 로드
        print("Loading dataset...")
        ds = load_from_disk(path)
        print(f"Original size: {len(ds):,}")

        # pandas DataFrame으로 변환
        print("Converting to pandas DataFrame...")
        df = ds.to_pandas()

        # Task 분포 확인
        print("Original task distribution:")
        task_counts = df['task'].value_counts()
        for task, count in task_counts.items():
            print(f"  {task}: {count:,}")

        # pandas로 필터링 (매우 빠름)
        print(f"\nFiltering for tasks: {target_tasks}")
        mask = df['task'].isin(target_tasks_set)
        filtered_df = df[mask].copy()

        print(f"Filtered size: {len(filtered_df):,}")

        if len(filtered_df) == 0:
            print(f"경고: {split_name}에서 필터링된 데이터가 없습니다.")
            continue

        # 필터링된 데이터셋의 Task 분포
        print("Filtered task distribution:")
        filtered_task_counts = filtered_df['task'].value_counts()
        for task, count in filtered_task_counts.items():
            print(f"  {task}: {count:,}")

        # Dataset으로 변환
        print("Converting back to HuggingFace Dataset...")
        filtered_ds = Dataset.from_pandas(filtered_df, preserve_index=False)

        # 개별 폴더에 저장
        output_path = os.path.join(output_base_dir, f"{split_name}")
        os.makedirs(output_path, exist_ok=True)

        print(f"Saving to {output_path}...")
        filtered_ds.save_to_disk(output_path)
        print(f"Saved {split_name} dataset!")

    print(f"\n{'='*50}")
    print("Done!")
    print(f"Output directory: {output_base_dir}")


def extract_all_tasks_separately(
    base_dir: str,
    output_base_dir: str,
):
    """
    모든 Task를 각각 별도의 폴더로 추출

    Args:
        base_dir: 원본 데이터셋이 있는 기본 디렉토리
        output_base_dir: 새로운 Dataset을 저장할 기본 디렉토리
                        각 task별로 {output_base_dir}/{task_name}/ 폴더가 생성됨
    """

    dataset_paths = get_dataset_paths(base_dir)

    # 먼저 모든 데이터셋을 로드하고 DataFrame으로 변환
    all_dfs = {}
    all_tasks_in_data = set()

    for split_name, path in dataset_paths.items():
        print(f"\n{'='*50}")
        print(f"Loading {split_name} split...")
        print(f"Path: {path}")

        if not os.path.exists(path):
            print(f"경고: {path}가 존재하지 않습니다. 건너뜁니다.")
            continue

        ds = load_from_disk(path)
        print(f"Size: {len(ds):,}")

        print("Converting to pandas DataFrame...")
        df = ds.to_pandas()
        all_dfs[split_name] = df

        # 이 split에 있는 task들 수집
        tasks_in_split = set(df['task'].unique())
        all_tasks_in_data.update(tasks_in_split)

        print(f"Tasks in {split_name}: {len(tasks_in_split)}")

    print(f"\n{'='*50}")
    print(f"Total unique tasks found: {len(all_tasks_in_data)}")
    print("Tasks:", sorted(all_tasks_in_data))

    # 각 task별로 필터링 및 저장
    for task_name in sorted(all_tasks_in_data):
        print(f"\n{'='*50}")
        print(f"Extracting task: {task_name}")

        task_output_dir = os.path.join(output_base_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)

        for split_name, df in all_dfs.items():
            # 해당 task만 필터링
            filtered_df = df[df['task'] == task_name].copy()

            if len(filtered_df) == 0:
                print(f"  {split_name}: 0 samples (skipped)")
                continue

            print(f"  {split_name}: {len(filtered_df):,} samples")

            # Dataset으로 변환
            filtered_ds = Dataset.from_pandas(filtered_df, preserve_index=False)

            # split 이름 결정 (train/val/test 추출)
            if 'train' in split_name.lower():
                output_split_name = 'train'
            elif 'val' in split_name.lower():
                output_split_name = 'val'
            elif 'test' in split_name.lower():
                output_split_name = 'test'
            else:
                output_split_name = split_name

            # 저장 폴더명: GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_{split}_512_Truncation_{task}
            output_folder_name = f"GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_{output_split_name}_512_Truncation_{task_name}"
            output_path = os.path.join(task_output_dir, output_folder_name)
            filtered_ds.save_to_disk(output_path)

        print(f"  Saved to: {task_output_dir}")

    print(f"\n{'='*50}")
    print("Done!")
    print(f"Output directory: {output_base_dir}")
    print(f"Created {len(all_tasks_in_data)} task folders")


def list_tasks(base_dir: str):
    """데이터셋의 모든 Task와 개수를 출력"""
    dataset_paths = get_dataset_paths(base_dir)

    if not dataset_paths:
        print(f"Error: No HuggingFace dataset found in {base_dir}")
        return None

    # 첫 번째 데이터셋 사용
    path = list(dataset_paths.values())[0]

    print(f"Loading dataset from {path}...")
    ds = load_from_disk(path)

    task_counts = Counter(ds['task'])

    print("\nAvailable Tasks:")
    print("-" * 50)
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"  {task}: {count:,}")

    return task_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract specific tasks from HuggingFace Dataset")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/app/Mol-LLM_Custom/dataset",
        help="Base directory containing the datasets"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Tasks to extract. If not specified, all tasks will be extracted to separate folders."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output base directory for the filtered datasets"
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="List all available tasks and exit"
    )

    args = parser.parse_args()

    if args.list_tasks:
        list_tasks(args.base_dir)
    elif args.tasks:
        # 특정 task만 추출
        extract_tasks(
            base_dir=args.base_dir,
            target_tasks=args.tasks,
            output_base_dir=args.output_dir,
        )
    else:
        # 모든 task를 각각 별도 폴더로 추출
        print("No tasks specified. Extracting all tasks to separate folders...")
        extract_all_tasks_separately(
            base_dir=args.base_dir,
            output_base_dir=args.output_dir,
        )
