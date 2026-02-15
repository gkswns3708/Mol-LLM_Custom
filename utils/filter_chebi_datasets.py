from datasets import load_from_disk
import os

# Validation set 샘플링 비율
VAL_SAMPLE_RATIO = 0.1
SEED = 42

# 경로 설정
BASE_DIR = "/home/jovyan/CHJ/Mol-LLM_Custom/dataset/train_official"
OUTPUT_DIR = "/home/jovyan/CHJ/Mol-LLM_Custom/dataset/train_official"

# 필터링할 task 목록
TARGET_TASKS = ["chebi-20-mol2text", "chebi-20-text2mol"]

# split 종류
SPLITS = ["train", "test", "val"]

# 파일명 패턴
PREFIX = "GSAI-ML-LLaDA-8B-Instruct_string+graph_q32"
SUFFIX = "512_Truncation"

# 출력 파일명 suffix
OUTPUT_SUFFIX = "chebi_mol2text_chebi_text2mol"


def filter_dataset_for_split(split: str):
    """특정 split에 대해 chebi task만 필터링"""

    # 입력 경로
    input_name = f"{PREFIX}_{split}_{SUFFIX}"
    input_path = os.path.join(BASE_DIR, input_name)

    if not os.path.exists(input_path):
        print(f"  [에러] 경로 없음: {input_path}")
        return None

    print(f"  로딩: {input_path}")
    ds = load_from_disk(input_path)
    print(f"    - 전체 샘플 수: {len(ds)}")

    # Dataset.filter()를 사용하여 원본 features 스키마 유지 (batched + multiprocessing)
    print(f"  필터링 중... (tasks: {TARGET_TASKS})")
    target_set = set(TARGET_TASKS)
    filtered_ds = ds.filter(
        lambda batch: [t in target_set for t in batch['task']],
        batched=True,
        batch_size=10000,
        num_proc=50
    )
    print(f"    - 필터링 후 샘플 수: {len(filtered_ds)}")

    if len(filtered_ds) == 0:
        print(f"  [경고] 필터링 결과가 비어있습니다.")
        return None

    # task별 개수 출력
    print(f"    - Task별 개수:")
    task_counts = {}
    for task in filtered_ds['task']:
        task_counts[task] = task_counts.get(task, 0) + 1
    for task in TARGET_TASKS:
        print(f"      {task}: {task_counts.get(task, 0)}")

    return filtered_ds


def main():
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Chebi 데이터셋 필터링 시작")
    print(f"대상 Task: {TARGET_TASKS}")
    print("=" * 60)

    for split in SPLITS:
        print(f"\n[{split.upper()}] 필터링 중...")

        filtered_dataset = filter_dataset_for_split(split)

        if filtered_dataset is not None:
            # Validation set은 10%만 랜덤 샘플링
            if split == "val":
                original_size = len(filtered_dataset)
                sample_size = int(original_size * VAL_SAMPLE_RATIO)
                print(f"  Validation set 샘플링: {original_size} -> {sample_size} ({VAL_SAMPLE_RATIO*100:.0f}%)")
                filtered_dataset = filtered_dataset.shuffle(seed=SEED).select(range(sample_size))

                # 샘플링 후 task별 개수 재출력
                print(f"    - 샘플링 후 Task별 개수:")
                task_counts = {}
                for task in filtered_dataset['task']:
                    task_counts[task] = task_counts.get(task, 0) + 1
                for task in TARGET_TASKS:
                    print(f"      {task}: {task_counts.get(task, 0)}")

            # 출력 경로 설정
            output_name = f"{PREFIX}_{split}_{SUFFIX}_{OUTPUT_SUFFIX}"
            output_path = os.path.join(OUTPUT_DIR, output_name)

            # 저장
            print(f"  저장 중: {output_path}")
            filtered_dataset.save_to_disk(output_path)
            print(f"  저장 완료!")

    print("\n" + "=" * 60)
    print("모든 필터링 완료!")
    print("=" * 60)

    # 결과 확인
    print("\n[결과 확인]")
    for split in SPLITS:
        output_name = f"{PREFIX}_{split}_{SUFFIX}_{OUTPUT_SUFFIX}"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        if os.path.exists(output_path):
            ds = load_from_disk(output_path)
            print(f"  {split}: {len(ds)} 샘플")


if __name__ == "__main__":
    main()
