from datasets import load_from_disk, concatenate_datasets
import os

# 경로 설정
BASE_DIR = "/app/Mol-LLM_Custom/dataset/filtered_dataset"
OUTPUT_DIR = "/app/Mol-LLM_Custom/dataset/merged_dataset"

# 데이터셋 이름과 경로
DATASETS = {
    "bace": f"{BASE_DIR}/bace",
    "chebi-20-mol2text": f"{BASE_DIR}/chebi-20-mol2text",
    "chebi-20-text2mol": f"{BASE_DIR}/chebi-20-text2mol",
    "qm9_homo": f"{BASE_DIR}/qm9_homo",
}

# split 종류
SPLITS = ["train", "test", "val"]

# 파일명 패턴
PREFIX = "GSAI-ML-LLaDA-8B-Instruct_string+graph_q32"
SUFFIX = "512_Truncation"

# 출력 파일명 suffix
OUTPUT_SUFFIX = "merged_bace_chebi_mol2text_chebi_text2mol_qm9_homo"


def merge_datasets_for_split(split: str):
    """특정 split에 대해 4개 데이터셋을 병합"""
    datasets_to_merge = []

    for dataset_name, dataset_path in DATASETS.items():
        # 각 데이터셋의 split별 경로 구성
        # 예: bace -> GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_train_512_Truncation_bace
        split_dir = f"{PREFIX}_{split}_{SUFFIX}_{dataset_name}"
        full_path = os.path.join(dataset_path, split_dir)

        if os.path.exists(full_path):
            print(f"  로딩: {dataset_name} ({split})")
            ds = load_from_disk(full_path)
            print(f"    - 샘플 수: {len(ds)}")
            datasets_to_merge.append(ds)
        else:
            print(f"  [경고] 경로 없음: {full_path}")

    if not datasets_to_merge:
        print(f"  [에러] {split}에 대해 병합할 데이터셋이 없습니다.")
        return None

    # 데이터셋 병합
    merged = concatenate_datasets(datasets_to_merge)
    print(f"  병합 완료: 총 {len(merged)} 샘플")

    return merged


def main():
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("데이터셋 병합 시작")
    print("=" * 60)

    for split in SPLITS:
        print(f"\n[{split.upper()}] 병합 중...")

        merged_dataset = merge_datasets_for_split(split)

        if merged_dataset is not None:
            # 출력 경로 설정
            # 예: GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_train_512_Truncation_merged_bace_chebi_mol2text_chebi_text2mol_qm9_homo
            output_name = f"{PREFIX}_{split}_{SUFFIX}_{OUTPUT_SUFFIX}"
            output_path = os.path.join(OUTPUT_DIR, output_name)

            # 저장
            print(f"  저장 중: {output_path}")
            merged_dataset.save_to_disk(output_path)
            print(f"  저장 완료!")

    print("\n" + "=" * 60)
    print("모든 병합 완료!")
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
