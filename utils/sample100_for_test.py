import os
import pandas as pd
from datasets import load_from_disk

# 1. 설정
dataset_path = "/home/jovyan/CHJ/Mol-LLM_Custom/dataset/train_official/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_test_512_Truncation"
save_path = dataset_path + "_100_sampled"

print(f"Loading dataset from: {dataset_path}")
dataset = load_from_disk(dataset_path)

# 2. 'task' 컬럼만 Pandas DataFrame으로 변환 (메모리 절약 및 속도 향상)
# 전체 데이터를 to_pandas()하면 그래프/텍스트 데이터 변환 때문에 느려질 수 있습니다.
print("Extracting task column to DataFrame...")
df_task = dataset.select_columns(['task']).to_pandas()

# 3. Pandas Groupby를 사용하여 Task별 상위 100개 인덱스 추출
print("Grouping and selecting indices...")
# 각 task 그룹에서 상위 100개의 인덱스(행 번호)를 가져옵니다.
selected_indices = df_task.groupby('task').head(100).index.tolist()
# 인덱스 정렬 (선택 사항, 저장 순서 유지를 위해)
selected_indices.sort()

print(f"Selected {len(selected_indices)} samples out of {len(dataset)}.")

# 4. 추출된 인덱스로 원본 데이터셋 필터링 (select는 복사 비용이 적어 빠름)
final_dataset = dataset.select(selected_indices)

# 5. 저장
print(f"Saving sampled dataset to: {save_path}")
final_dataset.save_to_disk(save_path)

print("Done.")