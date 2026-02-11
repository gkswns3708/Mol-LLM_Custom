#!/usr/bin/env python3
"""
METEOR Score 비교 테스트: Model Tokenizer vs NLTK word_tokenize
실제 모델 출력 데이터 (ft-step20655_semi_ar-outputs.json) 사용

- Model Tokenizer: LLaDA의 subword tokenizer (SELFIES 토큰 추가)
- NLTK word_tokenize: whole word 기반 (WordNet 유의어 매칭 가능)
"""

import os
import json
from collections import defaultdict

# NLTK 설정
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer

# ============================================================
# 설정
# ============================================================
JSON_PATH = "/app/ft-step20655_semi_ar-outputs.json"
SELFIES_DICT_PATH = "/app/Mol-LLM_Custom/model/selfies_dict.txt"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

# 특수 토큰 목록
CUSTOM_SPECIAL_TOKENS = [
    '<|startoftext|>', '<|endoftext|>', '<DESCRIPTION>', '</DESCRIPTION>',
    '<SELFIES>', '</SELFIES>', '<BOOLEAN>', '</BOOLEAN>', '<FLOAT>', '</FLOAT>',
    '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>',
    '<rxn>', '</rxn>', '<MOLBLOCK>', '</MOLBLOCK>', '<SMILES>', '</SMILES>',
]

def load_selfies_tokens(path):
    """SELFIES dict 파일에서 토큰 로드"""
    if not os.path.exists(path):
        print(f"[Warning] SELFIES dict not found at {path}")
        return []
    with open(path, 'r') as f:
        tokens = f.read().splitlines()
    return [t.strip() for t in tokens if t.strip()]

def clean_target(target):
    """Target에서 <|eot_id|> 제거"""
    return target.replace("<|eot_id|>", "").strip()

# ============================================================
# 데이터 로드
# ============================================================
print("=" * 80)
print("METEOR Score 비교: Model Tokenizer vs NLTK word_tokenize")
print("=" * 80)

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

print(f"✓ 데이터 로드: {len(data)}개 샘플")

# Task별 분포 확인
task_counts = defaultdict(int)
for item in data:
    task_counts[item['task']] += 1

print(f"\nTask 분포:")
for task, count in sorted(task_counts.items()):
    print(f"  {task}: {count}")

# ============================================================
# Tokenizer 로드 + SELFIES 토큰 추가
# ============================================================
print("\n" + "=" * 80)
print("LLaDA Tokenizer 로드 + SELFIES/Special Tokens 추가")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

original_vocab_size = len(tokenizer)
print(f"✓ 기본 Tokenizer 로드: {tokenizer.__class__.__name__}")
print(f"  원본 Vocab size: {original_vocab_size}")

# Special Tokens + SELFIES 토큰 추가
tokens_to_add = list(set(CUSTOM_SPECIAL_TOKENS))
selfies_tokens = load_selfies_tokens(SELFIES_DICT_PATH)
tokens_to_add.extend(selfies_tokens)

# 이미 vocab에 있는 토큰 제외
existing_vocab = set(tokenizer.get_vocab().keys())
final_tokens = [t for t in set(tokens_to_add) if t not in existing_vocab]

if final_tokens:
    num_added = tokenizer.add_tokens(final_tokens)
    print(f"  추가된 토큰 수: {num_added}")
    print(f"  새 Vocab size: {len(tokenizer)}")

# ============================================================
# METEOR 계산 함수
# ============================================================
def calculate_meteor_both(prediction, target, tokenizer):
    """Model Tokenizer와 NLTK word_tokenize 두 가지로 METEOR 계산"""

    # Model tokenizer (원본 케이스 유지 - SELFIES 등)
    model_pred_tokens = tokenizer.tokenize(prediction)
    model_ref_tokens = tokenizer.tokenize(target)

    # NLTK word_tokenize (lowercase)
    nltk_pred_tokens = word_tokenize(prediction.lower())
    nltk_ref_tokens = word_tokenize(target.lower())

    # METEOR scores
    try:
        meteor_model = meteor_score([model_ref_tokens], model_pred_tokens)
    except Exception:
        meteor_model = 0.0

    try:
        meteor_nltk = meteor_score([nltk_ref_tokens], nltk_pred_tokens)
    except Exception:
        meteor_nltk = 0.0

    return {
        "meteor_model": meteor_model,
        "meteor_nltk": meteor_nltk,
        "diff": meteor_nltk - meteor_model,
        "model_pred_tokens": len(model_pred_tokens),
        "model_ref_tokens": len(model_ref_tokens),
        "nltk_pred_tokens": len(nltk_pred_tokens),
        "nltk_ref_tokens": len(nltk_ref_tokens),
    }

# ============================================================
# Task별 METEOR 계산
# ============================================================
print("\n" + "=" * 80)
print("Task별 METEOR Score 계산")
print("=" * 80)

# Task별 결과 저장
task_results = defaultdict(list)

for i, item in enumerate(data):
    task = item['task']
    prediction = item['prediction']
    target = clean_target(item['target'])  # <|eot_id|> 제거

    scores = calculate_meteor_both(prediction, target, tokenizer)
    scores['task'] = task
    scores['index'] = i
    scores['prediction'] = prediction
    scores['target'] = target

    task_results[task].append(scores)

    # 진행 상황 표시
    if (i + 1) % 500 == 0:
        print(f"  처리 중... {i + 1}/{len(data)}")

print(f"✓ 모든 샘플 처리 완료")

# ============================================================
# Task별 결과 요약
# ============================================================
print("\n" + "=" * 80)
print("Task별 METEOR Score 비교 요약")
print("=" * 80)

print(f"\n{'Task':<40} | {'샘플수':>6} | {'Model Tok':>10} | {'NLTK':>10} | {'차이':>10}")
print("-" * 90)

all_results = []
for task in sorted(task_results.keys()):
    results = task_results[task]
    avg_model = sum(r['meteor_model'] for r in results) / len(results)
    avg_nltk = sum(r['meteor_nltk'] for r in results) / len(results)
    avg_diff = avg_nltk - avg_model

    task_display = task[:38] + '..' if len(task) > 40 else task
    print(f"{task_display:<40} | {len(results):>6} | {avg_model:>10.4f} | {avg_nltk:>10.4f} | {avg_diff:>+10.4f}")

    all_results.extend(results)

# 전체 평균
print("-" * 90)
total_model = sum(r['meteor_model'] for r in all_results) / len(all_results)
total_nltk = sum(r['meteor_nltk'] for r in all_results) / len(all_results)
total_diff = total_nltk - total_model
print(f"{'전체 평균':<40} | {len(all_results):>6} | {total_model:>10.4f} | {total_nltk:>10.4f} | {total_diff:>+10.4f}")

# ============================================================
# Task별 샘플 예시 (토큰화 비교 포함)
# ============================================================
print("\n" + "=" * 80)
print("Task별 샘플 예시 (토큰화 비교)")
print("=" * 80)

for task in sorted(task_results.keys()):
    results = task_results[task]
    sample = results[0]

    prediction = sample['prediction']
    target = sample['target']

    # 토큰화 결과
    model_pred = tokenizer.tokenize(prediction)
    model_ref = tokenizer.tokenize(target)
    nltk_pred = word_tokenize(prediction.lower())
    nltk_ref = word_tokenize(target.lower())

    print(f"\n{'=' * 80}")
    print(f"[{task}]")
    print("=" * 80)
    print(f"\nPrediction: {prediction[:120]}{'...' if len(prediction) > 120 else ''}")
    print(f"Target:     {target[:120]}{'...' if len(target) > 120 else ''}")

    print(f"\n[토큰화 비교]")
    print(f"  Model Tokenizer:")
    print(f"    - Prediction 토큰 수: {len(model_pred)}")
    print(f"    - Reference 토큰 수: {len(model_ref)}")
    print(f"    - Prediction 토큰: {model_pred[:15]}{'...' if len(model_pred) > 15 else ''}")

    print(f"\n  NLTK word_tokenize:")
    print(f"    - Prediction 토큰 수: {len(nltk_pred)}")
    print(f"    - Reference 토큰 수: {len(nltk_ref)}")
    print(f"    - Prediction 토큰: {nltk_pred[:15]}{'...' if len(nltk_pred) > 15 else ''}")

    print(f"\n[METEOR Score 비교]")
    print(f"  Model Tokenizer:    {sample['meteor_model']:.4f} ({sample['meteor_model'] * 100:.2f}%)")
    print(f"  NLTK word_tokenize: {sample['meteor_nltk']:.4f} ({sample['meteor_nltk'] * 100:.2f}%)")
    print(f"  차이:               {sample['diff']:+.4f} ({sample['diff'] * 100:+.2f}%p)")

# ============================================================
# 결론
# ============================================================
print("\n" + "=" * 80)
print("결론")
print("=" * 80)

print(f"""
[전체 결과 요약]
- 총 샘플 수: {len(all_results)}개
- Task 수: {len(task_results)}개

[METEOR Score 비교]
                      Model Tokenizer    NLTK word_tokenize    차이
  전체 평균:          {total_model:.4f}              {total_nltk:.4f}              {total_diff:+.4f}

[분석]
1. Model Tokenizer (LLaDA):
   - BPE 기반 subword 토큰화 ("comprising" → ["compris", "ing"])
   - SELFIES 토큰은 단일 토큰으로 인식 ("[C]" → ["[C]"])
   - WordNet에서 subword를 인식하지 못해 유의어 매칭 불가

2. NLTK word_tokenize:
   - 완전한 단어 단위 토큰화 ("comprising" → ["comprising"])
   - WordNet 유의어 매칭 정상 작동
   - SELFIES 토큰은 대괄호 기준 분리 ("[C]" → ["[", "C", "]"])

[Task별 특성]
- DESCRIPTION 생성 (mol2text, captioning): NLTK가 유의어 매칭에 유리
- SELFIES 생성 (text2mol, reaction): 토큰화 방식에 따른 영향
- FLOAT 예측 (qm9, property): 숫자 형식이므로 차이 미미

[권장사항]
- METEOR/BLEU 평가 시 NLTK word_tokenize 사용 (현재 help_funcs.py에 적용됨)
- 이는 NLP 평가 메트릭의 표준 관행
""")
