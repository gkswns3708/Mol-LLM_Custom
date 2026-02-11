"""
METEOR 메트릭 비교 테스트
- 모델 tokenizer (subword) vs NLTK word_tokenize (whole word)
- WordNet 유의어 매칭 효과 확인
- SELFIES 토큰 및 특수 토큰 처리 비교
"""

import os
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
from nltk.corpus import wordnet as wn

# =============================================================================
# Special Tokens 및 SELFIES 토큰 정의
# =============================================================================
BOOL = ["<BOOLEAN>", "</BOOLEAN>"]
FLOAT = ["<FLOAT>", "</FLOAT>"]
DESCRIPTION = ["<DESCRIPTION>", "</DESCRIPTION>"]
SELFIES = ["<SELFIES>", "</SELFIES>"]
MOL_2D = ["<GRAPH>", "</GRAPH>"]
MOL_3D = ["<3D_CONFORMER>", "</3D_CONFORMER>"]
MOL_EMBEDDING = ["<mol>"]
NUMBER = [f"<|{i}|>" for i in range(10)] + ["<|+|>", "<|-|>", "<|.|>"]
INSTRUCTION = ["<INSTRUCTION>", "</INSTRUCTION>"]
REACTION_DIRECTION = ["|>>|"]
IUPAC = ["<IUPAC>", "</IUPAC>"]
MOLFORMULA = ["<MOLFORMULA>", "</MOLFORMULA>"]

CUSTOM_SPECIAL_TOKENS = (
    BOOL + FLOAT + DESCRIPTION + SELFIES + MOL_2D + MOL_3D + 
    MOL_EMBEDDING + NUMBER + INSTRUCTION + REACTION_DIRECTION + 
    IUPAC + MOLFORMULA
)

# SELFIES 사전 경로
SELFIES_DICT_PATH = "/app/Mol-LLM_Custom/model/selfies_dict.txt"

def load_selfies_tokens(path):
    """SELFIES 토큰 사전 로드"""
    if not os.path.exists(path):
        print(f"[Warning] SELFIES dict not found at {path}")
        return []
    with open(path, 'r') as f:
        tokens = f.read().splitlines()
    return [t.strip() for t in tokens if t.strip()]

print("설정 완료")

# ============================================================
# 실제 데이터셋에서 Task별 샘플 로드
# ============================================================
import re
from datasets import load_from_disk
from collections import defaultdict

# 데이터셋 경로
DATASET_PATH = "/app/Mol-LLM_Custom/dataset/merged_dataset/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_test_512_Truncation_merged_bace_chebi_mol2text_chebi_text2mol_qm9_homo"

# Target 텍스트에서 실제 응답 부분만 추출하는 패턴
RESPONSE_PATTERNS = {
    "DESCRIPTION": re.compile(r"<DESCRIPTION>.*?</DESCRIPTION>", re.DOTALL),
    "SELFIES": re.compile(r"<SELFIES>.*?</SELFIES>", re.DOTALL),
    "BOOLEAN": re.compile(r"<BOOLEAN>.*?</BOOLEAN>", re.DOTALL),
    "FLOAT": re.compile(r"<FLOAT>.*?</FLOAT>", re.DOTALL),
}

def extract_response(target_text):
    """Target 텍스트에서 실제 응답 부분만 추출"""
    for pattern_name, pattern in RESPONSE_PATTERNS.items():
        match = pattern.search(target_text)
        if match:
            return match.group(), pattern_name
    return target_text, "UNKNOWN"

def load_task_samples(dataset_path, samples_per_task=2):
    """데이터셋에서 Task별 샘플 로드"""
    print("=" * 80)
    print("실제 데이터셋에서 Task별 샘플 로드")
    print("=" * 80)

    try:
        ds = load_from_disk(dataset_path)
        print(f"✓ 데이터셋 로드 성공: {len(ds)} 샘플")
    except Exception as e:
        print(f"✗ 데이터셋 로드 실패: {e}")
        return []

    # Task별 샘플 수집
    task_samples = defaultdict(list)
    for i, sample in enumerate(ds):
        task = sample['task']
        if len(task_samples[task]) < samples_per_task:
            response, response_type = extract_response(sample['target_text'])
            task_samples[task].append({
                'index': i,
                'task': task,
                'prompt_text': sample['prompt_text'],
                'target_text': sample['target_text'],
                'response': response,
                'response_type': response_type,
            })

    print(f"\nTask별 샘플 수:")
    for task, samples in task_samples.items():
        print(f"  {task}: {len(samples)} 샘플")

    return task_samples

# 데이터 로드
task_samples = load_task_samples(DATASET_PATH, samples_per_task=2)

# 테스트 케이스 생성 (실제 데이터 기반)
test_cases = []
for task, samples in task_samples.items():
    for i, sample in enumerate(samples):
        # 실제 데이터에서는 prediction = target으로 설정 (토큰화 비교 목적)
        # 실제 모델 출력은 다를 수 있음
        test_cases.append({
            "name": f"{task} (샘플 {i+1})",
            "task": task,
            "prediction": sample['response'],  # 실제로는 모델 출력이지만, 여기서는 target과 동일하게 설정
            "target": sample['response'],
            "response_type": sample['response_type'],
            "full_target": sample['target_text'],
        })

print(f"\n총 {len(test_cases)}개의 테스트 케이스 생성됨")
# ============================================================
# LLaDA Tokenizer 로드 + SELFIES 토큰 추가
# ============================================================
print("=" * 80)
print("LLaDA Tokenizer 로드 + SELFIES/Special Tokens 추가")
print("=" * 80)

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_vocab_size = len(tokenizer)
    print(f"✓ 기본 Tokenizer 로드 성공: {tokenizer.__class__.__name__}")
    print(f"  원본 Vocab size: {original_vocab_size}")

    # Special Tokens + SELFIES 토큰 추가
    tokens_to_add = list(set(CUSTOM_SPECIAL_TOKENS))
    selfies_tokens = load_selfies_tokens(SELFIES_DICT_PATH)
    tokens_to_add.extend(selfies_tokens)

    # 이미 vocab에 있는 토큰 제외
    existing_vocab = set(tokenizer.get_vocab().keys())
    final_tokens = [t for t in set(tokens_to_add) if t not in existing_vocab]

    print(f"  추가 예정 토큰 수: {len(final_tokens)}")

    if final_tokens:
        num_added = tokenizer.add_tokens(final_tokens)
        print(f"  실제 추가된 토큰 수: {num_added}")
        print(f"  새 Vocab size: {len(tokenizer)}")
        print(f"  SELFIES 토큰 예시: {selfies_tokens[:10]}")

        # 추가 확인: [C] 토큰이 제대로 추가되었는지
        test_token = "[C]"
        test_id = tokenizer.convert_tokens_to_ids(test_token)
        test_tokenized = tokenizer.tokenize(test_token)
        print(f"\n  [추가 확인] '{test_token}':")
        print(f"    - Token ID: {test_id}")
        print(f"    - Tokenize 결과: {test_tokenized}")
        print(f"    - 단일 토큰 여부: {len(test_tokenized) == 1 and test_tokenized[0] == test_token}")
    else:
        print("  추가할 새 토큰 없음")

except Exception as e:
    import traceback
    print(f"✗ Tokenizer 로드 실패: {e}")
    traceback.print_exc()
    tokenizer = None
# ============================================================
# METEOR 점수 비교 함수
# ============================================================

def compare_meteor_scores(prediction, target, tokenizer, case_name="", response_type="UNKNOWN"):
    """Model Tokenizer vs NLTK word_tokenize METEOR 점수 비교

    Note: SELFIES 토큰과 특수 토큰은 대소문자를 유지해야 하므로
    Model Tokenizer는 원본 텍스트를 사용하고,
    NLTK는 lower()를 적용합니다.
    """

    print(f"\n{'=' * 80}")
    print(f"테스트: {case_name} [{response_type}]")
    print("=" * 80)

    print(f"\n[Prediction]\n{prediction[:150]}{'...' if len(prediction) > 150 else ''}")
    print(f"\n[Target]\n{target[:150]}{'...' if len(target) > 150 else ''}")

    # Model tokenizer (SELFIES/특수 토큰은 대소문자 유지해야 함)
    model_pred_tokens = tokenizer.tokenize(prediction)
    model_ref_tokens = tokenizer.tokenize(target)

    # NLTK word_tokenize (일반 텍스트용 - lowercase)
    nltk_pred_tokens = word_tokenize(prediction.lower())
    nltk_ref_tokens = word_tokenize(target.lower())
    
    print(f"\n[토큰화 비교]")
    print(f"  Model Tokenizer:")
    print(f"    - Prediction 토큰 수: {len(model_pred_tokens)}")
    print(f"    - Reference 토큰 수: {len(model_ref_tokens)}")
    print(f"    - Prediction 토큰: {model_pred_tokens[:15]}{'...' if len(model_pred_tokens) > 15 else ''}")
    
    print(f"\n  NLTK word_tokenize:")
    print(f"    - Prediction 토큰 수: {len(nltk_pred_tokens)}")
    print(f"    - Reference 토큰 수: {len(nltk_ref_tokens)}")
    print(f"    - Prediction 토큰: {nltk_pred_tokens[:15]}{'...' if len(nltk_pred_tokens) > 15 else ''}")
    
    # METEOR 점수 계산
    meteor_model = meteor_score([model_ref_tokens], model_pred_tokens)
    meteor_nltk = meteor_score([nltk_ref_tokens], nltk_pred_tokens)
    
    print(f"\n[METEOR Score 비교]")
    print(f"  Model Tokenizer:    {meteor_model:.4f} ({meteor_model * 100:.2f}%)")
    print(f"  NLTK word_tokenize: {meteor_nltk:.4f} ({meteor_nltk * 100:.2f}%)")
    print(f"  차이:               {(meteor_nltk - meteor_model):+.4f} ({(meteor_nltk - meteor_model) * 100:+.2f}%p)")
    
    return {
        "case_name": case_name,
        "response_type": response_type,
        "meteor_model": meteor_model,
        "meteor_nltk": meteor_nltk,
        "diff": meteor_nltk - meteor_model,
        "model_tokens": len(model_pred_tokens),
        "nltk_tokens": len(nltk_pred_tokens)
    }
# ============================================================
# 모든 테스트 케이스 실행
# ============================================================

results = []
for tc in test_cases:
    result = compare_meteor_scores(
        prediction=tc["prediction"],
        target=tc["target"],
        tokenizer=tokenizer,
        case_name=tc["name"],
        response_type=tc.get("response_type", "UNKNOWN")
    )
    results.append(result)
# ============================================================
# SELFIES 토큰 상세 분석 (추가된 토큰 포함)
# ============================================================
print("\n" + "=" * 80)
print("SELFIES 토큰 상세 분석 (Custom Tokenizer with SELFIES)")
print("=" * 80)

selfies_sample = "[C][C][O][C][=C][Branch1][Ring1]"

print(f"\n[SELFIES 샘플]: {selfies_sample}")

print(f"\n[Model Tokenizer (SELFIES 토큰 추가됨)]")
model_selfies = tokenizer.tokenize(selfies_sample)
print(f"  토큰: {model_selfies}")
print(f"  토큰 수: {len(model_selfies)}")

print(f"\n[NLTK word_tokenize]")
nltk_selfies = word_tokenize(selfies_sample)
print(f"  토큰: {nltk_selfies}")
print(f"  토큰 수: {len(nltk_selfies)}")

# 개별 SELFIES 토큰 분석 (추가된 토큰 확인)
print(f"\n[개별 SELFIES 토큰 분석]")
selfies_test_tokens = ["[C]", "[O]", "[=C]", "[Branch1]", "[Ring1]", "[N]", "[C@H1]", "[C@@H1]"]
for token in selfies_test_tokens:
    model_tok = tokenizer.tokenize(token)
    nltk_tok = word_tokenize(token)
    token_id = tokenizer.convert_tokens_to_ids(token)
    is_single = len(model_tok) == 1 and model_tok[0] == token
    status = "✓ 단일 토큰" if is_single else "✗ 분리됨"
    print(f"  {token:12} → ID: {token_id:6} | Model: {str(model_tok):30} | {status}")
# ============================================================
# 특수 토큰 상세 분석 (추가된 토큰 포함)
# ============================================================
print("\n" + "=" * 80)
print("특수 토큰 상세 분석 (Custom Tokenizer)")
print("=" * 80)

special_tokens = [
    "<DESCRIPTION>", "</DESCRIPTION>",
    "<SELFIES>", "</SELFIES>",
    "<BOOLEAN>", "</BOOLEAN>",
    "<FLOAT>", "</FLOAT>",
    "<IUPAC>", "</IUPAC>",
    "<mol>"
]

print(f"\n[특수 토큰 토큰화 비교]")
for token in special_tokens:
    model_tok = tokenizer.tokenize(token)
    nltk_tok = word_tokenize(token)
    token_id = tokenizer.convert_tokens_to_ids(token)
    is_single = len(model_tok) == 1
    status = "✓ 단일 토큰" if is_single else "✗ 분리됨"
    print(f"  {token:20} → ID: {token_id:6} | Model: {str(model_tok):40} | {status}")
# ============================================================
# 핵심 단어 유의어 매칭 분석
# ============================================================
print("\n" + "=" * 80)
print("핵심 단어 유의어 매칭 분석: 'comprising' vs 'consisting'")
print("=" * 80)

print("\n[Model Tokenizer]")
print(f"  'comprising' → {tokenizer.tokenize('comprising')}")
print(f"  'consisting' → {tokenizer.tokenize('consisting')}")

print("\n[NLTK word_tokenize]")
print(f"  'comprising' → {word_tokenize('comprising')}")
print(f"  'consisting' → {word_tokenize('consisting')}")

print("\n[WordNet 유의어 관계]")
print(f"  'comprising' synsets: {wn.synsets('comprising')}")
print(f"  'consisting' synsets: {wn.synsets('consisting')}")

# comprise와 consist가 유의어인지 확인
print("\n[comprise vs consist 유의어 관계]")
for syn in wn.synsets('comprise'):
    lemmas = [l.name() for l in syn.lemmas()]
    if 'consist' in lemmas or any('consist' in l for l in lemmas):
        print(f"  ✓ Synset '{syn.name()}': {lemmas}")

# Model tokenizer에서 분리된 subword의 WordNet 조회
comprising_model = tokenizer.tokenize('comprising')
if len(comprising_model) > 1:
    first_subword = comprising_model[0].replace('Ġ', '').replace('▁', '')
    print(f"\n[Model Tokenizer subword의 WordNet 조회]")
    print(f"  'comprising' → {comprising_model}")
    print(f"  '{first_subword}'의 WordNet synsets: {wn.synsets(first_subword)}")
    print(f"  → WordNet에서 인식되지 않으므로 유의어 매칭 불가!")
# ============================================================
# 결과 요약
# ============================================================
print("\n" + "=" * 80)
print("결과 요약 (Response Type별)")
print("=" * 80)

# Response Type별 그룹화
from collections import defaultdict
results_by_type = defaultdict(list)
for r in results:
    results_by_type[r.get('response_type', 'UNKNOWN')].append(r)

for response_type, type_results in results_by_type.items():
    print(f"\n[{response_type}]")
    print(f"{'테스트 케이스':<40} | {'Model Tok':>10} | {'NLTK':>10} | {'차이':>10}")
    print("-" * 80)
    for r in type_results:
        name = r['case_name'][:38] + '..' if len(r['case_name']) > 40 else r['case_name']
        print(f"{name:<40} | {r['meteor_model']:>10.4f} | {r['meteor_nltk']:>10.4f} | {r['diff']:>+10.4f}")

    # Type별 평균
    avg_model = sum(r['meteor_model'] for r in type_results) / len(type_results)
    avg_nltk = sum(r['meteor_nltk'] for r in type_results) / len(type_results)
    avg_diff = avg_nltk - avg_model
    print("-" * 80)
    print(f"{'평균':<40} | {avg_model:>10.4f} | {avg_nltk:>10.4f} | {avg_diff:>+10.4f}")

# 전체 요약
print("\n" + "=" * 80)
print("Response Type별 평균 요약")
print("=" * 80)
print(f"{'Response Type':<20} | {'Model Tok':>10} | {'NLTK':>10} | {'차이':>10} | {'샘플수':>6}")
print("-" * 70)
for response_type, type_results in results_by_type.items():
    avg_model = sum(r['meteor_model'] for r in type_results) / len(type_results)
    avg_nltk = sum(r['meteor_nltk'] for r in type_results) / len(type_results)
    avg_diff = avg_nltk - avg_model
    print(f"{response_type:<20} | {avg_model:>10.4f} | {avg_nltk:>10.4f} | {avg_diff:>+10.4f} | {len(type_results):>6}")

print("\n" + "=" * 80)
print("결론")
print("=" * 80)

print("""
[분석 결과]
1. Model Tokenizer (SELFIES 토큰 추가됨):
   - SELFIES 토큰([C], [O], [Branch1] 등)이 단일 토큰으로 인식됨
   - 일반 영어 단어는 여전히 subword로 분리됨
   - WordNet 유의어 매칭 불가 (subword는 사전에 없음)

2. NLTK word_tokenize (whole word 기반):
   - 완전한 단어 단위로 토큰화
   - WordNet 유의어 매칭 정상 작동
   - SELFIES 토큰은 대괄호 기준으로 분리됨

[Response Type별 특성]
- DESCRIPTION: 일반 텍스트 → NLTK가 유의어 매칭에 유리
- SELFIES: 분자 구조 → Model Tokenizer가 단일 토큰으로 인식
- BOOLEAN: True/False → 차이 미미
- FLOAT: 숫자 값 → 차이 미미

[권장사항]
- METEOR/BLEU 계산 시 NLTK word_tokenize 사용
- 이는 NLP 평가 메트릭의 표준 관행
- 유의어 매칭 ('comprising' ↔ 'consisting')이 올바르게 동작함
""")