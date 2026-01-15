#!/usr/bin/env python3
"""
LLaDA-Instruct 데이터 포맷 검증 스크립트

이 스크립트는 현재 데이터셋이 LLaMA 3 / LLaDA Instruct 포맷을 올바르게 따르고 있는지 확인합니다.

검증 항목:
1. 특수 토큰 존재 여부 및 위치
   - <|startoftext|> (126080): 시퀀스 시작
   - <|start_header_id|> (126346): 역할 헤더 시작
   - <|end_header_id|> (126347): 역할 헤더 종료
   - <|eot_id|> (126348): 턴 종료 토큰 (EOT)
   - <|endoftext|> (126081): 전체 종료 토큰 (EOS)

2. 프롬프트 구조 검증
   - system/user/assistant 역할 구분
   - 올바른 템플릿 형식

3. Target(Output) 검증
   - <|eot_id|> 토큰이 target_text 끝에 포함되어 있는지
   - Loss 계산 시 EOT 토큰이 포함되는지 확인
"""

import os
import re
import json
import pandas as pd
import pyarrow.parquet as pq
from collections import Counter
from datasets import load_from_disk
from transformers import AutoTokenizer

# LLaDA 특수 토큰 정의
SPECIAL_TOKENS = {
    'startoftext': {'id': 126080, 'text': '<|startoftext|>'},
    'endoftext': {'id': 126081, 'text': '<|endoftext|>'},
    'start_header_id': {'id': 126346, 'text': '<|start_header_id|>'},
    'end_header_id': {'id': 126347, 'text': '<|end_header_id|>'},
    'eot_id': {'id': 126348, 'text': '<|eot_id|>'},
}


def print_section(title):
    """섹션 제목 출력"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def load_dataset_as_dataframe(dataset_path):
    """데이터셋을 pandas DataFrame으로 로드"""
    arrow_file = os.path.join(dataset_path, 'data-00000-of-00001.arrow')

    if os.path.exists(arrow_file):
        import pyarrow as pa
        try:
            # Arrow IPC Stream format (HuggingFace datasets default)
            with pa.memory_map(arrow_file, 'r') as source:
                reader = pa.ipc.open_stream(source)
                table = reader.read_all()
            df = table.to_pandas()
            return df
        except Exception:
            pass

    # fallback: HuggingFace datasets 사용
    ds = load_from_disk(dataset_path)
    return ds.to_pandas()


def check_special_tokens_in_text(text, sample_type="unknown"):
    """텍스트에서 특수 토큰 존재 여부 확인"""
    results = {}
    for name, info in SPECIAL_TOKENS.items():
        count = text.count(info['text'])
        results[name] = {
            'present': count > 0,
            'count': count,
        }
    return results


def analyze_prompt_structure(prompt_text):
    """프롬프트 구조 분석"""
    roles_found = []

    # 각 역할의 메시지 추출
    pattern = r'<\|start_header_id\|>(\w+)<\|end_header_id\|>\n\n(.*?)(?=<\|eot_id\|>|$)'
    matches = re.findall(pattern, prompt_text, re.DOTALL)

    for role, content in matches:
        roles_found.append({
            'role': role,
            'content_preview': content[:100] + '...' if len(content) > 100 else content,
            'content_length': len(content)
        })

    return roles_found


def check_dataset(dataset_path, num_samples=5):
    """데이터셋 검증 (pandas DataFrame 사용)"""
    print_section(f"데이터셋 검증: {os.path.basename(dataset_path)}")

    try:
        df = load_dataset_as_dataframe(dataset_path)
        print(f"✓ 데이터셋 로드 성공: {len(df)} 샘플")
        print(f"  컬럼: {list(df.columns)}")
    except Exception as e:
        print(f"✗ 데이터셋 로드 실패: {e}")
        return

    eot_token = SPECIAL_TOKENS['eot_id']['text']

    # 샘플별 상세 분석
    print_section("샘플별 상세 분석")

    for i in range(min(num_samples, len(df))):
        row = df.iloc[i]
        print(f"\n--- 샘플 {i} ---")

        # prompt_text 분석
        if 'prompt_text' in df.columns:
            prompt = row['prompt_text']
            print(f"\n[Prompt 구조 분석]")

            # 특수 토큰 확인
            token_check = check_special_tokens_in_text(prompt, "prompt")
            for name, info in token_check.items():
                status = "✓" if info['present'] else "✗"
                print(f"  {status} {SPECIAL_TOKENS[name]['text']}: {info['count']}개")

            # 역할 구조 분석
            roles = analyze_prompt_structure(prompt)
            print(f"\n  역할 구조:")
            for role_info in roles:
                print(f"    - {role_info['role']}: {role_info['content_length']} chars")

        # target_text 분석
        if 'target_text' in df.columns:
            target = row['target_text']
            print(f"\n[Target 분석]")
            print(f"  원본: {repr(target)}")

            ends_with_eot = target.strip().endswith(eot_token)
            eot_count = target.count(eot_token)

            if ends_with_eot:
                print(f"  ✓ <|eot_id|>로 끝남")
            else:
                print(f"  ✗ <|eot_id|>로 끝나지 않음 - 문제 가능!")

            print(f"  <|eot_id|> 개수: {eot_count}")

    # 전체 통계 (벡터화된 연산 사용)
    print_section("전체 데이터셋 통계")

    # pandas 벡터화 연산으로 빠르게 계산
    df['target_ends_with_eot'] = df['target_text'].str.strip().str.endswith(eot_token)
    df['target_eot_count'] = df['target_text'].str.count(re.escape(eot_token))

    total_with_eot = df['target_ends_with_eot'].sum()
    total_without_eot = (~df['target_ends_with_eot']).sum()

    print(f"Target에 <|eot_id|> 포함된 샘플: {total_with_eot} ({100*total_with_eot/len(df):.1f}%)")
    print(f"Target에 <|eot_id|> 누락된 샘플: {total_without_eot} ({100*total_without_eot/len(df):.1f}%)")

    if total_without_eot > 0:
        print("\n⚠️  경고: <|eot_id|>가 누락된 샘플이 있습니다!")
        print("   모델이 생성을 멈추지 않는 문제가 발생할 수 있습니다.")

        # 누락된 샘플 예시 출력
        missing_samples = df[~df['target_ends_with_eot']].head(3)
        print("\n  누락된 샘플 예시:")
        for idx, row in missing_samples.iterrows():
            print(f"    - [{idx}] {repr(row['target_text'][:50])}")

    # Prompt 특수 토큰 통계
    print_section("Prompt 특수 토큰 통계")

    for name, info in SPECIAL_TOKENS.items():
        token = info['text']
        has_token = df['prompt_text'].str.contains(re.escape(token), regex=True)
        count = has_token.sum()
        print(f"  {token}: {count}/{len(df)} ({100*count/len(df):.1f}%)")

    # <|eot_id|> 개수 분포 (Prompt 내)
    df['prompt_eot_count'] = df['prompt_text'].str.count(re.escape(eot_token))
    eot_distribution = df['prompt_eot_count'].value_counts().sort_index()
    print(f"\n  Prompt 내 <|eot_id|> 개수 분포:")
    for count, num in eot_distribution.items():
        print(f"    {count}개: {num} 샘플")

    return {
        'total_samples': len(df),
        'with_eot': int(total_with_eot),
        'without_eot': int(total_without_eot),
        'dataframe': df
    }


def analyze_tokenization_details(dataset_path, tokenizer_path="GSAI-ML/LLaDA-8B-Instruct", num_samples=3):
    """토큰화 상세 분석"""
    print_section("토큰화 상세 분석")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"✓ 토크나이저 로드 성공: {tokenizer_path}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    except Exception as e:
        print(f"✗ 토크나이저 로드 실패: {e}")
        print("  로컬 분석만 진행합니다.")
        tokenizer = None

    df = load_dataset_as_dataframe(dataset_path)

    for i in range(min(num_samples, len(df))):
        row = df.iloc[i]
        print(f"\n--- 샘플 {i} 토큰화 분석 ---")

        if tokenizer:
            # prompt_text 토큰화
            prompt_tokens = tokenizer.encode(row['prompt_text'], add_special_tokens=False)
            target_tokens = tokenizer.encode(row['target_text'], add_special_tokens=False)

            print(f"\n[Prompt 토큰화]")
            print(f"  토큰 수: {len(prompt_tokens)}")
            print(f"  처음 10 토큰: {prompt_tokens[:10]}")
            print(f"  마지막 10 토큰: {prompt_tokens[-10:]}")

            # EOT 토큰 위치 확인
            eot_positions = [j for j, t in enumerate(prompt_tokens) if t == SPECIAL_TOKENS['eot_id']['id']]
            print(f"  <|eot_id|> (126348) 위치: {eot_positions}")

            print(f"\n[Target 토큰화]")
            print(f"  토큰 수: {len(target_tokens)}")
            print(f"  전체 토큰: {target_tokens}")

            # Target의 마지막 토큰 확인
            if target_tokens:
                last_token = target_tokens[-1]
                is_eot = last_token == SPECIAL_TOKENS['eot_id']['id']
                print(f"  마지막 토큰: {last_token} ({'<|eot_id|>' if is_eot else '다른 토큰'})")
                if is_eot:
                    print(f"  ✓ Target이 <|eot_id|> (126348)로 끝남 - 정상")
                else:
                    print(f"  ✗ Target이 <|eot_id|>로 끝나지 않음 - Loss 계산 시 문제 발생 가능")

            # 디코딩 확인
            print(f"\n[디코딩 확인]")
            decoded_target = tokenizer.decode(target_tokens)
            print(f"  디코딩된 Target: {repr(decoded_target)}")


def check_full_sequence_structure(dataset_path, num_samples=3):
    """전체 시퀀스(prompt + target) 구조 상세 분석"""
    print_section("전체 시퀀스 구조 분석")

    df = load_dataset_as_dataframe(dataset_path)

    eot_token = SPECIAL_TOKENS['eot_id']['text']

    for i in range(min(num_samples, len(df))):
        row = df.iloc[i]
        print(f"\n--- 샘플 {i}: 전체 시퀀스 구조 ---")

        prompt = row['prompt_text']
        target = row['target_text']
        full_sequence = prompt + target

        print(f"\n[전체 시퀀스 길이]")
        print(f"  Prompt: {len(prompt)} chars")
        print(f"  Target: {len(target)} chars")
        print(f"  Total: {len(full_sequence)} chars")

        # <|eot_id|> 위치 분석
        eot_positions = [m.start() for m in re.finditer(re.escape(eot_token), full_sequence)]
        prompt_eot_positions = [m.start() for m in re.finditer(re.escape(eot_token), prompt)]
        target_eot_positions = [m.start() for m in re.finditer(re.escape(eot_token), target)]

        print(f"\n[<|eot_id|> 위치 분석]")
        print(f"  전체 시퀀스 내 위치: {eot_positions}")
        print(f"  Prompt 내 개수: {len(prompt_eot_positions)}")
        print(f"  Target 내 개수: {len(target_eot_positions)}")

        # 기대 구조 검증
        print(f"\n[기대 구조 검증]")

        # 1. <|startoftext|>로 시작하는지
        starts_with_sot = prompt.startswith(SPECIAL_TOKENS['startoftext']['text'])
        print(f"  {'✓' if starts_with_sot else '✗'} <|startoftext|>로 시작")

        # 2. system, user, assistant 역할이 있는지
        has_system = '<|start_header_id|>system<|end_header_id|>' in prompt
        has_user = '<|start_header_id|>user<|end_header_id|>' in prompt
        has_assistant = '<|start_header_id|>assistant<|end_header_id|>' in prompt

        print(f"  {'✓' if has_system else '✗'} system 역할 존재")
        print(f"  {'✓' if has_user else '✗'} user 역할 존재")
        print(f"  {'✓' if has_assistant else '✗'} assistant 역할 존재")

        # 3. Target이 <|eot_id|>로 끝나는지
        target_ends_eot = target.strip().endswith(eot_token)
        print(f"  {'✓' if target_ends_eot else '✗'} Target이 <|eot_id|>로 끝남")

        # 4. 전체 구조 미리보기
        print(f"\n[시퀀스 미리보기]")
        preview_len = 200
        print(f"  처음 {preview_len}자:")
        print(f"    {repr(prompt[:preview_len])}")
        print(f"  마지막 {preview_len}자 (Target 포함):")
        print(f"    {repr(full_sequence[-preview_len:])}")


def generate_format_summary(results_dict):
    """포맷 검증 결과 요약 생성"""
    print_section("포맷 검증 결과 요약")

    print("""
[LLaMA 3 / LLaDA Instruct 기대 포맷]

<|startoftext|><|start_header_id|>system<|end_header_id|>

{System Prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{User Question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{답변}<|eot_id|>  ← Target은 반드시 <|eot_id|>로 끝나야 함!

[핵심 특수 토큰]
""")
    for name, info in SPECIAL_TOKENS.items():
        print(f"  {info['text']}: ID {info['id']}")

    all_good = True
    for path, result in results_dict.items():
        if result and result['without_eot'] > 0:
            all_good = False

    if all_good:
        print("\n✓ 모든 데이터셋이 올바른 형식을 따르고 있습니다!")
    else:
        print("\n" + "=" * 70)
        print(" 해결 방안")
        print("=" * 70)
        print("""
Target 데이터 형식 수정이 필요합니다:

[현재 형식 - 문제]
<BOOLEAN> True </BOOLEAN>

[권장 형식 - 해결]
<BOOLEAN> True </BOOLEAN><|eot_id|>

이렇게 하면 모델이 답변 후 <|eot_id|> (126348) 토큰을
예측하도록 학습되어, 생성 시 자연스럽게 멈추게 됩니다.

Finetuning 시 Loss mask에서 <|eot_id|> 토큰 위치도
반드시 Loss 계산에 포함시켜야 합니다.
        """)


def main():
    """메인 실행"""
    print("=" * 70)
    print(" LLaDA-Instruct 데이터 포맷 검증 스크립트 (pandas 기반)")
    print("=" * 70)

    # 데이터셋 경로들
    dataset_paths = [
        '/app/Mol-LLM_Custom/dataset/train_official/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_train_512_Truncation_bace',
        '/app/Mol-LLM_Custom/dataset/train_official/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_test_512_Truncation_bace',
    ]

    results = {}
    for path in dataset_paths:
        if os.path.exists(path):
            results[path] = check_dataset(path, num_samples=3)

    # 전체 시퀀스 구조 분석
    if dataset_paths and os.path.exists(dataset_paths[0]):
        check_full_sequence_structure(dataset_paths[0], num_samples=2)

    # 토큰화 분석 (첫 번째 데이터셋)
    if dataset_paths and os.path.exists(dataset_paths[0]):
        analyze_tokenization_details(dataset_paths[0], num_samples=2)

    # 최종 요약
    generate_format_summary(results)


if __name__ == "__main__":
    main()
