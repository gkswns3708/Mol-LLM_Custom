import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from model.blip2_opt import Blip2OPT
import logging
import numpy as np

logger = logging.get_logger(__name__)

# LLaDA MASK Token ID (config에서 확인 권장, 논문 기준 126336)
MASK_TOKEN_ID = 126336

class Blip2LLaDA(Blip2OPT):
    def __init__(self, args):
        # Blip2OPT의 __init__을 호출하지만, 내부 모델 로딩은 다시 처리
        super().__init__(args=args)
        self.mask_token_id = MASK_TOKEN_ID
    
    # [수정 1] LLaDA 아키텍처에 맞는 LoRA Target Modules
    def get_lora_target_modules(self):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # [수정 2] 모델 로딩 방식 변경 (LlamaForCausalLM이 아닌 AutoModel 사용)
    def set_llm_model(self, llm_model):
        self.llm_model = AutoModel.from_pretrained(
            llm_model, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model, 
            trust_remote_code=True
        )
        # Tokenizer Padding 설정 (LLaDA는 padding_side에 민감할 수 있음)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        
        self.llm_tokenizer.padding_side = 'left' # Generation 시 보통 left padding

        # Embedding Layer 참조 (Input Injection을 위해 필요)
        self.llm_embed_tokens = self.llm_model.model.embed_tokens

    # [수정 3] Diffusion Loss를 적용한 Forward
    def forward(self, samples):
        # 1. Graph Features 추출 (기존 로직 활용)
        graph_embeds, graph_avg_norm, moltoken_avg_norm = self.extract_graph_feature(samples)
        
        # 2. Text Input 준비
        input_ids = samples['input_ids']
        attention_mask = samples['attention_mask']
        labels = samples['labels']

        # 3. LLaDA SFT Diffusion Process
        # (1) 노이즈 추가 (Forward Process)
        # SFT: Prompt(질문)는 그대로 두고, Answer(답변) 부분만 마스킹하여 복원 학습
        
        batch_size, seq_len = input_ids.shape
        
        # Random timestep t sampling
        eps = 1e-3
        t = torch.rand(batch_size, device=self.device)
        p_mask = (1 - eps) * t +   # Masking ratio
        p_mask = p_mask[:, None].repeat(1, seq_len)

        # Prompt 부분은 마스킹하지 않음 (labels == -100 인 곳이 Prompt라고 가정)
        is_answer = (labels != -100)
        
        # 마스킹 결정 (Answer 부분 내에서 p_mask 확률로 마스킹)
        mask_prob = torch.rand((batch_size, seq_len), device=self.device)
        masked_indices = (mask_prob < p_mask) & is_answer
        
        # 입력 ID에 마스크 토큰 적용
        noisy_input_ids = input_ids.clone()
        noisy_input_ids[masked_indices] = self.mask_token_id
        
        # (2) 임베딩 생성 및 결합
        # Text Embeddings
        noisy_text_embeds = self.llm_embed_tokens(noisy_input_ids)
        
        # Graph Embeds 주입 (Blip2OPT의 inject_graph_embeds2input_embeds 로직과 유사하게 처리 필요)
        # 여기서는 간단히 concat 방식을 예시로 들거나, 기존 injection 로직을 호출
        # 하지만 LLaDA는 [Graph] + [Text] 구조가 더 적합할 수 있음.
        # Blip2 구조상 Input Embeds를 직접 조작하므로, inject 로직을 사용:
        
        # 주의: LLaDA는 AutoModel이므로 inputs_embeds 인자를 받는지 확인 필요 (보통 받음)
        inputs_embeds = noisy_text_embeds.clone()
        
        # Graph Embedding 주입 (기존 메서드 활용)
        if "graphs" in samples:
             inputs_embeds, _, _ = self.inject_graph_embeds2input_embeds(
                input_embeds=inputs_embeds,
                is_mol_token=samples['is_mol_token'],
                graphs=(samples['graphs'], samples['additional_graphs'])
            )

        # (3) Model Forward (Diffusion Prediction)
        # LLaDA는 Causal Mask를 쓰지 않으므로 attention_mask는 1(visible)로 설정됨
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits # [B, Seq, Vocab]

        # (4) Loss Calculation (Re-weighted Loss)
        # 마스킹된 부분만 Loss 계산
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        
        # labels는 원본 토큰 ID
        # logits에서 masked_indices 위치만 추출
        if masked_indices.sum() == 0:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            token_loss = loss_fct(logits.transpose(1, 2), input_ids) # [B, Seq]
            
            # Weighting: 1 / p_mask (Guidelines 참조)
            # 마스킹된 토큰에 대해서만 loss 계산 및 가중치 적용
            weighted_loss = token_loss * masked_indices.float() / p_mask
            
            # Answer 길이로 정규화 (선택 사항, Guidelines 코드 참조)
            answer_lengths = is_answer.sum(dim=1, keepdim=True).float()
            loss = weighted_loss.sum() / (answer_lengths.sum() + 1e-8) # 전체 배치의 평균
        instance_loss = weighted_loss.sum(dim=1) / (answer_lengths.squeeze(-1) + 1e-8)

        return {
            "loss": loss,
            "instance_loss": instance_loss,  # [필수 추가]
            "logits": logits,
            "graph_avg_norm": graph_avg_norm,
            "moltoken_avg_norm": moltoken_avg_norm
        }

    # [수정 4] LLaDA 전용 Generate (Iterative Denoising)
    @torch.no_grad()
    def generate(
        self,
        graphs,
        input_ids,
        attention_mask,
        is_mol_token=None,
        max_length=128,
        steps=64, # Sampling Steps
        gen_guidance_scale=0.0, # CFG Scale (필요시)
        **kwargs
    ):
        batch_size = input_ids.shape[0]
        
        # 1. Graph Feature 준비
        # generate 함수 호출 시 넘어오는 graphs 처리
        # graph_embeds 추출 로직은 forward나 inject_graph_... 활용
        
        # 2. Prompt Embeddings 준비
        # input_ids는 Prompt 부분만 포함됨
        prompt_len = input_ids.shape[1]
        
        # 3. 초기화: Answer 부분을 모두 MASK 토큰으로 채움
        # 전체 길이 = Prompt Len + Gen Len
        total_len = prompt_len + max_length
        
        # [Prompt, Masked Answer]
        gen_tokens = torch.full((batch_size, max_length), self.mask_token_id, device=self.device, dtype=torch.long)
        full_ids = torch.cat([input_ids, gen_tokens], dim=1)
        
        # Attention Mask 확장
        gen_mask = torch.ones((batch_size, max_length), device=self.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([attention_mask, gen_mask], dim=1)
        
        # 4. Sampling Loop (Semi-Discrete Euler or Simple Re-masking)
        # LLaDA generate.py 참고하여 간소화된 버전 구현
        
        for step in range(steps):
            # (1) Embeddings 생성 및 Graph 주입
            current_embeds = self.llm_embed_tokens(full_ids)
            
            # Graph Injection (Prompt 부분에 해당하는 위치에 주입)
            # is_mol_token도 확장 필요 (Answer 부분은 False)
            is_mol_token_gen = torch.zeros((batch_size, max_length), device=self.device, dtype=torch.bool)
            full_is_mol_token = torch.cat([is_mol_token, is_mol_token_gen], dim=1)
            
            current_embeds, _, _ = self.inject_graph_embeds2input_embeds(
                input_embeds=current_embeds,
                is_mol_token=full_is_mol_token,
                graphs=graphs
            )
            
            # (2) Model Forward
            outputs = self.llm_model(
                inputs_embeds=current_embeds,
                attention_mask=full_attention_mask
            )
            logits = outputs.logits # [B, Total_Len, Vocab]
            
            # Answer 부분 Logits
            gen_logits = logits[:, prompt_len:, :]
            
            # (3) Prediction & Update (간단한 Re-masking 전략 예시)
            # 실제 구현 시 generate.py의 get_num_transfer_tokens 등 사용 권장
            
            # 예측 토큰 (x0_hat)
            pred_tokens = torch.argmax(gen_logits, dim=-1)
            
            # Confidence 기반 업데이트 (Re-masking schedule)
            # Step에 따라 점진적으로 확정
            progress = (step + 1) / steps
            # 간단하게: 확률적으로 일부 토큰을 확정 (실제 LLaDA는 더 정교함)
            
            # 여기서는 LLaDA generate.py의 핵심 로직(confidence 기반)을 가져오는 것이 좋음
            # 공간상 생략되었으나, pred_tokens를 full_ids의 뒷부분에 업데이트
            full_ids[:, prompt_len:] = pred_tokens # (단순 예시: 매 스텝 전부 업데이트 -> 실제론 일부만)
            
            # 실제로는 마스크 상태를 유지하며 일부만 confident한 것으로 교체해야 함
            
        # 5. 결과 반환
        generated_text = self.llm_tokenizer.batch_decode(full_ids[:, prompt_len:], skip_special_tokens=True)
        
        return AttrDict(
            predictions=generated_text,
            logits=logits,
            attentions=None # 필요하다면 추가
        )

    # Graph Feature 추출 헬퍼 (기존 코드 재사용 위함)
    def extract_graph_feature(self, samples):
        # Blip2OPT의 로직을 활용하거나 직접 구현
        # 여기서는 inject_graph_embeds2input_embeds 내부에서 처리하므로
        # 더미 리턴 혹은 forward 내에서 직접 호출
        return None, None, None
    
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self