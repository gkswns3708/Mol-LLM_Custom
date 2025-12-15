# model/blip2_llada.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from model.blip2_opt import Blip2OPT
import logging
import numpy as np

logger = logging.get_logger(__name__)

# LLaDA MASK Token ID
MASK_TOKEN_ID = 126336

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

class Blip2LLaDA(Blip2OPT):
    def __init__(self, args):
        super().__init__(args=args)
        self.system_prompt = "You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation."
        self.mask_token_id = MASK_TOKEN_ID

    def set_llm_model(self, llm_model):
        # LLaDA 모델 로드 (trust_remote_code=True 필수)
        self.llm_model = AutoModel.from_pretrained(
            llm_model, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model, trust_remote_code=True
        )
        # ! TODO: Padding 방향 확인
        if self.llm_tokenizer.padding_side != 'left':
            self.llm_tokenizer.padding_side = 'left'
        
        # 모델의 Embedding Layer 참조 (Input Embedding 조작을 위해 필요)
        self.llm_embed_tokens = self.llm_model.model.embed_tokens

    def forward(self, samples):
        # 1. Graph Encoder & Q-Former를 통한 Soft Prompt 생성
        # [Batch, Num_Query_Tokens, Hidden_Dim]
        graph_embeds = self.get_context_emb(samples) 
        
        # 2. Text Input 처리 (Prompt + Answer)
        # samples['input_ids']는 Prompt, samples['labels']는 Answer를 포함한다고 가정
        # (DataModule의 구성에 따라 조정 필요, 여기서는 일반적인 구조로 작성)
        
        # text_input_ids: Prompt + Answer 전체 시퀀스
        text_input_ids = samples['input_ids'] 
        attention_mask = samples['attention_mask']
        
        # LLaDA 학습 (SFT Guidelines 참조)
        # Prompt 부분은 Noise를 주지 않고, Answer 부분만 Masking 수행
        
        # Prompt 길이 계산 (실제 구현 시 DataModule에서 prompt_lengths를 넘겨주는 것이 정확함)
        # 여기서는 간단히 graph_embeds 이후의 텍스트 부분을 처리한다고 가정
        
        input_embeddings = self.llm_embed_tokens(text_input_ids)
        
        # SFT Forward Process: Answer 부분에만 Masking 적용
        # LLaDA는 전체 시퀀스를 입력받아 처리 (Graph Soft Prompt + Text Prompt + Masked Answer)
        
        batch_size, seq_len = text_input_ids.shape
        
        # 마스킹 확률 설정 (Linear schedule)
        eps = 1e-3
        t = torch.rand(batch_size, device=self.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, seq_len)
        
        # Prompt 영역 마스킹 방지 (labels가 -100인 곳이 Prompt라고 가정)
        # DataModule에서 labels의 prompt 영역을 -100으로 설정했다고 가정
        labels = samples['labels']
        is_answer = (labels != -100)
        
        # 마스킹 적용
        masked_indices = (torch.rand((batch_size, seq_len), device=self.device) < p_mask) & is_answer
        
        # 입력 ID에 마스크 토큰 적용
        noisy_input_ids = torch.where(masked_indices, self.mask_token_id, text_input_ids)
        noisy_text_embeds = self.llm_embed_tokens(noisy_input_ids)
        
        # 전체 입력 임베딩 결합: [Graph Embeds, Noisy Text Embeds]
        inputs_embeds = torch.cat([graph_embeds, noisy_text_embeds], dim=1)
        
        # Attention Mask 확장
        # Graph 부분에 대한 Attention Mask (1) 추가
        graph_atts = torch.ones(graph_embeds.size()[:-1], dtype=torch.long, device=self.device)
        extended_attention_mask = torch.cat([graph_atts, attention_mask], dim=1)
        
        # Forward Pass
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask)
        logits = outputs.logits
        
        # Logits는 Graph 부분이 앞에 포함되어 있으므로 제거
        # logits shape: [B, Graph_Len + Text_Len, Vocab] -> Text 부분만 슬라이싱
        text_logits = logits[:, graph_embeds.shape[1]:, :]
        
        # Loss 계산 (Masked Tokens에 대해서만)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_loss = loss_fct(text_logits[masked_indices], text_input_ids[masked_indices])
        
        # Weighting by p_mask (from LLaDA paper/guidelines)
        weighted_loss = token_loss / p_mask[masked_indices]
        loss = weighted_loss.mean()
        
        return {"loss": loss, "logits": text_logits}

    @torch.no_grad()
    def generate(
        self,
        graphs,
        input_ids,
        attention_mask,
        num_beams=1, # LLaDA는 Beam Search 대신 Sampling step을 사용하므로 무시되거나 steps로 활용
        max_length=128,
        min_length=1,
        **kwargs
    ):
        # LLaDA Generation Logic (Diffusion Sampling)
        
        # 1. Graph Embeddings 생성
        # inputs format conversion logic similar to forward
        samples = {'graphs': graphs[0], 'additional_graphs': graphs[1]} # 구조에 맞게 조정
        graph_embeds = self.get_context_emb(samples)
        
        # 2. Text Prompt Embeddings
        prompt_embeds = self.llm_embed_tokens(input_ids)
        
        # 3. Generation 초기화 (Mask Tokens로 채워진 Answer 생성)
        batch_size = input_ids.shape[0]
        gen_length = max_length
        steps = 64 # Sampling Steps (Hyperparameter, Config에서 받아오도록 수정 권장)
        block_length = 32
        
        # 초기 상태: [Prompt, Masked_Answer]
        # Answer 부분은 모두 MASK_TOKEN_ID로 초기화
        answer_ids = torch.full((batch_size, gen_length), self.mask_token_id, dtype=torch.long, device=self.device)
        
        # Attention Mask 확장 (Prompt + Gen)
        gen_attention_mask = torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=self.device)
        full_text_mask = torch.cat([attention_mask, gen_attention_mask], dim=1)
        graph_atts = torch.ones(graph_embeds.size()[:-1], dtype=torch.long, device=self.device)
        final_attention_mask = torch.cat([graph_atts, full_text_mask], dim=1)
        
        # Generation Loop (Simplified from LLaDA generate.py)
        # 전체 텍스트 시퀀스(ID)를 유지하면서 반복 업데이트
        # Prompt 부분은 고정, Answer 부분(answer_ids)만 업데이트
        
        current_answer_ids = answer_ids.clone()
        
        # Blockwise Generation or Simple Generation Loop
        # 여기서는 Simple Generation 예시 (전체 길이 한 번에 Denoising)
        # 실제 LLaDA generate.py의 block 로직을 적용하려면 복잡도가 증가하므로, 기본 로직 구현
        
        mask_index = (current_answer_ids == self.mask_token_id)
        num_transfer_tokens = get_num_transfer_tokens(mask_index, steps)
        
        for i in range(steps):
            # 현재 상태의 Embeddings 생성
            answer_embeds = self.llm_embed_tokens(current_answer_ids)
            full_embeds = torch.cat([graph_embeds, prompt_embeds, answer_embeds], dim=1)
            
            # Model Forward
            outputs = self.llm_model(inputs_embeds=full_embeds, attention_mask=final_attention_mask)
            
            # Logits에서 Answer 부분만 추출
            # [B, Graph_Len + Prompt_Len + Gen_Len, Vocab]
            logits = outputs.logits[:, -gen_length:, :]
            
            # Gumbel Noise & Prediction
            logits_with_noise = add_gumbel_noise(logits, temperature=0.0) # Temp 조절 가능
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Confidence 계산
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            
            # Mask된 부분만 업데이트할 후보 선정
            current_mask_indices = (current_answer_ids == self.mask_token_id)
            
            # x0는 예측된 원본 토큰, current_answer_ids는 현재 노이즈 상태
            # Confidence 기반으로 일부만 확정(Transfer)
            
            confidence = torch.where(current_mask_indices, x0_p, -np.inf)
            
            transfer_index = torch.zeros_like(current_answer_ids, dtype=torch.bool)
            for j in range(batch_size):
                if num_transfer_tokens[j, i] > 0:
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
            
            # 업데이트
            current_answer_ids[transfer_index] = x0[transfer_index]
            
        return AttrDict(
            predictions=self.llm_tokenizer.batch_decode(current_answer_ids, skip_special_tokens=True),
            logits=None, # 필요시 반환
            attentions=None
        )

    def get_context_emb(self, samples):
        # 기존 Blip2 로직 활용하여 Graph Embedding 추출
        graphs = samples['graphs']
        image_embeds = self.ln_graph(self.graph_encoder(graphs))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_llm = self.llm_proj(query_output.last_hidden_state)
        return inputs_llm

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self