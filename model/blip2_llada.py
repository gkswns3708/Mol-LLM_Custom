import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from model.blip2_opt import Blip2OPT, split_batch_by_components
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
import model.added_tokens as added_tokens
import numpy as np
import json
import os
from datetime import datetime, timezone, timedelta

# 한국 시간대 (KST = UTC+9)
KST = timezone(timedelta(hours=9))
# [중요] Python 기본 logging 대신 transformers의 logging을 사용
logger = logging.get_logger(__name__)

# LLaDA MASK Token ID
MASK_TOKEN_ID = 126336 # <|mdm_mask|> id

class Blip2LLaDA(Blip2OPT):
    def __init__(self, args):
        super().__init__(args=args)
        self.mask_token_id = MASK_TOKEN_ID

        # [수정] LLaDA는 기존 Attention Logging 로직과 호환되지 않으므로,
        # Stage 3 평가 시 에러 방지를 위해 강제로 False로 설정합니다.
        if hasattr(self.args, 'log_attn_score'):
            self.args.log_attn_score = False
            logger.info("LLaDA model detected: Disabled 'log_attn_score' to avoid incompatibility.")

        # ==================== Debug Logging Configuration ====================
        # config에서 로깅 설정 가져오기 (없으면 기본값 사용)
        self._log_embedding_status = getattr(args, 'log_embedding_status', False)
        self._embedding_log_interval = getattr(args, 'embedding_log_interval', 500)
        self._log_model_init_details = getattr(args, 'log_model_init_details', False)
        self._log_nan_details = getattr(args, 'log_nan_details', True)
        self._nan_log_dir = getattr(args, 'nan_log_dir', './nan_logs')

        # Special token embedding 로깅을 위한 카운터
        self._log_step_counter = 0
        self._initial_embedding_norms = None  # 초기 embedding norm 저장용

        # 새 토큰 디버깅 로깅 설정
        self._log_new_token_debug = getattr(args, 'log_new_token_debug', False)
        self._new_token_debug_interval = getattr(args, 'new_token_debug_interval', 100)
        # =====================================================================

    def get_lora_target_modules(self):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def set_llm_model(self, llm_model):
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model, 
            trust_remote_code=True
        )
        
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        
        self.llm_tokenizer.padding_side = 'left' 

        # Embedding Layer 참조 (안전하게 가져오기)
        try:
            self.llm_embed_tokens = self.llm_model.get_input_embeddings()
        except AttributeError:
            if hasattr(self.llm_model, 'embed_tokens'):
                self.llm_embed_tokens = self.llm_model.embed_tokens
            elif hasattr(self.llm_model, 'model') and hasattr(self.llm_model.model, 'embed_tokens'):
                self.llm_embed_tokens = self.llm_model.model.embed_tokens
            else:
                raise AttributeError("Cannot find embedding layer.")
        
        self.check_and_add_special_tokens()
        
    def check_and_add_special_tokens(self):
        # 1. 기본 특수 토큰 리스트 취합
        special_tokens_list = (
            added_tokens.BOOL + 
            added_tokens.FLOAT + 
            added_tokens.DESCRIPTION + 
            added_tokens.SELFIES +
            added_tokens.MOL_2D + 
            added_tokens.MOL_3D + 
            added_tokens.MOL_EMBEDDING +
            added_tokens.NUMBER + 
            added_tokens.INSTRUCTION + 
            added_tokens.REACTION_DIRECTION +
            added_tokens.IUPAC + 
            added_tokens.MOLFORMULA
        )

        # 2. SELFIES Dictionary 파일에서 토큰 읽어오기
        if getattr(self.args, "add_selfies_tokens", False):
            if hasattr(self.args, "selfies_token_path") and self.args.selfies_token_path:
                try:
                    with open(self.args.selfies_token_path, "r") as f:
                        selfies_tokens = f.readlines()
                        selfies_tokens = [token.strip() for token in selfies_tokens]
                    
                    special_tokens_list.extend(selfies_tokens)
                    # config에서 로깅 설정 가져오기 (초기화 순서 때문에 getattr 사용)
                    if getattr(self.args, 'log_model_init_details', False):
                        logger.info(f"[Token Check] Loaded {len(selfies_tokens)} SELFIES tokens from {self.args.selfies_token_path}")
                    self.llm_tokenizer.added_selfies_tokens = selfies_tokens

                except Exception as e:
                    logger.error(f"[Token Check] Failed to load SELFIES tokens from file: {e}")

        # config에서 로깅 설정 가져오기 (초기화 순서 때문에 getattr 사용)
        _log_details = getattr(self.args, 'log_model_init_details', False)

        # 3. 토크나이저에 모든 토큰 추가
        num_added_toks = self.llm_tokenizer.add_tokens(special_tokens_list)
        if _log_details:
            logger.info(f"[Token Check] Added {num_added_toks} special tokens to tokenizer.")

        # 4. 모델 임베딩 크기 조정 (Input Embedding)
        new_vocab_size = len(self.llm_tokenizer)
        if _log_details:
            logger.info(f"[DEBUG] Resizing Input Embeddings to {new_vocab_size}")
        self.llm_model.resize_token_embeddings(new_vocab_size)

        # ==============================================================================
        # [중요] 5. 출력 레이어(LM Head) 강제 리사이징 (이 부분이 핵심 해결책입니다)
        # ==============================================================================
        output_embeddings = self.llm_model.get_output_embeddings()

        if output_embeddings is not None and output_embeddings.weight.shape[0] != new_vocab_size:
            if _log_details:
                logger.info(f"[DEBUG] Output embedding size mismatch! Input: {new_vocab_size}, Output: {output_embeddings.weight.shape[0]}")
                logger.info("[DEBUG] Forcing resize of output embeddings (lm_head)...")

            # 새로운 출력 헤드 생성 (기존 가중치 복사)
            new_lm_head = nn.Linear(
                output_embeddings.in_features,
                new_vocab_size,
                bias=output_embeddings.bias is not None
            ).to(self.llm_model.device).to(output_embeddings.weight.dtype)

            # 기존 가중치 복사
            n_orig = output_embeddings.weight.shape[0]
            with torch.no_grad():
                new_lm_head.weight[:n_orig, :] = output_embeddings.weight
                if output_embeddings.bias is not None:
                    new_lm_head.bias[:n_orig] = output_embeddings.bias

            # 모델에 새로운 헤드 설정
            self.llm_model.set_output_embeddings(new_lm_head)

            # NOTE: requires_grad 설정은 PEFT 적용 후에 blip2_opt.py에서 처리됨
            # 여기서 설정해도 get_peft_model() 호출 시 래핑되면서 무시될 수 있음
            if _log_details:
                logger.info(f"[DEBUG] Output embedding force resize complete. New shape: {new_lm_head.weight.shape}")
                logger.info(f"[DEBUG] Set lm_head.weight.requires_grad = True")
        else:
            if _log_details:
                logger.info(f"[DEBUG] Output embedding size is correct: {output_embeddings.weight.shape[0]}")
        # ==============================================================================

        # 6. <mol> 토큰 ID 저장
        self.mol_token_id = self.llm_tokenizer.convert_tokens_to_ids(added_tokens.MOL_EMBEDDING)[0]

        # 7. SELFIES Token ID 저장
        if getattr(self.args, "add_selfies_tokens", False) and hasattr(self.llm_tokenizer, "added_selfies_tokens"):
            self.llm_tokenizer.selfies_token_ids = [
                self.llm_tokenizer.convert_tokens_to_ids(token)
                for token in self.llm_tokenizer.added_selfies_tokens
            ]
    
    # [Stage 1 에러 수정] 그래프가 None일 경우 예외 처리 추가된 버전
    def inject_graph_embeds2input_embeds(self, input_embeds, is_mol_token, graphs):
        # graphs 튜플 언패킹 (Main Graph, Additional Graph)
        mol_graphs, mol2_graphs = graphs

        mol_token_sequence = []

        for graphs in [mol_graphs, mol2_graphs]:
            # 그래프가 None이면 건너뛰기
            if graphs is None:
                continue

            mol_x = graphs["x"]
            mol_edge_index = graphs["edge_index"]
            mol_edge_attr = graphs["edge_attr"]
            mol_batch = graphs["batch"]

            if self.args.process_disjoint:
                num_graph_list = []
                graph_list = []
                for graph in graphs.to_data_list():
                    tmp_batch = Batch.from_data_list([graph])
                    tmp_batch = split_batch_by_components(tmp_batch)
                    graph_list.extend(tmp_batch)
                    num_graph_list.append(len(tmp_batch))
                
                graph_batch = Batch.from_data_list(graph_list)
                graph_embeds, graph_masks = self.graph_encoder(
                    graph_batch.x,
                    graph_batch.edge_index,
                    graph_batch.edge_attr,
                    graph_batch.batch,
                )
                mol_embeds_list = []
                mol_mask_list = []

                graph_embeds = torch.split(graph_embeds, num_graph_list, dim=0)
                graph_masks = torch.split(graph_masks, num_graph_list, dim=0)
                for graph_embed, graph_mask in zip(graph_embeds, graph_masks):
                    mol_embeds_list.append(graph_embed[graph_mask])
                    mol_mask_list.append(graph_mask[graph_mask])
                mol_embeds = pad_sequence(mol_embeds_list, batch_first=True)
                mol_masks = pad_sequence(mol_mask_list, batch_first=True)
            else:
                mol_embeds, mol_masks = self.graph_encoder(
                    mol_x, mol_edge_index, mol_edge_attr, mol_batch
                )

            if not self.tune_gnn:
                mol_embeds = mol_embeds.detach()
            mol_embeds = self.ln_graph(mol_embeds, mol_masks)

            if self.args.projector_type == "qformer":
                query_tokens = self.query_tokens.expand(mol_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=mol_embeds,
                    encoder_attention_mask=mol_masks,
                    return_dict=True,
                )
                mol_tokens = self.opt_proj(query_output.last_hidden_state)
            else:
                mol_tokens = self.opt_proj(mol_embeds)
            mol_token_sequence.append(mol_tokens)

        # [핵심] 그래프 토큰이 생성되지 않았을 경우 처리
        if len(mol_token_sequence) == 0:
            batch_size = input_embeds.shape[0]
            device = input_embeds.device
            dummy_norm = torch.zeros(batch_size, device=device)
            return input_embeds, dummy_norm, dummy_norm

        # 결과 연결
        mol_tokens = torch.cat(mol_token_sequence, dim=1)
        moltoken_avg_norm = torch.norm(mol_tokens, p=1, dim=-1).mean(1)
        
        graph_avg_norm = torch.zeros(input_embeds.shape[0], device=input_embeds.device)
        
        num_mol_tokens_per_sample = is_mol_token.sum(dim=1)
        if (num_mol_tokens_per_sample > 0).any():
            mol_token_indices_full = (is_mol_token.cumsum(dim=1) - 1)
            batch_indices, token_indices = is_mol_token.nonzero(as_tuple=True)
            mol_token_indices = mol_token_indices_full[batch_indices, token_indices]
            
            input_embeds[batch_indices, token_indices, :] = mol_tokens[
                batch_indices, mol_token_indices, :
            ]

        return input_embeds, graph_avg_norm, moltoken_avg_norm

    def forward(self, samples):
        input_ids = samples['input_ids']
        attention_mask = samples['attention_mask']
        labels = samples['labels']

        batch_size, seq_len = input_ids.shape

        # ========================================================================
        # LLaDA Forward Process (SMDM 원본 구현 참조)
        #
        # 핵심: response 영역에서만 마스킹하고, 마스킹된 위치의 원본 토큰을 예측
        #
        # 1. labels != -100 인 위치가 response 영역 (is_answer)
        #    - DataLoader에서 target text 끝에 EOS 토큰이 추가됨 (data_utils.py)
        #    - 따라서 EOS 토큰도 labels에 포함되어 response 영역에 포함됨
        #    - LLaDA 논문 Section 2.3: "We treat |EOS| as a normal token during training"
        # 2. response 영역 내에서 랜덤하게 마스킹 (masked_indices)
        # 3. 마스킹된 위치에서만 cross-entropy loss 계산
        # 4. 1/p_mask로 importance weighting
        # ========================================================================

        eps = 1e-6
        t = torch.rand(batch_size, device=self.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, seq_len)

        # is_answer: response 영역 마스크 (labels != -100인 위치)
        is_answer = (labels != -100)

        mask_prob = torch.rand((batch_size, seq_len), device=self.device)
        # 오직 response 영역(is_answer)에서만 마스킹
        masked_indices = (mask_prob < p_mask) & is_answer

        # 원본 토큰 저장 (loss 계산용)
        original_tokens = input_ids.clone()

        # Noisy input 생성: 마스킹된 위치를 mask_token_id로 대체
        noisy_input_ids = input_ids.clone()
        noisy_input_ids[masked_indices] = self.mask_token_id

        noisy_text_embeds = self.llm_embed_tokens(noisy_input_ids)
        inputs_embeds = noisy_text_embeds.clone()

        graph_avg_norm = torch.zeros(batch_size, device=self.device)
        moltoken_avg_norm = torch.zeros(batch_size, device=self.device)

        if "graphs" in samples:
             inputs_embeds, graph_avg_norm, moltoken_avg_norm = self.inject_graph_embeds2input_embeds(
                input_embeds=inputs_embeds,
                is_mol_token=samples['is_mol_token'],
                graphs=(samples['graphs'], samples['additional_graphs'])
            )

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_loss = None  # NaN 로깅을 위해 미리 초기화

        if masked_indices.sum() == 0:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            instance_loss = torch.zeros(batch_size, device=self.device)
        else:
            # ================================================================
            # [핵심 수정] SMDM 원본 방식으로 loss 계산
            #
            # 원본: loss = CE(logits[mask], target[mask]) / p_mask[mask]
            #       loss = loss.sum() / (batch_size * seq_len)
            #
            # 마스킹된 위치에서만 loss를 계산하고, 1/p_mask로 weighting
            # ================================================================

            # logits: [batch, seq_len, vocab_size]
            # masked_indices: [batch, seq_len] boolean

            # 마스킹된 위치의 logits와 targets 추출
            masked_logits = logits[masked_indices]  # [num_masked, vocab_size]
            masked_targets = original_tokens[masked_indices]  # [num_masked]
            masked_p = p_mask[masked_indices]  # [num_masked]

            # Cross-entropy loss (마스킹된 위치만)
            token_loss = loss_fct(masked_logits, masked_targets)  # [num_masked]

            # Importance weighting: 1/p_mask
            weighted_loss = token_loss / masked_p  # [num_masked]

            # Normalization: response 길이의 합으로 나눔 (원본은 전체 seq_len이지만,
            # conditional generation이므로 response 길이 사용)
            answer_lengths = is_answer.sum(dim=1).float()  # [batch]
            total_answer_length = answer_lengths.sum()

            loss = weighted_loss.sum() / (total_answer_length + 1e-8)

            # Instance-level loss 계산 (per-sample)
            # 각 샘플별로 마스킹된 토큰의 weighted loss 합계
            instance_loss = torch.zeros(batch_size, device=self.device)
            batch_indices = torch.where(masked_indices)[0]  # 각 마스킹된 토큰이 어떤 배치에 속하는지
            instance_loss.scatter_add_(0, batch_indices, weighted_loss)
            instance_loss = instance_loss / (answer_lengths + 1e-8)
        if torch.isnan(loss) or torch.isinf(loss):
            # NaN 로깅 (config로 제어)
            if self._log_nan_details:
                self._log_nan_samples(
                    samples=samples,
                    logits=logits,
                    input_ids=input_ids,
                    labels=labels,
                    masked_indices=masked_indices,
                    p_mask=p_mask,
                    token_loss=token_loss,
                    graph_avg_norm=graph_avg_norm,
                    loss_value=loss
                )

            # [임시 조치] 학습이 터지는 것을 막기 위해 Loss를 0으로 강제 변환
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # ==============================================================================
        # [디버깅] Special Token Embedding 상태 로깅 (config로 제어)
        # ==============================================================================
        if self.training and self._log_embedding_status and self._embedding_log_interval > 0:
            self._log_step_counter += 1
            if self._log_step_counter % self._embedding_log_interval == 0:
                self._log_special_token_embedding_status()

        # ==============================================================================
        # [디버깅] 새 토큰 Loss 기여도 및 Gradient 분석
        # ==============================================================================
        new_token_debug_info = {}
        if self.training and self._log_new_token_debug:
            self._log_step_counter += 1
            if self._log_step_counter % self._new_token_debug_interval == 0:
                new_token_debug_info = self._analyze_new_token_contribution(
                    original_tokens=original_tokens,
                    masked_indices=masked_indices,
                    masked_targets=masked_targets if masked_indices.sum() > 0 else None,
                    token_loss=token_loss,
                    masked_logits=masked_logits if masked_indices.sum() > 0 else None,
                )

        return {
            "loss": loss,
            "instance_loss": instance_loss,
            "logits": logits,
            "graph_avg_norm": graph_avg_norm,
            "moltoken_avg_norm": moltoken_avg_norm,
            "new_token_debug": new_token_debug_info,
        }

    def forward_stepwise_teacher_forcing(self, samples, steps=32):
        """
        Step-wise Teacher Forcing 방식의 Generation Loss 계산

        LLaDA의 iterative denoising 과정 전체를 시뮬레이션:
        - 각 step별로 해당 마스킹 비율에 맞는 입력을 생성
        - 정답 토큰을 Teacher Forcing으로 제공 (실제 생성 대신)
        - 각 step의 loss에 importance weighting (1/p) 적용
        - 모든 step의 weighted loss를 평균

        이는 val_total_loss와 유사하지만, 전체 denoising trajectory를
        deterministic하게 시뮬레이션한다는 점이 다름.

        Args:
            samples: 배치 데이터 (input_ids, attention_mask, labels, graphs 등)
            steps: denoising step 수 (기본값: 32)

        Returns:
            dict: {
                "loss": 전체 평균 loss,
                "instance_loss": 샘플별 loss [batch_size]
            }
        """
        input_ids = samples['input_ids']
        attention_mask = samples['attention_mask']
        labels = samples['labels']

        batch_size, seq_len = input_ids.shape

        # response 영역 (labels != -100인 위치)
        is_answer = (labels != -100)
        answer_lengths = is_answer.sum(dim=1).float()  # [batch_size]

        # 원본 토큰 (정답) - Teacher Forcing에 사용
        original_tokens = input_ids.clone()

        loss_fct = nn.CrossEntropyLoss(reduction='none')

        # Step-wise loss 누적
        instance_weighted_loss_sum = torch.zeros(batch_size, device=self.device)

        for step in range(steps):
            # 현재 step의 마스킹 비율 계산 (1.0 → 0.0 선형 감소)
            # t = 1.0 - step / steps
            # 마스킹 비율 p = t (step 0에서 100%, step N-1에서 ~0%)
            t = 1.0 - step / steps
            p_mask = max(t, 1e-6)  # 0 방지

            # 각 샘플의 response 영역에서 p_mask 비율만큼 마스킹
            noisy_input_ids = input_ids.clone()

            # 샘플별로 마스킹 적용
            for b in range(batch_size):
                # 해당 샘플의 response 위치들
                answer_positions = torch.where(is_answer[b])[0]
                num_answer_tokens = len(answer_positions)

                if num_answer_tokens == 0:
                    continue

                # p_mask 비율만큼 마스킹할 토큰 수
                num_to_mask = int(num_answer_tokens * p_mask)
                num_to_mask = max(1, num_to_mask)  # 최소 1개는 마스킹

                # 랜덤하게 마스킹할 위치 선택
                perm = torch.randperm(num_answer_tokens, device=self.device)
                mask_indices = answer_positions[perm[:num_to_mask]]

                # 마스킹 적용
                noisy_input_ids[b, mask_indices] = self.mask_token_id

            # Embedding 생성
            noisy_text_embeds = self.llm_embed_tokens(noisy_input_ids)
            inputs_embeds = noisy_text_embeds.clone()

            # Graph embedding 주입 (있는 경우)
            if "graphs" in samples:
                inputs_embeds, _, _ = self.inject_graph_embeds2input_embeds(
                    input_embeds=inputs_embeds,
                    is_mol_token=samples['is_mol_token'],
                    graphs=(samples['graphs'], samples['additional_graphs'])
                )

            # Forward pass
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits

            # 마스킹된 위치에서만 loss 계산
            masked_indices = (noisy_input_ids == self.mask_token_id) & is_answer

            if masked_indices.sum() == 0:
                continue

            # 마스킹된 위치의 logits와 targets
            masked_logits = logits[masked_indices]
            masked_targets = original_tokens[masked_indices]

            # CE Loss
            token_loss = loss_fct(masked_logits, masked_targets)

            # Importance weighting: 1/p_mask
            weighted_token_loss = token_loss / p_mask

            # 샘플별로 weighted loss 누적
            batch_indices = torch.where(masked_indices)[0]
            for i, (b_idx, w_loss) in enumerate(zip(batch_indices, weighted_token_loss)):
                instance_weighted_loss_sum[b_idx] += w_loss

        # 각 샘플의 최종 loss: weighted_loss_sum / (answer_length * steps)
        # steps로 나누는 이유: 각 토큰이 여러 step에서 마스킹될 수 있으므로
        instance_loss = instance_weighted_loss_sum / ((answer_lengths + 1e-8) * steps)

        # 전체 평균 loss
        loss = instance_loss.mean()

        return {
            "loss": loss,
            "instance_loss": instance_loss,
        }

    def _log_special_token_embedding_status(self):
        """
        Special token embedding 상태 및 gradient 로깅
        - 새로 추가된 토큰들의 embedding이 학습되고 있는지 확인
        - requires_grad 상태, embedding norm, gradient norm 출력
        """
        special_tokens = [
            "<BOOLEAN>", "</BOOLEAN>",
            "<DESCRIPTION>", "</DESCRIPTION>",
            "<SELFIES>", "</SELFIES>",
            "<FLOAT>", "</FLOAT>",
            "<mol>",
        ]

        try:
            embed_layer = self.llm_model.get_input_embeddings()
            output_layer = self.llm_model.get_output_embeddings()

            logger.info("=" * 70)
            logger.info(f"[Step {self._log_step_counter}] Special Token Embedding Status")
            logger.info("=" * 70)

            # Embedding layer 상태
            logger.info(f"Input Embedding (embed_tokens):")
            logger.info(f"  - Shape: {embed_layer.weight.shape}")
            logger.info(f"  - requires_grad: {embed_layer.weight.requires_grad}")
            logger.info(f"  - has gradient: {embed_layer.weight.grad is not None}")

            if output_layer is not None:
                logger.info(f"Output Layer (lm_head):")
                logger.info(f"  - Shape: {output_layer.weight.shape}")
                logger.info(f"  - requires_grad: {output_layer.weight.requires_grad}")
                logger.info(f"  - has gradient: {output_layer.weight.grad is not None}")

            logger.info("-" * 70)
            logger.info("Special Token Details:")

            token_data = []
            for token in special_tokens:
                token_id = self.llm_tokenizer.convert_tokens_to_ids(token)

                # 토큰이 제대로 추가되었는지 확인
                if token_id is None or token_id == self.llm_tokenizer.unk_token_id:
                    logger.warning(f"  {token}: NOT FOUND in tokenizer (id={token_id})")
                    continue

                if token_id >= embed_layer.weight.shape[0]:
                    logger.warning(f"  {token}: id={token_id} OUT OF RANGE (vocab_size={embed_layer.weight.shape[0]})")
                    continue

                # Embedding norm
                embed_norm = embed_layer.weight[token_id].detach().norm().item()

                # Gradient norm (있으면)
                embed_grad_norm = "N/A"
                if embed_layer.weight.grad is not None:
                    embed_grad_norm = f"{embed_layer.weight.grad[token_id].norm().item():.6f}"

                # Output layer norm & grad
                output_norm = "N/A"
                output_grad_norm = "N/A"
                if output_layer is not None and token_id < output_layer.weight.shape[0]:
                    output_norm = f"{output_layer.weight[token_id].detach().norm().item():.4f}"
                    if output_layer.weight.grad is not None:
                        output_grad_norm = f"{output_layer.weight.grad[token_id].norm().item():.6f}"

                logger.info(f"  {token:15} (id={token_id:6}): "
                           f"embed_norm={embed_norm:.4f}, embed_grad={embed_grad_norm}, "
                           f"output_norm={output_norm}, output_grad={output_grad_norm}")

                token_data.append({
                    'token': token,
                    'id': token_id,
                    'embed_norm': embed_norm
                })

            # 초기 embedding norm 저장 (첫 로깅 시)
            if self._initial_embedding_norms is None and token_data:
                self._initial_embedding_norms = {d['token']: d['embed_norm'] for d in token_data}
                logger.info("-" * 70)
                logger.info("Initial embedding norms saved for comparison.")
            elif self._initial_embedding_norms:
                # 변화량 출력
                logger.info("-" * 70)
                logger.info("Embedding Norm Changes (from initial):")
                for d in token_data:
                    if d['token'] in self._initial_embedding_norms:
                        initial = self._initial_embedding_norms[d['token']]
                        current = d['embed_norm']
                        change = current - initial
                        pct_change = (change / initial * 100) if initial != 0 else 0
                        logger.info(f"  {d['token']:15}: {initial:.4f} -> {current:.4f} "
                                   f"(delta={change:+.4f}, {pct_change:+.2f}%)")

            # 새로 추가된 토큰 전체의 gradient 통계
            if embed_layer.weight.grad is not None:
                # LLaDA 원래 vocab size (대략적인 값, 실제로는 config에서 가져와야 함)
                orig_vocab_size = 128256
                if embed_layer.weight.shape[0] > orig_vocab_size:
                    new_token_grads = embed_layer.weight.grad[orig_vocab_size:]
                    grad_norms = new_token_grads.norm(dim=1)
                    logger.info("-" * 70)
                    logger.info(f"New Token Gradients (id >= {orig_vocab_size}):")
                    logger.info(f"  - Count: {new_token_grads.shape[0]}")
                    logger.info(f"  - Grad norm mean: {grad_norms.mean().item():.6f}")
                    logger.info(f"  - Grad norm max: {grad_norms.max().item():.6f}")
                    logger.info(f"  - Grad norm min: {grad_norms.min().item():.6f}")
                    logger.info(f"  - Non-zero grads: {(grad_norms > 1e-8).sum().item()}/{new_token_grads.shape[0]}")

            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"Error logging special token status: {e}")

    def _log_nan_samples(
        self,
        samples,
        logits,
        input_ids,
        labels,
        masked_indices,
        p_mask,
        token_loss,
        graph_avg_norm,
        loss_value
    ):
        """
        NaN/Inf loss 발생 시 상세 정보를 파일로 로깅합니다.
        - 어떤 샘플에서 발생했는지
        - 예측값 (argmax of logits)
        - 라벨값
        - 각종 통계 정보
        """
        timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S_%f")
        log_dir = getattr(self.args, 'nan_log_dir', './nan_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"nan_sample_{timestamp}.json")

        batch_size = input_ids.shape[0]

        # Task 정보
        task_info = samples.get("task", samples.get("dataset_name", ["Unknown"] * batch_size))
        if isinstance(task_info, str):
            task_info = [task_info] * batch_size

        # 콘솔 로그 출력
        logger.error("\n" + "="*60)
        logger.error(f"🚨 [NaN/Inf DETECTED] Logging to: {log_file}")
        logger.error(f"Loss Value: {loss_value.item() if torch.is_tensor(loss_value) else loss_value}")
        logger.error(f"Batch Size: {batch_size}")
        logger.error(f"Tasks: {task_info}")

        # Logit 통계
        logger.error(f"Logits Stats - Max: {logits.max().item():.6f}, Min: {logits.min().item():.6f}, Mean: {logits.mean().item():.6f}")
        logger.error(f"Logits has NaN: {torch.isnan(logits).any().item()}, has Inf: {torch.isinf(logits).any().item()}")

        # Token Loss 통계
        if token_loss is not None:
            logger.error(f"Token Loss Stats - Max: {token_loss.max().item():.6f}, Min: {token_loss.min().item():.6f}")
            logger.error(f"Token Loss has NaN: {torch.isnan(token_loss).any().item()}, has Inf: {torch.isinf(token_loss).any().item()}")

        # p_mask 통계
        logger.error(f"p_mask Stats - Max: {p_mask.max().item():.6f}, Min: {p_mask.min().item():.6f}")

        # Graph Norm
        logger.error(f"Graph Avg Norm: {graph_avg_norm.mean().item():.6f}")
        logger.error("="*60)

        # 상세 정보를 JSON으로 저장
        nan_log_data = {
            "timestamp": timestamp,
            "loss_value": float(loss_value.item()) if torch.is_tensor(loss_value) and not (torch.isnan(loss_value) or torch.isinf(loss_value)) else str(loss_value.item() if torch.is_tensor(loss_value) else loss_value),
            "batch_size": batch_size,
            "logits_stats": {
                "max": float(logits.max().item()),
                "min": float(logits.min().item()),
                "mean": float(logits.mean().item()),
                "has_nan": bool(torch.isnan(logits).any().item()),
                "has_inf": bool(torch.isinf(logits).any().item()),
                "nan_count": int(torch.isnan(logits).sum().item()),
                "inf_count": int(torch.isinf(logits).sum().item()),
            },
            "p_mask_stats": {
                "max": float(p_mask.max().item()),
                "min": float(p_mask.min().item()),
                "mean": float(p_mask.mean().item()),
            },
            "graph_avg_norm": float(graph_avg_norm.mean().item()),
            "samples": []
        }

        if token_loss is not None:
            nan_log_data["token_loss_stats"] = {
                "max": float(token_loss.max().item()),
                "min": float(token_loss.min().item()),
                "has_nan": bool(torch.isnan(token_loss).any().item()),
                "has_inf": bool(torch.isinf(token_loss).any().item()),
            }

        # 각 샘플별 상세 정보
        predictions = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

        for i in range(batch_size):
            sample_info = {
                "sample_idx": i,
                "task": task_info[i] if i < len(task_info) else "Unknown",
            }

            # 입력 텍스트 (prompt)
            if "prompt" in samples:
                prompt = samples["prompt"]
                sample_info["prompt"] = prompt[i] if isinstance(prompt, list) else prompt

            # 타겟 텍스트 (정답)
            if "target" in samples:
                target = samples["target"]
                sample_info["target_text"] = target[i] if isinstance(target, list) else target

            # 분자 정보
            if "smiles" in samples:
                smiles = samples["smiles"]
                sample_info["smiles"] = smiles[i] if isinstance(smiles, list) else smiles

            # 라벨 토큰 (마스킹 되지 않은 영역, labels != -100)
            label_mask = labels[i] != -100
            label_tokens = labels[i][label_mask].cpu().tolist()
            sample_info["label_token_ids"] = label_tokens
            sample_info["label_text"] = self.llm_tokenizer.decode(label_tokens, skip_special_tokens=False)

            # 예측 토큰 (마스킹된 위치에서의 예측)
            masked_positions = masked_indices[i]
            if masked_positions.any():
                pred_at_masked = predictions[i][masked_positions].cpu().tolist()
                label_at_masked = input_ids[i][masked_positions].cpu().tolist()
                sample_info["masked_positions_count"] = int(masked_positions.sum().item())
                sample_info["pred_token_ids_at_masked"] = pred_at_masked
                sample_info["label_token_ids_at_masked"] = label_at_masked
                sample_info["pred_text_at_masked"] = self.llm_tokenizer.decode(pred_at_masked, skip_special_tokens=False)
                sample_info["label_text_at_masked"] = self.llm_tokenizer.decode(label_at_masked, skip_special_tokens=False)

            # 해당 샘플의 logit 통계
            sample_logits = logits[i]
            sample_info["sample_logits_stats"] = {
                "max": float(sample_logits.max().item()),
                "min": float(sample_logits.min().item()),
                "has_nan": bool(torch.isnan(sample_logits).any().item()),
                "has_inf": bool(torch.isinf(sample_logits).any().item()),
            }

            # 해당 샘플의 token_loss 통계
            if token_loss is not None:
                sample_token_loss = token_loss[i]
                sample_info["sample_token_loss_stats"] = {
                    "max": float(sample_token_loss.max().item()),
                    "min": float(sample_token_loss.min().item()),
                    "has_nan": bool(torch.isnan(sample_token_loss).any().item()),
                    "has_inf": bool(torch.isinf(sample_token_loss).any().item()),
                }

                # NaN/Inf가 발생한 위치 찾기
                nan_positions = torch.where(torch.isnan(sample_token_loss))[0].cpu().tolist()
                inf_positions = torch.where(torch.isinf(sample_token_loss))[0].cpu().tolist()
                if nan_positions:
                    sample_info["nan_token_positions"] = nan_positions[:20]  # 최대 20개만
                    sample_info["nan_token_ids"] = input_ids[i][nan_positions[:20]].cpu().tolist()
                    sample_info["nan_tokens_text"] = self.llm_tokenizer.decode(input_ids[i][nan_positions[:20]].cpu().tolist())
                if inf_positions:
                    sample_info["inf_token_positions"] = inf_positions[:20]
                    sample_info["inf_token_ids"] = input_ids[i][inf_positions[:20]].cpu().tolist()
                    sample_info["inf_tokens_text"] = self.llm_tokenizer.decode(input_ids[i][inf_positions[:20]].cpu().tolist())

            nan_log_data["samples"].append(sample_info)

        # JSON 파일로 저장
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(nan_log_data, f, ensure_ascii=False, indent=2)
            logger.error(f"✅ NaN log saved to: {log_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save NaN log: {e}")

    def _analyze_new_token_contribution(
        self,
        original_tokens,
        masked_indices,
        masked_targets,
        token_loss,
        masked_logits,
    ):
        """
        새로 추가된 토큰(SELFIES 등)의 Loss 기여도 및 학습 상태 분석

        분석 항목:
        1. 전체 토큰 중 새 토큰 비율 (input, masked)
        2. 새 토큰 vs 기존 토큰의 Loss 비교
        3. 새 토큰에 대한 모델 예측 정확도
        4. Embedding gradient 분석
        """
        debug_info = {}

        try:
            # 기존 vocab size (LLaDA 기본)
            orig_vocab_size = getattr(self.args, 'original_vocab_size', 128256)

            batch_size, seq_len = original_tokens.shape

            # ================================================================
            # 1. 토큰 비율 분석
            # ================================================================
            # 전체 input에서 새 토큰 비율
            is_new_token_input = (original_tokens >= orig_vocab_size)
            total_tokens = original_tokens.numel()
            new_token_count_input = is_new_token_input.sum().item()
            new_token_ratio_input = new_token_count_input / total_tokens * 100

            debug_info['token_ratio/input_new_token_count'] = new_token_count_input
            debug_info['token_ratio/input_new_token_pct'] = new_token_ratio_input
            debug_info['token_ratio/input_total_tokens'] = total_tokens

            # Masked 토큰 중 새 토큰 비율
            if masked_targets is not None and len(masked_targets) > 0:
                is_new_token_masked = (masked_targets >= orig_vocab_size)
                masked_total = len(masked_targets)
                new_token_count_masked = is_new_token_masked.sum().item()
                new_token_ratio_masked = new_token_count_masked / masked_total * 100

                debug_info['token_ratio/masked_new_token_count'] = new_token_count_masked
                debug_info['token_ratio/masked_new_token_pct'] = new_token_ratio_masked
                debug_info['token_ratio/masked_total'] = masked_total

                # ================================================================
                # 2. Loss 분석 (새 토큰 vs 기존 토큰)
                # ================================================================
                if token_loss is not None and len(token_loss) > 0:
                    # 새 토큰에 대한 loss
                    if is_new_token_masked.sum() > 0:
                        new_token_loss = token_loss[is_new_token_masked]
                        debug_info['loss/new_token_mean'] = new_token_loss.mean().item()
                        debug_info['loss/new_token_max'] = new_token_loss.max().item()
                        debug_info['loss/new_token_min'] = new_token_loss.min().item()
                        debug_info['loss/new_token_sum'] = new_token_loss.sum().item()
                    else:
                        debug_info['loss/new_token_mean'] = 0.0
                        debug_info['loss/new_token_sum'] = 0.0

                    # 기존 토큰에 대한 loss
                    is_orig_token_masked = ~is_new_token_masked
                    if is_orig_token_masked.sum() > 0:
                        orig_token_loss = token_loss[is_orig_token_masked]
                        debug_info['loss/orig_token_mean'] = orig_token_loss.mean().item()
                        debug_info['loss/orig_token_max'] = orig_token_loss.max().item()
                        debug_info['loss/orig_token_min'] = orig_token_loss.min().item()
                        debug_info['loss/orig_token_sum'] = orig_token_loss.sum().item()
                    else:
                        debug_info['loss/orig_token_mean'] = 0.0
                        debug_info['loss/orig_token_sum'] = 0.0

                    # Loss 기여도 비율
                    total_loss = token_loss.sum().item()
                    if total_loss > 0:
                        debug_info['loss/new_token_contribution_pct'] = debug_info.get('loss/new_token_sum', 0) / total_loss * 100
                        debug_info['loss/orig_token_contribution_pct'] = debug_info.get('loss/orig_token_sum', 0) / total_loss * 100

                # ================================================================
                # 3. 예측 정확도 분석
                # ================================================================
                if masked_logits is not None and len(masked_logits) > 0:
                    predictions = masked_logits.argmax(dim=-1)

                    # 새 토큰에 대한 정확도
                    if is_new_token_masked.sum() > 0:
                        new_token_preds = predictions[is_new_token_masked]
                        new_token_targets = masked_targets[is_new_token_masked]
                        new_token_correct = (new_token_preds == new_token_targets).float().mean().item()
                        debug_info['accuracy/new_token'] = new_token_correct * 100

                        # 새 토큰이 새 토큰으로 예측되었는지 (vocab 범위 체크)
                        pred_is_new = (new_token_preds >= orig_vocab_size).float().mean().item()
                        debug_info['accuracy/new_token_pred_in_new_vocab_pct'] = pred_is_new * 100
                    else:
                        debug_info['accuracy/new_token'] = 0.0

                    # 기존 토큰에 대한 정확도
                    if is_orig_token_masked.sum() > 0:
                        orig_token_preds = predictions[is_orig_token_masked]
                        orig_token_targets = masked_targets[is_orig_token_masked]
                        orig_token_correct = (orig_token_preds == orig_token_targets).float().mean().item()
                        debug_info['accuracy/orig_token'] = orig_token_correct * 100
                    else:
                        debug_info['accuracy/orig_token'] = 0.0

            # ================================================================
            # 4. Embedding Gradient 분석
            # ================================================================
            embed_layer = self.llm_model.get_input_embeddings()
            output_layer = self.llm_model.get_output_embeddings()

            # Input Embedding gradient
            if embed_layer is not None:
                # 실제 weight 가져오기 (PEFT wrapper 처리)
                if hasattr(embed_layer, 'modules_to_save'):
                    # PEFT ModulesToSaveWrapper인 경우
                    actual_embed = embed_layer.modules_to_save.get('default', embed_layer)
                    if hasattr(actual_embed, 'weight'):
                        embed_weight = actual_embed.weight
                    else:
                        embed_weight = embed_layer.weight
                elif hasattr(embed_layer, 'weight'):
                    embed_weight = embed_layer.weight
                else:
                    embed_weight = None

                if embed_weight is not None and embed_weight.grad is not None:
                    vocab_size = embed_weight.shape[0]

                    if vocab_size > orig_vocab_size:
                        # 새 토큰 gradient
                        new_token_grad = embed_weight.grad[orig_vocab_size:]
                        new_grad_norms = new_token_grad.norm(dim=1)
                        debug_info['grad/embed_new_mean'] = new_grad_norms.mean().item()
                        debug_info['grad/embed_new_max'] = new_grad_norms.max().item()
                        debug_info['grad/embed_new_nonzero_count'] = (new_grad_norms > 1e-10).sum().item()
                        debug_info['grad/embed_new_nonzero_pct'] = (new_grad_norms > 1e-10).sum().item() / len(new_grad_norms) * 100

                        # 기존 토큰 gradient (비교용)
                        orig_token_grad = embed_weight.grad[:orig_vocab_size]
                        orig_grad_norms = orig_token_grad.norm(dim=1)
                        debug_info['grad/embed_orig_mean'] = orig_grad_norms.mean().item()
                        debug_info['grad/embed_orig_max'] = orig_grad_norms.max().item()

                        # Gradient 비율
                        if debug_info['grad/embed_orig_mean'] > 0:
                            debug_info['grad/embed_new_vs_orig_ratio'] = debug_info['grad/embed_new_mean'] / debug_info['grad/embed_orig_mean']

            # Output (LM Head) gradient
            if output_layer is not None:
                if hasattr(output_layer, 'modules_to_save'):
                    actual_output = output_layer.modules_to_save.get('default', output_layer)
                    if hasattr(actual_output, 'weight'):
                        output_weight = actual_output.weight
                    else:
                        output_weight = output_layer.weight
                elif hasattr(output_layer, 'weight'):
                    output_weight = output_layer.weight
                else:
                    output_weight = None

                if output_weight is not None and output_weight.grad is not None:
                    vocab_size = output_weight.shape[0]

                    if vocab_size > orig_vocab_size:
                        # 새 토큰 gradient
                        new_head_grad = output_weight.grad[orig_vocab_size:]
                        new_head_grad_norms = new_head_grad.norm(dim=1)
                        debug_info['grad/head_new_mean'] = new_head_grad_norms.mean().item()
                        debug_info['grad/head_new_max'] = new_head_grad_norms.max().item()
                        debug_info['grad/head_new_nonzero_count'] = (new_head_grad_norms > 1e-10).sum().item()

                        # 기존 토큰 gradient
                        orig_head_grad = output_weight.grad[:orig_vocab_size]
                        orig_head_grad_norms = orig_head_grad.norm(dim=1)
                        debug_info['grad/head_orig_mean'] = orig_head_grad_norms.mean().item()

            # ================================================================
            # 5. 콘솔 로깅 (요약)
            # ================================================================
            logger.info("\n" + "=" * 70)
            logger.info(f"[Step {self._log_step_counter}] NEW TOKEN DEBUG ANALYSIS")
            logger.info("=" * 70)
            logger.info(f"[Token Ratio]")
            logger.info(f"  Input:  {new_token_count_input}/{total_tokens} ({new_token_ratio_input:.2f}%)")
            if 'token_ratio/masked_total' in debug_info:
                logger.info(f"  Masked: {debug_info.get('token_ratio/masked_new_token_count', 0)}/{debug_info['token_ratio/masked_total']} ({debug_info.get('token_ratio/masked_new_token_pct', 0):.2f}%)")

            logger.info(f"\n[Loss Analysis]")
            logger.info(f"  New Token Loss Mean:  {debug_info.get('loss/new_token_mean', 'N/A')}")
            logger.info(f"  Orig Token Loss Mean: {debug_info.get('loss/orig_token_mean', 'N/A')}")
            logger.info(f"  New Token Loss Contribution: {debug_info.get('loss/new_token_contribution_pct', 'N/A'):.2f}%")

            logger.info(f"\n[Prediction Accuracy]")
            logger.info(f"  New Token Accuracy:  {debug_info.get('accuracy/new_token', 'N/A'):.2f}%")
            logger.info(f"  Orig Token Accuracy: {debug_info.get('accuracy/orig_token', 'N/A'):.2f}%")

            logger.info(f"\n[Gradient Analysis]")
            logger.info(f"  Embed New Grad Mean:  {debug_info.get('grad/embed_new_mean', 'N/A')}")
            logger.info(f"  Embed Orig Grad Mean: {debug_info.get('grad/embed_orig_mean', 'N/A')}")
            logger.info(f"  Embed New/Orig Ratio: {debug_info.get('grad/embed_new_vs_orig_ratio', 'N/A')}")
            logger.info(f"  Embed New Nonzero:    {debug_info.get('grad/embed_new_nonzero_count', 'N/A')}/{vocab_size - orig_vocab_size if 'grad/embed_new_mean' in debug_info else 'N/A'}")
            logger.info("=" * 70 + "\n")

        except Exception as e:
            logger.error(f"Error in _analyze_new_token_contribution: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return debug_info

    @staticmethod
    def add_gumbel_noise(logits, temperature):
        if temperature <= 1e-6:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (- torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    @staticmethod
    def get_num_transfer_tokens(mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1
        return num_transfer_tokens

    @torch.no_grad()
    def generate(
        self,
        graphs,
        input_ids,
        attention_mask,
        is_mol_token=None,
        max_length=128,
        steps=64,
        temperature=0.0,
        remasking_strategy='low_confidence',
        use_semi_ar=False,
        semi_ar_block_size=32,
        task_name=None,
        **kwargs
    ):
        """
        LLaDA Generation with configurable remasking strategy

        ========================================================================
        LLaDA 논문 Algorithm 4 & 5 구현 (Appendix A.3)
        ========================================================================

        remasking_strategy 옵션:
        - 'low_confidence': Algorithm 5 - 낮은 confidence 토큰을 다시 mask
        - 'random': Algorithm 4 - 랜덤하게 s/t 비율만큼 다시 mask
        - 'none': Remasking 없이 매 step마다 top-k 토큰만 unmask (기존 방식)

        use_semi_ar 옵션:
        - True: Semi-Autoregressive 모드 (블록 단위로 순차 생성)
        - False: 전체 영역 동시 생성

        Args:
            graphs: 분자 그래프 (tuple of main_graph, additional_graph)
            input_ids: 입력 토큰 ID [batch, prompt_len]
            attention_mask: 어텐션 마스크 [batch, prompt_len]
            is_mol_token: mol token 위치 마스크 [batch, prompt_len]
            max_length: 최대 생성 길이
            steps: Diffusion steps
            temperature: Gumbel noise temperature (0=greedy, >0=stochastic)
            remasking_strategy: 'low_confidence' | 'random' | 'none'
            use_semi_ar: Semi-Autoregressive 모드 사용 여부
            semi_ar_block_size: Semi-AR 블록 크기
            task_name: Task 이름 (Semi-AR 모드에서 format token 결정용)
        """
        # Semi-AR 모드 분기
        if use_semi_ar:
            return self._generate_semi_ar(
                graphs=graphs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                is_mol_token=is_mol_token,
                max_length=max_length,
                steps=steps,
                temperature=temperature,
                remasking_strategy=remasking_strategy,
                block_size=semi_ar_block_size,
                task_name=task_name,
                **kwargs
            )

        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        gen_len = max_length

        # 초기화: 생성 영역을 모두 MASK로 설정 (t=1 상태)
        gen_tokens = torch.full((batch_size, gen_len), self.mask_token_id, device=self.device, dtype=torch.long)
        full_ids = torch.cat([input_ids, gen_tokens], dim=1)

        gen_mask = torch.ones((batch_size, gen_len), device=self.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([attention_mask, gen_mask], dim=1)

        if is_mol_token is not None:
            is_mol_token_gen = torch.zeros((batch_size, gen_len), device=self.device, dtype=torch.bool)
            full_is_mol_token = torch.cat([is_mol_token, is_mol_token_gen], dim=1)
        else:
            full_is_mol_token = None

        # ================================================================
        # Remasking 전략에 따른 생성 로직
        # ================================================================

        if remasking_strategy == 'none':
            # 기존 방식: 매 step마다 top-k 토큰만 unmask (remasking 없음)
            return self._generate_no_remask(
                full_ids=full_ids,
                full_attention_mask=full_attention_mask,
                full_is_mol_token=full_is_mol_token,
                graphs=graphs,
                prompt_len=prompt_len,
                gen_len=gen_len,
                steps=steps,
                temperature=temperature,
            )
        else:
            # Algorithm 4 (random) 또는 Algorithm 5 (low_confidence)
            return self._generate_with_remask(
                full_ids=full_ids,
                full_attention_mask=full_attention_mask,
                full_is_mol_token=full_is_mol_token,
                graphs=graphs,
                prompt_len=prompt_len,
                gen_len=gen_len,
                steps=steps,
                temperature=temperature,
                remasking_strategy=remasking_strategy,
            )

    def _generate_no_remask(
        self,
        full_ids,
        full_attention_mask,
        full_is_mol_token,
        graphs,
        prompt_len,
        gen_len,
        steps,
        temperature,
    ):
        """
        Remasking 없는 생성 (기존 방식)

        매 step마다 가장 높은 confidence의 k개 토큰만 unmask.
        한번 unmask된 토큰은 변경되지 않음.
        """
        batch_size = full_ids.shape[0]

        mask_index = (full_ids[:, prompt_len:] == self.mask_token_id)
        num_transfer_tokens = self.get_num_transfer_tokens(mask_index, steps)

        for step in range(steps):
            cur_mask_index = (full_ids == self.mask_token_id)

            current_embeds = self.llm_embed_tokens(full_ids)

            if graphs is not None:
                current_embeds, _, _ = self.inject_graph_embeds2input_embeds(
                    input_embeds=current_embeds,
                    is_mol_token=full_is_mol_token,
                    graphs=graphs
                )

            outputs = self.llm_model(
                inputs_embeds=current_embeds,
                attention_mask=full_attention_mask,
                return_dict=True
            )
            logits = outputs.logits

            logits_with_noise = self.add_gumbel_noise(logits, temperature)
            x0_pred = torch.argmax(logits_with_noise, dim=-1)

            # Confidence 계산 (low_confidence 방식 사용)
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0_pred, -1)), -1
            )

            x0_p[:, :prompt_len] = -np.inf
            confidence = torch.where(cur_mask_index, x0_p, torch.tensor(-np.inf, device=self.device))

            # Top-k 선택하여 unmask
            transfer_mask = torch.zeros_like(full_ids, dtype=torch.bool)

            for b in range(batch_size):
                k = num_transfer_tokens[b, step]
                if k > 0:
                    _, select_indices = torch.topk(confidence[b], k=k)
                    transfer_mask[b, select_indices] = True

            full_ids[transfer_mask] = x0_pred[transfer_mask]

        generated_tokens = full_ids[:, prompt_len:]
        generated_text = self.llm_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return AttrDict(
            predictions=generated_text,
            sequences=full_ids,
            logits=logits,
            attentions=None
        )

    def _generate_with_remask(
        self,
        full_ids,
        full_attention_mask,
        full_is_mol_token,
        graphs,
        prompt_len,
        gen_len,
        steps,
        temperature,
        remasking_strategy,
    ):
        """
        LLaDA 논문 Algorithm 4 (random) & Algorithm 5 (low_confidence) 구현

        핵심 차이점 (기존 방식 vs 논문 방식):
        - 기존: 매 step마다 k개 토큰만 unmask, 나머지는 그대로 유지
        - 논문: 모든 mask 토큰을 예측 후, 낮은 confidence 토큰을 다시 mask

        Algorithm 5 (Low-Confidence Remasking):
        1. 모든 masked 토큰을 예측 (r0 = argmax)
        2. 이미 unmasked된 토큰은 confidence = 1로 설정
        3. nun = ⌊L(1-s)⌋ 개의 가장 높은 confidence 토큰만 unmask 상태 유지
        4. 나머지는 다시 MASK로 되돌림 (remasking)

        Algorithm 4 (Random Remasking):
        - Step 2에서 confidence 대신 random 값 사용
        """
        batch_size = full_ids.shape[0]

        for step in range(steps):
            # 현재 timestep t와 다음 timestep s 계산
            t = 1.0 - step / steps
            s = max(0.0, t - 1.0 / steps)

            # Forward pass
            current_embeds = self.llm_embed_tokens(full_ids)

            if graphs is not None:
                current_embeds, _, _ = self.inject_graph_embeds2input_embeds(
                    input_embeds=current_embeds,
                    is_mol_token=full_is_mol_token,
                    graphs=graphs
                )

            outputs = self.llm_model(
                inputs_embeds=current_embeds,
                attention_mask=full_attention_mask,
                return_dict=True
            )
            logits = outputs.logits

            # Gumbel noise로 예측
            logits_with_noise = self.add_gumbel_noise(logits, temperature)
            x0_pred = torch.argmax(logits_with_noise, dim=-1)

            # ============================================================
            # Confidence 계산 (Algorithm 5 line 8-9)
            # ============================================================
            cur_mask_index = (full_ids == self.mask_token_id)

            if remasking_strategy == 'low_confidence':
                # Algorithm 5: 예측 토큰의 확률을 confidence로 사용
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0_pred, -1)), -1
                )
            elif remasking_strategy == 'random':
                # Algorithm 4: 랜덤 값을 confidence로 사용
                x0_p = torch.rand_like(logits[:, :, 0])
            else:
                raise NotImplementedError(f"Unknown remasking strategy: {remasking_strategy}")

            # Confidence 초기화:
            # - 현재 masked인 위치: 예측 confidence
            # - 현재 unmasked인 위치: 1.0 (이미 확정된 토큰은 유지)
            confidence = torch.zeros_like(x0_p)
            confidence[:, :prompt_len] = np.inf  # prompt는 절대 건드리지 않음
            confidence[:, prompt_len:] = torch.where(
                cur_mask_index[:, prompt_len:],
                x0_p[:, prompt_len:],  # masked -> 예측 confidence
                torch.ones_like(x0_p[:, prompt_len:])  # unmasked -> 1.0 (유지)
            )

            # ============================================================
            # Step 1: 모든 masked 위치를 예측값으로 채움 (Algorithm 5 line 6-8)
            # ============================================================
            full_ids = torch.where(cur_mask_index, x0_pred, full_ids)

            # ============================================================
            # Step 2: Remasking (Algorithm 5 line 12-16)
            # nun = ⌊L(1-s)⌋ 개의 가장 높은 confidence 토큰만 유지
            # ============================================================
            if s > 0:  # 마지막 step이 아니면 remasking 수행
                # nun: unmask 상태로 유지할 토큰 수
                nun = int(gen_len * (1 - s))

                for b in range(batch_size):
                    # 생성 영역의 confidence만 고려
                    gen_confidence = confidence[b, prompt_len:]

                    if nun < gen_len:
                        # 가장 높은 confidence의 nun개만 유지, 나머지는 다시 mask
                        _, keep_indices = torch.topk(gen_confidence, k=nun, largest=True)

                        # 모든 생성 영역을 MASK로 설정
                        full_ids[b, prompt_len:] = self.mask_token_id

                        # 가장 높은 confidence의 토큰만 복원
                        full_ids[b, prompt_len + keep_indices] = x0_pred[b, prompt_len + keep_indices]

        generated_tokens = full_ids[:, prompt_len:]
        generated_text = self.llm_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return AttrDict(
            predictions=generated_text,
            sequences=full_ids,
            logits=logits,
            attentions=None
        )

    def _generate_semi_ar(
        self,
        graphs,
        input_ids,
        attention_mask,
        is_mol_token,
        max_length,
        steps,
        temperature,
        remasking_strategy,
        block_size,
        task_name=None,
        **kwargs
    ):
        """
        Semi-Autoregressive Generation (논문 Appendix A.3, Figure 4)

        시퀀스를 여러 블록으로 나누어 왼쪽에서 오른쪽으로 순차 생성.
        각 블록 내에서는 지정된 remasking_strategy 사용.

        Args:
            block_size: 각 블록의 크기 (토큰 수)
            remasking_strategy: 블록 내 remasking 전략
        """
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        gen_len = max_length

        # 초기화
        gen_tokens = torch.full((batch_size, gen_len), self.mask_token_id, device=self.device, dtype=torch.long)
        full_ids = torch.cat([input_ids, gen_tokens], dim=1)

        gen_mask = torch.ones((batch_size, gen_len), device=self.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([attention_mask, gen_mask], dim=1)

        if is_mol_token is not None:
            is_mol_token_gen = torch.zeros((batch_size, gen_len), device=self.device, dtype=torch.bool)
            full_is_mol_token = torch.cat([is_mol_token, is_mol_token_gen], dim=1)
        else:
            full_is_mol_token = None

        # 블록 수 계산
        num_blocks = (gen_len + block_size - 1) // block_size

        # 각 블록에 할당할 step 수
        steps_per_block = max(1, steps // num_blocks)

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_size
            block_end = min(prompt_len + (block_idx + 1) * block_size, prompt_len + gen_len)
            current_block_size = block_end - block_start

            if current_block_size <= 0:
                break

            # 현재 블록에 대해 diffusion 수행
            for step in range(steps_per_block):
                t = 1.0 - step / steps_per_block
                s = max(0.0, t - 1.0 / steps_per_block)

                # Forward pass (전체 시퀀스)
                current_embeds = self.llm_embed_tokens(full_ids)

                if graphs is not None:
                    current_embeds, _, _ = self.inject_graph_embeds2input_embeds(
                        input_embeds=current_embeds,
                        is_mol_token=full_is_mol_token,
                        graphs=graphs
                    )

                outputs = self.llm_model(
                    inputs_embeds=current_embeds,
                    attention_mask=full_attention_mask,
                    return_dict=True
                )
                logits = outputs.logits

                logits_with_noise = self.add_gumbel_noise(logits, temperature)
                x0_pred = torch.argmax(logits_with_noise, dim=-1)

                # 현재 블록의 mask 여부 확인
                cur_block_mask = (full_ids[:, block_start:block_end] == self.mask_token_id)

                # Confidence 계산
                if remasking_strategy == 'low_confidence':
                    p = F.softmax(logits[:, block_start:block_end, :], dim=-1)
                    block_pred = x0_pred[:, block_start:block_end]
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(block_pred, -1)), -1
                    )
                elif remasking_strategy == 'random':
                    x0_p = torch.rand((batch_size, current_block_size), device=self.device)
                else:
                    # 'none': confidence 기반 선택
                    p = F.softmax(logits[:, block_start:block_end, :], dim=-1)
                    block_pred = x0_pred[:, block_start:block_end]
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(block_pred, -1)), -1
                    )

                # Confidence 설정
                confidence = torch.where(
                    cur_block_mask,
                    x0_p,
                    torch.ones_like(x0_p)  # 이미 unmask된 토큰은 유지
                )

                # 모든 masked 위치를 예측값으로 채움
                full_ids[:, block_start:block_end] = torch.where(
                    cur_block_mask,
                    x0_pred[:, block_start:block_end],
                    full_ids[:, block_start:block_end]
                )

                # Remasking (마지막 step이 아닌 경우)
                if s > 0 and remasking_strategy != 'none':
                    nun = int(current_block_size * (1 - s))

                    for b in range(batch_size):
                        block_confidence = confidence[b]

                        if nun < current_block_size:
                            _, keep_indices = torch.topk(block_confidence, k=nun, largest=True)

                            # 블록 전체를 MASK로 설정
                            full_ids[b, block_start:block_end] = self.mask_token_id

                            # 높은 confidence 토큰만 복원
                            for idx in keep_indices:
                                full_ids[b, block_start + idx] = x0_pred[b, block_start + idx]

        generated_tokens = full_ids[:, prompt_len:]
        generated_text = self.llm_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return AttrDict(
            predictions=generated_text,
            sequences=full_ids,
            logits=logits,
            attentions=None
        )

    def extract_graph_feature(self, samples):
        return None, None, None

    # ==========================================================================
    # Semi-Autoregressive Generation
    # ==========================================================================
    # 핵심 아이디어:
    # 1. Task 유형에 따라 format tokens (<BOOLEAN>, <SELFIES> 등)을 먼저 결정
    # 2. Format tokens를 생성 시퀀스의 시작/끝 위치에 배치
    # 3. 나머지 content 위치에서만 diffusion 수행
    # ==========================================================================

    # Task -> Format Token 매핑
    TASK_FORMAT_MAPPING = {
        # Classification tasks -> BOOLEAN
        'bace': ('BOOLEAN', 'BOOLEAN'),
        'bbbp': ('BOOLEAN', 'BOOLEAN'),
        'hiv': ('BOOLEAN', 'BOOLEAN'),
        'clintox': ('BOOLEAN', 'BOOLEAN'),
        'tox21': ('BOOLEAN', 'BOOLEAN'),
        'sider': ('BOOLEAN', 'BOOLEAN'),
        'pcba': ('BOOLEAN', 'BOOLEAN'),
        'muv': ('BOOLEAN', 'BOOLEAN'),

        # Regression tasks -> FLOAT
        'esol': ('FLOAT', 'FLOAT'),
        'freesolv': ('FLOAT', 'FLOAT'),
        'lipo': ('FLOAT', 'FLOAT'),
        'qm7': ('FLOAT', 'FLOAT'),
        'qm8': ('FLOAT', 'FLOAT'),
        'qm9': ('FLOAT', 'FLOAT'),

        # Description tasks -> DESCRIPTION
        'chebi-20': ('DESCRIPTION', 'DESCRIPTION'),
        'mol2text': ('DESCRIPTION', 'DESCRIPTION'),
        'description': ('DESCRIPTION', 'DESCRIPTION'),

        # Molecule generation tasks -> SELFIES
        'text2mol': ('SELFIES', 'SELFIES'),
        'forward_synthesis': ('SELFIES', 'SELFIES'),
        'retrosynthesis': ('SELFIES', 'SELFIES'),
        'molecule_generation': ('SELFIES', 'SELFIES'),
        'selfies': ('SELFIES', 'SELFIES'),

        # IUPAC tasks
        'iupac': ('IUPAC', 'IUPAC'),
        'smiles2iupac': ('IUPAC', 'IUPAC'),

        # Molecular Formula
        'molformula': ('MOLFORMULA', 'MOLFORMULA'),
    }

    def _get_format_tokens_for_task(self, task_name):
        """
        Task 이름에 따라 적절한 format token pair를 반환합니다.

        Args:
            task_name: Task 이름 (예: 'bace', 'esol', 'chebi-20')

        Returns:
            Tuple[str, str]: (open_tag, close_tag) 예: ('<BOOLEAN>', '</BOOLEAN>')
        """
        if task_name is None:
            return None, None

        task_lower = task_name.lower().strip()

        # 정확한 매칭 먼저 시도
        if task_lower in self.TASK_FORMAT_MAPPING:
            format_type = self.TASK_FORMAT_MAPPING[task_lower][0]
            return f'<{format_type}>', f'</{format_type}>'

        # 부분 매칭 시도 (task 이름에 키워드가 포함된 경우)
        for key, (format_type, _) in self.TASK_FORMAT_MAPPING.items():
            if key in task_lower or task_lower in key:
                return f'<{format_type}>', f'</{format_type}>'

        # 키워드 기반 fallback
        if any(kw in task_lower for kw in ['class', 'binary', 'bool']):
            return '<BOOLEAN>', '</BOOLEAN>'
        elif any(kw in task_lower for kw in ['regress', 'predict', 'value', 'float']):
            return '<FLOAT>', '</FLOAT>'
        elif any(kw in task_lower for kw in ['desc', 'caption', 'text', 'explain']):
            return '<DESCRIPTION>', '</DESCRIPTION>'
        elif any(kw in task_lower for kw in ['mol', 'smiles', 'selfies', 'synth', 'retro']):
            return '<SELFIES>', '</SELFIES>'

        return None, None

    def _estimate_content_length(self, task_name, max_length):
        """
        Task에 따른 예상 content 길이를 반환합니다.
        Format tokens을 제외한 실제 content 영역의 길이입니다.

        Args:
            task_name: Task 이름
            max_length: 전체 최대 생성 길이

        Returns:
            int: 예상 content 길이
        """
        task_lower = task_name.lower().strip() if task_name else ''

        # Classification: "True" or "False" -> 매우 짧음
        if task_lower in ['bace', 'bbbp', 'hiv', 'clintox', 'tox21', 'sider', 'pcba', 'muv']:
            return min(8, max_length - 4)  # "True"/"False" + 여유

        # Regression: 숫자 -> 짧음
        if task_lower in ['esol', 'freesolv', 'lipo', 'qm7', 'qm8', 'qm9']:
            return min(16, max_length - 4)  # "-3.456" 정도

        # Description: 긴 텍스트
        if task_lower in ['chebi-20', 'mol2text', 'description']:
            return max_length - 4

        # Molecule generation: SELFIES 시퀀스
        if task_lower in ['text2mol', 'forward_synthesis', 'retrosynthesis', 'molecule_generation', 'selfies']:
            return max_length - 4

        # Default
        return max_length - 4

    @torch.no_grad()
    def generate_semi_ar(
        self,
        graphs,
        input_ids,
        attention_mask,
        is_mol_token=None,
        task_name=None,  # Task 이름 (format 결정용)
        max_length=128,
        steps=64,
        temperature=0.0,
        remasking_strategy='low_confidence',
        **kwargs
    ):
        """
        Semi-Autoregressive Generation for LLaDA

        Format tokens를 먼저 고정한 후, content만 diffusion으로 생성합니다.
        이를 통해 multi-task 환경에서 format 혼란 문제를 해결합니다.

        Args:
            graphs: 분자 그래프 (tuple of main_graph, additional_graph)
            input_ids: 입력 토큰 ID [batch, prompt_len]
            attention_mask: 어텐션 마스크 [batch, prompt_len]
            is_mol_token: mol token 위치 마스크 [batch, prompt_len]
            task_name: Task 이름 또는 Task 리스트 (batch별로 다를 수 있음)
            max_length: 최대 생성 길이
            steps: Diffusion steps
            temperature: Gumbel noise temperature
            remasking_strategy: 'low_confidence' 또는 'random'

        Returns:
            AttrDict with predictions, sequences, logits, attentions
        """
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]

        # Task name을 batch 형태로 정규화
        if task_name is None:
            task_names = [None] * batch_size
        elif isinstance(task_name, str):
            task_names = [task_name] * batch_size
        else:
            task_names = list(task_name)
            if len(task_names) < batch_size:
                task_names.extend([task_names[-1]] * (batch_size - len(task_names)))

        # ============================================================
        # Step 1: Format tokens 결정 및 배치
        # ============================================================

        # 각 샘플별 format token 정보 수집
        format_info = []
        for i, tn in enumerate(task_names):
            open_tag, close_tag = self._get_format_tokens_for_task(tn)

            if open_tag is not None:
                open_id = self.llm_tokenizer.convert_tokens_to_ids(open_tag)
                close_id = self.llm_tokenizer.convert_tokens_to_ids(close_tag)

                # 토큰이 제대로 변환되었는지 확인
                if open_id == self.llm_tokenizer.unk_token_id or close_id == self.llm_tokenizer.unk_token_id:
                    logger.warning(f"[Semi-AR] Format tokens not found in vocab: {open_tag}, {close_tag}")
                    format_info.append({'has_format': False})
                else:
                    content_len = self._estimate_content_length(tn, max_length)
                    format_info.append({
                        'has_format': True,
                        'open_tag': open_tag,
                        'close_tag': close_tag,
                        'open_id': open_id,
                        'close_id': close_id,
                        'content_len': content_len
                    })
            else:
                format_info.append({'has_format': False})

        # ============================================================
        # Step 2: 생성 시퀀스 초기화 (format tokens 포함)
        # ============================================================

        gen_len = max_length
        gen_tokens = torch.full((batch_size, gen_len), self.mask_token_id, device=self.device, dtype=torch.long)

        # Format tokens 배치
        for i, info in enumerate(format_info):
            if info['has_format']:
                # 시작 위치: 0번 인덱스
                gen_tokens[i, 0] = info['open_id']

                # 종료 위치: content_len + 1 (content 바로 뒤)
                # 또는 max_length - 1 (끝 위치)
                end_pos = min(info['content_len'] + 1, gen_len - 1)
                gen_tokens[i, end_pos] = info['close_id']

                # 저장해둠 (나중에 mask에서 제외용)
                info['open_pos'] = 0
                info['close_pos'] = end_pos

        # 전체 시퀀스 구성
        full_ids = torch.cat([input_ids, gen_tokens], dim=1)

        gen_mask = torch.ones((batch_size, gen_len), device=self.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([attention_mask, gen_mask], dim=1)

        if is_mol_token is not None:
            is_mol_token_gen = torch.zeros((batch_size, gen_len), device=self.device, dtype=torch.bool)
            full_is_mol_token = torch.cat([is_mol_token, is_mol_token_gen], dim=1)
        else:
            full_is_mol_token = None

        # ============================================================
        # Step 3: Format tokens을 제외한 mask index 생성
        # ============================================================

        # 기본 mask: 생성 영역의 mask token
        mask_index = (full_ids[:, prompt_len:] == self.mask_token_id)

        # Format tokens 위치는 mask에서 제외 (이미 고정됨)
        # (위에서 gen_tokens에 이미 format token을 넣었으므로,
        #  mask_token_id가 아닌 위치는 자동으로 mask_index=False가 됨)

        num_transfer_tokens = self.get_num_transfer_tokens(mask_index, steps)

        # ============================================================
        # Step 4: Iterative Denoising (content만)
        # ============================================================

        for step in range(steps):
            # 현재 mask 상태 (prompt 제외, format tokens도 mask 아님)
            cur_mask_index = (full_ids == self.mask_token_id)

            current_embeds = self.llm_embed_tokens(full_ids)

            # 그래프 주입
            if graphs is not None:
                current_embeds, _, _ = self.inject_graph_embeds2input_embeds(
                    input_embeds=current_embeds,
                    is_mol_token=full_is_mol_token,
                    graphs=graphs
                )

            outputs = self.llm_model(
                inputs_embeds=current_embeds,
                attention_mask=full_attention_mask,
                return_dict=True
            )
            logits = outputs.logits

            logits_with_noise = self.add_gumbel_noise(logits, temperature)
            x0_pred = torch.argmax(logits_with_noise, dim=-1)

            # Confidence 계산
            if remasking_strategy == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0_pred, -1)), -1
                )
            elif remasking_strategy == 'random':
                x0_p = torch.rand_like(logits[:, :, 0])
            else:
                raise NotImplementedError(f"Unknown remasking strategy: {remasking_strategy}")

            # Prompt 영역은 confidence -inf (수정 안함)
            x0_p[:, :prompt_len] = -np.inf

            # Format token 위치도 -inf (이미 고정됨)
            for i, info in enumerate(format_info):
                if info['has_format']:
                    x0_p[i, prompt_len + info['open_pos']] = -np.inf
                    x0_p[i, prompt_len + info['close_pos']] = -np.inf

            confidence = torch.where(cur_mask_index, x0_p, torch.tensor(-np.inf, device=self.device))

            # Top-k 선택하여 unmask
            transfer_mask = torch.zeros_like(full_ids, dtype=torch.bool)

            for b in range(batch_size):
                k = num_transfer_tokens[b, step]
                if k > 0:
                    _, select_indices = torch.topk(confidence[b], k=k)
                    transfer_mask[b, select_indices] = True

            full_ids[transfer_mask] = x0_pred[transfer_mask]

        # ============================================================
        # Step 5: 결과 후처리
        # ============================================================

        generated_tokens = full_ids[:, prompt_len:]
        generated_text = self.llm_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return AttrDict(
            predictions=generated_text,
            sequences=full_ids,
            logits=logits,
            attentions=None,
            format_info=format_info  # 디버깅용
        )

    @torch.no_grad()
    def generate(
        self,
        graphs,
        input_ids,
        attention_mask,
        is_mol_token=None,
        max_length=128,
        steps=64,
        temperature=0.0,
        remasking_strategy='low_confidence',
        use_semi_ar=False,  # Semi-AR 모드 활성화 플래그
        task_name=None,     # Semi-AR용 task 이름
        **kwargs
    ):
        """
        LLaDA Generation (기본 또는 Semi-AR 모드)

        Args:
            use_semi_ar: True이면 semi-autoregressive 모드 사용
            task_name: Semi-AR 모드에서 format 결정에 사용
            (나머지 인자는 기존과 동일)
        """
        # Semi-AR 모드 분기
        if use_semi_ar and task_name is not None:
            return self.generate_semi_ar(
                graphs=graphs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                is_mol_token=is_mol_token,
                task_name=task_name,
                max_length=max_length,
                steps=steps,
                temperature=temperature,
                remasking_strategy=remasking_strategy,
                **kwargs
            )

        # 기존 generation 로직
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        gen_len = max_length

        gen_tokens = torch.full((batch_size, gen_len), self.mask_token_id, device=self.device, dtype=torch.long)
        full_ids = torch.cat([input_ids, gen_tokens], dim=1)

        gen_mask = torch.ones((batch_size, gen_len), device=self.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([attention_mask, gen_mask], dim=1)

        if is_mol_token is not None:
            is_mol_token_gen = torch.zeros((batch_size, gen_len), device=self.device, dtype=torch.bool)
            full_is_mol_token = torch.cat([is_mol_token, is_mol_token_gen], dim=1)
        else:
            full_is_mol_token = None

        mask_index = (full_ids[:, prompt_len:] == self.mask_token_id)
        num_transfer_tokens = self.get_num_transfer_tokens(mask_index, steps)

        for step in range(steps):
            cur_mask_index = (full_ids == self.mask_token_id)

            current_embeds = self.llm_embed_tokens(full_ids)

            # 그래프 주입 (오버라이딩된 메서드 사용)
            if graphs is not None:
                current_embeds, _, _ = self.inject_graph_embeds2input_embeds(
                    input_embeds=current_embeds,
                    is_mol_token=full_is_mol_token,
                    graphs=graphs
                )

            outputs = self.llm_model(
                inputs_embeds=current_embeds,
                attention_mask=full_attention_mask,
                return_dict=True
            )
            logits = outputs.logits

            logits_with_noise = self.add_gumbel_noise(logits, temperature)
            x0_pred = torch.argmax(logits_with_noise, dim=-1)

            if remasking_strategy == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0_pred, -1)), -1
                )
            elif remasking_strategy == 'random':
                x0_p = torch.rand_like(logits[:, :, 0])
            else:
                raise NotImplementedError(f"Unknown remasking strategy: {remasking_strategy}")

            x0_p[:, :prompt_len] = -np.inf
            confidence = torch.where(cur_mask_index, x0_p, torch.tensor(-np.inf, device=self.device))

            transfer_mask = torch.zeros_like(full_ids, dtype=torch.bool)

            for b in range(batch_size):
                k = num_transfer_tokens[b, step]
                if k > 0:
                    _, select_indices = torch.topk(confidence[b], k=k)
                    transfer_mask[b, select_indices] = True

            full_ids[transfer_mask] = x0_pred[transfer_mask]

        generated_tokens = full_ids[:, prompt_len:]
        generated_text = self.llm_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # [수정] attentions=None 추가하여 에러 방지
        return AttrDict(
            predictions=generated_text,
            sequences=full_ids,
            logits=logits,
            attentions=None
        )

    # ==========================================================================
    # LLaDA Paper Eq. 6: Monte Carlo Likelihood Estimation
    # ==========================================================================
    #
    # 논문 Section 2.4 Inference, Algorithm 3 (Appendix A), Appendix B.5 참조
    #
    # Eq. 6: log p(y|x) ≈ (1/K) Σ_{k=1}^{K} Σ_{i=1}^{|y|} log p(y_i | x, y^{M_k})
    #
    # 핵심 아이디어:
    # 1. K개의 Monte Carlo 샘플에서 마스킹 비율 t_k ~ Uniform(0,1) 샘플링
    # 2. 응답 토큰 중 t_k 비율만큼 랜덤하게 마스킹
    # 3. Forward pass로 마스킹된 위치의 log-probability 계산
    # 4. 평균 log-likelihood 반환
    # ==========================================================================

    @torch.no_grad()
    def compute_response_likelihood(
        self,
        graphs,
        input_ids,
        attention_mask,
        response_ids,
        response_attention_mask,
        is_mol_token=None,
        num_samples=128,  # 논문 Appendix B.5: K=128 for evaluation
    ):
        """
        LLaDA 논문 Eq. 6에 따른 Monte Carlo Likelihood 추정

        Args:
            graphs: 분자 그래프 (tuple of main_graph, additional_graph)
            input_ids: 프롬프트 토큰 ID [batch, prompt_len]
            attention_mask: 프롬프트 어텐션 마스크 [batch, prompt_len]
            response_ids: 응답 토큰 ID [batch, response_len]
            response_attention_mask: 응답 어텐션 마스크 [batch, response_len]
            is_mol_token: mol token 위치 마스크 [batch, prompt_len]
            num_samples: Monte Carlo 샘플 수 (default: 128)

        Returns:
            torch.Tensor: 각 샘플의 평균 log-likelihood [batch]
        """
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        response_len = response_ids.shape[1]

        # 전체 시퀀스 구성: [prompt | response]
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        full_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=1)

        if is_mol_token is not None:
            is_mol_token_resp = torch.zeros((batch_size, response_len), device=self.device, dtype=torch.bool)
            full_is_mol_token = torch.cat([is_mol_token, is_mol_token_resp], dim=1)
        else:
            full_is_mol_token = None

        # 응답 영역 마스크 (response_attention_mask가 1인 위치)
        response_mask = response_attention_mask.bool()  # [batch, response_len]
        response_lengths = response_mask.sum(dim=1)  # [batch]

        # Monte Carlo 샘플링으로 log-likelihood 추정
        total_log_likelihood = torch.zeros(batch_size, device=self.device)

        for _ in range(num_samples):
            # Step 1: 각 배치별로 마스킹 비율 t ~ Uniform(0, 1) 샘플링
            t = torch.rand(batch_size, device=self.device)

            # Step 2: 응답 영역에서 t 비율만큼 랜덤하게 마스킹
            noisy_full_ids = full_ids.clone()
            mask_probs = torch.rand((batch_size, response_len), device=self.device)

            # 각 샘플별로 마스킹 비율 적용
            for b in range(batch_size):
                # 응답 영역 내에서만 마스킹
                valid_response = response_mask[b]
                should_mask = (mask_probs[b] < t[b]) & valid_response

                # 마스킹 적용 (prompt_len 이후가 response 영역)
                noisy_full_ids[b, prompt_len:][should_mask] = self.mask_token_id

            # Step 3: Forward pass
            current_embeds = self.llm_embed_tokens(noisy_full_ids)

            if graphs is not None:
                current_embeds, _, _ = self.inject_graph_embeds2input_embeds(
                    input_embeds=current_embeds,
                    is_mol_token=full_is_mol_token,
                    graphs=graphs
                )

            outputs = self.llm_model(
                inputs_embeds=current_embeds,
                attention_mask=full_attention_mask,
                return_dict=True
            )
            logits = outputs.logits  # [batch, seq_len, vocab_size]

            # Step 4: 마스킹된 위치의 log-probability 계산
            # 응답 영역의 logits만 추출
            response_logits = logits[:, prompt_len:, :]  # [batch, response_len, vocab_size]
            log_probs = F.log_softmax(response_logits, dim=-1)  # [batch, response_len, vocab_size]

            # 원본 응답 토큰의 log-probability 추출
            # gather를 사용해 각 위치의 정답 토큰에 대한 log-prob 추출
            target_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=response_ids.unsqueeze(-1)
            ).squeeze(-1)  # [batch, response_len]

            # 마스킹된 위치만 합산 (실제 응답 토큰이 있는 위치)
            # 현재 샘플에서 마스킹된 위치 = noisy_full_ids에서 mask_token_id인 위치
            masked_in_response = (noisy_full_ids[:, prompt_len:] == self.mask_token_id) & response_mask

            # 마스킹된 위치의 log-prob 합산
            sample_log_likelihood = (target_log_probs * masked_in_response.float()).sum(dim=1)

            # 마스킹된 토큰 수로 정규화 (0 division 방지)
            num_masked = masked_in_response.sum(dim=1).float()
            sample_log_likelihood = sample_log_likelihood / (num_masked + 1e-8)

            # 마스킹된 토큰이 없는 경우 (t≈0) 처리
            sample_log_likelihood = torch.where(
                num_masked > 0,
                sample_log_likelihood,
                torch.zeros_like(sample_log_likelihood)
            )

            total_log_likelihood += sample_log_likelihood

        # Monte Carlo 평균
        avg_log_likelihood = total_log_likelihood / num_samples

        return avg_log_likelihood

    @torch.no_grad()
    def compute_binary_prob_likelihood(
        self,
        graphs,
        input_ids,
        attention_mask,
        is_mol_token=None,
    ):
        """
        Binary classification (True/False)에 대한 확률을 Likelihood 비교로 계산

        짧은 응답(True/False)에 최적화된 버전:
        - 전체 응답을 마스킹하고 1회 forward pass로 log-likelihood 계산
        - 128번 Monte Carlo 샘플링은 긴 응답에만 의미 있음
        - 3~4 토큰짜리 응답에서는 전체 마스킹이 더 효율적이고 정확

        Args:
            graphs: 분자 그래프 (tuple)
            input_ids: 프롬프트 토큰 ID [batch, prompt_len]
            attention_mask: 프롬프트 어텐션 마스크 [batch, prompt_len]
            is_mol_token: mol token 위치 마스크 [batch, prompt_len]

        Returns:
            torch.Tensor: [P(False), P(True)] 확률 [batch, 2]
        """
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]

        # 후보 응답 토큰화 (Training target 형식과 일치시킴)
        # Training target: "<BOOLEAN> True </BOOLEAN><|eot_id|>" (공백 포함)
        # Likelihood 비교도 동일한 형식 사용해야 정확한 확률 추정 가능
        true_response = "<BOOLEAN> True </BOOLEAN>"
        false_response = "<BOOLEAN> False </BOOLEAN>"

        true_tokens = self.llm_tokenizer.encode(true_response, add_special_tokens=False)
        false_tokens = self.llm_tokenizer.encode(false_response, add_special_tokens=False)

        # 두 응답의 길이를 맞춤 (더 긴 쪽에 맞춰 padding)
        max_resp_len = max(len(true_tokens), len(false_tokens))

        # Padding (mask_token_id 사용 - 어차피 전체 마스킹할 것이므로)
        true_tokens_padded = true_tokens + [self.mask_token_id] * (max_resp_len - len(true_tokens))
        false_tokens_padded = false_tokens + [self.mask_token_id] * (max_resp_len - len(false_tokens))

        true_ids = torch.tensor([true_tokens_padded] * batch_size, device=self.device, dtype=torch.long)
        false_ids = torch.tensor([false_tokens_padded] * batch_size, device=self.device, dtype=torch.long)

        # 실제 토큰 위치 마스크 (padding 제외)
        true_valid_mask = torch.zeros((batch_size, max_resp_len), device=self.device, dtype=torch.bool)
        false_valid_mask = torch.zeros((batch_size, max_resp_len), device=self.device, dtype=torch.bool)
        true_valid_mask[:, :len(true_tokens)] = True
        false_valid_mask[:, :len(false_tokens)] = True

        # ================================================================
        # 전체 마스킹 후 1회 forward pass로 log-likelihood 계산
        # ================================================================

        def compute_full_mask_likelihood(response_ids, valid_mask):
            """전체 응답을 마스킹하고 log-likelihood 계산"""
            # 전체 시퀀스 구성: [prompt | masked_response]
            masked_response = torch.full_like(response_ids, self.mask_token_id)
            full_ids = torch.cat([input_ids, masked_response], dim=1)
            full_attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, max_resp_len), device=self.device, dtype=attention_mask.dtype)
            ], dim=1)

            if is_mol_token is not None:
                is_mol_token_resp = torch.zeros((batch_size, max_resp_len), device=self.device, dtype=torch.bool)
                full_is_mol_token = torch.cat([is_mol_token, is_mol_token_resp], dim=1)
            else:
                full_is_mol_token = None

            # Forward pass
            current_embeds = self.llm_embed_tokens(full_ids)

            if graphs is not None:
                current_embeds, _, _ = self.inject_graph_embeds2input_embeds(
                    input_embeds=current_embeds,
                    is_mol_token=full_is_mol_token,
                    graphs=graphs
                )

            outputs = self.llm_model(
                inputs_embeds=current_embeds,
                attention_mask=full_attention_mask,
                return_dict=True
            )
            logits = outputs.logits  # [batch, seq_len, vocab_size]

            # 응답 영역의 log-probability 계산
            response_logits = logits[:, prompt_len:, :]  # [batch, max_resp_len, vocab_size]
            log_probs = F.log_softmax(response_logits, dim=-1)

            # 원본 응답 토큰의 log-probability 추출
            target_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=response_ids.unsqueeze(-1)
            ).squeeze(-1)  # [batch, max_resp_len]

            # valid한 위치만 합산 (padding 제외)
            log_likelihood = (target_log_probs * valid_mask.float()).sum(dim=1)

            # 토큰 수로 정규화 (길이가 다른 응답 간 공정한 비교)
            num_tokens = valid_mask.sum(dim=1).float()
            log_likelihood = log_likelihood / (num_tokens + 1e-8)

            return log_likelihood

        # True/False 각각의 log-likelihood 계산
        true_log_likelihood = compute_full_mask_likelihood(true_ids, true_valid_mask)
        false_log_likelihood = compute_full_mask_likelihood(false_ids, false_valid_mask)

        # Log-likelihood를 확률로 변환 (softmax)
        # [batch, 2] where dim=1 is [P(False), P(True)]
        log_likelihoods = torch.stack([false_log_likelihood, true_log_likelihood], dim=1)
        probs = F.softmax(log_likelihoods, dim=1)

        return probs


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self