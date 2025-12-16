import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, logging
from model.blip2_opt import Blip2OPT, split_batch_by_components
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
import model.added_tokens as added_tokens
import numpy as np
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

    def get_lora_target_modules(self):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

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
        # 1. 추가할 토큰 리스트 취합
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
        
        # 2. 토크나이저에 추가
        num_added_toks = self.llm_tokenizer.add_tokens(special_tokens_list)
        if num_added_toks > 0:
            logger.info(f"[Token Check] Added {num_added_toks} special tokens to tokenizer.")
        else:
            logger.info("[Token Check] Special tokens already exist.")

        # 3. 모델 임베딩 크기 조정 (필수)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        
        # 4. <mol> 토큰 ID 저장 (임베딩 주입 시 사용)
        self.mol_token_id = self.llm_tokenizer.convert_tokens_to_ids(added_tokens.MOL_EMBEDDING)[0]
    
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
        
        eps = 1e-6
        t = torch.rand(batch_size, device=self.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, seq_len)

        is_answer = (labels != -100)
        
        mask_prob = torch.rand((batch_size, seq_len), device=self.device)
        masked_indices = (mask_prob < p_mask) & is_answer
        
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
        
        if masked_indices.sum() == 0:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            instance_loss = torch.zeros(batch_size, device=self.device)
        else:
            token_loss = loss_fct(logits.transpose(1, 2), input_ids)
            weighted_loss = token_loss * masked_indices.float() / p_mask
            answer_lengths = is_answer.sum(dim=1, keepdim=True).float()
            loss = weighted_loss.sum() / (answer_lengths.sum() + 1e-8) 
            instance_loss = weighted_loss.sum(dim=1) / (answer_lengths.squeeze(-1) + 1e-8)

        return {
            "loss": loss,
            "instance_loss": instance_loss,
            "logits": logits,
            "graph_avg_norm": graph_avg_norm,
            "moltoken_avg_norm": moltoken_avg_norm
        }

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
        **kwargs
    ):
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

    def extract_graph_feature(self, samples):
        return None, None, None
    
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self