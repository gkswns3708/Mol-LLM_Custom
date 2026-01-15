import os
import math
from typing import Any, Dict
import torch
from model.blip2_opt import Blip2OPT
from model.blip2_llama import Blip2Llama
from model.blip2_mistral import Blip2Mistral
from model.blip2_t5 import Blip2T5
from model.blip2_llada import Blip2LLaDA
import pytorch_lightning as pl
from torch import optim
from model.scheduler import (
    LinearWarmupCosineLRScheduler, 
    LinearWarmupStepLRScheduler, 
    LinearWarmupConstantLRScheduler, 
    WarmupStableDecayLRScheduler
)
import json
from model.help_funcs import (
    per_device_evaluate,
    total_device_evaluate,
    AttrDict,
    convert_logit2binary_prob,
)
from transformers import Adafactor
import json
from data_utils import (
    CLASSIFICATION_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
    REACTION_BENCHMARKS,
    id2task,
)
from transformers.utils import logging

from torch.nn import CrossEntropyLoss

logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}

    # try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)


def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1 :]
            if key == "":
                return value
            module_state_dict[key] = value
    return module_state_dict


def subset_batch_by_indices(batch, indices, device=None):
    """
    배치에서 특정 인덱스의 샘플만 추출하는 헬퍼 함수.
    혼합 배치에서 Classification/Generation 샘플을 분리할 때 사용.

    Args:
        batch: 원본 배치 (AttrDict 또는 dict-like)
        indices: 추출할 샘플 인덱스 리스트
        device: 텐서를 이동할 디바이스 (None이면 원본 유지)

    Returns:
        subset_batch: 인덱스에 해당하는 샘플만 포함한 새 배치
    """
    if len(indices) == 0:
        return None

    indices_tensor = torch.tensor(indices, dtype=torch.long)
    if device is not None:
        indices_tensor = indices_tensor.to(device)

    subset = AttrDict()

    for key in batch.keys():
        value = batch[key]
        if value is None:
            subset[key] = None
        elif isinstance(value, torch.Tensor):
            # 텐서인 경우 인덱싱
            if device is not None:
                subset[key] = value[indices_tensor].to(device)
            else:
                subset[key] = value[indices_tensor]
        elif isinstance(value, list):
            # 리스트인 경우 인덱싱
            subset[key] = [value[i] for i in indices]
        elif isinstance(value, dict):
            # Graph 데이터 등 dict인 경우 (torch_geometric Batch)
            # 이 경우는 복잡하므로 None 처리 (graph는 별도 처리 필요)
            subset[key] = value  # 일단 전체 전달 (필요시 별도 처리)
        else:
            # 기타 타입은 그대로 전달
            subset[key] = value

    return subset


class Blip2Stage3(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        to_be_removed = []
        for key, value in checkpoint["state_dict"].items():
            # graph encoder parameters
            if (
                "running_mean" in key
                or "running_var" in key
                or "num_batches_tracked" in key
                or "lora" in key
            ):
                continue

            # [CRITICAL FIX] modules_to_save (embed_tokens, lm_head, wte, ff_out)는 frozen이어도 저장
            # Stage 2 (Q-Former pretraining)에서 LLM이 frozen이어도, 새 vocab이 추가된
            # embed_tokens/lm_head는 반드시 보존해야 Stage 3에서 계속 학습 가능
            # LLaDA: wte (embedding), ff_out (lm_head)
            # PEFT modules_to_save: *.modules_to_save.default.* 패턴
            is_module_to_save = (
                "embed_tokens" in key or "lm_head" in key or  # 일반 모델
                "wte" in key or "ff_out" in key or  # LLaDA 모델
                "modules_to_save" in key  # PEFT modules_to_save wrapper
            )

            try:
                # modules_to_save가 아니고, frozen이면 삭제
                if not self.get_parameter(key).requires_grad and not is_module_to_save:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint["state_dict"].pop(key)

        if hasattr(self, "task_specific_chosen_reward"):
            checkpoint[f"task_specific_chosen_reward"] = (
                self.task_specific_chosen_reward
            )

        # DISABLED: self.log_model_parameters()
        # This logs 8B+ parameters to WandB/TensorBoard which takes hours
        # Only enable for debugging specific parameter issues

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "task_specific_chosen_reward" in checkpoint:
            self.task_specific_chosen_reward = checkpoint["task_specific_chosen_reward"]

        # [CRITICAL FIX] 체크포인트 로드 후 modules_to_save 모듈을 강제로 trainable 설정
        # 이전 체크포인트가 requires_grad=False로 저장되었을 수 있으므로
        logger.info("\n" + "="*70)
        logger.info("[CHECKPOINT LOAD FIX] Re-enabling gradients for modules_to_save...")
        self._fix_modules_to_save_gradients()
        logger.info("="*70 + "\n")

    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)
        self.args = args
        self.debug = getattr(args, 'debug', False)
        if self.debug:
            print(args, " - args")
        self.num_beams = args.num_beams
        self.gen_max_len = args.gen_max_len
        self.min_len = args.min_len
        self.tune_llm = args.tune_llm
        self.on_second_stage = False
        # set strict_loading to False to load model in a lightweight way
        self.strict_loading = False
        # 학습 완료 시점의 global_step을 저장 (test 시 파일명에 사용)
        self._trained_global_step = None

        # [Fix 2.2] Gradient 로깅을 위한 파라미터 캐싱 (오버헤드 최소화)
        self._embed_tokens_param = None
        self._lm_head_param = None

        # Weight Norm Logging 설정
        self._log_weight_norm_layers = getattr(args, 'log_weight_norm_layers', [])
        self._log_weight_norm_interval = getattr(args, 'log_weight_norm_interval', 100)
        self._weight_norm_param_cache = {}  # layer별 파라미터 캐싱
        self._initial_weight_norms = {}  # 초기 weight norm 저장 (변화량 추적)

        # 상위 빈도 SELFIES 토큰 norm 추적 설정
        self._top_selfies_tokens = getattr(args, 'top_selfies_tokens', [
            '[C]', '[O]', '[Branch1]', '[Ring1]', '[=C]',
            '[=Branch1]', '[N]', '[=O]', '[C@H1]', '[Ring2]',
            '[C@@H1]', '[Branch2]', '[=N]', '[#Branch1]', '[/C]'
        ])
        self._top_selfies_token_ids = None  # 토큰 ID 캐싱 (첫 호출 시 초기화)
        self._top_selfies_initial_norms = {}  # 초기 norm 저장
        if "galactica" in args.llm_model:
            blip2model = Blip2OPT
        elif "llama" in args.llm_model:
            blip2model = Blip2Llama
        elif "mistral" in args.llm_model:
            blip2model = Blip2Mistral
        elif "t5" in args.llm_model:
            blip2model = Blip2T5
        elif "llada" in args.llm_model or "LLaDA" in args.llm_model: # [추가]
            blip2model = Blip2LLaDA
        else:
            raise NotImplementedError()

        self.blip2model = blip2model(args)
        self.tokenizer = self.blip2model.init_tokenizer()
        self.save_hyperparameters(args)

    def load_from_stage1_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        graph_encoder_dict = get_module_state_dict(
            state_dict, "blip2qformer.graph_encoder"
        )
        qformer_dict = get_module_state_dict(state_dict, "blip2qformer.Qformer")
        ln_graph_dict = get_module_state_dict(state_dict, "blip2qformer.ln_graph")
        qs_weight = get_module_state_dict(state_dict, "blip2qformer.query_tokens")
        load_ignore_unexpected(self.blip2model.Qformer, qformer_dict)
        self.blip2model.graph_encoder.load_state_dict(graph_encoder_dict)
        self.blip2model.ln_graph.load_state_dict(ln_graph_dict)
        self.blip2model.query_tokens.data.copy_(qs_weight)
        return self

    def configure_optimizers(self):
        """
        각 파라미터 그룹별로 다른 Learning Rate를 적용:
        1. LoRA: 2e-4 (args.lr_lora, default: 2e-4)
        2. WTE (Token Embedding Layer):
           - 2.1 기존 Vocab embedding: 1e-5 (args.lr_embed_orig, default: 1e-5)
           - 2.2 새로 추가되는 Vocab embedding: 1e-4 (args.lr_embed_new, default: 1e-4)
        3. Classifier (LM Head / ff_out):
           - 3.1 기존 vocab weight matrix: 1e-5 (args.lr_head_orig, default: 1e-5)
           - 3.2 새로 추가되는 Vocab weight matrix: 1e-4 (args.lr_head_new, default: 1e-4)
        4. 기타 파라미터: args.init_lr
        """
        # LR 설정 (args에서 가져오거나 기본값 사용)
        lr_lora = getattr(self.args, 'lr_lora', 2e-4)
        lr_embed_orig = getattr(self.args, 'lr_embed_orig', 1e-5)
        lr_embed_new = getattr(self.args, 'lr_embed_new', 1e-4)
        lr_head_orig = getattr(self.args, 'lr_head_orig', 1e-5)
        lr_head_new = getattr(self.args, 'lr_head_new', 1e-4)
        lr_other = getattr(self.args, 'init_lr', lr_lora)  # 기타 파라미터는 lr_lora와 동일하게

        # Original vocab size (LLaDA Llama-3 8B 기준, args에서 오버라이드 가능)
        original_vocab_size = getattr(self.args, 'original_vocab_size', 126349)

        # 파라미터 그룹 분류
        params_lora = []
        params_embed_orig = []
        params_embed_new = []
        params_head_orig = []
        params_head_new = []
        params_other = []

        # Embedding/LM Head 파라미터 찾기 및 분리
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            name_lower = name.lower()

            # 1. LoRA 파라미터
            if 'lora' in name_lower:
                params_lora.append(param)
                if self.debug:
                    print(f"  [Optimizer] LoRA: {name}")

            # 2. WTE (Embedding) 파라미터 - wte 또는 embed_tokens
            elif ('wte' in name_lower or 'embed_tokens' in name_lower) and 'original_module' not in name_lower:
                # Embedding weight를 original/new vocab으로 분리하여 wrapper로 처리
                # 실제로는 하나의 weight지만, 학습 시 다른 LR 적용을 위해 별도 처리 필요
                # 여기서는 전체 파라미터를 등록하고, scheduler에서 처리
                params_embed_orig.append({'param': param, 'name': name, 'split_idx': original_vocab_size})
                if self.debug:
                    print(f"  [Optimizer] Embed (will split): {name}, shape={param.shape}")

            # 3. Classifier (LM Head / ff_out) 파라미터
            elif ('lm_head' in name_lower or ('ff_out' in name_lower and 'blocks' not in name_lower)) and 'original_module' not in name_lower:
                params_head_orig.append({'param': param, 'name': name, 'split_idx': original_vocab_size})
                if self.debug:
                    print(f"  [Optimizer] Head (will split): {name}, shape={param.shape}")

            # 4. 기타 파라미터
            else:
                params_other.append(param)
                if self.debug:
                    print(f"  [Optimizer] Other: {name}")

        # Embedding과 Head 파라미터의 original/new 분리 처리
        # PyTorch optimizer는 같은 파라미터를 여러 그룹에 넣을 수 없으므로,
        # 전체 파라미터를 하나의 그룹에 넣되, 커스텀 훅으로 gradient scaling 적용
        embed_params_all = [p['param'] for p in params_embed_orig]
        head_params_all = [p['param'] for p in params_head_orig]

        # Embedding/Head의 original/new vocab 인덱스 정보 저장 (gradient scaling용)
        self._embed_head_split_info = {
            'original_vocab_size': original_vocab_size,
            'embed_params': [(p['param'], p['name']) for p in params_embed_orig],
            'head_params': [(p['param'], p['name']) for p in params_head_orig],
            'lr_ratio_embed': lr_embed_new / lr_embed_orig if lr_embed_orig > 0 else 1.0,
            'lr_ratio_head': lr_head_new / lr_head_orig if lr_head_orig > 0 else 1.0,
        }

        if self.debug:
            print(f"\n[Optimizer Summary]")
            if self.debug:
                print(f"  LoRA params: {len(params_lora)} (lr={lr_lora})")
            if self.debug:
                print(f"  Embed params: {len(embed_params_all)} (orig_lr={lr_embed_orig}, new_lr={lr_embed_new})")
            if self.debug:
                print(f"  Head params: {len(head_params_all)} (orig_lr={lr_head_orig}, new_lr={lr_head_new})")
            if self.debug:
                print(f"  Other params: {len(params_other)} (lr={lr_other})")
            if self.debug:
                print(f"  Original vocab size: {original_vocab_size}")

        # 로깅
        logger.info("="*70)
        logger.info("[Optimizer] Parameter groups with different learning rates:")
        logger.info(f"  1. LoRA:        {len(params_lora):4d} params, lr={lr_lora}")
        logger.info(f"  2. Embed (orig): lr={lr_embed_orig} (idx < {original_vocab_size})")
        logger.info(f"  3. Embed (new):  lr={lr_embed_new} (idx >= {original_vocab_size})")
        logger.info(f"  4. Head (orig):  lr={lr_head_orig} (idx < {original_vocab_size})")
        logger.info(f"  5. Head (new):   lr={lr_head_new} (idx >= {original_vocab_size})")
        logger.info(f"  6. Other:       {len(params_other):4d} params, lr={lr_other}")
        logger.info("="*70)

        if self.args.optimizer == "adafactor":
            if self.debug:
                print("Using adafactor optimizer")
            # Adafactor는 param group을 지원하지만, 여기서는 기본 설정 유지
            optimizer = Adafactor(
                self.parameters(),
                lr=1e-3,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
            self.scheduler = None
        else:
            self.trainer.fit_loop.setup_data()

            # 파라미터 그룹 구성
            # Embed/Head의 경우 base_lr로 orig LR을 사용하고, new vocab은 gradient scaling으로 처리
            param_groups = []

            if params_lora:
                param_groups.append({
                    'params': params_lora,
                    'lr': lr_lora,
                    'name': 'lora'
                })

            if embed_params_all:
                param_groups.append({
                    'params': embed_params_all,
                    'lr': lr_embed_orig,
                    'name': 'embed',
                    'lr_new': lr_embed_new,  # 커스텀 필드: new vocab용 LR
                })

            if head_params_all:
                param_groups.append({
                    'params': head_params_all,
                    'lr': lr_head_orig,
                    'name': 'head',
                    'lr_new': lr_head_new,  # 커스텀 필드: new vocab용 LR
                })

            if params_other:
                param_groups.append({
                    'params': params_other,
                    'lr': lr_other,
                    'name': 'other'
                })

            optimizer = optim.AdamW(
                param_groups,
                weight_decay=self.args.weight_decay,
            )

            self.steps_per_epoch = (
                len(self.trainer.train_dataloader) / self.args.accumulate_grad_batches
            )

            max_step = int(self.args.max_epochs * self.steps_per_epoch)
            if hasattr(self.args, "warmup_steps") and self.args.warmup_steps > 0:
                warmup_steps = self.args.warmup_steps
            # 2. 아니면 args.warmup_epochs를 기준으로 계산
            elif hasattr(self.args, "warmup_epochs") and self.args.warmup_epochs > 0:
                warmup_steps = int(self.steps_per_epoch * self.args.warmup_epochs)
            # 3. 둘 다 없으면 0
            else:
                warmup_steps = 0

            if self.args.scheduler == "linear_warmup_cosine_lr":
                self.scheduler = LinearWarmupCosineLRScheduler(
                    optimizer=optimizer,
                    max_step=max_step,
                    min_lr=self.args.min_lr,
                    init_lr=self.args.init_lr,
                    warmup_steps=warmup_steps,
                    warmup_start_lr=self.args.warmup_lr,
                )
            elif self.args.scheduler == "linear_warmup_step_lr":
                self.scheduler = LinearWarmupStepLRScheduler(
                    optimizer,
                    self.args.max_epochs,
                    self.args.min_lr,
                    self.args.init_lr,
                    self.args.lr_decay_rate,
                    self.args.warmup_lr,
                    self.args.warmup_steps,
                )
            elif self.args.scheduler == "None":
                self.scheduler = None
            elif self.args.scheduler == "warmup_stable_decay_lr":
                # config에서 직접 비율 지정
                min_lr_ratio = getattr(self.args, 'min_lr_ratio', 0.1)
                decay_ratio = getattr(self.args, 'decay_ratio', 0.1)

                self.scheduler = WarmupStableDecayLRScheduler(
                    optimizer=optimizer,
                    max_step=max_step,
                    init_lr=lr_lora,  # 기준값으로 lr_lora 사용
                    min_lr=lr_lora * min_lr_ratio,
                    warmup_steps=warmup_steps,
                    decay_ratio=decay_ratio,
                    min_lr_ratio=min_lr_ratio,
                )
            else:
                raise NotImplementedError()
        return optimizer

    def save_predictions(
        self,
        predictions,
        targets,
        tasks,
        prompts,
        input_mol_strings,
        token_ids=None,  #! 추가해봄. (optional로 변경)
        probs=None,
        filename="predictions.json",
    ):
        assert len(predictions) == len(targets)
        assert len(predictions) == len(tasks)
        assert len(predictions) == len(prompts)
        assert len(predictions) == len(input_mol_strings)
        if probs is not None:
            assert len(predictions) == len(probs)
        instances = []
        for i in range(len(predictions)):
            instance = {
                "task": tasks[i],
                "prediction": predictions[i],
                "target": targets[i],
                "prompt": prompts[i],
                "input_mol_strings": input_mol_strings[i],
            }
            if token_ids is not None:
                instance["token_ids"] = token_ids[i]
            if tasks[i] in CLASSIFICATION_BENCHMARKS and probs is not None:
                instance["prob"] = probs[i]
            instances.append(instance)
        os.makedirs(self.logger.log_dir, exist_ok=True)

        with open(os.path.join(self.logger.log_dir, filename), "w") as f:
            json.dump(instances, f, ensure_ascii=False, indent=4)

    def on_test_epoch_start(self) -> None:
        # test 시작 시 학습된 global_step 저장 (trainer.test()가 step을 리셋하기 전)
        if self._trained_global_step is None:
            self._trained_global_step = self.global_step
        self.on_evaluation_epoch_start()

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluation_step(batch, batch_idx, dataloader_idx, mode="test")

    def on_test_epoch_end(self):
        self.on_evaluation_epoch_end(mode="test")

    def on_validation_epoch_start(self) -> None:
        self.on_evaluation_epoch_start()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluation_step(batch, batch_idx, dataloader_idx, mode="val")

    def on_validation_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(mode="val")

    def apply_separated_stage(self):
        if (
            self.trainer.global_step
            >= self.args.second_stage_start_epoch * self.steps_per_epoch
            and not self.on_second_stage
        ):
            self.blip2model.set_params_requires_grads(
                model=self.blip2model.llm_model,
                keyword="lora",
                grad=True,
                IsPrint=False,
            )
            self.on_second_stage = True
            if self.debug:
                print("set lora weights trainable")

    # a batch of 3 tuples
    # sft tuple (gw, sw, q, y)
    # molpo chosen tuple (gw, sl, q, y)
    # molpo rejected tuple (gl, sl, q, y)
    def get_total_molpo_loss(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        molpo_labels: torch.LongTensor,
        instance_loss: torch.FloatTensor,
        tasks,
        is_train=True,
        molpo_batch_division=2,
        config=None
    ):
        out = concatenated_forward(
            all_logits=logits,
            all_labels=molpo_labels,
            label_pad_token_id=-100,
            instance_loss=instance_loss,
            molpo_batch_division=molpo_batch_division,
            config=config
        )
        sft_instance_loss = out["sft_instance_loss"]

        policy_chosen_logps = out["chosen_logps"]
        chosen_loss_mask = out["chosen_loss_mask"]

        policy_rejected_logps = out["rejected_logps"]
        rejected_loss_mask = out["rejected_loss_mask"]

        # calculate rewards
        chosen_rewards = self.args.beta * policy_chosen_logps
        rejected_rewards = self.args.beta * policy_rejected_logps

        # calculate sft loss
        sft_loss_bug = None
        if molpo_batch_division == 2:
            assert (
                labels.shape[0] % molpo_batch_division == 0
            ), "batch_size(labels.shape[0]) should be divisible by molpo_batch_division"
            sft_loss_mask = labels[: labels.shape[0] // 2, :] != -100
            sft_loss = (sft_instance_loss * sft_loss_mask.sum(-1))[
                sft_loss_mask.sum(-1) > 0
            ].sum() / sft_loss_mask.sum()

        elif molpo_batch_division == 3:
            policy_sft_logps = out["sft_logps"]
            sft_loss_mask = out["sft_loss_mask"]
            sft_rewards = self.args.beta * policy_sft_logps

            sft_loss = (sft_instance_loss * sft_loss_mask.sum(-1))[
                sft_loss_mask.sum(-1) > 0
            ].sum() / sft_loss_mask.sum()

        # update and get task specific chosen rewards
        if is_train:
            self.update_task_specific_chosen_rewards_avg(
                chosen_rewards=chosen_rewards,
                tasks=tasks,
                task_specific_outputs=self.task_specific_chosen_reward,
                alpha=0.99,
            )

        # get average chosen rewards
        avg_chosen_rewards_list = []
        for task in tasks:
            if task in self.task_specific_chosen_reward:
                avg_chosen_rewards_list.append(self.task_specific_chosen_reward[task])
            else:
                avg_chosen_rewards_list.append(0.0)

        avg_chosen_rewards = torch.tensor(
            avg_chosen_rewards_list,
            device=logits.device,
        )

        # calculate molpo loss
        loss_molpo, losses_molpo = molpo_loss(
            chosen_rewards=chosen_rewards,
            chosen_loss_mask=chosen_loss_mask,
            rejected_rewards=rejected_rewards,
            rejected_loss_mask=rejected_loss_mask,
            loss_type=self.args.loss_type,
            beta=self.args.beta,
            gamma_beta_ratio=self.args.gamma_beta_ratio,
            molpo_lambda=self.args.molpo_lambda,
            avg_chosen_rewards=avg_chosen_rewards,
            margin_clip_scale=self.args.margin_clip_scale,
        )

        # calculate anchor losses
        anchor_rejected_losses = anchor_loss(
            avg_chosen_rewards=avg_chosen_rewards,
            rejected_rewards=rejected_rewards,
            rejected_lambda=self.args.rejected_lambda,
            loss_type=self.args.anc_loss_type,
        )
        # apply loss clipping to anchor losses
        anchor_rejected_loss = anchor_rejected_losses.mean()
        if self.args.anc_reject_clip > 0:
            anchor_rejected_loss = torch.clamp(
                anchor_rejected_loss, max=self.args.anc_reject_clip
            )
        if self.args.molpo_weight > 0.0:
            loss = (
                self.args.sft_weight * sft_loss
                + self.args.molpo_weight * loss_molpo
                + self.args.anc_rejected_weight * anchor_rejected_loss
            )
        else:
            loss = self.args.sft_weight * sft_loss

        if torch.isnan(loss):
            assert not torch.isnan(loss), "loss is nan"

        metrics = {}
        metrics[f"rewards/chosen"] = chosen_rewards.cpu()
        metrics[f"rewards/rejected"] = rejected_rewards.cpu()
        metrics[f"rewards/accuracies"] = (
            (chosen_rewards > rejected_rewards).float().cpu()
        )
        metrics[f"rewards/margins"] = (chosen_rewards - rejected_rewards).cpu()

        metrics[f"logps/chosen"] = policy_chosen_logps.clone().detach().cpu()
        metrics[f"logps/rejected"] = policy_rejected_logps.clone().detach().cpu()

        metrics[f"sft_loss"] = sft_loss.clone().detach().cpu()

        metrics[f"instance_loss"] = sft_instance_loss.clone().detach().cpu()
        metrics[f"molpo_loss"] = losses_molpo.clone().detach().cpu()
        metrics[f"anchor_loss/rejected"] = anchor_rejected_losses.clone().detach().cpu()
        metrics["rl_over_bar_rw"] = rejected_rewards.cpu() / avg_chosen_rewards.cpu()

        if molpo_batch_division == 3:
            metrics[f"rewards/sft"] = sft_rewards.cpu()
            metrics["logps/sft"] = policy_sft_logps.clone().detach().cpu()

        return loss, metrics

    def update_task_specific_chosen_rewards_avg(
        self,
        chosen_rewards,
        tasks,
        task_specific_outputs: Dict[str, Dict[str, list]],
        alpha=0.99,
    ):
        for i in range(chosen_rewards.shape[0]):
            if torch.isnan(chosen_rewards[i]):
                continue

            task = tasks[i]
            task_specific_outputs.setdefault(task, chosen_rewards[i].item())
            task_specific_outputs[task] = (
                alpha * task_specific_outputs[task]
                + (1 - alpha) * chosen_rewards[i].item()
            )

    def training_step(self, batch, batch_idx):
        if self.args.llava_pretraining:
            self.apply_separated_stage()

        if self.scheduler:
            self.scheduler.step(cur_step=self.trainer.global_step)

        # [중요] 전역 id2task 함수 사용 (함수 내 import 제거)
        tasks = [id2task(task_id.item()) for task_id in batch.tasks]

        outputs = self.blip2model(batch)
        
        # [요청하신 값 추출 방식]
        # ModelOutput은 dict를 상속하지 않으므로 hasattr로 체크
        if hasattr(outputs, 'pop'):
            logits = outputs.pop("logits", None)
            loss = outputs.pop("loss", None)
            # instance_loss는 MolPO 계산에 필요하므로 보존하거나 get으로 접근
        else:
            logits = None
            loss = outputs

        # =================================================================
        # [DEBUG] NaN / Inf 발생 시 상세 디버깅 정보 및 샘플 출력
        # =================================================================
        if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
            if self.debug:
                print(f"\n{'='*20} [CRITICAL ERROR] Loss is NaN/Inf {'='*20}")
            if self.debug:
                print(f"Global Step: {self.global_step}, Batch Index: {batch_idx}")
            if self.debug:
                print(f"Current Batch Tasks: {tasks}")
            
            if self.debug:
                print("\n[Possible Causes Candidates]")
            if self.debug:
                print("1. Learning Rate Explosion: 초기 LR이 너무 높거나 Warmup이 부족할 수 있습니다.")
            if self.debug:
                print("2. Gradient Explosion: gradient_clip_val 설정을 확인하세요.")
            if self.debug:
                print("3. Invalid Data/Labels: Label이 전부 -100이거나 Input에 NaN이 있을 수 있습니다.")
            if self.debug:
                print("4. Logit Instability: 모델 출력 Logit이 발산했는지 확인하세요.")

            # 1. Label 통계 확인
            if "labels" in batch:
                labels = batch.labels
                valid_labels = (labels != -100).sum()
                if self.debug:
                    print(f"\n[Label Statistics] Total: {labels.numel()}, Valid(!=-100): {valid_labels.item()}")
                if valid_labels == 0:
                    if self.debug:
                        print("!!! Warning: All labels are -100 (Ignore Index). Loss becomes 0 or NaN. !!!")
            
            # 2. Logits 통계 확인
            if logits is not None:
                if self.debug:
                    print(f"\n[Logits Statistics] Max: {logits.max().item()}, Min: {logits.min().item()}, Mean: {logits.mean().item()}")
                if torch.isnan(logits).any():
                    if self.debug:
                        print("!!! Logits contain NaN values !!!")
            
            # 3. 입력 샘플 디코딩하여 출력 (데이터 문제 확인용)
            try:
                if self.debug:
                    print("\n[Sample Input Decoding]")
                tokenizer = self.blip2model.llm_tokenizer
                # batch 객체 구조에 따라 input_ids 가져오기
                input_ids = batch.input_ids if hasattr(batch, 'input_ids') else batch.prompt_input_ids
                if input_ids is not None:
                    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                    if self.debug:
                        print(f"Decoded Input (truncated 500 chars): {decoded[:500]} ...")
            except Exception as e:
                if self.debug:
                    print(f"Failed to decode sample: {e}")
            
            if self.debug:
                print("="*60 + "\n")
            # 필요 시 에러를 발생시켜 학습 중단: raise ValueError("Training stopped due to NaN")
        for i, t in enumerate(tasks):
            if "bace" in t or "chebi" in t:
                valid_len = (batch.labels[i] != -100).sum()
                if valid_len == 0:
                    if self.debug:
                        print(f"[WARNING] Task {t} has NO valid labels (all -100). This causes NaN instance loss.")
                if hasattr(self.args, "train_molpo") and self.args.train_molpo:
                    compute_loss_context_manager = torch.amp.autocast
                    len_tuple = batch.labels.shape[0] // self.args.molpo_batch_division
                    tasks = tasks[:len_tuple]

                    with compute_loss_context_manager(device_type="cuda"):
                        # outputs가 dict인 경우 instance_loss 가져오기
                        inst_loss = outputs.get("instance_loss", None) if isinstance(outputs, dict) else None
                        
                        loss, metrics = self.get_total_molpo_loss(
                            logits=logits,
                            labels=batch.labels,
                            molpo_labels=batch.molpo_labels,
                            instance_loss=inst_loss,
                            tasks=tasks,
                            is_train=True,
                            molpo_batch_division=self.args.molpo_batch_division,
                            config=self.args
                        )
                    outputs.update(metrics)

                    if "graph_avg_norm" in outputs:
                        graph_keys = ["graph_avg_norm", "moltoken_avg_norm"]
                        for k in graph_keys:
                            avg_norm = outputs.pop(k)
                            if self.args.molpo_batch_division == 2:
                                chosen_avg_norm = avg_norm[:len_tuple]
                                reject_avg_norm = avg_norm[len_tuple:]
                            elif self.args.molpo_batch_division == 3:
                                sft_avg_norm = avg_norm[:len_tuple]
                                chosen_avg_norm = avg_norm[len_tuple : 2 * len_tuple]
                                reject_avg_norm = avg_norm[2 * len_tuple :]

                                outputs[f"{k}/sft"] = sft_avg_norm

                            outputs[f"{k}/chosen"] = chosen_avg_norm
                            outputs[f"{k}/reject"] = reject_avg_norm

        # 5개 LR 그룹 모두 로깅
        self._log_all_learning_rates()

        # loss를 progress bar에 표시
        loss_value = loss.clone().detach().item() if loss is not None else 0.0
        self.log(
            "loss",
            loss_value,
            batch_size=self.args.batch_size,
            sync_dist=False,
            prog_bar=True,
        )

        for k, v in outputs.items():
            # logits는 너무 큰 텐서이므로 제외
            if k in ["logits"]:
                continue

            # None 체크 추가
            if v is None:
                continue

            # new_token_debug는 별도 처리
            if k == "new_token_debug":
                continue

            val_to_log = v.mean() if isinstance(v, torch.Tensor) else v

            # instance_loss는 progress bar에도 표시
            show_in_prog_bar = (k == "instance_loss")

            self.log(
                f"train/{k}",
                float(val_to_log),
                batch_size=self.args.batch_size,
                sync_dist=False,
                on_step=True,
                on_epoch=False,
                logger=True,
                prog_bar=show_in_prog_bar,
            )

        # ==============================================================================
        # [NEW] 새 토큰 디버깅 정보 wandb 로깅
        # ==============================================================================
        if "new_token_debug" in outputs and outputs["new_token_debug"]:
            debug_info = outputs["new_token_debug"]
            for k, v in debug_info.items():
                if v is not None and not isinstance(v, str):
                    self.log(
                        f"new_token_debug/{k}",
                        float(v),
                        batch_size=self.args.batch_size,
                        sync_dist=False,
                        on_step=True,
                        on_epoch=False,
                        logger=True,
                    )

        if not hasattr(self, "task_specific_outputs"):
            self.task_specific_outputs = {}
        self.task_specific_logging(
            outputs=outputs,
            tasks=tasks,
            mode="train",
            epoch_end=False,
            task_specific_outputs=self.task_specific_outputs,
            num_moving_samples=32,
        )
        if self.args.train_molpo:
            # bar r logging
            for k, v in self.task_specific_chosen_reward.items():
                self.log(
                    f"train/{k}/bar_reward",
                    v,
                    batch_size=self.args.batch_size,
                    sync_dist=False,
                )

        # [Fix 2.2] embed_tokens 및 lm_head gradient 로깅
        self._log_embedding_gradients()

        # Weight Norm Logging (설정된 interval마다, batch_idx 기준)
        self._log_weight_norms(batch_idx)

        # [Fix 2.3] Training sample token-level logging
        if self.global_step % self.trainer.log_every_n_steps == 0:
            self._log_sample_predictions(batch, outputs, tasks, batch_idx, mode="train")

        return loss

    def _cache_critical_params(self):
        """첫 실행 시 한 번만 embed_tokens/lm_head 파라미터 찾기 (오버헤드 최소화)"""
        if not hasattr(self.blip2model, 'llm_model'):
            return

        for name, param in self.blip2model.llm_model.named_parameters():
            if 'embed_tokens' in name and self._embed_tokens_param is None:
                self._embed_tokens_param = param
            if 'lm_head' in name and self._lm_head_param is None:
                self._lm_head_param = param

    def _log_all_learning_rates(self):
        """5개 LR 그룹 모두 wandb에 로깅

        - lr/lora: LoRA 파라미터 LR
        - lr/embed_orig: 기존 vocab embedding LR
        - lr/embed_new: 새 vocab embedding LR (effective)
        - lr/head_orig: 기존 vocab head LR
        - lr/head_new: 새 vocab head LR (effective)
        """
        optimizer = self.trainer.optimizers[0]

        # param_groups 순서: [lora, embed, head, other] (configure_optimizers에서 설정)
        for group in optimizer.param_groups:
            group_name = group.get('name', 'unknown')
            base_lr = group['lr']

            if group_name == 'lora':
                self.log("lr/lora", base_lr, batch_size=self.args.batch_size, sync_dist=False)
            elif group_name == 'embed':
                # embed_orig는 base LR, embed_new는 lr_new 필드 사용
                self.log("lr/embed_orig", base_lr, batch_size=self.args.batch_size, sync_dist=False)
                lr_new = group.get('lr_new', base_lr)
                self.log("lr/embed_new", lr_new, batch_size=self.args.batch_size, sync_dist=False)
            elif group_name == 'head':
                # head_orig는 base LR, head_new는 lr_new 필드 사용
                self.log("lr/head_orig", base_lr, batch_size=self.args.batch_size, sync_dist=False)
                lr_new = group.get('lr_new', base_lr)
                self.log("lr/head_new", lr_new, batch_size=self.args.batch_size, sync_dist=False)
            elif group_name == 'other':
                self.log("lr/other", base_lr, batch_size=self.args.batch_size, sync_dist=False)

        # 기존 'lr' 키도 유지 (backward compatibility, LoRA LR 사용)
        lora_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
        self.log("lr", lora_lr, batch_size=self.args.batch_size, sync_dist=False)

    def _log_embedding_gradients(self):
        """embed_tokens 및 lm_head의 gradient norm 로깅"""
        # 첫 실행 시에만 파라미터 캐싱
        if self._embed_tokens_param is None or self._lm_head_param is None:
            self._cache_critical_params()
            if self._embed_tokens_param is None:  # 여전히 None이면 스킵
                return

        # Gradient norm 계산
        embed_grad_norm = 0.0
        lm_head_grad_norm = 0.0

        if self._embed_tokens_param.grad is not None:
            embed_grad_norm = self._embed_tokens_param.grad.norm(2).item()

        if self._lm_head_param.grad is not None:
            lm_head_grad_norm = self._lm_head_param.grad.norm(2).item()

        # WandB/TensorBoard에 로깅
        self.log("train/embed_tokens_grad_norm", embed_grad_norm,
                 batch_size=self.args.batch_size, sync_dist=False)
        self.log("train/lm_head_grad_norm", lm_head_grad_norm,
                 batch_size=self.args.batch_size, sync_dist=False)

    def _cache_weight_norm_params(self):
        """Weight norm 로깅을 위한 파라미터 캐싱 (첫 호출 시 한 번만)"""
        if self._weight_norm_param_cache:
            return  # 이미 캐싱됨

        if not hasattr(self.blip2model, 'llm_model'):
            return

        import logging
        log = logging.getLogger(__name__)

        # 먼저 모든 파라미터 이름 출력 (디버깅용, 첫 호출 시만)
        log.info("\n" + "=" * 70)
        log.info("[Weight Norm] Scanning all trainable parameters...")
        log.info("=" * 70)

        all_param_names = []
        for name, param in self.blip2model.llm_model.named_parameters():
            if param.requires_grad:
                all_param_names.append(name)
        for name, param in self.blip2model.named_parameters():
            if param.requires_grad and name not in all_param_names:
                all_param_names.append(f"blip2model.{name}")

        log.info(f"Total trainable params: {len(all_param_names)}")
        log.info("Sample param names (first 20):")
        for name in all_param_names[:20]:
            log.info(f"  - {name}")
        if len(all_param_names) > 20:
            log.info(f"  ... and {len(all_param_names) - 20} more")
        log.info("-" * 70)

        # config에 적은 이름을 직접 패턴으로 사용
        # 예: ["wte", "ff_out", "lora"] → wte, ff_out, lora_A/lora_B 파라미터 매칭
        # "lora"는 특수 처리: lora_A, lora_B 둘 다 매칭
        #
        # [IMPORTANT] PEFT modules_to_save 처리:
        # - PEFT가 modules_to_save로 지정된 모듈을 ModulesToSaveWrapper로 래핑
        # - 실제 trainable weights는 "modules_to_save.default" 안에 있음
        # - original_module은 frozen 상태로 유지됨
        # - 예: wte.modules_to_save.default.weight (trainable)
        #       wte.original_module.weight (frozen)

        # 각 layer 패턴에 해당하는 파라미터 수집
        # [확장된 패턴 매칭]
        # - "lora": lora_A, lora_B 매칭
        # - "layers.0": model.layers.0.* 전체 매칭
        # - "layers.0.self_attn": model.layers.0.self_attn.* 매칭
        # - "blocks.0.attn_out": LLaDA alias → layers.0.self_attn.o_proj 매칭
        # - "o_proj": 모든 레이어의 o_proj 매칭
        #
        # Alias 매핑 (사용자 친화적 이름 → 실제 모델 경로)
        LAYER_ALIASES = {
            "blocks.0.attn_out": "layers.0.self_attn.o_proj",
            "blocks.0.attn": "layers.0.self_attn",
            "blocks.0.mlp": "layers.0.mlp",
            "blocks.0": "layers.0",
            "attn_out": "o_proj",  # LLaDA/Llama에서 attention output projection
        }

        for layer_name in self._log_weight_norm_layers:
            # Alias 변환
            resolved_pattern = LAYER_ALIASES.get(layer_name, layer_name)

            # lora는 lora_A, lora_B 둘 다 매칭
            if layer_name == "lora":
                patterns = ["lora_A", "lora_B"]
            else:
                patterns = [resolved_pattern]

            params = []
            # LLM 모델 파라미터
            for name, param in self.blip2model.llm_model.named_parameters():
                if any(p in name for p in patterns):
                    # [PEFT FIX] modules_to_save의 경우 original_module은 제외하고
                    # modules_to_save.default만 포함 (실제 trainable weights)
                    if "original_module" in name:
                        continue  # frozen copy, skip
                    params.append((name, param))

            # blip2model 직접 파라미터도 검색 (graph_encoder, opt_proj, ln_graph 등)
            for name, param in self.blip2model.named_parameters():
                # llm_model 하위는 이미 위에서 처리했으므로 제외
                if "llm_model" in name:
                    continue
                if any(p in name for p in patterns):
                    if "original_module" in name:
                        continue  # frozen copy, skip
                    params.append((name, param))

            if params:
                self._weight_norm_param_cache[layer_name] = params
                trainable_count = sum(1 for _, p in params if p.requires_grad)
                # Alias 변환 정보 출력
                alias_info = f" (resolved: '{resolved_pattern}')" if resolved_pattern != layer_name else ""
                # 콘솔에 직접 출력 (log.info는 로그 레벨에 따라 안 보일 수 있음)
                if self.debug:
                    print(f"\n[Weight Norm Cache] [{layer_name}]{alias_info} Found {len(params)} params (trainable={trainable_count})")
                log.info(f"[{layer_name}]{alias_info} Found {len(params)} params (trainable={trainable_count})")
                for name, param in params[:5]:  # 처음 5개만 출력
                    if self.debug:
                        print(f"    - {name} (shape={list(param.shape)}, requires_grad={param.requires_grad})")
                    log.info(f"    - {name} (shape={list(param.shape)}, requires_grad={param.requires_grad})")
                if len(params) > 5:
                    if self.debug:
                        print(f"    ... and {len(params) - 5} more")
                    log.info(f"    ... and {len(params) - 5} more")
                # [DEBUG] trainable이 0이면 경고
                if trainable_count == 0:
                    if self.debug:
                        print(f"    ⚠️ WARNING: No trainable params found! Check PEFT modules_to_save config.")
                    log.warning(f"    ⚠️ WARNING: No trainable params found! Check PEFT modules_to_save config.")
            else:
                if self.debug:
                    print(f"\n[Weight Norm Cache] [{layer_name}] No params found matching patterns: {patterns}")
                log.warning(f"[{layer_name}] No params found matching patterns: {patterns}")

        log.info("=" * 70 + "\n")

    def _log_weight_norms(self, batch_idx):
        """
        설정된 layer들의 weight norm과 gradient norm 로깅 (batch_idx 기준)

        wte/ff_out (embed/head)의 경우 5개 그룹으로 분리:
        - lora: LoRA 파라미터
        - embed_orig: 기존 vocab embedding (idx < original_vocab_size)
        - embed_new: 새 vocab embedding (idx >= original_vocab_size)
        - head_orig: 기존 vocab head (idx < original_vocab_size)
        - head_new: 새 vocab head (idx >= original_vocab_size)
        """
        if not self._log_weight_norm_layers:
            return

        # batch_idx 기준으로 interval 체크 (mini-batch 단위)
        if batch_idx % self._log_weight_norm_interval != 0:
            return

        # 첫 호출 시 파라미터 캐싱
        if not self._weight_norm_param_cache:
            self._cache_weight_norm_params()
            if not self._weight_norm_param_cache:
                return

        import logging
        log = logging.getLogger(__name__)

        # original_vocab_size 가져오기
        original_vocab_size = getattr(self.args, 'original_vocab_size', 126349)

        log.info("\n" + "=" * 70)
        log.info(f"[Batch {batch_idx}] Weight Norm Logging (5-group split)")
        log.info("=" * 70)

        for layer_name, params in self._weight_norm_param_cache.items():
            # wte 또는 ff_out인 경우 orig/new로 분리
            is_embed = 'wte' in layer_name.lower() or 'embed_tokens' in layer_name.lower()
            is_head = 'ff_out' in layer_name.lower() or 'lm_head' in layer_name.lower()

            if is_embed or is_head:
                # orig/new 분리 로깅
                self._log_split_weight_norms(
                    layer_name, params, original_vocab_size,
                    is_embed, batch_idx, log
                )
            else:
                # 기존 방식 (전체 합산)
                self._log_single_weight_norm(layer_name, params, batch_idx, log)

        # 상위 빈도 SELFIES 토큰 norm 로깅
        self._log_top_selfies_token_norms(batch_idx, log)

        log.info("=" * 70 + "\n")

    def _log_split_weight_norms(self, layer_name, params, original_vocab_size, is_embed, batch_idx, log):
        """wte/ff_out을 orig/new vocab으로 분리하여 로깅 (개별 토큰 norm의 평균 방식)"""
        prefix = "embed" if is_embed else "head"

        for name, param in params:
            if not param.requires_grad:
                continue

            # 2D weight: [vocab_size, hidden_dim] 또는 [hidden_dim, vocab_size]
            if param.dim() < 2:
                continue

            # vocab 차원 찾기 (보통 첫 번째 또는 마지막)
            vocab_dim = 0 if param.shape[0] > param.shape[-1] else -1
            vocab_size = param.shape[vocab_dim]

            if vocab_size <= original_vocab_size:
                # 분리 불가 (새 vocab 없음)
                self._log_single_weight_norm(layer_name, [(name, param)], batch_idx, log)
                continue

            # orig/new 분리
            if vocab_dim == 0:
                orig_weight = param.data[:original_vocab_size]
                new_weight = param.data[original_vocab_size:]
                orig_grad = param.grad[:original_vocab_size] if param.grad is not None else None
                new_grad = param.grad[original_vocab_size:] if param.grad is not None else None
            else:
                orig_weight = param.data[..., :original_vocab_size]
                new_weight = param.data[..., original_vocab_size:]
                orig_grad = param.grad[..., :original_vocab_size] if param.grad is not None else None
                new_grad = param.grad[..., original_vocab_size:] if param.grad is not None else None

            # orig 로깅 (개별 토큰 norm의 평균)
            # orig_weight: [num_orig_tokens, hidden_dim] 또는 [hidden_dim, num_orig_tokens]
            if vocab_dim == 0:
                orig_token_norms = orig_weight.norm(2, dim=1)  # 각 토큰별 L2 norm
                orig_grad_token_norms = orig_grad.norm(2, dim=1) if orig_grad is not None else None
            else:
                orig_token_norms = orig_weight.norm(2, dim=0)  # [hidden_dim, vocab] → dim=0
                orig_grad_token_norms = orig_grad.norm(2, dim=0) if orig_grad is not None else None

            orig_avg_norm = orig_token_norms.mean().item()
            orig_avg_grad_norm = orig_grad_token_norms.mean().item() if orig_grad_token_norms is not None else 0.0
            orig_key = f"{prefix}_orig"

            if orig_key not in self._initial_weight_norms:
                self._initial_weight_norms[orig_key] = orig_avg_norm
            orig_init = self._initial_weight_norms[orig_key]
            orig_change = orig_avg_norm - orig_init
            orig_pct = (orig_change / orig_init * 100) if orig_init != 0 else 0

            num_orig_tokens = orig_weight.shape[0] if vocab_dim == 0 else orig_weight.shape[-1]
            log.info(f"  [{orig_key}] n_tokens={num_orig_tokens}, "
                     f"avg_norm={orig_avg_norm:.4f} (init={orig_init:.4f}, {orig_pct:+.2f}%), "
                     f"avg_grad_norm={orig_avg_grad_norm:.6f}")

            self.log(f"weight_norm/{orig_key}/avg_norm", orig_avg_norm,
                     batch_size=self.args.batch_size, sync_dist=False)
            self.log(f"weight_norm/{orig_key}/avg_grad_norm", orig_avg_grad_norm,
                     batch_size=self.args.batch_size, sync_dist=False)
            self.log(f"weight_norm/{orig_key}/avg_norm_change_pct", orig_pct,
                     batch_size=self.args.batch_size, sync_dist=False)

            # new 로깅 (개별 토큰 norm의 평균)
            if vocab_dim == 0:
                new_token_norms = new_weight.norm(2, dim=1)
                new_grad_token_norms = new_grad.norm(2, dim=1) if new_grad is not None else None
            else:
                new_token_norms = new_weight.norm(2, dim=0)
                new_grad_token_norms = new_grad.norm(2, dim=0) if new_grad is not None else None

            new_avg_norm = new_token_norms.mean().item()
            new_avg_grad_norm = new_grad_token_norms.mean().item() if new_grad_token_norms is not None else 0.0
            new_key = f"{prefix}_new"

            if new_key not in self._initial_weight_norms:
                self._initial_weight_norms[new_key] = new_avg_norm
            new_init = self._initial_weight_norms[new_key]
            new_change = new_avg_norm - new_init
            new_pct = (new_change / new_init * 100) if new_init != 0 else 0

            num_new_tokens = new_weight.shape[0] if vocab_dim == 0 else new_weight.shape[-1]
            log.info(f"  [{new_key}] n_tokens={num_new_tokens}, "
                     f"avg_norm={new_avg_norm:.4f} (init={new_init:.4f}, {new_pct:+.2f}%), "
                     f"avg_grad_norm={new_avg_grad_norm:.6f}")

            self.log(f"weight_norm/{new_key}/avg_norm", new_avg_norm,
                     batch_size=self.args.batch_size, sync_dist=False)
            self.log(f"weight_norm/{new_key}/avg_grad_norm", new_avg_grad_norm,
                     batch_size=self.args.batch_size, sync_dist=False)
            self.log(f"weight_norm/{new_key}/avg_norm_change_pct", new_pct,
                     batch_size=self.args.batch_size, sync_dist=False)

    def _log_single_weight_norm(self, layer_name, params, batch_idx, log):
        """단일 layer의 weight norm 로깅 (기존 방식)"""
        total_weight_norm = 0.0
        total_grad_norm = 0.0
        has_grad = False
        param_count = 0

        for name, param in params:
            if param.requires_grad:
                param_count += 1
                weight_norm = param.data.norm(2).item()
                total_weight_norm += weight_norm ** 2

                if param.grad is not None:
                    has_grad = True
                    grad_norm = param.grad.norm(2).item()
                    total_grad_norm += grad_norm ** 2

        if param_count == 0:
            return

        # L2 norm 합산
        total_weight_norm = total_weight_norm ** 0.5
        total_grad_norm = total_grad_norm ** 0.5 if has_grad else 0.0

        # 초기 weight norm 저장 (첫 로깅 시)
        if layer_name not in self._initial_weight_norms:
            self._initial_weight_norms[layer_name] = total_weight_norm

        initial_norm = self._initial_weight_norms[layer_name]
        norm_change = total_weight_norm - initial_norm
        pct_change = (norm_change / initial_norm * 100) if initial_norm != 0 else 0

        # 콘솔 로깅
        trainable_count = sum(1 for _, p in params if p.requires_grad)
        msg = (f"[{layer_name}] params={len(params)} (trainable={trainable_count}), "
               f"weight_norm={total_weight_norm:.4f} (init={initial_norm:.4f}, "
               f"delta={norm_change:+.4f}, {pct_change:+.2f}%), "
               f"grad_norm={total_grad_norm:.6f}, has_grad={has_grad}")
        if self.debug:
            print(f"\n[Batch {batch_idx}] {msg}")
        log.info(msg)

        # TensorBoard/WandB 로깅
        self.log(f"weight_norm/{layer_name}/weight_norm", total_weight_norm,
                 batch_size=self.args.batch_size, sync_dist=False)
        self.log(f"weight_norm/{layer_name}/grad_norm", total_grad_norm,
                 batch_size=self.args.batch_size, sync_dist=False)
        self.log(f"weight_norm/{layer_name}/weight_norm_change_pct", pct_change,
                 batch_size=self.args.batch_size, sync_dist=False)

    def _log_top_selfies_token_norms(self, batch_idx, log):
        """
        상위 빈도 SELFIES 토큰들의 Embedding/LM Head weight norm 추적

        wandb에 Top5/Top10/Top15/Total 그룹별 평균 norm과 변화율 로깅
        - top5/top10/top15: 상위 빈도 SELFIES 토큰
        - total: 전체 새 토큰 (idx >= original_vocab_size)
        """
        if not self._top_selfies_tokens:
            return

        # 첫 호출 시 토큰 ID 캐싱
        if self._top_selfies_token_ids is None:
            self._initialize_top_selfies_token_ids()
            if self._top_selfies_token_ids is None:
                return

        # Embedding, LM Head 파라미터 찾기
        embed_param = None
        head_param = None

        for layer_name, params in self._weight_norm_param_cache.items():
            if 'wte' in layer_name.lower() or 'embed_tokens' in layer_name.lower():
                for name, param in params:
                    if param.requires_grad and param.dim() >= 2:
                        embed_param = param
                        break
            elif 'ff_out' in layer_name.lower() or 'lm_head' in layer_name.lower():
                for name, param in params:
                    if param.requires_grad and param.dim() >= 2:
                        head_param = param
                        break

        if embed_param is None and head_param is None:
            return

        log.info("-" * 50)
        log.info("[Top SELFIES Token Norm Tracking]")

        # original_vocab_size 가져오기
        original_vocab_size = getattr(self.args, 'original_vocab_size', 126349)

        # 전체 새 토큰 ID 리스트 생성 (idx >= original_vocab_size)
        if embed_param is not None:
            vocab_size = embed_param.shape[0]
            total_new_token_ids = list(range(original_vocab_size, vocab_size))
        elif head_param is not None:
            is_vocab_first = head_param.shape[0] > head_param.shape[-1]
            vocab_size = head_param.shape[0] if is_vocab_first else head_param.shape[-1]
            total_new_token_ids = list(range(original_vocab_size, vocab_size))
        else:
            total_new_token_ids = []

        # 그룹별 토큰 인덱스 (top5, top10, top15, total)
        groups = {
            'top5': self._top_selfies_token_ids[:5],
            'top10': self._top_selfies_token_ids[:10],
            'top15': self._top_selfies_token_ids[:15],
            'total': total_new_token_ids  # 전체 새 토큰 (idx >= original_vocab_size)
        }

        for group_name, token_ids in groups.items():
            valid_ids = [tid for tid in token_ids if tid is not None]
            if not valid_ids:
                continue

            # Embedding norm 계산
            if embed_param is not None:
                embed_norms = self._compute_token_norms(embed_param, valid_ids, is_vocab_first=True)
                if embed_norms:
                    avg_embed_norm = sum(embed_norms) / len(embed_norms)
                    key = f"embed_{group_name}"

                    if key not in self._top_selfies_initial_norms:
                        self._top_selfies_initial_norms[key] = avg_embed_norm
                    init_norm = self._top_selfies_initial_norms[key]
                    pct_change = ((avg_embed_norm - init_norm) / init_norm * 100) if init_norm != 0 else 0

                    log.info(f"  Embed {group_name}: avg_norm={avg_embed_norm:.4f} "
                             f"(init={init_norm:.4f}, {pct_change:+.2f}%)")

                    self.log(f"top_selfies/{group_name}/embed_norm", avg_embed_norm,
                             batch_size=self.args.batch_size, sync_dist=False)
                    self.log(f"top_selfies/{group_name}/embed_change_pct", pct_change,
                             batch_size=self.args.batch_size, sync_dist=False)

            # LM Head norm 계산
            if head_param is not None:
                # LM Head는 [hidden_dim, vocab_size] 형태일 수 있음
                is_vocab_first = head_param.shape[0] > head_param.shape[-1]
                head_norms = self._compute_token_norms(head_param, valid_ids, is_vocab_first=is_vocab_first)
                if head_norms:
                    avg_head_norm = sum(head_norms) / len(head_norms)
                    key = f"head_{group_name}"

                    if key not in self._top_selfies_initial_norms:
                        self._top_selfies_initial_norms[key] = avg_head_norm
                    init_norm = self._top_selfies_initial_norms[key]
                    pct_change = ((avg_head_norm - init_norm) / init_norm * 100) if init_norm != 0 else 0

                    log.info(f"  Head  {group_name}: avg_norm={avg_head_norm:.4f} "
                             f"(init={init_norm:.4f}, {pct_change:+.2f}%)")

                    self.log(f"top_selfies/{group_name}/head_norm", avg_head_norm,
                             batch_size=self.args.batch_size, sync_dist=False)
                    self.log(f"top_selfies/{group_name}/head_change_pct", pct_change,
                             batch_size=self.args.batch_size, sync_dist=False)

            # Gradient norm (학습 진행 상태 확인용)
            if embed_param is not None and embed_param.grad is not None:
                embed_grad_norms = self._compute_token_norms(embed_param.grad, valid_ids, is_vocab_first=True)
                if embed_grad_norms:
                    avg_grad_norm = sum(embed_grad_norms) / len(embed_grad_norms)
                    self.log(f"top_selfies/{group_name}/embed_grad_norm", avg_grad_norm,
                             batch_size=self.args.batch_size, sync_dist=False)

            if head_param is not None and head_param.grad is not None:
                is_vocab_first = head_param.shape[0] > head_param.shape[-1]
                head_grad_norms = self._compute_token_norms(head_param.grad, valid_ids, is_vocab_first=is_vocab_first)
                if head_grad_norms:
                    avg_head_grad_norm = sum(head_grad_norms) / len(head_grad_norms)
                    self.log(f"top_selfies/{group_name}/head_grad_norm", avg_head_grad_norm,
                             batch_size=self.args.batch_size, sync_dist=False)

    def _initialize_top_selfies_token_ids(self):
        """상위 SELFIES 토큰의 ID를 토크나이저에서 조회하여 캐싱"""
        import logging
        log = logging.getLogger(__name__)

        # LLaDA의 경우 llm_tokenizer에 SELFIES 토큰이 추가됨
        tokenizer = None
        if hasattr(self, 'blip2model') and hasattr(self.blip2model, 'llm_tokenizer'):
            tokenizer = self.blip2model.llm_tokenizer
        elif hasattr(self, 'tokenizer') and self.tokenizer is not None:
            tokenizer = self.tokenizer

        if tokenizer is None:
            log.warning("[Top SELFIES] Tokenizer not available")
            if self.debug:
                print("[Top SELFIES] WARNING: Tokenizer not available")
            return

        self._top_selfies_token_ids = []
        log.info("[Top SELFIES Token ID Mapping]")
        if self.debug:
            print("\n[Top SELFIES Token ID Mapping]")

        for token in self._top_selfies_tokens:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                # UNK 토큰이면 None으로 설정
                if token_id == tokenizer.unk_token_id:
                    log.warning(f"  {token}: UNK (not found)")
                    if self.debug:
                        print(f"  {token}: UNK (not found)")
                    self._top_selfies_token_ids.append(None)
                else:
                    log.info(f"  {token}: {token_id}")
                    if self.debug:
                        print(f"  {token}: {token_id}")
                    self._top_selfies_token_ids.append(token_id)
            except Exception as e:
                log.warning(f"  {token}: Error - {e}")
                if self.debug:
                    print(f"  {token}: Error - {e}")
                self._top_selfies_token_ids.append(None)

        # 유효한 토큰 ID 개수 확인
        valid_count = sum(1 for tid in self._top_selfies_token_ids if tid is not None)
        log.info(f"[Top SELFIES] Initialized {valid_count}/{len(self._top_selfies_tokens)} token IDs")
        if self.debug:
            print(f"[Top SELFIES] Initialized {valid_count}/{len(self._top_selfies_tokens)} token IDs\n")

    def _compute_token_norms(self, param, token_ids, is_vocab_first=True):
        """특정 토큰들의 weight norm 계산"""
        norms = []
        vocab_size = param.shape[0] if is_vocab_first else param.shape[-1]

        for tid in token_ids:
            if tid is None or tid >= vocab_size:
                continue

            if is_vocab_first:
                # [vocab_size, hidden_dim] 형태
                token_weight = param.data[tid]
            else:
                # [hidden_dim, vocab_size] 형태
                token_weight = param.data[..., tid]

            norms.append(token_weight.norm(2).item())

        return norms

    def _log_sample_predictions(self, batch, outputs, tasks, batch_idx, mode="train",
                                 num_samples=2, predictions=None, targets=None, prompts=None, generated_ids=None):
        """통합 샘플 예측 로깅 (Training & Validation)"""
        import logging
        logger = logging.getLogger(__name__)

        # Mode별 제목
        mode_str = "Training" if mode == "train" else "Validation"

        logger.info("\n" + "="*80)
        logger.info(f"[{mode_str} Sample Log] Step {self.global_step}, Batch {batch_idx}")
        logger.info("="*80)

        # Training mode: logits에서 예측 생성
        if mode == "train":
            if isinstance(outputs, dict):
                logits = outputs.get('logits')
            else:
                logits = getattr(outputs, 'logits', None)

            if logits is None or not hasattr(batch, 'labels'):
                return

            pred_ids = torch.argmax(logits, dim=-1)  # [batch, seq_len]
            labels = batch.labels
            num_samples_to_log = min(num_samples, len(tasks))

            for i in range(num_samples_to_log):
                task_name = tasks[i] if isinstance(tasks[i], str) else f"task_{tasks[i]}"
                logger.info(f"\n--- Sample {i} | Task: {task_name} ---")

                # [NEW] Input Token IDs
                if hasattr(batch, 'input_ids'):
                    input_ids_list = batch.input_ids[i].tolist()
                    logger.info(f"Input Token IDs (len={len(input_ids_list)}): {input_ids_list}")

                    try:
                        input_text = self.blip2model.llm_tokenizer.decode(
                            batch.input_ids[i], skip_special_tokens=False
                        )
                        logger.info(f"Input Text: {input_text[:200]}...")
                    except Exception as e:
                        logger.warning(f"Could not decode input: {e}")

                # Token-by-token breakdown
                label_seq = labels[i]
                pred_seq = pred_ids[i]
                answer_mask = (label_seq != -100)
                answer_indices = answer_mask.nonzero(as_tuple=True)[0]

                if len(answer_indices) > 0:
                    # [NEW] Prediction Token IDs
                    pred_answer_ids = pred_seq[answer_mask].tolist()
                    logger.info(f"\nPrediction Token IDs (len={len(pred_answer_ids)}): {pred_answer_ids}")

                    logger.info("\nToken-by-Token Breakdown:")
                    logger.info(f"{'Pos':<5} {'Label':<10} {'Pred':<10} {'Label Tok':<25} {'Pred Tok':<25} {'Match':<5}")
                    logger.info("-" * 90)

                    max_tokens_to_show = min(20, len(answer_indices))
                    for j in range(max_tokens_to_show):
                        pos = answer_indices[j].item()
                        label_id = label_seq[pos].item()
                        pred_id = pred_seq[pos].item()

                        try:
                            label_tok = self.blip2model.llm_tokenizer.decode([label_id])
                            pred_tok = self.blip2model.llm_tokenizer.decode([pred_id])
                        except Exception:
                            label_tok = f"<id:{label_id}>"
                            pred_tok = f"<id:{pred_id}>"

                        match = "✓" if label_id == pred_id else "✗"
                        logger.info(f"{pos:<5} {label_id:<10} {pred_id:<10} {label_tok:<25} {pred_tok:<25} {match:<5}")

                    if len(answer_indices) > max_tokens_to_show:
                        logger.info(f"... (showing {max_tokens_to_show}/{len(answer_indices)} tokens)")

                    # 정확도
                    correct = (pred_seq[answer_mask] == label_seq[answer_mask]).sum().item()
                    total = answer_mask.sum().item()
                    acc = 100.0 * correct / total if total > 0 else 0.0
                    logger.info(f"\nSample Accuracy: {acc:.1f}% ({correct}/{total} tokens)")

                    # Instance loss
                    if isinstance(outputs, dict) and 'instance_loss' in outputs:
                        try:
                            inst_loss = outputs['instance_loss'][i].item()
                            logger.info(f"Instance Loss: {inst_loss:.4f}")
                        except Exception:
                            pass

        # Validation mode: 이미 생성된 predictions 사용
        else:  # mode == "val"
            if predictions is None or targets is None or prompts is None:
                return

            # Task별 카운팅 (validation은 5개 제한)
            if not hasattr(self, 'debug_task_counts'):
                self.debug_task_counts = {}

            num_samples_to_log = min(num_samples, len(tasks))

            for i in range(num_samples_to_log):
                task_name = tasks[i] if isinstance(tasks[i], str) else f"task_{tasks[i]}"

                # Task별 제한 (validation)
                if task_name not in self.debug_task_counts:
                    self.debug_task_counts[task_name] = 0

                if self.debug_task_counts[task_name] >= 5:
                    continue

                self.debug_task_counts[task_name] += 1
                logger.info(f"\n--- Sample {i} | Task: {task_name} ---")

                # Prompt (Input)
                logger.info(f"Prompt Text: {prompts[i]}")

                # [NEW] Prompt Token IDs
                try:
                    prompt_ids = self.blip2model.llm_tokenizer.encode(prompts[i], add_special_tokens=False)
                    logger.info(f"Prompt Token IDs (len={len(prompt_ids)}): {prompt_ids}")
                except Exception as e:
                    logger.warning(f"Could not encode prompt: {e}")

                # Target
                logger.info(f"\nTarget: {targets[i]}")

                # Prediction
                logger.info(f"Prediction: {predictions[i]}")

                # [NEW] Generated Token IDs (raw output from model)
                if generated_ids is not None:
                    try:
                        gen_id_list = generated_ids[i].tolist() if hasattr(generated_ids[i], 'tolist') else generated_ids[i]
                        logger.info(f"\n[Generated Output] Token IDs (len={len(gen_id_list)}): {gen_id_list}")

                        # Decode generated IDs
                        generated_text = self.blip2model.llm_tokenizer.decode(generated_ids[i], skip_special_tokens=False)
                        logger.info(f"[Generated Output] Decoded Text: {generated_text}")

                        # Token breakdown (filter out-of-vocab)
                        vocab_size = len(self.blip2model.llm_tokenizer)
                        valid_gen_ids = [tid for tid in gen_id_list if 0 <= tid < vocab_size]
                        gen_tokens = self.blip2model.llm_tokenizer.convert_ids_to_tokens(valid_gen_ids)
                        logger.info(f"[Generated Output] Tokens: {' || '.join(gen_tokens[:50])}{'...' if len(gen_tokens) > 50 else ''}")
                    except Exception as e:
                        logger.warning(f"Could not process generated_ids: {e}")

                # [OLD] Prediction Token IDs (from re-encoded prediction string)
                try:
                    pred_ids = self.blip2model.llm_tokenizer.encode(predictions[i], add_special_tokens=False)
                    logger.info(f"\n[Re-encoded Prediction] Token IDs (len={len(pred_ids)}): {pred_ids}")

                    # Token breakdown (filter out-of-vocab)
                    vocab_size = len(self.blip2model.llm_tokenizer)
                    valid_pred_ids = [tid for tid in pred_ids if 0 <= tid < vocab_size]
                    pred_tokens = self.blip2model.llm_tokenizer.convert_ids_to_tokens(valid_pred_ids)
                    logger.info(f"[Re-encoded Prediction] Tokens: {' || '.join(pred_tokens)}")
                except Exception as e:
                    logger.warning(f"Token debug error: {e}")

        logger.info("="*80 + "\n")

    def task_specific_logging(
        self,
        outputs,
        tasks,
        mode,
        epoch_end,
        task_specific_outputs: Dict[str, Dict[str, list]],
        num_moving_samples=32,
    ):

        if mode == "train":
            assert epoch_end == False

        if (mode == "train") or not epoch_end:
            # log dataset specific losses
            new_outputs = {
                k: v for k, v in outputs.items()
                if isinstance(v, torch.Tensor) and v.shape != torch.Size([])
            }

            for task in tasks:
                if task not in task_specific_outputs:
                    task_specific_outputs[task] = {}
                for k in new_outputs.keys():
                    if k not in task_specific_outputs[task]:
                        task_specific_outputs[task][k] = []

            for metric, v in new_outputs.items():

                for i in range(v.shape[0]):
                    if torch.isnan(v[i]):
                        if i < 5:  # 로그 폭주 방지용
                            if self.debug:
                                print(f"[DEBUG] NaN detected for task: {tasks[i]} in metric: {metric}")
                        continue

                    task = tasks[i]
                    task_specific_outputs[task][metric].append(v[i].item())

                    if num_moving_samples is not None and (
                        len(self.task_specific_outputs[task][metric])
                        > num_moving_samples
                    ):
                        task_specific_outputs[task][metric].pop(0)

        if mode == "train" or epoch_end:
            for task, metric_dict in task_specific_outputs.items():
                for metric, vs in metric_dict.items():
                    if vs:
                        self.log(
                            f"{mode}/{task}/{metric}",
                            sum(vs) / len(vs),
                            batch_size=len(vs),
                            sync_dist=False,
                        )

    def _fix_modules_to_save_gradients(self):
        """
        [CRITICAL FIX] modules_to_save (embed_tokens, lm_head)를 강제로 trainable 설정

        체크포인트 로드 후 또는 초기화 후에 호출하여
        PEFT modules_to_save로 지정된 모듈들이 실제로 학습 가능한지 확인하고 수정합니다.
        """
        if not hasattr(self.blip2model, 'llm_model'):
            logger.warning("No llm_model found, skipping gradient fix")
            return

        # LoRA 모드가 아니면 스킵
        if self.tune_llm != "lora":
            return

        fixed_params = []

        # [FIX] 정확한 경로 지정으로 INPUT embedding과 OUTPUT head만 학습
        # blocks 내부의 ff_out은 제외하여 메모리 절약
        for name, param in self.blip2model.llm_model.named_parameters():
            # Skip if it's inside transformer blocks
            if '.blocks.' in name or 'transformer.blocks' in name:
                continue

            # [CRITICAL] PEFT modules_to_save가 적용된 경우:
            # - modules_to_save.default: 실제 학습되는 weights (trainable로 설정)
            # - original_module: frozen copy (forward에서 사용 안됨, trainable로 설정하면 안됨)
            if 'original_module' in name:
                continue  # PEFT original_module은 건드리지 않음

            # LLaDA: wte (input embedding), ff_out (output head at transformer level ONLY)
            # Standard: embed_tokens, lm_head
            name_lower = name.lower()
            if ('wte' in name_lower or 'embed_tokens' in name_lower or
                'lm_head' in name_lower or 'ff_out' in name_lower):
                if not param.requires_grad:
                    param.requires_grad = True
                    fixed_params.append(name)

        if fixed_params:
            logger.info(f"  Fixed {len(fixed_params)} parameters to requires_grad=True:")
            for name in fixed_params[:5]:  # 처음 5개만 출력
                logger.info(f"    - {name}")
            if len(fixed_params) > 5:
                logger.info(f"    ... and {len(fixed_params) - 5} more")
        else:
            logger.info("  All embed/lm_head parameters already have requires_grad=True")

    def on_train_epoch_start(self) -> None:
        self.task_specific_outputs = {}
        # if not hasattr(self, "task_specific_chosen_reward"):
        self.task_specific_chosen_reward = {}

        # [CRITICAL FIX] 매 epoch 시작마다 modules_to_save gradients 재확인
        # 특히 epoch 0에서 체크포인트가 로드된 경우 필수
        if self.current_epoch == 0:
            self._fix_modules_to_save_gradients()

        self.train_list_predictions = []
        self.train_list_targets = []
        self.train_list_prompts = []
        self.train_list_tasks = []
        self.train_list_probs = []
        self.train_total_avg_loss = 0.0
        self.train_total_seen_data_size = 0

        # ======================================================================
        # Epoch 단위 validation metric 누적을 위한 초기화
        # ======================================================================
        self.epoch_val_metrics = {}  # {metric_key: [values]}
        self.epoch_val_count = 0  # epoch 내 validation 횟수

        self.trainer.train_dataloader.collate_fn.current_epoch = (
            self.trainer.current_epoch
        )

        # DISABLED: self.log_model_parameters()
        # This logs 8B+ parameters to WandB/TensorBoard which takes hours
        # Only enable for debugging specific parameter issues

        # [Fix 2.1] Epoch 0에서 LoRA 설정 검증
        if self.current_epoch == 0 and self.global_step == 0:
            self._log_lora_verification()

    def _log_lora_verification(self):
        """Epoch 0에서 LoRA 및 modules_to_save 설정 검증"""
        import logging
        logger = logging.getLogger(__name__)

        logger.info("\n" + "="*70)
        logger.info("[LoRA Verification] Checking modules_to_save setup...")
        logger.info("="*70)

        # 1. PEFT config 확인
        if not hasattr(self.blip2model, 'llm_model'):
            logger.warning("❌ blip2model has no llm_model attribute!")
            logger.info("="*70 + "\n")
            return

        if not hasattr(self.blip2model.llm_model, 'peft_config'):
            logger.warning("❌ Model does not have PEFT config! (Not using LoRA?)")
            logger.info("="*70 + "\n")
            return

        # peft_config는 dict[adapter_name, PeftConfig] 형태
        peft_config_dict = self.blip2model.llm_model.peft_config
        if isinstance(peft_config_dict, dict):
            # 기본 어댑터는 "default" 키를 사용
            adapter_name = list(peft_config_dict.keys())[0] if peft_config_dict else None
            if adapter_name:
                peft_cfg = peft_config_dict[adapter_name]
                if hasattr(peft_cfg, 'modules_to_save') and peft_cfg.modules_to_save:
                    logger.info(f"✅ modules_to_save configured: {peft_cfg.modules_to_save}")
                    logger.info(f"   (LLaDA uses: model.transformer.wte, model.transformer.ff_out)")
                else:
                    logger.warning("⚠️  No modules_to_save in PEFT config!")
            else:
                logger.warning("⚠️  No adapters found in peft_config!")
        else:
            # 이전 버전 호환성
            peft_cfg = peft_config_dict
            if hasattr(peft_cfg, 'modules_to_save') and peft_cfg.modules_to_save:
                logger.info(f"✅ modules_to_save configured: {peft_cfg.modules_to_save}")
            else:
                logger.warning("⚠️  No modules_to_save in PEFT config!")

        # 2. embed_tokens 및 lm_head 상태 확인
        # PEFT 모델에서는 직접 get_input_embeddings / get_output_embeddings 사용이 더 안전
        try:
            embed_layer = self.blip2model.llm_model.get_input_embeddings()
            lm_head_layer = self.blip2model.llm_model.get_output_embeddings()

            embed_tokens_found = embed_layer is not None
            lm_head_found = lm_head_layer is not None

            if embed_tokens_found:
                embed_tokens_trainable = embed_layer.weight.requires_grad
                embed_size = embed_layer.weight.shape
                logger.info(f"  📊 Input Embeddings (embed_tokens)")
                logger.info(f"      Shape: {embed_size}, requires_grad: {embed_tokens_trainable}")
                logger.info(f"      Type: {type(embed_layer).__name__}")

            if lm_head_found:
                lm_head_trainable = lm_head_layer.weight.requires_grad
                lm_head_size = lm_head_layer.weight.shape
                logger.info(f"  📊 Output Embeddings (lm_head)")
                logger.info(f"      Shape: {lm_head_size}, requires_grad: {lm_head_trainable}")
                logger.info(f"      Type: {type(lm_head_layer).__name__}")

            # 추가: named_parameters로 실제 저장된 이름 확인
            logger.info("\n  Checking parameter names containing 'embed' or 'lm_head':")
            found_params = []
            for name, param in self.blip2model.llm_model.named_parameters():
                if 'embed' in name.lower() or 'lm_head' in name.lower():
                    found_params.append(f"{name} (grad={param.requires_grad})")

            if found_params:
                for param_info in found_params[:10]:  # 최대 10개만 출력
                    logger.info(f"    - {param_info}")
                if len(found_params) > 10:
                    logger.info(f"    ... and {len(found_params) - 10} more")
            else:
                logger.info("    No parameters found with 'embed' or 'lm_head' in name")

        except Exception as e:
            logger.error(f"  Error checking embeddings: {e}")
            # Fallback: 이전 방식 사용
            embed_tokens_found = False
            lm_head_found = False
            embed_tokens_trainable = False
            lm_head_trainable = False

            for name, param in self.blip2model.llm_model.named_parameters():
                if 'embed_tokens' in name or 'embed_token' in name:
                    embed_tokens_found = True
                    embed_tokens_trainable = embed_tokens_trainable or param.requires_grad
                if 'lm_head' in name:
                    lm_head_found = True
                    lm_head_trainable = lm_head_trainable or param.requires_grad

        # 3. 상태 요약
        if embed_tokens_found and lm_head_found:
            if embed_tokens_trainable and lm_head_trainable:
                logger.info("✅ Both embed_tokens and lm_head are TRAINABLE")
            else:
                logger.warning(f"⚠️  Training status - embed_tokens: {embed_tokens_trainable}, lm_head: {lm_head_trainable}")
                if not embed_tokens_trainable or not lm_head_trainable:
                    logger.error("❌ CRITICAL: modules_to_save 모듈이 학습 불가능 상태입니다!")
        else:
            logger.error(f"❌ Missing modules! embed_tokens: {embed_tokens_found}, lm_head: {lm_head_found}")
            logger.error("❌ CRITICAL: 이 상태에서는 새로운 토큰을 학습할 수 없습니다!")

        # 4. Vocab size 일관성 확인
        try:
            tokenizer_size = len(self.blip2model.llm_tokenizer)
            model_embed_size = self.blip2model.llm_model.get_input_embeddings().weight.shape[0]
            model_lm_head_size = self.blip2model.llm_model.get_output_embeddings().weight.shape[0]

            logger.info(f"\n  Vocabulary Sizes:")
            logger.info(f"    Tokenizer:   {tokenizer_size}")
            logger.info(f"    Embed layer: {model_embed_size}")
            logger.info(f"    LM head:     {model_lm_head_size}")

            if tokenizer_size == model_embed_size == model_lm_head_size:
                logger.info("✅ Size consistency check PASSED")
            else:
                logger.error("❌ Size MISMATCH detected! This will cause training issues.")
        except Exception as e:
            logger.warning(f"⚠️  Could not verify vocab sizes: {e}")

        logger.info("="*70 + "\n")

        # [Fix 2.1] Epoch 0에서 LoRA 설정 검증
        if self.current_epoch == 0 and self.global_step == 0:
            self._log_lora_verification()

    def _log_lora_verification(self):
        """Epoch 0에서 LoRA 및 modules_to_save 설정 검증"""
        import logging
        logger = logging.getLogger(__name__)

        logger.info("\n" + "="*70)
        logger.info("[LoRA Verification] Checking modules_to_save setup...")
        logger.info("="*70)

        # 1. PEFT config 확인
        if not hasattr(self.blip2model, 'llm_model'):
            logger.warning("❌ blip2model has no llm_model attribute!")
            logger.info("="*70 + "\n")
            return

        if not hasattr(self.blip2model.llm_model, 'peft_config'):
            logger.warning("❌ Model does not have PEFT config! (Not using LoRA?)")
            logger.info("="*70 + "\n")
            return

        peft_cfg = self.blip2model.llm_model.peft_config
        if hasattr(peft_cfg, 'modules_to_save') and peft_cfg.modules_to_save:
            logger.info(f"✅ modules_to_save configured: {peft_cfg.modules_to_save}")
        else:
            logger.warning("⚠️  No modules_to_save in PEFT config!")

        # 2. embed_tokens 및 lm_head 상태 확인
        embed_tokens_found = False
        lm_head_found = False
        embed_tokens_trainable = False
        lm_head_trainable = False
        embed_size = None
        lm_head_size = None

        for name, param in self.blip2model.llm_model.named_parameters():
            if 'embed_tokens' in name:
                embed_tokens_found = True
                embed_tokens_trainable = param.requires_grad
                embed_size = param.shape
                logger.info(f"  📊 {name}")
                logger.info(f"      Shape: {param.shape}, requires_grad: {param.requires_grad}")
            if 'lm_head' in name:
                lm_head_found = True
                lm_head_trainable = param.requires_grad
                lm_head_size = param.shape
                logger.info(f"  📊 {name}")
                logger.info(f"      Shape: {param.shape}, requires_grad: {param.requires_grad}")

        # 3. 상태 요약
        if embed_tokens_found and lm_head_found:
            if embed_tokens_trainable and lm_head_trainable:
                logger.info("✅ Both embed_tokens and lm_head are TRAINABLE")
            else:
                logger.warning(f"⚠️  Training status - embed_tokens: {embed_tokens_trainable}, lm_head: {lm_head_trainable}")
        else:
            logger.error(f"❌ Missing modules! embed_tokens: {embed_tokens_found}, lm_head: {lm_head_found}")

        # 4. Vocab size 일관성 확인
        try:
            tokenizer_size = len(self.blip2model.llm_tokenizer)
            model_embed_size = self.blip2model.llm_model.get_input_embeddings().weight.shape[0]
            model_lm_head_size = self.blip2model.llm_model.get_output_embeddings().weight.shape[0]

            logger.info(f"\n  Vocabulary Sizes:")
            logger.info(f"    Tokenizer:   {tokenizer_size}")
            logger.info(f"    Embed layer: {model_embed_size}")
            logger.info(f"    LM head:     {model_lm_head_size}")

            if tokenizer_size == model_embed_size == model_lm_head_size:
                logger.info("✅ Size consistency check PASSED")
            else:
                logger.error("❌ Size MISMATCH detected! This will cause training issues.")
        except Exception as e:
            logger.warning(f"⚠️  Could not verify vocab sizes: {e}")

        logger.info("="*70 + "\n")

    def on_evaluation_epoch_start(self):
        # Print validation start indicator
        if self.debug:
            print(f"\n{'='*70}")
        if self.debug:
            print(f"🔍 [VALIDATION] Starting validation at step {self.global_step}")
        if self.debug:
            print(f"🔍 [VALIDATION] Current epoch: {self.current_epoch}")
        if self.debug:
            print(f"🔍 [VALIDATION] Trainer state: {self.trainer.state.stage if hasattr(self.trainer, 'state') else 'N/A'}")
        if self.debug:
            print(f"{'='*70}\n")
        import sys
        sys.stdout.flush()

        # ======================================================================
        # Multi-Strategy Validation 설정
        # ======================================================================
        is_llada = "llada" in self.args.llm_model.lower()
        val_strategies = getattr(self.args, "val_strategies", None)

        if is_llada and val_strategies is not None and len(val_strategies) > 0:
            # Multi-strategy 모드
            self.active_val_strategies = val_strategies
            if self.debug:
                print(f"🔍 [Multi-Strategy Validation] Active strategies: {self.active_val_strategies}")
        else:
            # 단일 전략 모드 (기존 호환)
            self.active_val_strategies = ["default"]

        # 각 전략별 list_logs 초기화
        self.strategy_list_logs = {}
        for strategy in self.active_val_strategies:
            self.strategy_list_logs[strategy] = {
                "predictions": [],
                "targets": [],
                "tasks": [],
                "probs": [],
                "prompts": [],
                "input_mol_strings": [],
            }

        # LLaDA Classification 최적화용 likelihood 전략 추가
        self.strategy_list_logs["likelihood"] = {
            "predictions": [],
            "targets": [],
            "tasks": [],
            "probs": [],
            "prompts": [],
            "input_mol_strings": [],
        }

        # 기존 호환성을 위한 기본 list_logs (첫 번째 전략 참조)
        self.list_logs = self.strategy_list_logs[self.active_val_strategies[0]]

        self.debug_task_counts = {}

        self.task_subtask_name_pairs = self.trainer.datamodule.task_subtask_name_pairs

        # 전략별 generation loss 저장용 (생성 결과 기반 loss)
        self.strategy_total_gen_loss = {strategy: 0.0 for strategy in self.active_val_strategies}
        self.strategy_total_gen_loss_count = {strategy: 0 for strategy in self.active_val_strategies}
        self.strategy_dataset_gen_losses = {
            strategy: {
                task_subtask_pair: {"gen_loss": 0.0, "num_instances": 0}
                for task_subtask_pair in self.task_subtask_name_pairs
            }
            for strategy in self.active_val_strategies
        }

        self.eval_task_specific_outputs = {}

        if not hasattr(self, "task_specific_chosen_reward"):
            self.task_specific_chosen_reward = {}

        # DISABLED: self.log_model_parameters()
        # This logs 8B+ parameters to WandB/TensorBoard which takes hours
        # Only enable for debugging specific parameter issues
        # Comment added to fix hang at iteration 31

    def log_model_parameters(self):
        mean_params = []
        for name, param in self.state_dict().items():
            try:
                mean_val = param.float().mean()
                name += "_mean"
            except:
                mean_val = param

            mean_params.append((name, mean_val))

        if not mean_params:
            return

        current_step = self.global_step

        if self.global_rank == 0:
            for logger in self.trainer.loggers:
                if isinstance(logger, pl.loggers.TensorBoardLogger):
                    for name, val in mean_params:
                        logger.experiment.add_scalar(
                            f"parameters/{name}", val, current_step
                        )

                elif isinstance(logger, pl.loggers.WandbLogger):
                    run = logger.experiment
                    run.define_metric("parameters/*", step_metric="global_step")
                    log_dict = {f"parameters/{name}": val for name, val in mean_params}
                    log_dict["global_step"] = current_step
                    logger.experiment.log(log_dict)

                elif isinstance(logger, pl.loggers.CSVLogger):
                    for name, val in mean_params:
                        logger.log_metrics(
                            {f"parameters/{name}": val}, step=current_step
                        )

    def evaluation_step(self, batch, batch_idx, dataloader_idx, mode="val"):
        # ----------------------------------------------------------------------
        # [Progress Indicator] Show validation progress every 100 batches
        # ----------------------------------------------------------------------
        if batch_idx % 100 == 0 and batch_idx > 0:
            if self.debug:
                print(f"📊 Validation progress: batch {batch_idx}...")

        # ----------------------------------------------------------------------
        # [Step 1] 초기 데이터 및 변수 설정
        # ----------------------------------------------------------------------
        # Graph 데이터 분리
        if "graph" in self.args.mol_representation:
            graphs = batch["graphs"]
            additional_graphs = batch["additional_graphs"]
            is_mol_token = batch["prompt_is_mol_token"]
        else:
            graphs = None
            additional_graphs = None
            is_mol_token = None

        # 변수 안전 초기화 (UnboundLocalError 방지)
        gen_logits = None
        forward_logits = None
        attentions = None

        is_llada = "llada" in self.args.llm_model.lower()

        # Task 이름 추출 (multi-strategy에서 공통 사용)
        task_names = [id2task(task_id.item()) for task_id in batch.tasks]

        # ===========================================================================
        # [LLaDA 혼합 배치 처리] Classification과 Generation 샘플 분리
        #
        # 혼합 배치 (Classification + Generation Task가 섞인 경우):
        # - Classification 샘플: Likelihood 비교 방식으로 평가
        # - Generation 샘플: 실제 Generation으로 평가
        # - 각각의 결과를 적절한 strategy_list_logs에 누적
        #
        # LLaDA 논문 Appendix B.5 (MMLU 평가 방식):
        # - Classification 태스크에서는 generation 없이 Likelihood 비교만 사용
        # - 각 후보(True/False)의 log-likelihood를 계산하여 argmax로 예측
        # ===========================================================================

        # 샘플별 분류
        cls_indices = [i for i, t in enumerate(task_names) if t in CLASSIFICATION_BENCHMARKS]
        gen_indices = [i for i, t in enumerate(task_names) if t not in CLASSIFICATION_BENCHMARKS]

        is_all_classification = len(gen_indices) == 0
        is_all_generation = len(cls_indices) == 0
        is_mixed_batch = not is_all_classification and not is_all_generation

        # [혼합 배치 처리] Classification과 Generation을 각각 처리
        if is_llada and is_mixed_batch:
            # -----------------------------------------------------------------------
            # [혼합 배치] Classification 샘플 처리 (Likelihood 방식)
            # -----------------------------------------------------------------------
            if len(cls_indices) > 0:
                cls_task_names = [task_names[i] for i in cls_indices]

                # Classification 샘플에 대해 Likelihood 계산
                with torch.no_grad():
                    # 배치에서 Classification 샘플만 추출하여 처리
                    cls_prompt_input_ids = batch.prompt_input_ids[cls_indices]
                    cls_prompt_attention_mask = batch.prompt_attention_mask[cls_indices]
                    cls_is_mol_token = is_mol_token[cls_indices] if is_mol_token is not None else None

                    cls_probs = self.blip2model.compute_binary_prob_likelihood(
                        graphs=(graphs, additional_graphs),  # 그래프는 전체 전달 (내부에서 처리)
                        input_ids=cls_prompt_input_ids,
                        attention_mask=cls_prompt_attention_mask,
                        is_mol_token=cls_is_mol_token,
                    )
                    cls_probs_list = cls_probs.cpu().tolist()

                    # Likelihood에서 predictions 도출
                    cls_predictions = []
                    for p in cls_probs_list:
                        if p[1] > p[0]:
                            cls_predictions.append("<BOOLEAN> True </BOOLEAN>")
                        else:
                            cls_predictions.append("<BOOLEAN> False </BOOLEAN>")

                # Classification 샘플의 targets 추출
                cls_gen_labels = batch.gen_labels[cls_indices]
                cls_target_ids = torch.where(
                    cls_gen_labels == -100,
                    self.blip2model.llm_tokenizer.pad_token_id,
                    cls_gen_labels,
                )
                cls_targets = self.blip2model.llm_tokenizer.batch_decode(cls_target_ids)
                cls_targets = [t.replace(self.blip2model.llm_tokenizer.pad_token, "") for t in cls_targets]

                # Classification 샘플의 prompts/input_mol_strings 추출
                cls_input_ids = batch.input_ids[cls_indices]
                cls_prompts = self.blip2model.llm_tokenizer.batch_decode(cls_input_ids, skip_special_tokens=False)
                cls_prompts = [p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in cls_prompts]

                cls_input_mol_strings_raw = batch.input_mol_strings[cls_indices]
                cls_input_mol_strings = self.blip2model.llm_tokenizer.batch_decode(cls_input_mol_strings_raw)
                cls_input_mol_strings = [p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in cls_input_mol_strings]

                # likelihood 전략에 Classification 결과 누적
                self.strategy_list_logs["likelihood"]["predictions"].extend(cls_predictions)
                self.strategy_list_logs["likelihood"]["targets"].extend(cls_targets)
                self.strategy_list_logs["likelihood"]["tasks"].extend(cls_task_names)
                self.strategy_list_logs["likelihood"]["probs"].extend(cls_probs_list)
                self.strategy_list_logs["likelihood"]["prompts"].extend(cls_prompts)
                self.strategy_list_logs["likelihood"]["input_mol_strings"].extend(cls_input_mol_strings)

            # -----------------------------------------------------------------------
            # [혼합 배치] Generation 샘플 처리 (실제 Generation 방식)
            # -----------------------------------------------------------------------
            if len(gen_indices) > 0:
                gen_task_names = [task_names[i] for i in gen_indices]

                # Generation 샘플 추출
                gen_prompt_input_ids = batch.prompt_input_ids[gen_indices]
                gen_prompt_attention_mask = batch.prompt_attention_mask[gen_indices]
                gen_is_mol_token = is_mol_token[gen_indices] if is_mol_token is not None else None
                gen_gen_labels = batch.gen_labels[gen_indices]

                # Step-wise 로깅용 target_label, input_text 계산 (첫 번째 샘플만)
                _log_target_ids = torch.where(
                    gen_gen_labels[0:1] == -100,
                    self.blip2model.llm_tokenizer.pad_token_id,
                    gen_gen_labels[0:1],
                )
                _log_target_label = self.blip2model.llm_tokenizer.decode(_log_target_ids[0]).replace(self.blip2model.llm_tokenizer.pad_token, "").strip()
                _log_input_text = self.blip2model.llm_tokenizer.decode(gen_prompt_input_ids[0], skip_special_tokens=False).replace(self.blip2model.llm_tokenizer.pad_token, "").strip()

                # 각 전략별 Generation 수행
                for strategy in self.active_val_strategies:
                    gen_kwargs = {
                        "graphs": (graphs, additional_graphs),
                        "input_ids": gen_prompt_input_ids,
                        "attention_mask": gen_prompt_attention_mask,
                        "is_mol_token": gen_is_mol_token,
                        "max_length": self.gen_max_len,
                        "target_label": _log_target_label,
                        "input_text": _log_input_text,
                        "num_gpus": self.trainer.world_size,
                        "total_dataset_size": len(self.trainer.val_dataloaders.dataset) if hasattr(self.trainer, 'val_dataloaders') and self.trainer.val_dataloaders else None,
                        "global_rank": self.trainer.global_rank,  # GPU rank 0에서만 step-wise 로깅
                        "mode": mode,  # val/test mode 전달 (step-wise 로깅 디렉토리 분리용)
                        "strategy": strategy,  # step-wise 로깅 전략별 독립 카운터용
                        "global_step": self.trainer.global_step,  # 학습 global step (로그 파일명용)
                    }

                    # LLaDA 전용 옵션
                    gen_kwargs["steps"] = getattr(self.args, "sampling_steps", 64)
                    gen_kwargs["gen_length"] = self.gen_max_len

                    if strategy == "default":
                        gen_kwargs["remasking_strategy"] = getattr(self.args, "remasking_strategy", "random")
                        if getattr(self.args, "use_semi_ar", False):
                            gen_kwargs["use_semi_ar"] = True
                            gen_kwargs["task_name"] = gen_task_names
                    elif strategy == "random":
                        gen_kwargs["remasking_strategy"] = getattr(self.args, "remasking_strategy", "random")
                        gen_kwargs["use_semi_ar"] = False
                    elif strategy == "semi_ar":
                        gen_kwargs["remasking_strategy"] = getattr(self.args, "remasking_strategy", "random")
                        gen_kwargs["use_semi_ar"] = True
                        gen_kwargs["task_name"] = gen_task_names
                    elif strategy == "low_confidence":
                        gen_kwargs["remasking_strategy"] = "low_confidence"
                        gen_kwargs["use_semi_ar"] = False
                    elif strategy == "semi_ar_low_confidence":
                        gen_kwargs["remasking_strategy"] = "low_confidence"
                        gen_kwargs["use_semi_ar"] = True
                        gen_kwargs["task_name"] = gen_task_names
                    else:
                        gen_kwargs["remasking_strategy"] = "random"

                    with torch.no_grad():
                        gen_outputs = self.blip2model.generate(**gen_kwargs)

                    gen_predictions = gen_outputs.predictions
                    gen_predictions = [p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in gen_predictions]

                    # Generation 샘플의 targets
                    gen_target_ids = torch.where(
                        gen_gen_labels == -100,
                        self.blip2model.llm_tokenizer.pad_token_id,
                        gen_gen_labels,
                    )
                    gen_targets = self.blip2model.llm_tokenizer.batch_decode(gen_target_ids)
                    gen_targets = [t.replace(self.blip2model.llm_tokenizer.pad_token, "") for t in gen_targets]

                    # Generation 샘플의 prompts/input_mol_strings
                    gen_input_ids = batch.input_ids[gen_indices]
                    gen_prompts = self.blip2model.llm_tokenizer.batch_decode(gen_input_ids, skip_special_tokens=False)
                    gen_prompts = [p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in gen_prompts]

                    gen_input_mol_strings_raw = batch.input_mol_strings[gen_indices]
                    gen_input_mol_strings = self.blip2model.llm_tokenizer.batch_decode(gen_input_mol_strings_raw)
                    gen_input_mol_strings = [p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in gen_input_mol_strings]

                    # Probs 계산 (Generation의 경우 likelihood 기반)
                    try:
                        gen_probs = self.blip2model.compute_binary_prob_likelihood(
                            graphs=(graphs, additional_graphs),
                            input_ids=gen_prompt_input_ids,
                            attention_mask=gen_prompt_attention_mask,
                            is_mol_token=gen_is_mol_token,
                        )
                        gen_probs_list = gen_probs.cpu().tolist()
                    except Exception:
                        gen_probs_list = [[0.5, 0.5]] * len(gen_predictions)

                    # 전략별 로그에 Generation 결과 누적
                    self.strategy_list_logs[strategy]["predictions"].extend(gen_predictions)
                    self.strategy_list_logs[strategy]["targets"].extend(gen_targets)
                    self.strategy_list_logs[strategy]["tasks"].extend(gen_task_names)
                    self.strategy_list_logs[strategy]["probs"].extend(gen_probs_list)
                    self.strategy_list_logs[strategy]["prompts"].extend(gen_prompts)
                    self.strategy_list_logs[strategy]["input_mol_strings"].extend(gen_input_mol_strings)

            # 혼합 배치 처리 완료 - 나머지 로직 건너뛰기
            return

        # ===========================================================================
        # [기존 로직] 배치 내 모든 샘플이 동일한 유형인 경우
        # ===========================================================================
        if is_llada and is_all_classification:
            # LLaDA Classification: Generation 건너뛰고 Likelihood 비교만 수행
            with torch.no_grad():
                llada_probs = self.blip2model.compute_binary_prob_likelihood(
                    graphs=(graphs, additional_graphs),
                    input_ids=batch.prompt_input_ids,
                    attention_mask=batch.prompt_attention_mask,
                    is_mol_token=is_mol_token,
                )
                # probs: [batch, 2] = [P(False), P(True)]
                probs = llada_probs.cpu().tolist()

                # Likelihood에서 직접 predictions 도출 (argmax)
                # Training target 형식과 일치: "<BOOLEAN> True </BOOLEAN>" (공백 포함)
                predictions = []
                for p in probs:
                    if p[1] > p[0]:  # P(True) > P(False)
                        predictions.append("<BOOLEAN> True </BOOLEAN>")
                    else:
                        predictions.append("<BOOLEAN> False </BOOLEAN>")

            # 나머지 필요한 변수들 초기화 (generation을 건너뛰었으므로)
            gen_outputs = None
            gen_logits = None
            generated_ids = None
            strategy_outputs = {}  # 빈 딕셔너리 (multi-strategy 미사용)

            # Forward pass for loss calculation
            with torch.no_grad():
                forward_outputs = self.blip2model(batch)

            if isinstance(forward_outputs, dict):
                forward_loss = forward_outputs.get("loss")
                forward_logits = forward_outputs.get("logits", None)

                if "instance_loss" in forward_outputs:
                    forward_instance_loss = forward_outputs["instance_loss"]
                else:
                    forward_instance_loss = torch.full(
                        (batch.prompt_input_ids.shape[0],),
                        forward_loss.item(),
                        device=self.device
                    )
            else:
                forward_loss = forward_outputs
                forward_instance_loss = torch.full(
                    (batch.prompt_input_ids.shape[0],),
                    forward_loss.item(),
                    device=self.device
                )

            gen_labels = batch.gen_labels
            attentions = None

            # Skip to Step 7 (Decoding)
            # 아래 코드에서 goto 대신 플래그 사용
            skip_generation_loop = True
        else:
            skip_generation_loop = False

        # ----------------------------------------------------------------------
        # [Step 3] Generation (추론) - Multi-Strategy Support
        # ----------------------------------------------------------------------
        # 각 전략별 결과를 저장할 딕셔너리
        if not skip_generation_loop:
            strategy_outputs = {}

        # Step-wise 로깅용 target_label, input_text 계산 (첫 번째 샘플만)
        if not skip_generation_loop and batch.gen_labels is not None and len(batch.gen_labels) > 0:
            _log_target_ids_2 = torch.where(
                batch.gen_labels[0:1] == -100,
                self.blip2model.llm_tokenizer.pad_token_id,
                batch.gen_labels[0:1],
            )
            _log_target_label_2 = self.blip2model.llm_tokenizer.decode(_log_target_ids_2[0]).replace(self.blip2model.llm_tokenizer.pad_token, "").strip()
            _log_input_text_2 = self.blip2model.llm_tokenizer.decode(batch.prompt_input_ids[0], skip_special_tokens=False).replace(self.blip2model.llm_tokenizer.pad_token, "").strip()
        else:
            _log_target_label_2 = None
            _log_input_text_2 = None

        for strategy in self.active_val_strategies if not skip_generation_loop else []:
            gen_kwargs = {
                "graphs": (graphs, additional_graphs),
                "input_ids": batch.prompt_input_ids,
                "attention_mask": batch.prompt_attention_mask,
                "is_mol_token": is_mol_token,
                "max_length": self.gen_max_len,
                "target_label": _log_target_label_2,
                "input_text": _log_input_text_2,
                "num_gpus": self.trainer.world_size,
                "total_dataset_size": len(self.trainer.val_dataloaders.dataset) if hasattr(self.trainer, 'val_dataloaders') and self.trainer.val_dataloaders else None,
                "global_rank": self.trainer.global_rank,  # GPU rank 0에서만 step-wise 로깅
                "mode": mode,  # val/test mode 전달 (step-wise 로깅 디렉토리 분리용)
                "strategy": strategy,  # step-wise 로깅 전략별 독립 카운터용
                "global_step": self.trainer.global_step,  # 학습 global step (로그 파일명용)
            }

            if is_llada:
                # LLaDA 전용 옵션
                gen_kwargs["steps"] = getattr(self.args, "sampling_steps", 64)
                gen_kwargs["gen_length"] = self.gen_max_len

                # 전략에 따른 설정
                # 전략 종류:
                #   - "default": use_semi_ar config 설정 따름
                #   - "random": 전체 diffusion + random remasking
                #   - "semi_ar": Semi-AR + random remasking
                #   - "low_confidence": 전체 diffusion + low_confidence remasking
                #   - "semi_ar_low_confidence": Semi-AR + low_confidence remasking
                if strategy == "default":
                    # 기존 config 설정 사용
                    gen_kwargs["remasking_strategy"] = getattr(self.args, "remasking_strategy", "random")
                    use_semi_ar = getattr(self.args, "use_semi_ar", False)
                    if use_semi_ar:
                        gen_kwargs["use_semi_ar"] = True
                        gen_kwargs["task_name"] = task_names
                elif strategy == "random":
                    gen_kwargs["remasking_strategy"] = getattr(self.args, "remasking_strategy", "random")
                    gen_kwargs["use_semi_ar"] = False
                elif strategy == "semi_ar":
                    gen_kwargs["remasking_strategy"] = getattr(self.args, "remasking_strategy", "random")
                    gen_kwargs["use_semi_ar"] = True
                    gen_kwargs["task_name"] = task_names
                elif strategy == "low_confidence":
                    gen_kwargs["remasking_strategy"] = "low_confidence"
                    gen_kwargs["use_semi_ar"] = False
                elif strategy == "semi_ar_low_confidence":
                    gen_kwargs["remasking_strategy"] = "low_confidence"
                    gen_kwargs["use_semi_ar"] = True
                    gen_kwargs["task_name"] = task_names
                else:
                    # 알 수 없는 전략은 기본값 사용
                    logger.warning(f"Unknown validation strategy: {strategy}, using default")
                    gen_kwargs["remasking_strategy"] = "random"
            else:
                # AR 모델 전용 옵션 (전략 무시)
                gen_kwargs["num_beams"] = self.num_beams
                gen_kwargs["min_length"] = self.min_len
                gen_kwargs["output_attentions"] = self.args.log_attn_score

            # Generation 실행
            with torch.no_grad():
                gen_outputs = self.blip2model.generate(**gen_kwargs)

            # 결과 저장
            strategy_outputs[strategy] = {
                "gen_outputs": gen_outputs,
                "gen_logits": gen_outputs.logits if hasattr(gen_outputs, "logits") else None,
                "generated_ids": gen_outputs.sequences if hasattr(gen_outputs, "sequences") else None,
            }

            # 첫 번째 배치에서 전략별 결과 로깅 (GPU 0에서만)
            if batch_idx == 0 and self.trainer.global_rank == 0:
                if self.debug:
                    print(f"\n📊 [Strategy: {strategy}] Sample predictions:")
                for k in range(min(2, len(gen_outputs.predictions))):
                    if self.debug:
                        print(f"  [{k}] {gen_outputs.predictions[k][:100]}...")

        # 기존 호환성: 첫 번째 전략의 결과를 기본으로 사용
        # (LLaDA Classification 최적화 경로에서는 이미 변수들이 설정됨)
        if not skip_generation_loop:
            primary_strategy = self.active_val_strategies[0]
            gen_outputs = strategy_outputs[primary_strategy]["gen_outputs"]
            gen_logits = strategy_outputs[primary_strategy]["gen_logits"]
            generated_ids = strategy_outputs[primary_strategy]["generated_ids"]

        gen_labels = batch.gen_labels

        # ----------------------------------------------------------------------
        # [Step 3.5] 디버깅 로그 출력 - Generated Sequence (첫 번째 배치만, GPU 0만)
        # ----------------------------------------------------------------------
        # LLaDA Classification 최적화 경로에서는 generation을 건너뛰었으므로 별도 로그 출력
        if skip_generation_loop and batch_idx == 0 and self.trainer.global_rank == 0:
            if self.debug:
                print(f"\n{'='*80}")
            if self.debug:
                print(f"[LLaDA Classification] Generation skipped - using Likelihood comparison")
            if self.debug:
                print(f"  Tasks: {task_names[:3]}...")
            if self.debug:
                print(f"  Predictions (from probs): {predictions[:3]}")
            if self.debug:
                print(f"  Probs: {probs[:3]}")
            if self.debug:
                print(f"{'='*80}\n")

        if batch_idx == 0 and self.args.custom_log and self.trainer.global_rank == 0 and not skip_generation_loop:
            tokenizer = self.blip2model.llm_tokenizer
            if self.debug:
                print(f"\n{'='*80}")
            if self.debug:
                print(f"{'='*25} [DEBUG: Generation Analysis] {'='*25}")
            if self.debug:
                print(f"{'='*80}")

            for k in range(min(2, batch.prompt_input_ids.shape[0])):
                if self.debug:
                    print(f"\n{'─'*80}")
                if self.debug:
                    print(f"[Sample {k}]")
                if self.debug:
                    print(f"{'─'*80}")

                if generated_ids is not None and k < len(generated_ids):
                    # Full sequence (Input + Output)
                    full_ids = generated_ids[k]
                    input_len = len(batch.prompt_input_ids[k])

                    # Split into Input and Output
                    input_part_ids = full_ids[:input_len]
                    output_part_ids = full_ids[input_len:]

                    # === Input Part ===
                    if self.debug:
                        print(f"\n📥 [INPUT PART] Token IDs (Length: {len(input_part_ids)}):")
                    if self.debug:
                        print(f"{input_part_ids.tolist()}")

                    input_part_decoded = tokenizer.decode(input_part_ids, skip_special_tokens=False)
                    if self.debug:
                        print(f"\n📥 [INPUT PART] Decoded String:")
                    if self.debug:
                        print(input_part_decoded)

                    if self.debug:
                        print(f"\n{'-'*80}")

                    # === Output Part (Generated Only) ===
                    if self.debug:
                        print(f"\n📤 [OUTPUT PART - GENERATED ONLY] Token IDs (Length: {len(output_part_ids)}):")
                    if self.debug:
                        print(f"{output_part_ids.tolist()}")

                    # Filter out-of-vocab tokens to avoid OverflowError
                    vocab_size = len(tokenizer)
                    valid_output_ids = output_part_ids[(output_part_ids >= 0) & (output_part_ids < vocab_size)]
                    output_tokens = tokenizer.convert_ids_to_tokens(valid_output_ids.tolist())
                    if self.debug:
                        print(f"\n📤 [OUTPUT PART] Token-wise List:")
                    if self.debug:
                        print(output_tokens)

                    output_part_decoded = tokenizer.decode(output_part_ids, skip_special_tokens=False)
                    if self.debug:
                        print(f"\n📤 [OUTPUT PART] Decoded String:")
                    if self.debug:
                        print(output_part_decoded)
                    
                    # === Label Part ===
                    label_ids = gen_labels[k]
                    if self.debug:
                        print(f"\n📄 [LABEL PART] Token IDs (Length: {len(label_ids)}):")
                    if self.debug:
                        print(f"{label_ids.tolist()}")

                    # Filter out invalid token IDs (e.g., -100 ignore_index, or out-of-vocab)
                    vocab_size = len(tokenizer)
                    valid_label_ids = label_ids[(label_ids >= 0) & (label_ids < vocab_size)]
                    label_tokens = tokenizer.convert_ids_to_tokens(valid_label_ids.tolist())
                    if self.debug:
                        print(f"\n📄 [LABEL PART] Token-wise List:")
                    if self.debug:
                        print(label_tokens)
                    
                    label_decoded = tokenizer.decode(valid_label_ids, skip_special_tokens=False)
                    if self.debug:
                        print(f"\n📄 [LABEL PART] Decoded String:")
                    if self.debug:
                        print(label_decoded)

                if self.debug:
                    print(f"\n{'─'*80}")

            if self.debug:
                print(f"\n{'='*80}\n")

        # ----------------------------------------------------------------------
        # [Step 4] Forward Pass (Loss 계산)
        # ----------------------------------------------------------------------
        # LLaDA Classification 최적화 경로에서는 이미 forward pass 완료 (Step 3에서)
        if not skip_generation_loop:
            with torch.no_grad():
                forward_outputs = self.blip2model(batch)

            # 모델 타입별 Output 처리
            if is_llada:
                if isinstance(forward_outputs, dict):
                    forward_loss = forward_outputs.get("loss")
                    forward_logits = forward_outputs.get("logits", None)

                    if "instance_loss" in forward_outputs:
                        forward_instance_loss = forward_outputs["instance_loss"]
                    else:
                        forward_instance_loss = torch.full(
                            (batch.prompt_input_ids.shape[0],),
                            forward_loss.item(),
                            device=self.device
                        )
                else:
                    # 딕셔너리가 아닌 경우 (Loss 스칼라만 반환된 경우)
                    forward_loss = forward_outputs
                    forward_instance_loss = torch.full(
                            (batch.prompt_input_ids.shape[0],),
                            forward_loss.item(),
                            device=self.device
                        )
            else:
                # Autoregressive 모델 처리
                if isinstance(forward_outputs, dict) and "logits" in forward_outputs:
                     forward_logits = forward_outputs["logits"]

                # forward_logits가 없으면 outputs에서 get 시도
                logits_to_use = forward_logits if forward_logits is not None else forward_outputs.get("logits")

                forward_loss_dict = get_instance_loss(
                    logits=logits_to_use,
                    labels=batch.labels
                )
                forward_instance_loss = forward_loss_dict["instance_loss"]
                forward_loss = forward_loss_dict["loss"]

        # ----------------------------------------------------------------------
        # [Step 5] MolPO Metrics 계산 (메모리 누수 방지 로직 적용)
        # ----------------------------------------------------------------------
        metrics = {}
        if self.args.eval_molpo:
            len_tuple = gen_labels.shape[0] // self.args.molpo_batch_division
            tasks = [id2task(task_id.item()) for task_id in batch.tasks][:len_tuple]

            compute_loss_context_manager = torch.amp.autocast

            # MolPO Loss 계산
            with torch.no_grad():
                with compute_loss_context_manager(device_type="cuda"):
                    _, raw_metrics = self.get_total_molpo_loss(
                        logits=forward_logits,
                        labels=batch.labels,
                        molpo_labels=batch.molpo_labels,
                        tasks=tasks,
                        instance_loss=forward_instance_loss,
                        is_train=False,
                        molpo_batch_division=self.args.molpo_batch_division,
                        config=self.args
                    )

            # [핵심] Metrics 내부의 모든 텐서를 스칼라(Python float)로 변환
            # 이렇게 해야 GPU 그래프가 끊기고 메모리가 해제됩니다.
            for k, v in raw_metrics.items():
                if isinstance(v, torch.Tensor):
                    metrics[k] = v.item()
                else:
                    metrics[k] = v

            # Visualization을 위한 Slicing (앞부분 데이터만 사용)
            if gen_logits is not None:
                gen_logits = gen_logits[:len_tuple]

            gen_labels = gen_labels[:len_tuple]
            forward_instance_loss = forward_instance_loss[:len_tuple]

            # LLaDA Classification 최적화 경로에서는 gen_outputs가 None
            if skip_generation_loop:
                # predictions는 이미 설정됨 (Step 3에서 probs -> argmax)
                predictions = predictions[:len_tuple]
                attentions = None
            else:
                if hasattr(gen_outputs, "attentions"):
                    attentions = gen_outputs.attentions
                else:
                    attentions = None
                predictions = gen_outputs.predictions[:len_tuple]

            prompt_input_ids = batch.prompt_input_ids[:len_tuple]
            input_ids = batch.input_ids[:len_tuple]
            if generated_ids is not None:
                generated_ids = generated_ids[:len_tuple]
        else:
            tasks = [id2task(task_id.item()) for task_id in batch.tasks]

            # LLaDA Classification 최적화 경로에서는 gen_outputs가 None
            if skip_generation_loop:
                # predictions는 이미 설정됨 (Step 3에서 probs -> argmax)
                attentions = None
            else:
                if hasattr(gen_outputs, "attentions"):
                    attentions = gen_outputs.attentions
                else:
                    attentions = None
                predictions = gen_outputs.predictions

            prompt_input_ids = batch.prompt_input_ids
            input_ids = batch.input_ids

        # ----------------------------------------------------------------------
        # [Step 6] Attention Score Logging (옵션)
        # ----------------------------------------------------------------------
        if self.args.log_attn_score and attentions is not None:
            self.log_attn_score(
                prompt_input_ids=prompt_input_ids,
                mode=mode,
                is_mol_token=is_mol_token,
                attentions=attentions,
            )

        # ----------------------------------------------------------------------
        # [Step 7] Decoding 및 Prediction 정리
        # ----------------------------------------------------------------------
        prompts = self.blip2model.llm_tokenizer.batch_decode(
            input_ids, skip_special_tokens=False
        )
        predictions = [
            p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in predictions
        ]
        target_ids = torch.where(
            gen_labels == -100,
            self.blip2model.llm_tokenizer.pad_token_id,
            gen_labels,
        )
        targets = self.blip2model.llm_tokenizer.batch_decode(target_ids)
        targets = [
            t.replace(self.blip2model.llm_tokenizer.pad_token, "") for t in targets
        ]

        # ----------------------------------------------------------------------
        # [Step 8] Probs 계산 및 거대 텐서 즉시 삭제 (메모리 확보 핵심)
        # ----------------------------------------------------------------------
        # LLaDA Classification: 이미 Step 3에서 probs 계산 완료 (skip_generation_loop=True)
        # LLaDA Non-Classification: 논문 Eq.6 방식 (Likelihood 비교)으로 prob 계산
        # AR 모델: 기존 방식 (logit에서 직접 추출)
        # ----------------------------------------------------------------------
        if skip_generation_loop and is_llada and is_all_classification:
            # LLaDA Classification: probs는 이미 계산됨 (Step 3에서)
            # predictions도 이미 도출됨 (argmax from probs)
            pass  # probs 변수가 이미 설정되어 있음
        else:
            with torch.no_grad():
                if is_llada:
                    # ================================================================
                    # [LLaDA] 논문 Eq.6: Likelihood 비교 방식
                    #
                    # - 전체 응답을 마스킹하고 forward pass로 log-likelihood 계산
                    # - True/False 각각의 likelihood를 비교하여 확률 산출
                    # - Appendix B.5: "단일 토큰만 예측하는 경우 Monte Carlo 1회면 충분"
                    # ================================================================
                    try:
                        llada_probs = self.blip2model.compute_binary_prob_likelihood(
                            graphs=(graphs, additional_graphs),
                            input_ids=batch.prompt_input_ids,
                            attention_mask=batch.prompt_attention_mask,
                            is_mol_token=is_mol_token,
                        )
                        # [batch, 2] -> [[P(False), P(True)], ...] 형태의 리스트로 변환
                        probs = llada_probs.cpu().tolist()
                    except Exception as e:
                        logger.warning(f"[LLaDA Prob] compute_binary_prob_likelihood failed: {e}")
                        logger.warning("[LLaDA Prob] Falling back to AR-style prob calculation")
                        # Fallback: AR 방식 (정확하지 않지만 동작은 함)
                        probs = convert_logit2binary_prob(
                            logits=gen_logits,
                            predictions=predictions,
                            tokenizer=self.blip2model.llm_tokenizer,
                        )
                else:
                    # AR 모델: 기존 방식 (logit에서 직접 추출)
                    probs = convert_logit2binary_prob(
                        logits=gen_logits,
                        predictions=predictions,
                        tokenizer=self.blip2model.llm_tokenizer,
                    )
        
        # [중요] 사용 끝난 거대 텐서 즉시 삭제
        if gen_logits is not None:
            del gen_logits
        if 'forward_outputs' in dir() and forward_outputs is not None:
            del forward_outputs
        if forward_logits is not None:
            del forward_logits
        
        # [중요] Probs를 Python List로 변환 (GPU 메모리 해제)
        if isinstance(probs, torch.Tensor):
            probs_list = probs.detach().cpu().tolist()
        elif isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], torch.Tensor):
            probs_list = [p.item() for p in probs]
        else:
            probs_list = probs

        # 입력 문자열 Decoding
        prompts = [
            p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in prompts
        ]
        input_mol_strings = self.blip2model.llm_tokenizer.batch_decode(
            batch.input_mol_strings
        )
        input_mol_strings = [
            p.replace(self.blip2model.llm_tokenizer.pad_token, "")
            for p in input_mol_strings
        ]

        # ----------------------------------------------------------------------
        # [Step 9] 디버깅용 샘플 출력 (통합 로깅 함수 사용)
        # ----------------------------------------------------------------------
        self._log_sample_predictions(
            batch=batch,
            outputs=None,  # Validation은 이미 생성된 predictions 사용
            tasks=tasks,
            batch_idx=batch_idx,
            mode="val",
            num_samples=len(tasks),  # 모든 샘플 시도 (함수 내부에서 task당 5개 제한)
            predictions=predictions,
            targets=targets,
            prompts=prompts,
            generated_ids=generated_ids  # [NEW] 생성된 토큰 ID 전달
        )
            

        # ----------------------------------------------------------------------
        # [Step 10] 로그 리스트 누적 (Multi-Strategy Support)
        # ----------------------------------------------------------------------
        if skip_generation_loop:
            # LLaDA Classification 최적화 경로: 직접 predictions 사용
            # (generation 없이 probs에서 도출된 predictions)
            # "likelihood" 전략으로 저장하여 _likelihood suffix가 붙도록 함

            # MolPO 처리 시 slicing
            if self.args.eval_molpo:
                len_tuple = gen_labels.shape[0] // self.args.molpo_batch_division
                predictions_to_log = predictions[:len_tuple]
            else:
                predictions_to_log = predictions

            # likelihood 전략에 누적 (Classification 최적화 전용)
            self.strategy_list_logs["likelihood"]["predictions"].extend(predictions_to_log)
            self.strategy_list_logs["likelihood"]["targets"].extend(targets)
            self.strategy_list_logs["likelihood"]["tasks"].extend(tasks)
            self.strategy_list_logs["likelihood"]["probs"].extend(probs_list)
            self.strategy_list_logs["likelihood"]["prompts"].extend(prompts)
            self.strategy_list_logs["likelihood"]["input_mol_strings"].extend(input_mol_strings)

            # 기존 호환성을 위한 list_logs 업데이트
            self.list_logs = self.strategy_list_logs["likelihood"]
        else:
            # 각 전략별로 predictions 수집
            for strategy in self.active_val_strategies:
                strategy_gen_outputs = strategy_outputs[strategy]["gen_outputs"]
                strategy_predictions = strategy_gen_outputs.predictions
                strategy_predictions = [
                    p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in strategy_predictions
                ]

                # MolPO 처리 시 slicing
                if self.args.eval_molpo:
                    len_tuple = gen_labels.shape[0] // self.args.molpo_batch_division
                    strategy_predictions = strategy_predictions[:len_tuple]

                # 각 전략별 로그에 누적
                self.strategy_list_logs[strategy]["predictions"].extend(strategy_predictions)
                self.strategy_list_logs[strategy]["targets"].extend(targets)
                self.strategy_list_logs[strategy]["tasks"].extend(tasks)
                self.strategy_list_logs[strategy]["probs"].extend(probs_list)
                self.strategy_list_logs[strategy]["prompts"].extend(prompts)
                self.strategy_list_logs[strategy]["input_mol_strings"].extend(input_mol_strings)

            # 기존 호환성을 위한 list_logs 업데이트 (첫 번째 전략 참조)
            self.list_logs = self.strategy_list_logs[self.active_val_strategies[0]]

        # ----------------------------------------------------------------------
        # [Step 10.5] Generation Loss 계산 (LLaDA 전용) - Step-wise Teacher Forcing
        # ----------------------------------------------------------------------
        # LLaDA의 iterative denoising 전체 과정을 시뮬레이션:
        # - 각 step별로 해당 마스킹 비율에 맞는 입력 생성
        # - 정답 토큰으로 Teacher Forcing (실제 생성 결과 대신)
        # - 각 step의 loss에 importance weighting (1/p) 적용
        # - val_total_loss와 유사하지만 전체 trajectory를 deterministic하게 시뮬레이션
        # LLaDA Classification 최적화 경로에서는 generation이 없으므로 건너뜀
        if is_llada and not skip_generation_loop:
            with torch.no_grad():
                # Step-wise Teacher Forcing: 32 step 전체 시뮬레이션
                tf_outputs = self.blip2model.forward_stepwise_teacher_forcing(batch, steps=32)
                instance_gen_losses = tf_outputs["instance_loss"]

                # 전체 평균
                batch_gen_loss = instance_gen_losses.mean().item()

                # 모든 전략에 동일한 gen_loss 기록 (Teacher Forcing은 전략 무관)
                for strategy in self.active_val_strategies:
                    # 전략별 총 gen_loss 누적
                    curr_count = self.strategy_total_gen_loss_count[strategy]
                    new_count = curr_count + len(instance_gen_losses)
                    self.strategy_total_gen_loss[strategy] = (
                        self.strategy_total_gen_loss[strategy] * curr_count + batch_gen_loss * len(instance_gen_losses)
                    ) / new_count
                    self.strategy_total_gen_loss_count[strategy] = new_count

                    # 태스크별 gen_loss 누적
                    instance_gen_losses_cpu = instance_gen_losses.detach().cpu()
                    for i in range(len(instance_gen_losses_cpu)):
                        gen_loss_val = instance_gen_losses_cpu[i].item()

                        if gen_loss_val != gen_loss_val:  # NaN check
                            continue

                        task_subtask_pair = tasks[i]
                        if task_subtask_pair not in self.strategy_dataset_gen_losses[strategy]:
                            self.strategy_dataset_gen_losses[strategy][task_subtask_pair] = {
                                "gen_loss": 0.0,
                                "num_instances": 0,
                            }

                        curr_dict = self.strategy_dataset_gen_losses[strategy][task_subtask_pair]
                        n = curr_dict["num_instances"]
                        curr_dict["gen_loss"] = (curr_dict["gen_loss"] * n + gen_loss_val) / (n + 1)
                        curr_dict["num_instances"] = n + 1

        # ----------------------------------------------------------------------
        # [Step 11] MolPO Logging
        # ----------------------------------------------------------------------
        if self.args.eval_molpo:
            self.task_specific_logging(
                outputs=metrics,
                tasks=tasks,
                mode=mode,
                epoch_end=False,
                task_specific_outputs=self.eval_task_specific_outputs,
                num_moving_samples=None,
            )

            for k, v in self.task_specific_chosen_reward.items():
                self.log(
                    f"train/{k}/bar_reward",
                    v,
                    batch_size=self.args.batch_size,
                    sync_dist=False,
                )
        
        # ----------------------------------------------------------------------
        # [Step 12] 최종 로깅 및 리턴
        # ----------------------------------------------------------------------
        # val_total_loss 로깅 (ModelCheckpoint 모니터링용)
        if is_llada:
            if skip_generation_loop:
                # LLaDA Classification 최적화 경로: forward_loss 사용
                # Generation을 건너뛰었으므로 forward_loss를 val_total_loss로 로깅
                if forward_loss is not None:
                    self.log("val_total_loss", forward_loss.item(), sync_dist=True, prog_bar=True, logger=True)
            elif hasattr(self, 'strategy_total_gen_loss'):
                # 일반 LLaDA 경로: 전략별 gen_loss 단순 평균
                valid_gen_losses = []
                for strategy in self.active_val_strategies:
                    count = self.strategy_total_gen_loss_count[strategy]
                    if count > 0:
                        # 각 전략별 평균 gen_loss를 리스트에 추가
                        valid_gen_losses.append(self.strategy_total_gen_loss[strategy] / count)
                if valid_gen_losses:
                    # 전략별 평균 gen_loss의 단순 평균
                    avg_gen_loss = sum(valid_gen_losses) / len(valid_gen_losses)
                    self.log("val_total_loss", avg_gen_loss, sync_dist=True, prog_bar=True, logger=True)
        
        # [강제 메모리 정리]
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # [중요] Return None: PyTorch Lightning이 결과를 저장하지 않도록 함 (OOM 방지 최후의 수단)
        return None
    
    def log_attn_score(self, prompt_input_ids, mode, is_mol_token, attentions):
        num_steps = len(attentions)
        seq_lengths = prompt_input_ids.shape[1]
        all_layers_attn = [
            torch.stack(attentions[step_idx])[..., :seq_lengths]
            for step_idx in range(1, num_steps)
        ]

        # [num_steps, num_heads, max_generated_length, max_generated_length] -> [num_steps, batch_size, max_generated_length]
        full_attn_mean = torch.stack(all_layers_attn).mean(dim=(1, 3)).squeeze()

        # 토크나이저에서 동적으로 SELFIES 토큰 ID 가져오기
        tokenizer = self.blip2model.llm_tokenizer
        selfies_start_token_id = tokenizer.convert_tokens_to_ids("<SELFIES>")
        selfies_end_token_id = tokenizer.convert_tokens_to_ids("</SELFIES>")

        selfies_mask = torch.zeros_like(prompt_input_ids, dtype=torch.bool)
        st_batch_indices, start_indices = (
            prompt_input_ids == selfies_start_token_id
        ).nonzero(as_tuple=True)
        end_batch_indices, end_indices = (
            prompt_input_ids == selfies_end_token_id
        ).nonzero(as_tuple=True)

        valid_start_mask = torch.isin(st_batch_indices, end_batch_indices)
        st_batch_indices = st_batch_indices[valid_start_mask]
        start_indices = start_indices[valid_start_mask]

        start_indices += 1
        end_indices -= 1

        seq_range = torch.arange(seq_lengths, device=prompt_input_ids.device).unsqueeze(
            0
        )

        broadcasted_start = start_indices.unsqueeze(1)
        broadcasted_end = end_indices.unsqueeze(1)

        sequence_masks = (seq_range >= broadcasted_start) & (
            seq_range <= broadcasted_end
        )
        temp_mask = torch.zeros_like(prompt_input_ids, dtype=torch.bool)
        temp_mask.scatter_add_(
            0,
            end_batch_indices.unsqueeze(1).expand(-1, seq_lengths),
            sequence_masks,
        )
        selfies_mask = selfies_mask | temp_mask

        # mol_attn_score = full_attn_mean[:, mol_token_mask]

        mol_mean_scores = []
        mol_sum_scores = []
        selfies_mean_scores = []
        selfies_sum_scores = []

        # exception for batch size 1
        if len(full_attn_mean.shape) == 2:
            full_attn_mean = full_attn_mean.unsqueeze(1)

        for i in range(prompt_input_ids.shape[0]):
            if "graph" in self.args.mol_representation:
                mol_mean_scores.append(
                    full_attn_mean[:, i, is_mol_token[i]].mean(dim=-1).mean(dim=0)
                )
                mol_sum_scores.append(
                    full_attn_mean[:, i, is_mol_token[i]].sum(dim=-1).sum(dim=0)
                )
            if "string" in self.args.mol_representation:
                selfies_mean_scores.append(
                    full_attn_mean[:, i, selfies_mask[i]].mean(dim=-1).mean(dim=0)
                )
                selfies_sum_scores.append(
                    full_attn_mean[:, i, selfies_mask[i]].sum(dim=-1).sum(dim=0)
                )
        if "graph" in self.args.mol_representation:
            mol_mean_scores = torch.stack(mol_mean_scores)
            mol_mean_scores = torch.where(
                torch.isnan(mol_mean_scores),
                torch.zeros_like(mol_mean_scores),
                mol_mean_scores,
            )
            mol_sum_scores = torch.stack(mol_sum_scores)
            mol_sum_scores = torch.where(
                torch.isnan(mol_sum_scores),
                torch.zeros_like(mol_sum_scores),
                mol_sum_scores,
            )

            self.log(
                f"{mode}/graph_attn_mean_score",
                mol_mean_scores.mean().item(),
                sync_dist=False,
                batch_size=prompt_input_ids.shape[0],
            )
            self.log(
                f"{mode}/graph_attn_sum_score",
                mol_sum_scores.mean().item(),
                sync_dist=False,
                batch_size=prompt_input_ids.shape[0],
            )

        if "string" in self.args.mol_representation:
            selfies_mean_scores = torch.stack(selfies_mean_scores)
            selfies_mean_scores = torch.where(
                torch.isnan(selfies_mean_scores),
                torch.zeros_like(selfies_mean_scores),
                selfies_mean_scores,
            )
            selfies_sum_scores = torch.stack(selfies_sum_scores)
            selfies_sum_scores = torch.where(
                torch.isnan(selfies_sum_scores),
                torch.zeros_like(selfies_sum_scores),
                selfies_sum_scores,
            )

            self.log(
                f"{mode}/selfies_attn_mean_score",
                selfies_mean_scores.mean().item(),
                sync_dist=False,
                batch_size=prompt_input_ids.shape[0],
            )
            self.log(
                f"{mode}/selfies_attn_sum_score",
                selfies_sum_scores.mean().item(),
                sync_dist=False,
                batch_size=prompt_input_ids.shape[0],
            )

    def on_evaluation_epoch_end(self, mode="val") -> None:
        if self.debug:
            print(f"\nDevice {self.device} on_evaluation_epoch_end start")

        # test 모드에서는 학습 완료 시점의 global_step 사용, 그 외에는 현재 global_step 사용
        if mode == "test" and self._trained_global_step is not None:
            step_for_filename = self._trained_global_step
        else:
            step_for_filename = self.global_step

        if self.args.eval_molpo:
            self.task_specific_logging(
                outputs=None,
                tasks=None,
                mode=mode,
                epoch_end=True,
                task_specific_outputs=self.eval_task_specific_outputs,
                num_moving_samples=None,
            )

        # ======================================================================
        # Multi-Strategy Evaluation
        # ======================================================================
        all_strategy_results = {}

        # [수정] 모든 전략 평가: Generation 전략 + likelihood 전략 (데이터가 있으면)
        # 기존 버그: likelihood에 데이터가 있으면 다른 전략을 무시했음
        strategies_to_evaluate = list(self.active_val_strategies)
        if len(self.strategy_list_logs["likelihood"]["predictions"]) > 0:
            # likelihood 전략도 추가 (덮어쓰기가 아닌 추가!)
            strategies_to_evaluate.append("likelihood")

        for strategy in strategies_to_evaluate:
            strategy_logs = self.strategy_list_logs[strategy]
            strategy_suffix = f"_{strategy}" if strategy != "default" else ""

            if self.debug:
                print(f"\n{'='*70}")
            if self.debug:
                print(f"📊 [Strategy: {strategy}] Evaluating predictions...")
            if self.debug:
                print(f"{'='*70}")

            evaluation_results, failed_cases = per_device_evaluate(
                predictions=strategy_logs["predictions"],
                targets=strategy_logs["targets"],
                tasks=strategy_logs["tasks"],
                prompts=strategy_logs["prompts"],
                input_mol_strings=strategy_logs["input_mol_strings"],
                tokenizer=self.blip2model.llm_tokenizer,
                total_task_subtask_pairs=self.task_subtask_name_pairs,
            )
            
            all_strategy_results[strategy] = {
                "evaluation_results": evaluation_results,
                "failed_cases": failed_cases,
            }

            # 전략별 prediction 파일 저장
            self.save_predictions(
                predictions=strategy_logs["predictions"],
                targets=strategy_logs["targets"],
                tasks=strategy_logs["tasks"],
                prompts=strategy_logs["prompts"],
                probs=strategy_logs["probs"],
                input_mol_strings=strategy_logs["input_mol_strings"],
                filename=f"{self.args.mode}-step{step_for_filename}-{self.global_rank}{strategy_suffix}-outputs.json",
            )

            self.save_predictions(
                predictions=failed_cases["predictions"],
                targets=failed_cases["targets"],
                tasks=failed_cases["tasks"],
                prompts=failed_cases["prompts"],
                input_mol_strings=failed_cases["input_mol_strings"],
                filename=f"{self.args.mode}-step{step_for_filename}-{self.global_rank}{strategy_suffix}-failed_cases.json",
            )

        # evaluate classification tasks - prepare common structures
        self.cls_task_subtask_name_pair = [
            task_subtask_pair
            for task_subtask_pair in self.task_subtask_name_pairs
            if task_subtask_pair.split("/")[0] in CLASSIFICATION_BENCHMARKS
        ]
        # sort classification task_subtask_name_pairs in alphabetical order
        self.cls_task_subtask_name_pair.sort()
        self.cls_task_subtask_name_pair_dict = {
            task_subtask_pair: idx
            for idx, task_subtask_pair in enumerate(self.cls_task_subtask_name_pair)
        }
        # get inverse of self.cls_task_subtask_name_pair_dict
        self.cls_task_subtask_name_pair_dict_inv = {
            idx: task_subtask_pair
            for task_subtask_pair, idx in self.cls_task_subtask_name_pair_dict.items()
        }

        # evaluate the other tasks - now per strategy
        flattened_metric_keys = []
        flattened_metric_tensors = torch.empty(size=(0, 2), device=self.device)

        # Process each strategy's evaluation results with strategy suffix
        for strategy in strategies_to_evaluate:
            strategy_suffix = f"_{strategy}" if strategy != "default" else ""
            evaluation_results = all_strategy_results[strategy]["evaluation_results"]

            # tied to order of self.task_subtask_name_pairs
            for task_subtask_pair in evaluation_results:
                for metric in evaluation_results[task_subtask_pair]:
                    flattened_metric_keys.append(f"{mode}/{task_subtask_pair}/{metric}{strategy_suffix}")
                    metric_value = evaluation_results[task_subtask_pair][metric]
                    num_instance = evaluation_results[task_subtask_pair]["num_instances"]
                    metric_count_pair = [metric_value * num_instance, num_instance]

                    flattened_metric_tensors = torch.cat(
                        [
                            flattened_metric_tensors,
                            torch.tensor(
                                metric_count_pair,
                                device=self.device,
                            ).unsqueeze(0),
                        ],
                        dim=0,
                    )

        # 전략별 Generation Loss 로깅 (LLaDA 전용)
        # likelihood 전략은 generation을 건너뛰므로 gen_loss가 없음
        if hasattr(self, 'strategy_dataset_gen_losses'):
            for strategy in strategies_to_evaluate:
                if strategy not in self.strategy_dataset_gen_losses:
                    continue  # likelihood 등 generation 없는 전략은 스킵
                strategy_suffix = f"_{strategy}" if strategy != "default" else ""
                for dataset in self.strategy_dataset_gen_losses[strategy].keys():
                    gen_loss_data = self.strategy_dataset_gen_losses[strategy][dataset]
                    if gen_loss_data["num_instances"] > 0:
                        flattened_metric_keys.append(f"{mode}/{dataset}/gen_loss{strategy_suffix}")
                        metric_value = gen_loss_data["gen_loss"]
                        num_instance = gen_loss_data["num_instances"]
                        metric_count_pair = [metric_value * num_instance, num_instance]
                        flattened_metric_tensors = torch.cat(
                            [
                                flattened_metric_tensors,
                                torch.tensor(
                                    metric_count_pair,
                                    device=self.device,
                                ).unsqueeze(0),
                            ],
                            dim=0,
                        )

        # Prepare per-strategy classification tensors
        self.num_per_device_cls = 10000
        self.strategy_per_device_cls_tensors = {}
        for strategy in strategies_to_evaluate:
            strategy_logs = self.strategy_list_logs[strategy]
            per_device_cls_tensor = torch.zeros(
                size=(self.num_per_device_cls, 4), device=self.device, dtype=torch.float
            )
            cls_idx = 0
            for i in range(len(strategy_logs["tasks"])):
                task_subtask_pair = strategy_logs["tasks"][i]
                if task_subtask_pair in self.cls_task_subtask_name_pair_dict.keys():
                    probs = strategy_logs["probs"][i]
                    label = int(
                        "True" in strategy_logs["targets"][i]
                        or "true" in strategy_logs["targets"][i]
                    )
                    pair_ids = self.cls_task_subtask_name_pair_dict[task_subtask_pair]
                    per_device_cls_tensor[cls_idx] = torch.tensor(
                        [probs[0], probs[1], pair_ids, label],
                        device=self.device,
                        dtype=torch.float,
                    )
                    cls_idx += 1
            self.strategy_per_device_cls_tensors[strategy] = per_device_cls_tensor

        # For backward compatibility, use first evaluated strategy's tensor
        self.per_device_cls_tensor = self.strategy_per_device_cls_tensors[strategies_to_evaluate[0]]

        assert flattened_metric_tensors.shape[0] == len(
            flattened_metric_keys
        ), f"flattened_metric_tensors.shape[0]: {flattened_metric_tensors.shape[0]}, len(flattened_metric_keys): {len(flattened_metric_keys)}"
        # prepare idx to sort the flattened_metric_keys in alphabetical order
        indexed_flattened_metric_keys = list(enumerate(flattened_metric_keys))
        # get indices to sort the flattened_metric_keys in alphabetical order
        sorted_idx = [
            idx
            for idx, key in sorted(indexed_flattened_metric_keys, key=lambda x: x[1])
        ]
        flattened_metric_keys = [flattened_metric_keys[idx] for idx in sorted_idx]
        flattened_metric_tensors = flattened_metric_tensors[sorted_idx]

        if self.trainer.world_size > 1:
            if self.debug:
                print("gather the metrics across devices")
            raw_gathered_flattened_metric_tensors = self.all_gather(
                flattened_metric_tensors
            )  # [world_size, num_metrics, metric_value * per_device_instance_count, per_device_instance_count]

            # get rid of nan values (nan )
            gathered_flattened_metric_tensors = torch.where(
                torch.isnan(raw_gathered_flattened_metric_tensors),
                torch.zeros_like(raw_gathered_flattened_metric_tensors),
                raw_gathered_flattened_metric_tensors,
            )
            scaled_flattened_metric_tensors = gathered_flattened_metric_tensors[
                :, :, 0
            ].sum(dim=0)

            # total_instance_count = gathered_flattened_metric_tensors[:, :, 1].sum(dim=0)
            total_instance_count = torch.where(
                torch.isnan(gathered_flattened_metric_tensors[:, :, 0]),
                torch.zeros_like(gathered_flattened_metric_tensors[:, :, 1]),
                gathered_flattened_metric_tensors[:, :, 1],
            ).sum(dim=0)
            total_instance_count_include_nan = gathered_flattened_metric_tensors[
                :, :, 1
            ].sum(dim=0)

            # Gather per-strategy classification tensors
            strategy_uniform_cls_tensors = {}
            for strategy in strategies_to_evaluate:
                gathered_cls_tensor = self.all_gather(self.strategy_per_device_cls_tensors[strategy])
                strategy_uniform_cls_tensors[strategy] = torch.cat(
                    [cls_tensor for cls_tensor in gathered_cls_tensor], dim=0
                )
            # For backward compatibility
            uniform_cls_tensor = strategy_uniform_cls_tensors[strategies_to_evaluate[0]]
        else:
            scaled_flattened_metric_tensors = flattened_metric_tensors[:, 0]
            total_instance_count = flattened_metric_tensors[:, 1]
            total_instance_count_include_nan = total_instance_count

            strategy_uniform_cls_tensors = self.strategy_per_device_cls_tensors
            uniform_cls_tensor = self.per_device_cls_tensor

        # if total_instance_count is 0, set the metric to null value
        averaged_flattened_metric_tensors = torch.where(
            total_instance_count > 0,
            scaled_flattened_metric_tensors / total_instance_count,
            torch.tensor(float("nan"), device=self.device),
        )

        # evaluate classification tasks - now per strategy
        cls_flattened_metric_keys = []
        cls_flattented_metric_tensors = torch.empty(size=(0, 1), device=self.device)

        for strategy in strategies_to_evaluate:
            strategy_suffix = f"_{strategy}" if strategy != "default" else ""
            strategy_cls_tensor = strategy_uniform_cls_tensors[strategy]

            # get total_cls_tensor only where total_cls_tensor[:, :2].sum(-1) > 0
            actual_cls_tensor = strategy_cls_tensor[strategy_cls_tensor[:, :2].sum(-1) > 0]

            if actual_cls_tensor.shape[0] > 0:
                total_probs = actual_cls_tensor[:, :2].cpu()
                total_labels = actual_cls_tensor[:, 3].cpu().to(torch.long)
                tasks_subtask_idx = actual_cls_tensor[:, 2].to(torch.int32).tolist()
                # get task names using self.cls_task_subtask_name_pair_dict_inv
                total_tasks = [
                    self.cls_task_subtask_name_pair_dict_inv[idx] for idx in tasks_subtask_idx
                ]
                classification_evaluation_result = total_device_evaluate(
                    total_labels=total_labels,
                    total_probs=total_probs,
                    total_tasks=total_tasks,
                    classification_task_subtask_pairs=self.cls_task_subtask_name_pair,
                )

                # convert classification_evaluation_result to flattened_metric_keys and flattened_metric_tensors
                for task_subtask_pair in classification_evaluation_result:
                    for metric in classification_evaluation_result[task_subtask_pair]:
                        cls_flattened_metric_keys.append(f"{mode}/{task_subtask_pair}/{metric}{strategy_suffix}")
                        metric_value = classification_evaluation_result[task_subtask_pair][
                            metric
                        ]

                        cls_flattented_metric_tensors = torch.cat(
                            [
                                cls_flattented_metric_tensors,
                                torch.tensor(
                                    [metric_value],
                                    device=self.device,
                                ).unsqueeze(0),
                            ],
                            dim=0,
                        )

        cls_flattented_metric_tensors = cls_flattented_metric_tensors.squeeze(-1)
        flattened_metric_keys += cls_flattened_metric_keys
        averaged_flattened_metric_tensors = torch.cat(
            [averaged_flattened_metric_tensors, cls_flattented_metric_tensors], dim=0
        )

        # sort flattened_metric_keys in alphabetical order
        indexed_flattened_metric_keys = list(enumerate(flattened_metric_keys))
        # get indices to sort the flattened_metric_keys in alphabetical order
        sorted_idx = [
            idx
            for idx, key in sorted(indexed_flattened_metric_keys, key=lambda x: x[1])
        ]
        flattened_metric_keys = [flattened_metric_keys[idx] for idx in sorted_idx]
        averaged_flattened_metric_tensors = averaged_flattened_metric_tensors[
            sorted_idx
        ]
        if self.debug:
            print(
            "============================== Evaluation Results =============================="
        )
        for i, key in enumerate(flattened_metric_keys):
            if (
                "num_instances" in key
            ):  # num_instance here is actually mean of quadratic of num_instance
                continue
            if self.debug:
                print(f"{key}: {averaged_flattened_metric_tensors[i]} ")
            self.log(
                key,
                averaged_flattened_metric_tensors[i],
                sync_dist=False,
                rank_zero_only=True,
            )
        if self.debug:
            print(
            "================================================================================="
        )

        result_path = os.path.join(
            self.logger.log_dir,
            f"{mode}-step{step_for_filename}-{self.global_rank}-results.json",
        )
        # zip the flattened metrics and averaged_flattened_metric_tensors
        result_dict = {}
        for i in range(len(flattened_metric_keys)):
            result_dict[flattened_metric_keys[i]] = averaged_flattened_metric_tensors[
                i
            ].item()
        # save result_dict in result_path
        with open(result_path, "w") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)

        # ======================================================================
        # Multi-Strategy Summary Metrics
        # ======================================================================
        if len(strategies_to_evaluate) >= 1:
            if self.debug:
                print(f"\n{'='*70}")
            if self.debug:
                print("📊 Multi-Strategy Comparison Summary")
            if self.debug:
                print(f"{'='*70}")

            # Strategy에서 remasking_strategy 매핑 (config 기반)
            config_remasking = getattr(self.args, "remasking_strategy", "random")
            strategy_to_remasking = {
                "default": config_remasking,
                "random": config_remasking,  # config의 remasking_strategy 사용
                "semi_ar": config_remasking,  # config의 remasking_strategy 사용
                "low_confidence": "low_confidence",  # 명시적 override
                "semi_ar_low_confidence": "low_confidence",  # 명시적 override
                "likelihood": "none",  # likelihood는 generation 안 함
            }

            strategy_summaries = {}
            for strategy in strategies_to_evaluate:
                strategy_eval_results = all_strategy_results[strategy]["evaluation_results"]
                strategy_failed = all_strategy_results[strategy]["failed_cases"]

                # remasking_strategy 결정
                remasking_strategy = strategy_to_remasking.get(strategy, "random")

                # Task 유형별 metric 집계
                mol2text_metrics = {"bleu2": [], "bleu4": [], "rouge1": [], "rouge2": [], "rougeL": [], "meteor": []}
                text2mol_metrics = {"validity_ratio": [], "exact_match_ratio": [], "MACCS_FTS": [], "RDK_FTS": [], "morgan_FTS": [], "bleu_smiles": []}
                regression_metrics = {"mae": [], "rmse": []}
                classification_metrics = {"accuracy": [], "f1": [], "roc_auc": []}

                total_samples = 0
                total_failure_rate = 0
                task_count = 0

                for task_pair, metrics in strategy_eval_results.items():
                    task_name = task_pair.split("/")[0]
                    num_instances = metrics.get("num_instances", 0)

                    if num_instances == 0:
                        continue

                    total_samples += num_instances

                    if "failure_rate" in metrics:
                        total_failure_rate += metrics["failure_rate"]
                        task_count += 1

                    # MOL2TEXT tasks (generation)
                    if task_name in MOL2TEXT_BENCHMARKS:
                        for key in mol2text_metrics.keys():
                            if key in metrics and not math.isnan(metrics[key]):
                                mol2text_metrics[key].append((metrics[key], num_instances))

                    # TEXT2MOL / REACTION tasks (molecule generation)
                    elif task_name in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS:
                        for key in text2mol_metrics.keys():
                            if key in metrics and not math.isnan(metrics[key]):
                                text2mol_metrics[key].append((metrics[key], num_instances))

                    # Regression tasks
                    elif task_name in REGRESSION_BENCHMARKS:
                        for key in regression_metrics.keys():
                            if key in metrics and not math.isnan(metrics[key]):
                                regression_metrics[key].append((metrics[key], num_instances))

                    # Classification tasks
                    elif task_name in CLASSIFICATION_BENCHMARKS:
                        for key in classification_metrics.keys():
                            if key in metrics and not math.isnan(metrics[key]):
                                classification_metrics[key].append((metrics[key], num_instances))

                # Weighted average 계산 함수
                def weighted_avg(metric_list):
                    if not metric_list:
                        return None
                    total_weight = sum(w for _, w in metric_list)
                    if total_weight == 0:
                        return None
                    return sum(v * w for v, w in metric_list) / total_weight

                avg_failure_rate = total_failure_rate / task_count if task_count > 0 else 0

                # 전략별 average_gen_loss 계산 (LLaDA 전용)
                avg_gen_loss = None
                if hasattr(self, 'strategy_total_gen_loss') and strategy in self.strategy_total_gen_loss:
                    gen_loss_count = self.strategy_total_gen_loss_count.get(strategy, 0)
                    if gen_loss_count > 0:
                        avg_gen_loss = self.strategy_total_gen_loss[strategy]

                # Strategy summary 구성
                strategy_summary = {
                    "strategy": strategy,
                    "remasking_strategy": remasking_strategy,
                    "total_samples": total_samples,
                    "num_failed_cases": len(strategy_failed["predictions"]),
                    "avg_failure_rate": avg_failure_rate,
                }

                # average_gen_loss 추가 (있는 경우에만)
                if avg_gen_loss is not None:
                    strategy_summary["average_gen_loss"] = avg_gen_loss

                # MOL2TEXT metrics
                mol2text_summary = {}
                for key, values in mol2text_metrics.items():
                    avg_val = weighted_avg(values)
                    if avg_val is not None:
                        mol2text_summary[f"avg_{key}"] = avg_val
                if mol2text_summary:
                    strategy_summary["mol2text"] = mol2text_summary

                # TEXT2MOL metrics
                text2mol_summary = {}
                for key, values in text2mol_metrics.items():
                    avg_val = weighted_avg(values)
                    if avg_val is not None:
                        text2mol_summary[f"avg_{key}"] = avg_val
                if text2mol_summary:
                    strategy_summary["text2mol"] = text2mol_summary

                # Regression metrics
                regression_summary = {}
                for key, values in regression_metrics.items():
                    avg_val = weighted_avg(values)
                    if avg_val is not None:
                        regression_summary[f"avg_{key}"] = avg_val
                if regression_summary:
                    strategy_summary["regression"] = regression_summary

                # Classification metrics
                classification_summary = {}
                for key, values in classification_metrics.items():
                    avg_val = weighted_avg(values)
                    if avg_val is not None:
                        classification_summary[f"avg_{key}"] = avg_val
                if classification_summary:
                    strategy_summary["classification"] = classification_summary

                strategy_summaries[strategy] = strategy_summary

                # Print summary
                if self.debug:
                    print(f"\n[{strategy}] (remasking: {remasking_strategy})")
                if self.debug:
                    print(f"  - Total Samples: {total_samples}")
                if self.debug:
                    print(f"  - Failed Cases: {len(strategy_failed['predictions'])}")
                if self.debug:
                    print(f"  - Average Failure Rate: {avg_failure_rate:.4f}")
                if avg_gen_loss is not None:
                    if self.debug:
                        print(f"  - Average Gen Loss: {avg_gen_loss:.4f}")

                if "mol2text" in strategy_summary:
                    if self.debug:
                        print(f"  [MOL2TEXT]")
                    for k, v in strategy_summary["mol2text"].items():
                        if self.debug:
                            print(f"    - {k}: {v:.4f}")

                if "text2mol" in strategy_summary:
                    if self.debug:
                        print(f"  [TEXT2MOL]")
                    for k, v in strategy_summary["text2mol"].items():
                        if self.debug:
                            print(f"    - {k}: {v:.4f}")

                if "regression" in strategy_summary:
                    if self.debug:
                        print(f"  [REGRESSION]")
                    for k, v in strategy_summary["regression"].items():
                        if self.debug:
                            print(f"    - {k}: {v:.4f}")

                if "classification" in strategy_summary:
                    if self.debug:
                        print(f"  [CLASSIFICATION]")
                    for k, v in strategy_summary["classification"].items():
                        if self.debug:
                            print(f"    - {k}: {v:.4f}")

                # Log strategy-specific summary metrics
                strategy_suffix = f"_{strategy}" if strategy != "default" else ""
                self.log(
                    f"{mode}/strategy{strategy_suffix}/avg_failure_rate",
                    avg_failure_rate,
                    sync_dist=False,
                    rank_zero_only=True,
                )

                # Log average_gen_loss with strategy and remasking_strategy
                if avg_gen_loss is not None:
                    self.log(
                        f"{mode}/average_gen_loss_{strategy}_{remasking_strategy}",
                        avg_gen_loss,
                        sync_dist=False,
                        rank_zero_only=True,
                    )

                # Log task-type specific metrics
                for task_type in ["mol2text", "text2mol", "regression", "classification"]:
                    if task_type in strategy_summary:
                        for metric_name, metric_value in strategy_summary[task_type].items():
                            self.log(
                                f"{mode}/strategy{strategy_suffix}/{task_type}/{metric_name}",
                                metric_value,
                                sync_dist=False,
                                rank_zero_only=True,
                            )

            # Save strategy comparison to file
            strategy_comparison_path = os.path.join(
                self.logger.log_dir,
                f"{mode}-step{step_for_filename}-{self.global_rank}-strategy_comparison.json",
            )
            with open(strategy_comparison_path, "w") as f:
                json.dump(strategy_summaries, f, ensure_ascii=False, indent=4)

            if self.debug:
                print(f"\n📁 Strategy comparison saved to: {strategy_comparison_path}")
            if self.debug:
                print(f"{'='*70}")

        # ======================================================================
        # Epoch 단위 metric 누적 (step별 validation 결과를 epoch 단위로 집계)
        # ======================================================================
        if hasattr(self, 'epoch_val_metrics'):
            for i, key in enumerate(flattened_metric_keys):
                if "num_instances" in key:
                    continue
                metric_value = averaged_flattened_metric_tensors[i].item()
                if not math.isnan(metric_value):
                    if key not in self.epoch_val_metrics:
                        self.epoch_val_metrics[key] = []
                    self.epoch_val_metrics[key].append(metric_value)
            self.epoch_val_count = getattr(self, 'epoch_val_count', 0) + 1

        if self.debug:
            print(f"\nDevice {self.device} on_evaluation_epoch_end end")

    def on_train_epoch_end(self) -> None:
        """Epoch 종료 시 epoch 전체에 대한 validation metric summary 로깅"""
        if not hasattr(self, 'epoch_val_metrics') or not self.epoch_val_metrics:
            if self.debug:
                print(f"[Epoch {self.current_epoch}] No validation metrics to summarize")
            return

        if self.debug:
            print(f"\n{'='*70}")
        if self.debug:
            print(f"📊 [EPOCH {self.current_epoch} SUMMARY] Aggregating {self.epoch_val_count} validation runs")
        if self.debug:
            print(f"{'='*70}")

        epoch_summary = {}
        for key, values in self.epoch_val_metrics.items():
            if len(values) > 0:
                # NaN 제거 후 평균 계산
                valid_values = [v for v in values if not math.isnan(v)]
                if len(valid_values) > 0:
                    avg_value = sum(valid_values) / len(valid_values)
                    # epoch suffix 추가한 새 key 생성
                    epoch_key = key.replace("/", "/epoch_") if key.count("/") >= 2 else key + "_epoch"
                    epoch_summary[epoch_key] = avg_value

                    # 로깅
                    self.log(
                        epoch_key,
                        avg_value,
                        sync_dist=False,
                        rank_zero_only=True,
                    )

        # Epoch summary를 파일로 저장 (rank 0에서만)
        if self.global_rank == 0 and hasattr(self, 'logger') and hasattr(self.logger, 'log_dir') and self.logger.log_dir:
            epoch_summary_path = os.path.join(
                self.logger.log_dir,
                f"epoch{self.current_epoch}-summary.json",
            )
            with open(epoch_summary_path, "w") as f:
                json.dump(epoch_summary, f, ensure_ascii=False, indent=4)
            if self.debug:
                print(f"📁 Epoch summary saved to: {epoch_summary_path}")

        # 주요 metric 출력
        if self.debug:
            print(f"\n[Epoch {self.current_epoch}] Key metrics (averaged over {self.epoch_val_count} validations):")
        for key, value in sorted(epoch_summary.items()):
            if any(m in key for m in ['accuracy', 'roc_auc', 'f1', 'total_loss']):
                if self.debug:
                    print(f"  {key}: {value:.4f}")

        if self.debug:
            print(f"{'='*70}\n")

    def on_before_optimizer_step(self, optimizer):
        """
        Optimizer step 직전에 gradient scaling 적용

        Embedding과 LM Head의 경우:
        - 기존 vocab (idx < original_vocab_size): base LR 사용
        - 새로운 vocab (idx >= original_vocab_size): gradient를 스케일링하여 더 높은 effective LR 적용

        이 방식은 같은 weight tensor에서 일부 row만 다른 LR을 적용하는 효과를 냄
        """
        # Gradient scaling 적용 (embed/head의 new vocab 부분)
        if hasattr(self, '_embed_head_split_info'):
            info = self._embed_head_split_info
            original_vocab_size = info['original_vocab_size']
            lr_ratio_embed = info['lr_ratio_embed']
            lr_ratio_head = info['lr_ratio_head']

            # Embedding 파라미터에 gradient scaling 적용
            for param, name in info['embed_params']:
                if param.grad is not None and param.shape[0] > original_vocab_size:
                    # new vocab 부분의 gradient를 lr_ratio만큼 스케일링
                    # effective_lr = base_lr * ratio 효과
                    param.grad[original_vocab_size:] *= lr_ratio_embed

                    if self.debug and self.global_step <= 5:
                        if self.debug:
                            print(f"  [Grad Scaling] {name}: scaled new vocab grads by {lr_ratio_embed:.2f}")

            # Head 파라미터에 gradient scaling 적용
            for param, name in info['head_params']:
                if param.grad is not None and param.shape[0] > original_vocab_size:
                    param.grad[original_vocab_size:] *= lr_ratio_head

                    if self.debug and self.global_step <= 5:
                        if self.debug:
                            print(f"  [Grad Scaling] {name}: scaled new vocab grads by {lr_ratio_head:.2f}")

        # 디버깅: 첫 몇 번의 optimizer step에서만
        if not hasattr(self, '_debug_weights_before'):
            self._debug_weights_before = {}

        if self.global_step > 5:
            return

        for name, param in self.named_parameters():
            if param.requires_grad:
                if ('wte' in name.lower() and 'original_module' not in name) or \
                   ('ff_out' in name.lower() and 'blocks' not in name and 'original_module' not in name):
                    self._debug_weights_before[name] = {
                        'weight_sum': param.data.sum().item(),
                        'weight_norm': param.data.norm(2).item(),
                        'grad_sum': param.grad.sum().item() if param.grad is not None else None,
                        'grad_norm': param.grad.norm(2).item() if param.grad is not None else None,
                    }

        if self._debug_weights_before:
            if self.debug:
                print(f"\n[DEBUG on_before_optimizer_step] Global Step {self.global_step}")
            for name, vals in self._debug_weights_before.items():
                if self.debug:
                    print(f"  {name}: weight_sum={vals['weight_sum']:.6f}, grad_sum={vals['grad_sum']}, grad_norm={vals['grad_norm']}")

    def on_after_backward(self):
        """Backward 직후 gradient norm을 5개 그룹으로 분리하여 로깅

        이 시점에서 gradient가 계산된 상태이므로 정확한 grad_norm을 얻을 수 있음.
        training_step에서는 backward() 전이므로 grad가 None이거나 이전 step 값임.
        """
        # 5개 그룹 gradient norm 로깅 (매 step)
        self._log_5group_grad_norms()

        # 첫 몇 번의 step에서만 디버깅 출력
        if self.global_step > 5:
            return

        wte_grad_info = []
        ff_out_grad_info = []

        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                if 'wte' in name.lower() and 'original_module' not in name:
                    wte_grad_info.append((name, param.grad.norm(2).item(), param.grad.sum().item()))
                elif 'ff_out' in name.lower() and 'blocks' not in name and 'original_module' not in name:
                    ff_out_grad_info.append((name, param.grad.norm(2).item(), param.grad.sum().item()))

        if wte_grad_info or ff_out_grad_info:
            if self.debug:
                print(f"\n[DEBUG on_after_backward] Global Step {self.global_step}")
            for name, norm, sumv in wte_grad_info:
                if self.debug:
                    print(f"  [wte] {name}: grad_norm={norm:.6f}, grad_sum={sumv:.6f}")
            for name, norm, sumv in ff_out_grad_info:
                if self.debug:
                    print(f"  [ff_out] {name}: grad_norm={norm:.6f}, grad_sum={sumv:.6f}")

    def _log_5group_grad_norms(self):
        """5개 파라미터 그룹의 gradient norm을 wandb에 로깅

        Groups:
        - grad_norm/lora: LoRA 파라미터
        - grad_norm/embed_orig: 기존 vocab embedding (idx < original_vocab_size)
        - grad_norm/embed_new: 새 vocab embedding (idx >= original_vocab_size)
        - grad_norm/head_orig: 기존 vocab head (idx < original_vocab_size)
        - grad_norm/head_new: 새 vocab head (idx >= original_vocab_size)
        """
        original_vocab_size = getattr(self.args, 'original_vocab_size', 126349)

        grad_norms = {
            'lora': 0.0,
            'embed_orig': 0.0,
            'embed_new': 0.0,
            'head_orig': 0.0,
            'head_new': 0.0,
        }

        for name, param in self.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            name_lower = name.lower()

            # LoRA 파라미터
            if 'lora' in name_lower:
                grad_norms['lora'] += param.grad.norm(2).item() ** 2

            # Embedding (wte / embed_tokens)
            elif ('wte' in name_lower or 'embed_tokens' in name_lower) and 'original_module' not in name_lower:
                if param.dim() >= 2 and param.shape[0] > original_vocab_size:
                    # vocab dimension이 첫 번째인 경우
                    orig_grad = param.grad[:original_vocab_size]
                    new_grad = param.grad[original_vocab_size:]
                    grad_norms['embed_orig'] += orig_grad.norm(2).item() ** 2
                    grad_norms['embed_new'] += new_grad.norm(2).item() ** 2
                else:
                    # 분리 불가 (새 vocab 없음)
                    grad_norms['embed_orig'] += param.grad.norm(2).item() ** 2

            # Head (ff_out / lm_head)
            elif ('ff_out' in name_lower or 'lm_head' in name_lower) and 'blocks' not in name_lower and 'original_module' not in name_lower:
                if param.dim() >= 2 and param.shape[0] > original_vocab_size:
                    orig_grad = param.grad[:original_vocab_size]
                    new_grad = param.grad[original_vocab_size:]
                    grad_norms['head_orig'] += orig_grad.norm(2).item() ** 2
                    grad_norms['head_new'] += new_grad.norm(2).item() ** 2
                else:
                    grad_norms['head_orig'] += param.grad.norm(2).item() ** 2

        # L2 norm 계산 및 로깅
        import math
        for key, squared_sum in grad_norms.items():
            norm = math.sqrt(squared_sum)
            self.log(f"grad_norm/{key}", norm, batch_size=self.args.batch_size, sync_dist=False)

    """
    def _on_after_backward_old(self):
        # Log the gradient norm for all parameters
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                self.log(
                    f"grad_norm/{name}",
                    grad_norm,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
    """
    


def check_model_parameters(model, keyword):
    from collections import OrderedDict

    # save as ordered dict
    trainable_params_dict = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad and keyword in name:
            trainable_params_dict[name] = param
    return trainable_params_dict


def get_instance_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, logits.shape[-1])
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)

    # custom forward to get not reduced loss
    loss_fct_not_reduced = CrossEntropyLoss(reduction="none")
    loss_not_reduced = loss_fct_not_reduced(shift_logits, shift_labels).view(
        labels.size(0), -1
    )
    # normalization exclude default ignore index -100
    instance_non_pad_tokens = torch.where(
        shift_labels != -100,
        torch.tensor(1).to(shift_labels.device),
        torch.tensor(0).to(shift_labels.device),
    ).view(labels.size(0), -1)
    instance_loss = (loss_not_reduced * instance_non_pad_tokens).sum(
        dim=-1
    ) / instance_non_pad_tokens.sum(dim=-1)
    loss = (
        loss_not_reduced * instance_non_pad_tokens
    ).sum() / instance_non_pad_tokens.sum()
    return {"loss": loss, "instance_loss": instance_loss}


import torch
from typing import Tuple
from torch.nn import functional as F


def molpo_loss(
    chosen_rewards: torch.FloatTensor,
    chosen_loss_mask: torch.BoolTensor,
    rejected_rewards: torch.FloatTensor,
    rejected_loss_mask: torch.BoolTensor,
    loss_type="sigmoid",
    beta=1.0,
    gamma_beta_ratio=0.0,
    molpo_lambda=None,
    avg_chosen_rewards=None,
    margin_clip_scale=-1,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the molpo loss for a batch of policy model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the molpo loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    # calculate molpo loss
    margin = chosen_rewards - rejected_rewards
    if margin_clip_scale > 0:
        max_clip = margin_clip_scale * torch.abs(avg_chosen_rewards)
        min_clip = -torch.abs(margin)
        margin = torch.clamp(margin, min=min_clip, max=max_clip)

    if molpo_lambda is not None or isinstance(molpo_lambda, str):
        assert molpo_lambda <= 0, f"molpo_lambda: {molpo_lambda} should be <= 0.0."
        logits = margin - molpo_lambda * avg_chosen_rewards
    else:
        logits = margin - beta * gamma_beta_ratio
    if loss_type == "sigmoid":
        losses = -F.logsigmoid(beta * logits)
    elif loss_type == "hinge":
        losses = torch.relu(1 - beta * logits)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge']"
        )
    loss_mask = torch.where(
        (rejected_loss_mask.sum(-1) > 0) & (chosen_loss_mask.sum(-1) > 0),
        True,
        False,
    )
    loss = losses[loss_mask].mean()

    return loss, losses


def anchor_loss(
    avg_chosen_rewards: torch.FloatTensor,
    rejected_rewards: torch.FloatTensor,
    rejected_lambda: float,
    loss_type: str = "sigmoid",
):
    assert (
        rejected_lambda >= 0.0
    ), f"rejected_lambda: {rejected_lambda} should be >= 0.0."
    rejected_logits = rejected_rewards - rejected_lambda * avg_chosen_rewards

    if loss_type == "sigmoid":
        anchor_rejected_losses = -F.logsigmoid(rejected_logits)
    elif loss_type == "hinge":
        anchor_rejected_losses = torch.relu(-rejected_logits)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge']"
        )
    return anchor_rejected_losses


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = True,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        label_pad_token_id: The label pad token id.
        is_encoder_decoder: Whether the model is an encoder-decoder model.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            "Logits (batch and sequence length dim) and labels must have the same shape."
        )

    if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    target_truncation_mask = torch.where(loss_mask.sum(-1) > 0, True, False)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    # just add one for target truncated instance.
    # the loss is not used for backprop, so it's fine to have a dummy loss for truncated instances.
    numerically_stable_mask = loss_mask.sum(-1) + ~target_truncation_mask * 1e-6

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / numerically_stable_mask
    else:
        return (per_token_logps * loss_mask).sum(-1)

    

# def concatenated_forward(
#     all_logits: torch.FloatTensor,
#     all_labels: torch.LongTensor,
#     instance_loss: torch.FloatTensor = None,
#     label_pad_token_id: int = -100,
#     molpo_batch_division: int = 2,
#     config=None
# ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
#     """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

#     We do this to avoid doing two forward passes, because it's faster for FSDP.
#     """

#     all_logps = get_batch_logps(
#         logits=all_logits,
#         labels=all_labels,
#         average_log_prob=True,
#         is_encoder_decoder=False,
#         label_pad_token_id=label_pad_token_id,
#     )
#     len_tuple = all_labels.shape[0] // molpo_batch_division

#     sft_instance_loss = instance_loss[:len_tuple]

#     if 'llada' in config.llm_model.lower():
#         ...
        
    
#     if molpo_batch_division == 2:
#         chosen_logps = all_logps[:len_tuple]
#         chosen_labels = all_labels[:len_tuple]
#         chosen_loss_mask = chosen_labels[:, 1:].clone() != -100

#         rejected_logps = all_logps[len_tuple:]
#         rejected_labels = all_labels[len_tuple:]
#         rejected_loss_mask = rejected_labels[:, 1:].clone() != -100

#         out_dict = {
#             "sft_instance_loss": sft_instance_loss,
#             "chosen_logps": chosen_logps,
#             "chosen_loss_mask": chosen_loss_mask,
#             "rejected_logps": rejected_logps,
#             "rejected_loss_mask": rejected_loss_mask,
#         }
#     elif molpo_batch_division == 3:
#         sft_logps = all_logps[:len_tuple]
#         sft_labels = all_labels[:len_tuple]
#         sft_loss_mask = sft_labels[:, 1:].clone() != -100

#         chosen_logps = all_logps[len_tuple : 2 * len_tuple]
#         chosen_labels = all_labels[len_tuple : 2 * len_tuple]
#         chosen_loss_mask = chosen_labels[:, 1:].clone() != -100

#         rejected_logps = all_logps[2 * len_tuple :]
#         rejected_labels = all_labels[2 * len_tuple :]
#         rejected_loss_mask = rejected_labels[:, 1:].clone() != -100

#         out_dict = {
#             "sft_instance_loss": sft_instance_loss,
#             "sft_logps": sft_logps,
#             "sft_loss_mask": sft_loss_mask,
#             "chosen_logps": chosen_logps,
#             "chosen_loss_mask": chosen_loss_mask,
#             "rejected_logps": rejected_logps,
#             "rejected_loss_mask": rejected_loss_mask,
#         }

#     return out_dict

from typing import Tuple
import torch

def concatenated_forward(
    all_logits: torch.FloatTensor,
    all_labels: torch.LongTensor,
    instance_loss: torch.FloatTensor = None,
    label_pad_token_id: int = -100,
    molpo_batch_division: int = 2,
    config=None
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    """
    is_llada = (config is not None) and hasattr(config, "llm_model") and ("llada" in config.llm_model.lower())

    if is_llada:
        # [LLaDA Path]
        # Diffusion 모델은 get_batch_logps(AR 방식)를 수행하지 않고 instance_loss를 바로 사용
        if instance_loss is None:
            raise ValueError("LLaDA requires instance_loss for MolPO in concatenated_forward().")
        
        # Loss는 낮을수록 좋으므로, Reward 관점에서는 (-)를 붙여야 함
        all_logps = -instance_loss 
    else:
        # [Autoregressive Path]
        # 기존 모델들은 Log Probability 계산 (Shift 연산 포함)
        all_logps = get_batch_logps(
            logits=all_logits,
            labels=all_labels,
            average_log_prob=True,
            is_encoder_decoder=False,
            label_pad_token_id=label_pad_token_id,
        )

    len_tuple = all_labels.shape[0] // molpo_batch_division
    sft_instance_loss = instance_loss[:len_tuple] if instance_loss is not None else None
    # =========================
    # 아래는 원본 구조/변수명 최대한 유지
    # =========================
    if molpo_batch_division == 2:
        chosen_logps = all_logps[:len_tuple]
        chosen_labels = all_labels[:len_tuple]

        rejected_logps = all_logps[len_tuple:]
        rejected_labels = all_labels[len_tuple:]

        # AR은 shift 마스크(기존 유지), LLaDA는 shift 없이 labels 기준이 더 자연스러움
        if is_llada:
            chosen_loss_mask = chosen_labels.clone() != label_pad_token_id
            rejected_loss_mask = rejected_labels.clone() != label_pad_token_id
        else:
            chosen_loss_mask = chosen_labels[:, 1:].clone() != label_pad_token_id
            rejected_loss_mask = rejected_labels[:, 1:].clone() != label_pad_token_id

        out_dict = {
            "sft_instance_loss": sft_instance_loss,
            "chosen_logps": chosen_logps,
            "chosen_loss_mask": chosen_loss_mask,
            "rejected_logps": rejected_logps,
            "rejected_loss_mask": rejected_loss_mask,
        }

    elif molpo_batch_division == 3:
        sft_logps = all_logps[:len_tuple]
        sft_labels = all_labels[:len_tuple]

        chosen_logps = all_logps[len_tuple : 2 * len_tuple]
        chosen_labels = all_labels[len_tuple : 2 * len_tuple]

        rejected_logps = all_logps[2 * len_tuple :]
        rejected_labels = all_labels[2 * len_tuple :]

        if is_llada:
            sft_loss_mask = sft_labels.clone() != label_pad_token_id
            chosen_loss_mask = chosen_labels.clone() != label_pad_token_id
            rejected_loss_mask = rejected_labels.clone() != label_pad_token_id
        else:
            sft_loss_mask = sft_labels[:, 1:].clone() != label_pad_token_id
            chosen_loss_mask = chosen_labels[:, 1:].clone() != label_pad_token_id
            rejected_loss_mask = rejected_labels[:, 1:].clone() != label_pad_token_id

        out_dict = {
            "sft_instance_loss": sft_instance_loss,
            "sft_logps": sft_logps,
            "sft_loss_mask": sft_loss_mask,
            "chosen_logps": chosen_logps,
            "chosen_loss_mask": chosen_loss_mask,
            "rejected_logps": rejected_logps,
            "rejected_loss_mask": rejected_loss_mask,
        }
    else:
        raise ValueError(f"Invalid molpo_batch_division={molpo_batch_division}")

    return out_dict
