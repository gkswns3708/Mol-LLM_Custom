import os
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
from data_utils import CLASSIFICATION_BENCHMARKS, id2task
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
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint["state_dict"].pop(key)

        if hasattr(self, "task_specific_chosen_reward"):
            checkpoint[f"task_specific_chosen_reward"] = (
                self.task_specific_chosen_reward
            )

        self.log_model_parameters()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "task_specific_chosen_reward" in checkpoint:
            self.task_specific_chosen_reward = checkpoint["task_specific_chosen_reward"]

    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)
        print(args, " - args")
        self.args = args
        self.num_beams = args.num_beams
        self.gen_max_len = args.gen_max_len
        self.min_len = args.min_len
        self.tune_llm = args.tune_llm
        self.on_second_stage = False
        # set strict_loading to False to load model in a lightweight way
        self.strict_loading = False
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
        if self.args.optimizer == "adafactor":
            print("Using adafactor optimizer")
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
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.args.init_lr,
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
                self.scheduler = WarmupStableDecayLRScheduler(
                    optimizer=optimizer,
                    max_step=max_step,
                    init_lr=self.args.init_lr,
                    min_lr=self.args.min_lr,
                    warmup_steps=self.args.warmup_steps,
                    decay_ratio=0.1, # 논문과 동일하게 마지막 10% 구간에서 Decay
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
            if tasks[i] in CLASSIFICATION_BENCHMARKS and probs is not None:
                instance["prob"] = probs[i]
            instances.append(instance)
        os.makedirs(self.logger.log_dir, exist_ok=True)

        with open(os.path.join(self.logger.log_dir, filename), "w") as f:
            json.dump(instances, f, ensure_ascii=False, indent=4)

    def on_test_epoch_start(self) -> None:
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
        # 안전한 처리를 위해 dict 여부 확인 후 pop 수행
        if isinstance(outputs, dict):
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
            print(f"\n{'='*20} [CRITICAL ERROR] Loss is NaN/Inf {'='*20}")
            print(f"Global Step: {self.global_step}, Batch Index: {batch_idx}")
            print(f"Current Batch Tasks: {tasks}")
            
            print("\n[Possible Causes Candidates]")
            print("1. Learning Rate Explosion: 초기 LR이 너무 높거나 Warmup이 부족할 수 있습니다.")
            print("2. Gradient Explosion: gradient_clip_val 설정을 확인하세요.")
            print("3. Invalid Data/Labels: Label이 전부 -100이거나 Input에 NaN이 있을 수 있습니다.")
            print("4. Logit Instability: 모델 출력 Logit이 발산했는지 확인하세요.")

            # 1. Label 통계 확인
            if "labels" in batch:
                labels = batch.labels
                valid_labels = (labels != -100).sum()
                print(f"\n[Label Statistics] Total: {labels.numel()}, Valid(!=-100): {valid_labels.item()}")
                if valid_labels == 0:
                    print("!!! Warning: All labels are -100 (Ignore Index). Loss becomes 0 or NaN. !!!")
            
            # 2. Logits 통계 확인
            if logits is not None:
                print(f"\n[Logits Statistics] Max: {logits.max().item()}, Min: {logits.min().item()}, Mean: {logits.mean().item()}")
                if torch.isnan(logits).any():
                    print("!!! Logits contain NaN values !!!")
            
            # 3. 입력 샘플 디코딩하여 출력 (데이터 문제 확인용)
            try:
                print("\n[Sample Input Decoding]")
                tokenizer = self.blip2model.llm_tokenizer
                # batch 객체 구조에 따라 input_ids 가져오기
                input_ids = batch.input_ids if hasattr(batch, 'input_ids') else batch.prompt_input_ids
                if input_ids is not None:
                    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                    print(f"Decoded Input (truncated 500 chars): {decoded[:500]} ...")
            except Exception as e:
                print(f"Failed to decode sample: {e}")
            
            print("="*60 + "\n")
            # 필요 시 에러를 발생시켜 학습 중단: raise ValueError("Training stopped due to NaN")

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

        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            batch_size=self.args.batch_size,
            sync_dist=False,
        )

        self.log(
            f"train_total_loss",
            loss.clone().detach().item() if loss is not None else 0.0,
            batch_size=self.args.batch_size,
            sync_dist=False,
        )

        for k, v in outputs.items():
            # 이미 pop 했거나 처리한 키 제외
            if k in ["logits", "loss", "instance_loss"]: continue
            
            val_to_log = v.mean() if isinstance(v, torch.Tensor) else v
            self.log(
                f"train/{k}",
                float(val_to_log),
                batch_size=self.args.batch_size,
                sync_dist=False,
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

        return loss

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
                k: v for k, v in outputs.items() if v.shape != torch.Size([])
            }

            for task in tasks:
                task_specific_outputs.setdefault(
                    task, {k: [] for k in new_outputs.keys()}
                )

            for metric, v in new_outputs.items():

                for i in range(v.shape[0]):
                    if torch.isnan(v[i]):
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

    def on_train_epoch_start(self) -> None:
        self.task_specific_outputs = {}
        # if not hasattr(self, "task_specific_chosen_reward"):
        self.task_specific_chosen_reward = {}

        self.train_list_predictions = []
        self.train_list_targets = []
        self.train_list_prompts = []
        self.train_list_tasks = []
        self.train_list_probs = []
        self.train_total_avg_loss = 0.0
        self.train_total_seen_data_size = 0

        self.trainer.train_dataloader.collate_fn.current_epoch = (
            self.trainer.current_epoch
        )

        self.log_model_parameters()

    def on_evaluation_epoch_start(self):
        self.list_logs = {
            "predictions": [],
            "targets": [],
            "tasks": [],
            "probs": [],
            "prompts": [],
            "input_mol_strings": [],
        }
        self.debug_task_counts = {}
        self.total_avg_loss = 0.0
        self.total_seen_data_size = 0
        # self.task_subtask_name_pairs = self.trainer.datamodule.dataset_split[
        #     "test"
        # ].task_subtask_name_pairs

        self.task_subtask_name_pairs = self.trainer.datamodule.task_subtask_name_pairs

        self.eval_dataset_losses = {
            task_subtask_pair: {"avg_loss": 0.0, "num_instances": 0}
            for task_subtask_pair in self.task_subtask_name_pairs
        }
        self.eval_task_specific_outputs = {}

        if not hasattr(self, "task_specific_chosen_reward"):
            self.task_specific_chosen_reward = {}

        self.log_model_parameters()

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
        
        # ----------------------------------------------------------------------
        # [Step 2] 디버깅 로그 출력 (첫 번째 배치만)
        # ----------------------------------------------------------------------
        if batch_idx == 0 and self.args.custom_log: 
            print(f"\n{'='*20} [DEBUG: Input Token Analysis] {'='*20}")
            tokenizer = self.blip2model.llm_tokenizer
            input_ids_batch = batch.prompt_input_ids
            
            for k in range(min(2, len(input_ids_batch))):
                ids = input_ids_batch[k]
                print(f"\n[Sample {k}] Input IDs (Length: {len(ids)}):")
                print(f"{ids.tolist()}")
                
                tokens = tokenizer.convert_ids_to_tokens(ids)
                print(f"[Sample {k}] Token-wise List:")
                print(tokens)
                
                decoded_text = tokenizer.decode(ids, skip_special_tokens=False)
                print(f"[Sample {k}] Full Decoded String:")
                print(decoded_text)
                print("-" * 60)
            print(f"{'='*60}\n")
        
        is_llada = "llada" in self.args.llm_model.lower()
        
        # ----------------------------------------------------------------------
        # [Step 3] Generation (추론) - 메모리 절약을 위해 no_grad 필수
        # ----------------------------------------------------------------------
        gen_kwargs = {
            "graphs": (graphs, additional_graphs),
            "input_ids": batch.prompt_input_ids,
            "attention_mask": batch.prompt_attention_mask,
            "is_mol_token": is_mol_token,
            "max_length": self.gen_max_len,
        }
        
        if is_llada:
            # LLaDA 전용 옵션
            gen_kwargs["steps"] = getattr(self.args, "sampling_steps", 64) 
            gen_kwargs["gen_length"] = self.gen_max_len
            gen_kwargs["remasking_strategy"] = getattr(self.args, "remasking_strategy", "low_confidence")
        else:
            # AR 모델 전용 옵션
            gen_kwargs["num_beams"] = self.num_beams
            gen_kwargs["min_length"] = self.min_len
            gen_kwargs["output_attentions"] = self.args.log_attn_score
        
        # Generation 실행
        with torch.no_grad():
            gen_outputs = self.blip2model.generate(**gen_kwargs)
        
        # Logits 안전하게 추출
        if hasattr(gen_outputs, "logits"):
            gen_logits = gen_outputs.logits
        else:
            gen_logits = None 

        gen_labels = batch.gen_labels
        
        # ----------------------------------------------------------------------
        # [Step 4] Forward Pass (Loss 계산)
        # ----------------------------------------------------------------------
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

            if hasattr(gen_outputs, "attentions"):
                attentions = gen_outputs.attentions
            else:
                attentions = None
                
            predictions = gen_outputs.predictions[:len_tuple]
            prompt_input_ids = batch.prompt_input_ids[:len_tuple]
            input_ids = batch.input_ids[:len_tuple]
        else:
            tasks = [id2task(task_id.item()) for task_id in batch.tasks]
            
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
        with torch.no_grad():
            probs = convert_logit2binary_prob(
                logits=gen_logits,
                predictions=predictions,
                tokenizer=self.blip2model.llm_tokenizer,
            )
        
        # [중요] 사용 끝난 거대 텐서 즉시 삭제
        del gen_logits
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
        # [Step 9] 디버깅용 샘플 출력
        # ----------------------------------------------------------------------
        for k, task_name in enumerate(tasks):
            if task_name not in self.debug_task_counts:
                self.debug_task_counts[task_name] = 0
            
            # 너무 많은 로그 방지 (Task 당 5개까지만)
            if self.debug_task_counts[task_name] >= 5:
                continue
            
            self.debug_task_counts[task_name] += 1
            print(f"\n[DEBUG] Rank {self.global_rank} | Batch {batch_idx} | Sample {k} | Task: {task_name}")
            print(f"Prompt_Text : {prompts[k]}")
            print(f"Target      : {targets[k]}")
            print(f"Prediction  : {predictions[k]}")
            print("-" * 40)

        # ----------------------------------------------------------------------
        # [Step 10] 로그 리스트 누적 (CPU 메모리만 사용)
        # ----------------------------------------------------------------------
        self.list_logs["predictions"].extend(predictions)
        self.list_logs["targets"].extend(targets)
        self.list_logs["tasks"].extend(tasks)
        self.list_logs["probs"].extend(probs_list)
        self.list_logs["prompts"].extend(prompts)
        self.list_logs["input_mol_strings"].extend(input_mol_strings)

        # ----------------------------------------------------------------------
        # [Step 11] Loss 누적 (반드시 .item() 사용)
        # ----------------------------------------------------------------------
        batch_size = len(input_ids)
        new_data_weight = batch_size / (self.total_seen_data_size + batch_size)
        
        # [중요] .item()을 사용하여 스칼라 값으로 변환 후 누적
        loss_val = forward_loss.item() if isinstance(forward_loss, torch.Tensor) else forward_loss
        self.total_avg_loss += (loss_val - self.total_avg_loss) * new_data_weight
        self.total_seen_data_size += batch_size

        # Instance Loss 처리
        # 텐서를 CPU로 옮겨서 처리 (GPU 접근 최소화)
        if isinstance(forward_instance_loss, torch.Tensor):
            inst_loss_cpu = forward_instance_loss.detach().cpu()
        else:
            inst_loss_cpu = forward_instance_loss

        for i in range(len(inst_loss_cpu)):
            val = inst_loss_cpu[i]
            if isinstance(val, torch.Tensor):
                val = val.item() # 스칼라 변환
            
            if val != val: # NaN check
                continue

            task_subtask_pair = tasks[i]
            if task_subtask_pair not in self.eval_dataset_losses:
                self.eval_dataset_losses[task_subtask_pair] = {
                    "avg_loss": 0.0,
                    "num_instances": 0,
                }
            
            # 평균 업데이트
            curr_dict = self.eval_dataset_losses[task_subtask_pair]
            curr_dict["avg_loss"] *= curr_dict["num_instances"] / (curr_dict["num_instances"] + 1)
            curr_dict["avg_loss"] += val / (curr_dict["num_instances"] + 1)
            curr_dict["num_instances"] += 1

        # ----------------------------------------------------------------------
        # [Step 12] MolPO Logging
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
        # [Step 13] 최종 로깅 및 리턴
        # ----------------------------------------------------------------------
        # Loss 로깅
        self.log("val_total_loss", loss_val, sync_dist=True, prog_bar=True, logger=True)
        
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

        selfies_start_token_id = 35743
        selfies_end_token_id = 35744

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
        print(f"\nDevice {self.device} on_evaluation_epoch_end start")

        if self.args.eval_molpo:
            self.task_specific_logging(
                outputs=None,
                tasks=None,
                mode=mode,
                epoch_end=True,
                task_specific_outputs=self.eval_task_specific_outputs,
                num_moving_samples=None,
            )

        evaluation_results, failed_cases = per_device_evaluate(
            predictions=self.list_logs["predictions"],
            targets=self.list_logs["targets"],
            tasks=self.list_logs["tasks"],
            prompts=self.list_logs["prompts"],
            input_mol_strings=self.list_logs["input_mol_strings"],
            tokenizer=self.blip2model.llm_tokenizer,
            total_task_subtask_pairs=self.task_subtask_name_pairs,
        )

        self.save_predictions(
            predictions=self.list_logs["predictions"],
            targets=self.list_logs["targets"],
            tasks=self.list_logs["tasks"],
            prompts=self.list_logs["prompts"],
            probs=self.list_logs["probs"],
            input_mol_strings=self.list_logs["input_mol_strings"],
            filename=(
                f"{self.args.mode}-step{self.global_step}-{self.global_rank}-outputs.json"
                if self.args.mode == "val"
                else f"{self.args.mode}-{self.global_rank}-outputs.json"
            ),
        )

        self.save_predictions(
            predictions=failed_cases["predictions"],
            targets=failed_cases["targets"],
            tasks=failed_cases["tasks"],
            prompts=failed_cases["prompts"],
            input_mol_strings=failed_cases["input_mol_strings"],
            filename=(
                f"{self.args.mode}-step{self.global_step}-{self.global_rank}-failed_cases.json"
                if self.args.mode == "val"
                else f"{self.args.mode}-{self.global_rank}-failed_cases.json"
            ),
        )

        self.log(
            f"{mode}/total_loss",
            self.total_avg_loss,
            sync_dist=True,
            batch_size=self.total_seen_data_size,
        )

        # evaluate classification tasks
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
        self.num_per_device_cls = 10000
        self.per_device_cls_tensor = torch.zeros(
            size=(self.num_per_device_cls, 4), device=self.device, dtype=torch.float
        )
        non_zero_count = 0
        cls_idx = 0
        for i in range(len(self.list_logs["tasks"])):
            task_subtask_pair = self.list_logs["tasks"][i]
            if task_subtask_pair in self.cls_task_subtask_name_pair_dict.keys():
                probs = self.list_logs["probs"][i]
                label = int(
                    "True" in self.list_logs["targets"][i]
                    or "true" in self.list_logs["targets"][i]
                )
                pair_ids = self.cls_task_subtask_name_pair_dict[task_subtask_pair]
                self.per_device_cls_tensor[cls_idx] = torch.tensor(
                    [probs[0], probs[1], pair_ids, label],
                    device=self.device,
                    dtype=torch.float,
                )
                non_zero_count += 1
                cls_idx += 1

        # evaluate the other tasks
        flattened_metric_keys = []
        flattened_metric_tensors = torch.empty(size=(0, 2), device=self.device)

        # tied to order of self.task_subtask_name_pairs
        for task_subtask_pair in evaluation_results:
            for metric in evaluation_results[task_subtask_pair]:
                flattened_metric_keys.append(f"{mode}/{task_subtask_pair}/{metric}")
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

        # tied to order of self.task_subtask_name_pairs
        for dataset in self.eval_dataset_losses.keys():
            flattened_metric_keys.append(f"{mode}/{dataset}/avg_loss")
            metric_value = self.eval_dataset_losses[dataset]["avg_loss"]
            num_instance = self.eval_dataset_losses[dataset]["num_instances"]
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

            gathered_cls_tensor = self.all_gather(self.per_device_cls_tensor)
            uniform_cls_tensor = torch.cat(
                [cls_tensor for cls_tensor in gathered_cls_tensor], dim=0
            )
        else:
            scaled_flattened_metric_tensors = flattened_metric_tensors[:, 0]
            total_instance_count = flattened_metric_tensors[:, 1]
            total_instance_count_include_nan = total_instance_count

            uniform_cls_tensor = self.per_device_cls_tensor

        # if total_instance_count is 0, set the metric to null value
        averaged_flattened_metric_tensors = torch.where(
            total_instance_count > 0,
            scaled_flattened_metric_tensors / total_instance_count,
            torch.tensor(float("nan"), device=self.device),
        )

        # evaluate classification tasks
        # get total_cls_tensor only where total_cls_tensor[:, :2].sum(-1) > 0
        actual_cls_tensor = uniform_cls_tensor[uniform_cls_tensor[:, :2].sum(-1) > 0]

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
        cls_flattened_metric_keys = []
        cls_flattented_metric_tensors = torch.empty(size=(0, 1), device=self.device)
        for task_subtask_pair in classification_evaluation_result:
            for metric in classification_evaluation_result[task_subtask_pair]:
                cls_flattened_metric_keys.append(f"{mode}/{task_subtask_pair}/{metric}")
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
        print(
            "============================== Evaluation Results =============================="
        )
        for i, key in enumerate(flattened_metric_keys):
            if (
                "num_instances" in key
            ):  # num_instance here is actually mean of quadratic of num_instance
                continue
            print(f"{key}: {averaged_flattened_metric_tensors[i]} ")
            self.log(
                key,
                averaged_flattened_metric_tensors[i],
                sync_dist=False,
                rank_zero_only=True,
            )
        print(
            "================================================================================="
        )

        result_path = os.path.join(
            self.logger.log_dir,
            f"{mode}-step{self.global_step}-{self.global_rank}-results.json",
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

        print(f"\nDevice {self.device} on_evaluation_epoch_end end")

    """
    def on_after_backward(self):
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
