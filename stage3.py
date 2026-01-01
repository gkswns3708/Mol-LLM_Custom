import faulthandler
faulthandler.enable()
import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from data_module import Stage3DM
from model.blip2_stage3 import Blip2Stage3
import json
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig 
from datetime import timedelta
import wandb
import nltk
from pprint import pprint
from datetime import datetime
nltk.download("wordnet")

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# for pyg bug
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", message=r".*Skipped loading .*")
warnings.filterwarnings("ignore", message=r".*No normalization for .*")
# for A5000 gpus
torch.set_float32_matmul_precision(
    "medium"
)  # can be medium (bfloat16), high (tensorfloat32), highest (float32)

class SaveInitCheckpointCallback(Callback):
    def __init__(self, filename="initial_checkpoint.ckpt"):
        self.filename = filename

    def on_fit_start(self, trainer, pl_module):
        # 저장 경로 생성
        save_path = os.path.join(trainer.logger.log_dir, self.filename)
        print(f"\n[INFO] Saving initial checkpoint before training/validation to: {save_path}")
        trainer.save_checkpoint(save_path)
        
class MyDDPStrategy(strategies.DDPStrategy):
    def __init__(
        self,
        find_unused_parameters=False,
        start_method="spawn",
        timeout=timedelta(minutes=90),
    ): 
        super().__init__(
            find_unused_parameters=find_unused_parameters,
            start_method=start_method,
            timeout=timeout,
        )

    def load_model_state_dict(self, checkpoint, strict=False):
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=strict)


def print_training_config_report(cfg, model):
    """통합된 학습 설정 및 파라미터 리포트 출력"""

    # 파라미터 분석
    all_params = 0
    trainable_params = 0
    component_params = {
        'lora': 0,
        'embedding': 0,
        'lm_head': 0,
        'qformer': 0,
        'graph': 0,
        'other': 0
    }

    trainable_layers = {
        'lora_modules': set(),
        'embedding_layers': [],
        'lm_head_layers': [],
        'qformer_layers': [],
        'graph_layers': [],
        'other_layers': []
    }

    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

            # 컴포넌트별 분류
            if 'lora' in name.lower():
                component_params['lora'] += param.numel()
                # LoRA 모듈명 추출 (예: blocks.0.q_proj)
                module_name = name.split('.lora_')[0] if '.lora_' in name else name
                base_module = '.'.join(module_name.split('.')[-3:])  # 마지막 3레벨만
                trainable_layers['lora_modules'].add(base_module)
            elif any(x in name.lower() for x in ['wte', 'embed_tokens', 'word_embeddings']):
                component_params['embedding'] += param.numel()
                trainable_layers['embedding_layers'].append(name.split('.')[-2] + '.' + name.split('.')[-1])
            elif any(x in name.lower() for x in ['ff_out', 'lm_head']) and 'blocks' not in name:
                component_params['lm_head'] += param.numel()
                trainable_layers['lm_head_layers'].append(name.split('.')[-2] + '.' + name.split('.')[-1])
            elif 'qformer' in name.lower():
                component_params['qformer'] += param.numel()
                trainable_layers['qformer_layers'].append('.'.join(name.split('.')[-3:]))
            elif 'graph' in name.lower():
                component_params['graph'] += param.numel()
                trainable_layers['graph_layers'].append('.'.join(name.split('.')[-3:]))
            else:
                component_params['other'] += param.numel()
                trainable_layers['other_layers'].append('.'.join(name.split('.')[-3:]))

    # 리포트 출력
    print("\n" + "="*100)
    print("🚀 TRAINING CONFIGURATION & PARAMETER REPORT".center(100))
    print("="*100)

    # 1. 기본 설정
    print("\n📋 [Configuration]")
    print(f"  Config File:        {HydraConfig.get().job.config_name}")
    print(f"  Mode:               {cfg.mode}")
    print(f"  Model:              {cfg.llm_model}")
    print(f"  Training Method:    LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout})")
    print(f"  Projector:          {cfg.projector_type}")
    print(f"  Precision:          {cfg.precision}")
    print(f"  Devices:            {cfg.devices}")
    print(f"  Seed:               {cfg.seed}")

    # 2. 학습 설정
    print("\n⚙️  [Training Settings]")
    print(f"  Max Epochs:         {cfg.max_epochs}")
    print(f"  Batch Size:         {cfg.batch_size} x {cfg.accumulate_grad_batches} (accum) = {cfg.total_batch_size} (effective)")
    print(f"  Learning Rate:      {cfg.init_lr} (init), {cfg.min_lr} (min)")
    print(f"  Warmup Steps:       {cfg.warmup_steps}")
    print(f"  Scheduler:          {cfg.scheduler}")
    print(f"  Optimizer:          {cfg.optimizer}")
    print(f"  Gradient Clip:      {cfg.gradient_clip_val}")
    print(f"  Weight Decay:       {cfg.weight_decay}")

    # 3. Checkpoint 설정
    print("\n💾 [Checkpoint Settings]")
    print(f"  Save Every:         {cfg.save_on_n_steps} steps")
    print(f"  Keep Top-K:         {cfg.save_top_k_checkpoints if hasattr(cfg, 'save_top_k_checkpoints') else 'All'} checkpoints")
    print(f"  Best Models:        Top {cfg.save_top_k_best if hasattr(cfg, 'save_top_k_best') else 3} (by val_loss)")
    print(f"  Directory:          {cfg.logging_dir}/{cfg.filename}")

    # 4. Resume/Pretrain 정보
    if cfg.ckpt_path or cfg.pretrained_ckpt_path:
        print("\n🔄 [Resume/Pretrain]")
        if cfg.ckpt_path:
            print(f"  Resume from:        {cfg.ckpt_path}")
        if cfg.pretrained_ckpt_path:
            print(f"  Pretrained:         {cfg.pretrained_ckpt_path}")

    # 5. 파라미터 통계
    print("\n📊 [Parameter Statistics]")
    print(f"  Total Parameters:   {all_params:,}")
    print(f"  Trainable:          {trainable_params:,} ({trainable_params/all_params*100:.2f}%)")
    print(f"  Frozen:             {all_params - trainable_params:,} ({(all_params - trainable_params)/all_params*100:.2f}%)")

    # 6. 컴포넌트별 파라미터
    print("\n🔧 [Trainable Components Breakdown]")
    if component_params['lora'] > 0:
        print(f"  ✅ LoRA Adapters:      {component_params['lora']:>12,} params  ({len(trainable_layers['lora_modules'])} unique modules)")
    if component_params['embedding'] > 0:
        print(f"  ✅ Embeddings:         {component_params['embedding']:>12,} params  ({len(trainable_layers['embedding_layers'])} layers)")
    if component_params['lm_head'] > 0:
        print(f"  ✅ LM Head:            {component_params['lm_head']:>12,} params  ({len(trainable_layers['lm_head_layers'])} layers)")
    if component_params['qformer'] > 0:
        print(f"  ✅ Q-Former:           {component_params['qformer']:>12,} params  ({len(set(trainable_layers['qformer_layers']))} layers)")
    if component_params['graph'] > 0:
        print(f"  ✅ Graph Encoder:      {component_params['graph']:>12,} params  ({len(set(trainable_layers['graph_layers']))} layers)")
    if component_params['other'] > 0:
        print(f"  ⚠️  Other:              {component_params['other']:>12,} params")

    # 7. 상세 레이어 정보 (간결하게)
    print("\n📝 [Trainable Layer Details]")

    if trainable_layers['lora_modules']:
        lora_sample = sorted(list(trainable_layers['lora_modules']))[:3]
        print(f"  LoRA Modules:       {', '.join(lora_sample)}")
        if len(trainable_layers['lora_modules']) > 3:
            print(f"                      ... and {len(trainable_layers['lora_modules']) - 3} more")

    if trainable_layers['embedding_layers']:
        print(f"  Embedding:          {', '.join(trainable_layers['embedding_layers'])}")

    if trainable_layers['lm_head_layers']:
        print(f"  LM Head:            {', '.join(trainable_layers['lm_head_layers'])}")

    if trainable_layers['qformer_layers']:
        qformer_sample = trainable_layers['qformer_layers'][:3]
        print(f"  Q-Former:           {', '.join(qformer_sample)}")
        if len(trainable_layers['qformer_layers']) > 3:
            print(f"                      ... and {len(trainable_layers['qformer_layers']) - 3} more")

    # 8. 경고 메시지
    if trainable_params < 10_000_000:
        print("\n⚠️  [WARNING] Trainable parameters are very low (< 10M)!")
        print("    → Check if LoRA target modules match the model architecture")
        print("    → Verify Q-Former/Graph encoder training settings")

    if component_params['lora'] == 0 and 'lora' in cfg.tune_llm.lower():
        print("\n❌ [ERROR] LoRA is enabled but no LoRA parameters found!")
        print("    → Check lora_config_llada.json target_modules")

    print("\n" + "="*100 + "\n")


@hydra.main(config_path="configs", config_name="train_llada.yaml", version_base=None)
def main(cfg):
    cfg = flatten_dictconfig(cfg)
    pl.seed_everything(cfg.seed)
    model = Blip2Stage3(cfg)

    # 통합 리포트 출력
    print_training_config_report(cfg, model)
        
    # when resuming training, load the current epoch information and argparse to datamodule
    if cfg.ckpt_path is not None:
        ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
        cfg.current_epoch = ckpt["epoch"]
        del ckpt
    else:
        cfg.current_epoch = 0

    # datamodule
    dm = Stage3DM(
        mode=cfg.mode,
        num_workers=cfg.num_workers,
        tokenizer=model.blip2model.llm_tokenizer,
        args=cfg,
    )
    #! 해당 아래 코드의 의도를 해결하고 주석 여부 결정하기
    # print("\n" + "="*50)
    # print("[DEBUG] Inspecting one batch from Dataloader...")
    # try:
    #     # 실행 모드(mode)에 따라 적절한 dataloader 선택
    #     if cfg.mode == "test":
    #         debug_loader = dm.test_dataloader()
    #         print("[DEBUG] Using Test Dataloader")
    #     else:
    #         debug_loader = dm.train_dataloader()
    #         print("[DEBUG] Using Train Dataloader")
        
    #     # 배치 하나 가져오기
    #     batch = next(iter(debug_loader))
        
    #     print(f"[DEBUG] Batch Keys: {list(batch.keys())}")
        
    #     # 키별 값 출력
    #     for key, value in batch.items():
    #         print(f"\n[Key]: {key}")
    #         if isinstance(value, torch.Tensor):
    #             pprint(f"  Type: Tensor")
    #             pprint(f"  Shape: {value.shape}")
    #             pprint(f"  Values:\n{value}")
    #         else:
    #             pprint(f"  Type: {type(value)}")
    #             pprint(f"  Values:\n{value}")
                
    # except Exception as e:
    #     print(f"[DEBUG] Failed to inspect dataloader: {e}")
    # print("="*50 + "\n")
    # [End] Dataloader Inspection Code
    callbacks = []
    today_date = datetime.now().strftime("%Y%m%d")

    # 1. [Step-based Checkpoint] 정기적으로 step마다 저장
    # save_top_k를 사용하면 monitor가 필요하므로, -1(모두 저장)일 때만 monitor 없이 사용
    save_top_k_checkpoints = cfg.save_top_k_checkpoints if hasattr(cfg, 'save_top_k_checkpoints') else -1

    train_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.logging_dir, cfg.filename),
        filename="{epoch:02d}-{step:06d}-train", # 파일명에 epoch와 step 모두 표시
        every_n_train_steps=cfg.save_on_n_steps if cfg.save_on_n_steps > 0 else None,
        save_last=True,                      # last.ckpt (최신 상태) 저장
        save_top_k=-1,                       # 모두 저장 (개수 제한 없음)
        save_on_train_epoch_end=True         # epoch 끝에도 저장
    )
    callbacks.append(train_checkpoint_callback)

    # 2. [Best Validation Checkpoint] Validation Loss 기준 최고 성능 모델 저장
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.logging_dir, cfg.filename),
        filename=f"best_{today_date}_{{epoch:02d}}_{{step:06d}}_loss={{val_total_loss:.4f}}",
        monitor="val_total_loss",  # [중요] 모델에서 log하는 metric 이름과 같아야 함
        mode="min",                # Loss니까 작을수록 좋음 (min)
        save_top_k=cfg.save_top_k_best if hasattr(cfg, 'save_top_k_best') else 3,  # 상위 N개 모델 유지
        save_last=False,
        auto_insert_metric_name=False
    )
    callbacks.append(best_checkpoint_callback)

    # 3. [Optional] Epoch 기반 정기 저장 (save_every_n_epochs가 설정된 경우)
    if hasattr(cfg, 'save_every_n_epochs') and cfg.save_every_n_epochs > 0:
        epoch_checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(cfg.logging_dir, cfg.filename),
            filename="{epoch:02d}-end",
            every_n_epochs=cfg.save_every_n_epochs,
            save_top_k=-1,  # 모든 epoch checkpoint 유지
            save_on_train_epoch_end=True
        )
        callbacks.append(epoch_checkpoint_callback)
        print(f"[INFO] Epoch checkpoint enabled: Saving every {cfg.save_every_n_epochs} epochs")

    print(f"[INFO] Checkpoint configuration:")
    print(f"  - Save every {cfg.save_on_n_steps} steps (keep {callbacks[0].save_top_k} checkpoints)")
    print(f"  - Save top {best_checkpoint_callback.save_top_k} best validation checkpoints")
    print(f"  - Checkpoint directory: {os.path.join(cfg.logging_dir, cfg.filename)}")

    if len(cfg.devices.split(",")) > 1:
        if cfg.strategy_name == "fsdp":
            strategy = strategies.DDPFullyShardedNativeStrategy()
        elif cfg.strategy_name == "deepspeed":
            strategy = strategies.DeepSpeedStrategy(stage=3)
        else:
            strategy = MyDDPStrategy(
                find_unused_parameters=cfg.find_unused_parameters,
                start_method="spawn",
                timeout=timedelta(minutes=90),
            )
    else:
        strategy = "auto"
        cfg.devices = [eval(cfg.devices)]

    logger = CSVLogger(save_dir=os.path.join(cfg.logging_dir, cfg.filename))

    wandb_logger = WandbLogger(
        name=cfg.filename,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        id=cfg.wandb_id,
        resume="allow",
    )

    tb_logger = TensorBoardLogger(
        os.path.join(cfg.logging_dir, "tensorboard"),
        name=cfg.filename,
    )

    world_size = len(cfg.devices.split(",")) if "," in cfg.devices else 1
    cfg.accumulate_grad_batches = cfg.total_batch_size // cfg.batch_size // world_size
    print("accumulate_grad_batches:", cfg.accumulate_grad_batches)

    # [FIX OOM] Set PyTorch memory allocator config for better fragmentation handling
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    trainer_args = {
        "accelerator": cfg.accelerator,
        "devices": cfg.devices,
        "precision": cfg.precision,
        "callbacks": callbacks,
        "strategy": strategy,
        "logger": [logger, wandb_logger, tb_logger],
        "max_steps": cfg.max_steps,
        "max_epochs": cfg.max_epochs,
        "val_check_interval": cfg.val_check_interval,
        "accumulate_grad_batches": cfg.accumulate_grad_batches,
        "check_val_every_n_epoch": cfg.check_val_every_n_epoch,
        "log_every_n_steps": cfg.log_every_n_steps,
        "gradient_clip_val": cfg.gradient_clip_val,
        "num_sanity_val_steps": cfg.num_sanity_val_steps,
        "limit_val_batches": cfg.limit_val_batches if hasattr(cfg, "limit_val_batches") else 1.0,
        "enable_progress_bar": True,  # Progress bar enabled
    }

    if cfg.skip_sanity_check:
        trainer_args["num_sanity_val_steps"] = 0
    if hasattr(cfg, "profiler"):
        trainer_args["profiler"] = cfg.profiler

    trainer = Trainer(**trainer_args)

    # load pretrained model for model parameter initialization
    if cfg.pretrained_ckpt_path is not None:
        assert cfg.ckpt_path is None, "only one ckpt path should be provided"
        ckpt = torch.load(cfg.pretrained_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"loaded pretrained model from {cfg.pretrained_ckpt_path}")

        # [CRITICAL FIX] 체크포인트 로드 후 modules_to_save를 강제로 trainable 설정
        print("\n" + "="*70)
        print("[CHECKPOINT LOAD] Fixing modules_to_save gradients after checkpoint load...")
        model._fix_modules_to_save_gradients()
        print("="*70 + "\n")

    if cfg.mode in {"ft"}:
        # Resume 정보 출력
        if cfg.ckpt_path is not None:
            print("\n" + "="*70)
            print(f"[RESUME] Resuming training from checkpoint:")
            print(f"  - Checkpoint path: {cfg.ckpt_path}")
            try:
                ckpt_info = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
                print(f"  - Epoch: {ckpt_info.get('epoch', 'N/A')}")
                print(f"  - Global step: {ckpt_info.get('global_step', 'N/A')}")
                if 'callbacks' in ckpt_info and 'ModelCheckpoint' in str(ckpt_info['callbacks']):
                    print(f"  - Best validation loss: {ckpt_info['callbacks'].get('ModelCheckpoint', {}).get('best_model_score', 'N/A')}")
                del ckpt_info
            except Exception as e:
                print(f"  - Could not read checkpoint info: {e}")
            print("="*70 + "\n")
        else:
            print("\n" + "="*70)
            print("[TRAINING] Starting training from scratch")
            print("="*70 + "\n")

        trainer.fit(model, datamodule=dm, ckpt_path=cfg.ckpt_path)
        outputs = trainer.test(model, datamodule=dm)
        assert "Training done"
    elif cfg.mode == "test":
        ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"loaded trained model from {cfg.ckpt_path}")
        outputs = trainer.test(model, datamodule=dm)
        if cfg.filename is not None:
            update_result_csv(
                logger_dir=trainer.logger.log_dir,
                outputs=outputs,
            )
        assert "Testing done"
    else:
        raise NotImplementedError()


def update_result_csv(outputs, logger_dir, task_names=None):
    # first, read the content in result_csv file

    performance_result_path = os.path.join(logger_dir, "benchmark_performance.json")

    os.makedirs(os.path.dirname(performance_result_path), exist_ok=True)
    # task average of output
    final_output = dict()
    tasks = [list(o.keys()) for o in outputs]
    tasks = list(set([item for sublist in tasks for item in sublist]))

    for task in tasks:
        final_output[task] = []
        for output in outputs:
            if task in output:
                final_output[task].append(output[task])
        final_output[task] = final_output[task][
            0
        ]  # identical, just replicated 4 dataloader

    with open(performance_result_path, "w") as f:
        json.dump(final_output, f, indent=4)


def flatten_dictconfig(config: DictConfig) -> DictConfig:
    """
    Flatten a nested DictConfig into a single level DictConfig with keys as the path to the original keys.

    Args:
    - config (DictConfig): The nested DictConfig to be flattened.
    - parent_key (str, optional): The base key to use for prefixing the keys. Defaults to ''.
    - separator (str, optional): The separator to use between keys. Defaults to '.'.

    Returns:
    - DictConfig: The flattened configuration.
    """

    # only flatten just first level
    items = []
    for k, v in config.items():
        new_key = k
        if isinstance(v, DictConfig):
            for kk, vv in v.items():
                items.append((f"{kk}", vv))
        else:
            items.append((new_key, v))
    return OmegaConf.create(dict(items))


if __name__ == "__main__":
    main()
