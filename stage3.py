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


@hydra.main(config_path="configs", config_name="train_llada.yaml", version_base=None)
def main(cfg):
    print(f"Loaded Config Name: {HydraConfig.get().job.config_name}")
    cfg = flatten_dictconfig(cfg)
    pl.seed_everything(cfg.seed)
    model = Blip2Stage3(cfg)
    print("total params:", sum(p.numel() for p in model.parameters()))
    
    
    # [디버깅 코드 시작] 학습 가능한 파라미터 상세 분석
    print("\n" + "="*80)
    print("[DEBUG] Inspecting Trainable Parameters")
    
    trainable_params = 0
    all_params = 0
    lora_modules = {}
    
    print(f"{'Module Name':<60} | {'Shape':<20} | {'Trainable'}")
    print("-" * 95)
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # LoRA 모듈인지 확인 (이름에 lora가 포함되는지)
            if "lora" in name:
                module_name = name.split(".lora_")[0] # 모듈 이름만 추출
                if module_name not in lora_modules:
                    lora_modules[module_name] = 0
                lora_modules[module_name] += param.numel()
            
            # 너무 길면 일부만 출력하거나 LoRA 관련만 출력
            if "lora" in name or "Qformer" in name or "graph" in name:
                print(f"{name:<60} | {str(list(param.shape)):<20} | True")
    
    print("-" * 95)
    print(f"Total Params: {all_params:,}")
    print(f"Trainable Params: {trainable_params:,} ({trainable_params/all_params:.2%})")
    
    if trainable_params < 10_000_000:
        print("\n[WARNING] Trainable parameters are extremely low (< 10M).")
        print("Possible causes:")
        print("1. Q-Former is FROZEN. (Should be trainable in Stage 3?)")
        print("2. LoRA target modules mismatch. Check 'lora_config_llada.json' vs Model Layer Names.")
        
    print("\n[LoRA Applied Modules Summary]")
    if not lora_modules:
        print("  No LoRA modules found! Check target_modules config.")
    else:
        for mod, count in list(lora_modules.items())[:5]: # 5개만 예시로 출력
            print(f"  {mod}: {count:,} params")
        print(f"  ... (Total {len(lora_modules)} modules applied)")
    
    print("="*80 + "\n")
    # [디버깅 코드 끝]
        
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
    print("\n" + "="*50)
    print("[DEBUG] Inspecting one batch from Dataloader...")
    try:
        # 실행 모드(mode)에 따라 적절한 dataloader 선택
        if cfg.mode == "test":
            debug_loader = dm.test_dataloader()
            print("[DEBUG] Using Test Dataloader")
        else:
            debug_loader = dm.train_dataloader()
            print("[DEBUG] Using Train Dataloader")
        
        # 배치 하나 가져오기
        batch = next(iter(debug_loader))
        
        print(f"[DEBUG] Batch Keys: {list(batch.keys())}")
        
        # 키별 값 출력
        for key, value in batch.items():
            print(f"\n[Key]: {key}")
            if isinstance(value, torch.Tensor):
                pprint(f"  Type: Tensor")
                pprint(f"  Shape: {value.shape}")
                pprint(f"  Values:\n{value}")
            else:
                pprint(f"  Type: {type(value)}")
                pprint(f"  Values:\n{value}")
                
    except Exception as e:
        print(f"[DEBUG] Failed to inspect dataloader: {e}")
    print("="*50 + "\n")
    # [End] Dataloader Inspection Code
    callbacks = []
    today_date = datetime.now().strftime("%Y%m%d")
    train_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.logging_dir, cfg.filename),
        filename="{epoch:02d}-{step}-train", # 파일명 구분
        # every_n_epochs=cfg.every_n_epochs,   # 설정된 epoch 마다
        every_n_train_steps=cfg.save_on_n_steps,
        save_last=True,                      # last.ckpt (최신 상태) 저장
        save_top_k=-1,                       # 모든 epoch 저장 (필요 없으면 0으로 설정)
        save_on_train_epoch_end=True         # [핵심] Validation 시작 전에 저장함
    )
    callbacks.append(train_checkpoint_callback)
    # 2. [Best Validation Checkpoint] Validation Loss 기준 최고 성능 모델 저장
    # 파일명 예시: best_20240520_val_loss=0.1234.ckpt
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.logging_dir, cfg.filename),
        filename=f"best_{today_date}_{{val_total_loss:.4f}}", # 중괄호 두 개{{}}는 f-string escape
        monitor="val_total_loss",  # [중요] 모델에서 log하는 metric 이름과 같아야 함
        mode="min",                # Loss니까 작을수록 좋음 (min)
        save_top_k=1,              # 가장 좋은 것 1개만 유지
        save_last=False,
        auto_insert_metric_name=False # 파일명에 'val_total_loss=' 자동 추가 방지 (원하는 포맷 유지를 위해)
    )
    callbacks.append(best_checkpoint_callback)

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
        "limit_val_batches": cfg.limit_val_batches if hasattr(cfg, "limit_val_batches") else 1.0
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

    if cfg.mode in {"ft"}:
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
