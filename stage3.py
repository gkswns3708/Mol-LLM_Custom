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
from datetime import timedelta
import wandb
import nltk
from pprint import pprint
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
    cfg = flatten_dictconfig(cfg)
    pl.seed_everything(cfg.seed)
    model = Blip2Stage3(cfg)
    print("total params:", sum(p.numel() for p in model.parameters()))

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
    callbacks.append(
        ModelCheckpoint(
            dirpath=os.path.join(cfg.logging_dir, cfg.filename),
            filename="{epoch:02d}-{step}",
            every_n_epochs=cfg.every_n_epochs,
            save_last=True,
            save_top_k=-1,
        )
    )

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
        "check_val_every_n_epoch": cfg.check_val_every_n_epoch, # 2epoch마다 validation 하도록 설정함.
        "log_every_n_steps": cfg.log_every_n_steps,
        "gradient_clip_val": cfg.gradient_clip_val,
        "num_sanity_val_steps": cfg.num_sanity_val_steps
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
