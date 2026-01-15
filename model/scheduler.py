"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import math

from lavis.common.registry import registry


@registry.register_lr_scheduler("linear_warmup_step_lr")
class LinearWarmupStepLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        decay_rate=1,
        warmup_start_lr=-1,
        warmup_steps=0,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.decay_rate = decay_rate

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            step_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate,
            )


@registry.register_lr_scheduler("linear_warmup_cosine_lr")
class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_step,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_step = max_step
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        assert self.warmup_steps < self.max_step, "warmup_steps should be less than max_step"
        assert self.min_lr < self.init_lr, "min_lr should be less than init_lr"

    def step(self, cur_step):
        # assuming the warmup iters less than one epoch
        if cur_step < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        elif cur_step >= self.max_step:
            min_lr_schedule(optimizer=self.optimizer, min_lr=self.min_lr)
        else:
            cosine_lr_schedule(
                step=cur_step - self.warmup_steps,
                optimizer=self.optimizer,  
                max_step=self.max_step - self.warmup_steps,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )


def cosine_lr_schedule(optimizer, step, max_step, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * step / max_step)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def min_lr_schedule(optimizer, min_lr):
    """Set the learning rate to min_lr"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = min_lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
@registry.register_lr_scheduler("linear_warmup_constant_lr")
class LinearWarmupConstantLRScheduler:
    def __init__(
        self,
        optimizer,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_step):
        if cur_step < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            # Warmup 이후에는 init_lr로 고정 (Decay 없음)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.init_lr
                
# [model/scheduler.py] 파일 하단에 추가

@registry.register_lr_scheduler("warmup_stable_decay_lr")
class WarmupStableDecayLRScheduler:
    """
    Warmup-Stable-Decay LR Scheduler (WSD)

    각 param group의 초기 LR 비율을 유지하면서 스케줄링:
    - Warmup: 0 -> group_lr (비율 유지)
    - Stable: group_lr 유지
    - Decay: group_lr -> group_lr × min_lr_ratio (비율 유지, Linear Decay)

    예: LoRA lr=2.5e-4, Embed lr=2.5e-5, min_lr_ratio=0.1인 경우
        -> Decay 후: LoRA=2.5e-5, Embed=2.5e-6 (10:1 비율 유지)
    """
    def __init__(
        self,
        optimizer,
        max_step,
        init_lr,
        min_lr,
        warmup_steps=50,
        decay_ratio=0.1,
        min_lr_ratio=None,  # 추가: 직접 비율 지정 가능
        **kwargs
    ):
        self.optimizer = optimizer
        self.max_step = max_step
        self.warmup_steps = warmup_steps

        # 전체 step의 마지막 decay_ratio 만큼만 감쇠 (논문은 10%)
        self.decay_start_step = int(max_step * (1 - decay_ratio))

        # min_lr_ratio 계산: 직접 지정되면 사용, 아니면 min_lr/init_lr로 계산
        if min_lr_ratio is not None:
            self.min_lr_ratio = min_lr_ratio
        elif init_lr > 0:
            self.min_lr_ratio = min_lr / init_lr
        else:
            self.min_lr_ratio = 0.1  # 기본값

        # 각 param group의 초기 LR 저장 (비율 계산용)
        self.initial_lrs = []
        self.initial_lrs_new = []  # embed/head의 new vocab용 LR
        for group in self.optimizer.param_groups:
            self.initial_lrs.append(group['lr'])
            # embed/head 그룹은 lr_new 필드가 있음
            self.initial_lrs_new.append(group.get('lr_new', group['lr']))

    def step(self, cur_step):
        # 스케줄링 비율 계산 (0.0 ~ 1.0)
        if cur_step < self.warmup_steps:
            # 1. Warmup Phase: 0 -> 1.0
            ratio = cur_step / max(1, self.warmup_steps)
        elif cur_step < self.decay_start_step:
            # 2. Stable Phase: 1.0 유지
            ratio = 1.0
        else:
            # 3. Decay Phase: 1.0 -> min_lr_ratio
            decay_steps = self.max_step - self.decay_start_step
            progress = (cur_step - self.decay_start_step) / max(1, decay_steps)
            progress = min(1.0, progress)
            # 1.0에서 min_lr_ratio까지 선형 감소
            ratio = 1.0 - (1.0 - self.min_lr_ratio) * progress

        # 각 param group에 비율 적용
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.initial_lrs[i]
            param_group["lr"] = base_lr * ratio

            # embed/head 그룹의 경우 lr_new도 업데이트
            if 'lr_new' in param_group:
                base_lr_new = self.initial_lrs_new[i]
                param_group["lr_new"] = base_lr_new * ratio