from dataclasses import dataclass
import logging
from typing import Any, Literal, cast
from torch import nn
import torch


def iter_named_params(*models: nn.Module | None):
    param_list: list[tuple[str, nn.Parameter]] = []
    for model in models:
        if model is not None:
            param_list.extend(model.named_parameters())
    return iter(param_list)


@dataclass
class OptimizerConfig:
    name: str
    modules: list[nn.Module | None]
    lr: float
    min_lr: float
    warmup_epochs: int
    delay_epochs: int
    enabled: bool
    lr_scheduler: Literal["cosine_anneal", "none"]


def set_weight_decay_per_param(
    *modules: nn.Module, weight_decay: float, skip_names=["bias", "norm", "ln", "bn"]
) -> list[dict[str, list[nn.Parameter] | float]]:
    logging.info(
        f"Setting weight decay to {weight_decay} for all parameters except biases and norms. Skipping parameters with names containing: {skip_names}"
    )

    no_decay: list[nn.Parameter] = []
    with_decay: list[nn.Parameter] = []

    for param_name, param in iter_named_params(*modules):
        if not param.requires_grad:
            continue

        if param.ndim < 2 or any(skip_name in param_name for skip_name in skip_names):
            no_decay.append(param)

        else:
            with_decay.append(param)

    params: list[dict[str, list[nn.Parameter] | float]] = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": with_decay, "weight_decay": weight_decay},
    ]
    return params


def get_optimizer_options(
    configs: list[OptimizerConfig],
    max_epochs: int,
    num_train_batches: int,
    modules_to_optimize: list[str] | None,
    optimizer_params: dict[str, Any],
):
    optimizer_options = []

    for config in configs:
        modules_to_opt = cast(
            list[nn.Module], [m for m in config.modules if m is not None]
        )

        if (
            (modules_to_optimize is not None and config.name not in modules_to_optimize)
            or (not config.enabled)
            or (not modules_to_opt)
        ):
            config.enabled = False
            for module in modules_to_opt:
                logging.info("Disabling optimization of module: %s")
                module.requires_grad_(False)
            continue

        logging.info(
            f"Creating optimizer: {config.name} - lr: {config.lr}, min_lr: {config.min_lr}, warmup_epochs: {config.warmup_epochs}, delay_epochs: {config.delay_epochs}"
        )
        warmup_steps = config.warmup_epochs * num_train_batches
        delay_steps = config.delay_epochs * num_train_batches
        total_steps = max_epochs * num_train_batches
        
        weight_decay = optimizer_params.pop("weight_decay") if "weight_decay" in optimizer_params else 0.0
        optimizer = torch.optim.AdamW(
            set_weight_decay_per_param(*modules_to_opt, weight_decay=weight_decay),
            lr=config.lr,
            **optimizer_params,
        )
        schedulers = []
        milestones = []

        if delay_steps > 0:
            schedulers.append(
                torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 0)
            )
            milestones.append(delay_steps + max(milestones or [0]))

        if delay_steps < total_steps and warmup_steps > 0:
            schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    total_iters=warmup_steps,
                )
            )
            milestones.append(warmup_steps + max(milestones or [0]))

        if delay_steps + warmup_steps < total_steps:
            if config.lr_scheduler == "cosine_anneal":
                schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=total_steps - max(milestones or [0]),
                        eta_min=config.min_lr,
                    )
                )
            elif config.lr_scheduler == "none":
                schedulers.append(
                    torch.optim.lr_scheduler.ConstantLR(
                        optimizer,
                        factor=1.0,
                    )
                )
            else:
                raise ValueError(f"Unknown lr_scheduler: {config.lr_scheduler}")

        milestones = milestones[: len(schedulers) - 1]

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=schedulers,
            milestones=milestones,
        )

        optimizer_options.append(
            {
                "name": config.name,
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        )

    return optimizer_options
