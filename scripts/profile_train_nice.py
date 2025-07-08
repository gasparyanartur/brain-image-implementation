import logging
from typing import Any
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from brain_image.configs import BaseConfig
from brain_image.trainer import NICETrainer, NICETrainerConfig
from brain_image.model import EEGEncoderConfig, NICEConfig
from brain_image.data import EEGDatasetConfig

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from brain_image.configs import GlobalConfig


class ProfileTrainNICEConfig(BaseConfig):
    """Configuration for NICE training profiling script."""

    script_name: str = "profile_train_nice"
    description: str = "Profiling script for NICE model training with PyTorch profiler"

    # Component composition - these will be populated by Hydra
    dataset: EEGDatasetConfig = EEGDatasetConfig()
    model: NICEConfig = NICEConfig(model_name="aligned_synclr")
    trainer: NICETrainerConfig = NICETrainerConfig()
    encoder: EEGEncoderConfig = EEGEncoderConfig()

    # Script-specific settings (not training parameters)
    checkpoint_path: str | None = None
    resume_training: bool = False
    run_test_after_training: bool = True
    save_final_model: bool = True

    # Profiling settings
    profile_output_dir: Path = Path("profiler_outputs")
    profile_activities: list[str] = ["cpu", "cuda"]  # ["cpu", "cuda", "kineto"]
    profile_record_shapes: bool = True
    profile_with_stack: bool = True
    profile_with_flops: bool = True
    profile_with_modules: bool = True
    profile_schedule: str = "default"  # "default", "wait", "warmup", "active", "repeat"
    profile_wait: int = 1
    profile_warmup: int = 1
    profile_active: int = 3
    profile_repeat: int = 1
    profile_export_chrome_trace: bool = True
    profile_export_stacks: bool = True


def get_profiler_schedule(config: ProfileTrainNICEConfig):
    """Get the profiler schedule based on configuration."""
    if config.profile_schedule == "default":
        return torch.profiler.schedule(
            wait=config.profile_wait,
            warmup=config.profile_warmup,
            active=config.profile_active,
            repeat=config.profile_repeat,
        )
    elif config.profile_schedule == "wait":
        return torch.profiler.schedule(
            wait=config.profile_wait, warmup=0, active=1, repeat=1
        )
    elif config.profile_schedule == "warmup":
        return torch.profiler.schedule(
            wait=0, warmup=config.profile_warmup, active=1, repeat=1
        )
    elif config.profile_schedule == "active":
        return torch.profiler.schedule(
            wait=0, warmup=0, active=config.profile_active, repeat=1
        )
    elif config.profile_schedule == "repeat":
        return torch.profiler.schedule(
            wait=0, warmup=0, active=1, repeat=config.profile_repeat
        )
    else:
        raise ValueError(f"Unknown profile schedule: {config.profile_schedule}")


def get_profiler_activities(config: ProfileTrainNICEConfig):
    """Get the profiler activities based on configuration."""
    activities = []
    for activity in config.profile_activities:
        if activity == "cpu":
            activities.append(ProfilerActivity.CPU)
        elif activity == "cuda":
            activities.append(ProfilerActivity.CUDA)
        elif activity == "kineto":
            # KINETO is not available in all PyTorch versions
            try:
                activities.append(ProfilerActivity.KINETO)
            except AttributeError:
                logging.warning(
                    "KINETO profiler activity not available in this PyTorch version"
                )
        else:
            logging.warning(f"Unknown profiler activity: {activity}")
    return activities


def profile_train_nice(
    trainer: NICETrainer,
    config: ProfileTrainNICEConfig,
    checkpoint_path: Path | None = None,
):
    """Profiled training function that takes a configured trainer."""

    # Create output directory
    config.profile_output_dir.mkdir(parents=True, exist_ok=True)

    # Get profiler configuration
    activities = get_profiler_activities(config)
    schedule = get_profiler_schedule(config)

    logging.info(
        f"Starting profiled training with activities: {config.profile_activities}"
    )
    logging.info(
        f"Profile schedule: wait={config.profile_wait}, warmup={config.profile_warmup}, active={config.profile_active}, repeat={config.profile_repeat}"
    )
    logging.info(f"Profile output directory: {config.profile_output_dir}")

    # Load checkpoint if provided
    if checkpoint_path and checkpoint_path.exists():
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)

    # Start profiling
    with profile(
        activities=activities,
        schedule=schedule,
        record_shapes=config.profile_record_shapes,
        with_stack=config.profile_with_stack,
        with_flops=config.profile_with_flops,
        with_modules=config.profile_with_modules,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(config.profile_output_dir)
        ),
    ) as prof:
        with record_function("train_nice_full_training"):
            # Start training
            logging.info("Starting training with profiler...")
            trainer.train()

            # Test the model if configured
            if config.run_test_after_training:
                logging.info("Running final test with profiler...")
                test_metrics = trainer.test()
            else:
                test_metrics = {}

        # Store profiler data before context ends
        cpu_time_table = prof.key_averages().table(
            sort_by="cpu_time_total", row_limit=10
        )
        cuda_time_table = prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=10
        )
        cpu_memory_table = prof.key_averages().table(
            sort_by="cpu_memory_usage", row_limit=10
        )
        cuda_memory_table = prof.key_averages().table(
            sort_by="cuda_memory_usage", row_limit=10
        )

        # Store detailed tables for file output
        detailed_cpu_time = prof.key_averages().table(
            sort_by="cpu_time_total", row_limit=20
        )
        detailed_cuda_time = prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=20
        )
        detailed_cpu_memory = prof.key_averages().table(
            sort_by="cpu_memory_usage", row_limit=20
        )
        detailed_cuda_memory = prof.key_averages().table(
            sort_by="cuda_memory_usage", row_limit=20
        )
        detailed_self_cpu_time = prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=20
        )
        detailed_self_cuda_time = prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=20
        )

    # Print profiler summary (after profiler context ends)
    logging.info("=== PROFILER SUMMARY ===")
    logging.info(f"Total CPU time: {cpu_time_table}")
    logging.info(f"Total CUDA time: {cuda_time_table}")
    logging.info(f"CPU memory: {cpu_memory_table}")
    logging.info(f"CUDA memory: {cuda_memory_table}")

    # Save detailed profile to file (after profiler context ends)
    profile_file = config.profile_output_dir / "profile_summary.txt"
    with open(profile_file, "w") as f:
        f.write("=== DETAILED PROFILER SUMMARY ===\n\n")
        f.write("CPU Time Summary:\n")
        f.write(detailed_cpu_time)
        f.write("\n\nCUDA Time Summary:\n")
        f.write(detailed_cuda_time)
        f.write("\n\nCPU Memory Summary:\n")
        f.write(detailed_cpu_memory)
        f.write("\n\nCUDA Memory Summary:\n")
        f.write(detailed_cuda_memory)
        f.write("\n\nSelf CPU Time Summary:\n")
        f.write(detailed_self_cpu_time)
        f.write("\n\nSelf CUDA Time Summary:\n")
        f.write(detailed_self_cuda_time)

    logging.info(f"Detailed profile saved to: {profile_file}")

    if config.profile_export_chrome_trace:
        logging.info(f"Chrome trace files saved to: {config.profile_output_dir}")

    if config.profile_export_stacks:
        logging.info(f"Stack traces saved to: {config.profile_output_dir}")

    return trainer.model, test_metrics


@hydra.main(
    config_path=str(GlobalConfig.CONFIGS_DIR),
    config_name="profile_train_nice",
    version_base=None,
)
def main(cfg: DictConfig):
    """Main function for NICE training profiling with modular configuration."""

    config = ProfileTrainNICEConfig.from_hydra_config(cfg)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Set torch precision
    torch.set_float32_matmul_precision("high")

    # Create trainer with composed components
    trainer = NICETrainer(
        config=config.trainer,
        model_config=config.model,
        dataset_config=config.dataset,
        encoder=config.encoder,
    )

    # Get checkpoint path if specified
    checkpoint_path = None
    if config.checkpoint_path:
        checkpoint_path = Path(config.checkpoint_path)

    model, test_metrics = profile_train_nice(trainer, config, checkpoint_path)

    if test_metrics:
        logging.info(f"Finished profiled training with test metrics:")
        for key, value in test_metrics.items():
            logging.info(f"  {key}: {value}")
    else:
        logging.info("Finished profiled training (no test metrics)")


if __name__ == "__main__":
    main()
