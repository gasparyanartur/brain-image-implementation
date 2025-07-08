#!/usr/bin/env python3
"""
Test script to demonstrate wandb integration with the trainer.
This script shows how to enable wandb logging in your training runs.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_image.trainer import NICETrainerConfig, NICETrainer


def test_wandb_integration():
    """Test wandb integration with a simple training configuration."""

    # Create a test configuration with wandb enabled
    config = NICETrainerConfig(
        run_name="test-wandb",
        num_epochs=1,  # Just 1 epoch for testing
        enable_wandb=True,
        wandb_project="brain-image-test",
        wandb_entity=None,  # Use your default entity
        wandb_tags=["test", "integration"],
        enable_barebones=False,
        overfit_batches=1,  # Overfit on 1 batch for quick testing
    )

    print("🧪 Testing wandb integration...")
    print(f"Run name: {config.run_name}")
    print(f"Wandb project: {config.wandb_project}")
    print(f"Wandb enabled: {config.enable_wandb}")

    try:
        # Create trainer
        trainer = config.create_trainer()
        print("✅ Trainer created successfully")

        # Check if wandb logger is present
        loggers = trainer.pl_trainer.loggers
        wandb_loggers = [logger for logger in loggers if hasattr(logger, "experiment")]

        if wandb_loggers:
            print("✅ Wandb logger found in trainer")
            for logger in wandb_loggers:
                print(f"   Logger: {type(logger).__name__}")
        else:
            print("❌ No wandb logger found")

        print("\n🎉 Wandb integration test completed!")
        print("\nTo run actual training with wandb:")
        print("1. Make sure you're logged into wandb: wandb login")
        print("2. Update your config to enable wandb")
        print("3. Run your training script")

    except Exception as e:
        print(f"❌ Error during wandb integration test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_wandb_integration()
