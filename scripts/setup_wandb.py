#!/usr/bin/env python3
"""
Setup script for Weights & Biases integration.
This script helps configure wandb authentication and project settings.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_wandb_installed():
    """Check if wandb is installed."""
    try:
        import wandb

        print("✅ wandb is already installed")
        return True
    except ImportError:
        print("❌ wandb is not installed")
        return False


def install_wandb():
    """Install wandb using uv."""
    print("Installing wandb...")
    try:
        subprocess.run([sys.executable, "-m", "uv", "add", "wandb"], check=True)
        print("✅ wandb installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install wandb")
        return False


def check_wandb_login():
    """Check if user is logged into wandb."""
    try:
        import wandb

        api = wandb.Api()
        # Try to access user info to check if logged in
        api.viewer
        print("✅ Already logged into wandb")
        return True
    except Exception:
        print("❌ Not logged into wandb")
        return False


def login_wandb():
    """Login to wandb."""
    print("Please login to wandb...")
    try:
        import wandb

        wandb.login()
        print("✅ Successfully logged into wandb")
        return True
    except Exception as e:
        print(f"❌ Failed to login to wandb: {e}")
        return False


def create_wandb_config():
    """Create a wandb configuration file."""
    config_content = """# Wandb Configuration
# Set your wandb username or team name here
wandb_entity: your_username_or_team

# Project name for your experiments
wandb_project: brain-image-nice

# Whether to log model artifacts
wandb_log_model: false

# Tags for organizing experiments
wandb_tags: []
"""

    config_path = Path("wandb_config.yaml")
    if not config_path.exists():
        with open(config_path, "w") as f:
            f.write(config_content)
        print(f"✅ Created {config_path}")
        print("📝 Please edit this file with your wandb settings")
    else:
        print(f"✅ {config_path} already exists")


def main():
    """Main setup function."""
    print("🚀 Setting up Weights & Biases integration...\n")

    # Check if wandb is installed
    if not check_wandb_installed():
        if not install_wandb():
            return

    # Check login status
    if not check_wandb_login():
        if not login_wandb():
            return

    # Create configuration file
    create_wandb_config()

    print("\n🎉 Wandb setup complete!")
    print("\nNext steps:")
    print("1. Edit wandb_config.yaml with your settings")
    print("2. Update your trainer config to enable wandb:")
    print("   enable_wandb: true")
    print("3. Run your training script")
    print("\nTo test wandb integration, run:")
    print("python -c \"import wandb; print('Wandb version:', wandb.__version__)\"")


if __name__ == "__main__":
    main()
