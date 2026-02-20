import wandb

print("Initializing wandb...")
run = wandb.init(project="test", name="connectivity-test", mode="online")
print(f"Success! Run: {run.url}")
wandb.finish()
