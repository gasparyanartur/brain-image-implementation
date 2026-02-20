import subprocess
import sys
import torch


def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return f"unavailable ({e})"


print("=== Python ===")
print(f"Version: {sys.version}")
print(f"Executable: {sys.executable}")

print("\n=== PyTorch ===")
print(f"Version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (torch): {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}  {props.total_memory // 1024**2} MB  compute {props.major}.{props.minor}")

print("\n=== System ===")
print(f"nvcc:  {run(['nvcc', '--version'])}")
print(f"nvidia-smi:\n{run(['nvidia-smi'])}")

print("\n=== Compute test ===")
nums = [1.0, 2.0, 3.0, 4.0, 5.0]
t = torch.tensor(nums, device="cuda")
print(f"Values: {nums}")
print(f"Sum on {t.device}: {t.sum().item()}")
