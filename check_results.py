import numpy as np
from pathlib import Path
import hashlib

RESULTS_ROOT = Path("results/mvtec+loco/CL")
METHODS = {
    "Finetune": "finetune_resnet18_pretrained",
    "Replay": "replay_resnet18_pretrained",
    "EWC": "ewc_resnet18_pretrained",
    "LwF": "lwf_resnet18_pretrained",
    "GPM": "gpm_resnet18_pretrained",
}

print("Checking Result Matrices...")
matrices = {}

for name, folder in METHODS.items():
    path = RESULTS_ROOT / folder
    files = sorted(path.glob(f"acc_matrix_*.npy"))
    if not files:
        print(f"[{name}] No matrix found in {path}")
        continue
    
    last_file = files[-1]
    data = np.load(last_file)
    data_hash = hashlib.md5(data.tobytes()).hexdigest()
    matrices[name] = {"file": last_file.name, "hash": data_hash, "data": data}
    print(f"[{name}] {last_file.name} | Hash: {data_hash} | Shape: {data.shape} | Mean: {data.mean():.4f}")

print("\n--- Comparison ---")
seen_hashes = {}
for name, info in matrices.items():
    h = info["hash"]
    if h in seen_hashes:
        print(f"WARNING: {name} has IDENTICAL data to {seen_hashes[h]}")
    else:
        seen_hashes[h] = name
