# cl_benchmark/cl_benchmark.py
# Option A: True anomaly detection via a pretrained feature extractor + memory bank per category
import os
import random
import argparse
import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime

# backbones
from cl_benchmark.backbones.models import (
    SimpleCNN,
    ResNet18,
    ResNet18_pretrained,
)

# dataset loaders
from cl_benchmark.datasets.mvtec_loader import load_mvtec_all_categories
from cl_benchmark.datasets.mvtec_loco_loader import load_mvtec_loco_all_categories

# utils
from cl_benchmark.utils.metrics_io import (
    ensure_dir,
    save_task_output,
    plot_heatmap,
    compute_auc_safe,
)
# DEFAULT CONFIG
DEFAULT_CFG = {
    "BACKBONE": "ResNet18_pretrained",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    # memory bank / features
    "MEMORY_SAMPLES_PER_TASK": 2000,
    "MEMORY_SUBSAMPLE": 1,  # keep every N-th sample when building memory
    # dataloading / speed
    "BATCH_SIZE": 64,
    "SEED": 42,
    # dataset roots
    "MVT_ROOT": "data/mvtec",
    "MVT_LOCO_ROOT": "data/mvtec-loco",
    "OUTDIR": "results/mvtec+loco/Anomaly",
}

# Utilities
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_device(device_cfg):
    if isinstance(device_cfg, torch.device):
        return device_cfg
    if isinstance(device_cfg, str):
        s = device_cfg.lower()
        if s == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features(model, loader, device):
    """Return numpy array of features for every image in loader (in order)."""
    model.eval()
    feats_list = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            x = model.features(imgs)              # (B, C, 1, 1) usually
            x = x.view(x.size(0), -1)             # (B, feat_dim)
            feats_list.append(x.cpu().numpy())
    if len(feats_list) == 0:
        return np.zeros((0, model.fc_input_features), dtype=np.float32)
    return np.concatenate(feats_list, axis=0)


def build_memory_bank_for_task(model, dataset, cfg, device):
    """
    Build memory bank for a task using only *normal* training images where label==0.
    Returns: memory_vectors (np.ndarray: N x D)
    """
    # filter dataset for normal images (label==0)
    normal_indices = [i for i in range(len(dataset)) if dataset.labels[i] == 0]
    if len(normal_indices) == 0:
        return np.zeros((0, model.fc_input_features), dtype=np.float32)

    # create subset loader manually to preserve transforms and ordering
    from torch.utils.data import Subset
    subset = Subset(dataset, normal_indices)

    loader = DataLoader(subset, batch_size=cfg["BATCH_SIZE"], shuffle=False)
    feats = extract_features(model, loader, device)  # (N, D)

    # subsample and cap
    if cfg.get("MEMORY_SUBSAMPLE", 1) > 1:
        feats = feats[:: cfg["MEMORY_SUBSAMPLE"]]

    cap = int(cfg.get("MEMORY_SAMPLES_PER_TASK", feats.shape[0]))
    if feats.shape[0] > cap:
        # random sample cap rows (deterministic seed)
        rng = np.random.RandomState(cfg["SEED"])
        idx = rng.choice(feats.shape[0], cap, replace=False)
        feats = feats[idx]

    # L2-normalize features (helps with euclidean/cosine)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats = feats / norms

    return feats.astype(np.float32)


def compute_anomaly_scores_from_memory(model, dataset, memory_vectors, cfg, device):
    """
    For each image in dataset, compute anomaly score = min Euclidean distance
    to any vector in memory_vectors (lower distance = more normal).
    Returns scores (list), labels (list).
    """
    loader = DataLoader(dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False)
    feats = extract_features(model, loader, device)  # (M, D)

    # normalize
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats = feats / norms

    if memory_vectors is None or memory_vectors.shape[0] == 0:
        # no memory -> fallback: use constant score (bad)
        scores = [0.0] * feats.shape[0]
    else:
        # compute pairwise distances efficiently: (a-b)^2 = a^2 + b^2 - 2ab
        # but dims might be small; do chunked computation if memory big
        mem = memory_vectors  # (K,D)
        # compute squared distances: for each feat, get min distance to mem
        # use chunking to avoid huge memory
        batch = 512
        scores = []
        for i in range(0, feats.shape[0], batch):
            chunk = feats[i : i + batch]  # (b,D)
            # squared distances: (b, K)
            # use dot product
            d2 = (
                np.sum(chunk * chunk, axis=1, keepdims=True)
                + np.sum(mem * mem, axis=1)[None, :]
                - 2.0 * np.matmul(chunk, mem.T)
            )
            d2 = np.maximum(d2, 0.0)
            mins = np.sqrt(d2.min(axis=1))  # Euclidean min
            scores.extend(mins.tolist())
    # labels from dataset
    labels = [int(l) for _, l in DataLoader(dataset, batch_size=1, shuffle=False)]
    # but above returns batches of tuple; simpler: iterate dataset for labels:
    labels = [int(dataset[idx][1]) for idx in range(len(dataset))]

    return scores, labels


# MAIN
def run(cfg: dict):
    set_seed(cfg["SEED"])
    device = normalize_device(cfg["DEVICE"])
    cfg["DEVICE"] = device

    # datasets
    mvtec_classes, mvtec_train, mvtec_test = load_mvtec_all_categories(cfg["MVT_ROOT"])
    loco_classes, loco_train, loco_test = load_mvtec_loco_all_categories(cfg["MVT_LOCO_ROOT"])

    all_categories = []
    for c in mvtec_classes:
        all_categories.append(("mvtec", c))
    for c in loco_classes:
        all_categories.append(("loco", c))

    # optional shuffle for experiment reproducibility
    random.shuffle(all_categories)

    print("[INFO] Task order:")
    for i, (t, c) in enumerate(all_categories):
        print(f"  Task {i+1}: {t}/{c}")

    # model (feature extractor)
    backbone = cfg["BACKBONE"].lower()
    if backbone == "simplecnn":
        model = SimpleCNN(input_channels=3).to(device)
    elif backbone == "resnet18":
        model = ResNet18(input_channels=3).to(device)
    elif backbone == "resnet18_pretrained":
        model = ResNet18_pretrained(input_channels=3).to(device)
    else:
        raise ValueError(f"Unknown BACKBONE: {cfg['BACKBONE']}")

    # freeze whole model (we're using pretrained features)
    for p in model.parameters():
        p.requires_grad = False

    outdir = cfg.get("OUTDIR", "results/mvtec+loco/Anomaly")
    ensure_dir(outdir)

    memory_banks = {}  # (task_idx) -> numpy array (K,D)
    accs = []
    auc_matrix = []

    # build and evaluate per task
    for task_id, (t, cname) in enumerate(all_categories):
        print(f"\n===== Building memory for Task {task_id+1}/{len(all_categories)}: {t}/{cname} =====")

        # select train/test sets
        if t == "mvtec":
            train_ds = mvtec_train[cname]
            test_ds = mvtec_test[cname]
        else:
            train_ds = loco_train[cname]
            test_ds = loco_test[cname]

        # Build memory from normal training images only
        mem = build_memory_bank_for_task(model, train_ds, cfg, device)
        memory_banks[task_id] = mem

        # save memory
        mem_path = os.path.join(outdir, f"memory_task_{task_id+1}_{t}_{cname}.npz")
        np.savez_compressed(mem_path, memory=mem)
        print(f"[INFO] Saved memory ({mem.shape[0]} vectors, dim={mem.shape[1] if mem.shape[0]>0 else 0}) -> {mem_path}")

        # evaluate on this task's test set using THIS task memory
        scores, labels = compute_anomaly_scores_from_memory(model, test_ds, mem, cfg, device)
        auc = compute_auc_safe(scores, labels)
        print(f"  AUC on {t}/{cname}: {auc:.4f}")

        # save outputs
        save_task_output(outdir, task_id + 1, task_id + 1, scores, labels)

        auc_matrix.append(auc)
        accs.append(auc)

    # summary
    print("\n=== SUMMARY ===")
    for i, (t, cname) in enumerate(all_categories):
        print(f"Task {i+1:02d} - {t}/{cname}: AUC = {auc_matrix[i]:.4f}")

    # Save AUC vector and timestamped file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(outdir, f"auc_per_task_{ts}.npy"), np.array(auc_matrix))
    try:
        # produce a small heatmap (single column vector)
        arr = np.array(auc_matrix).reshape(-1, 1) * 100.0
        plot_heatmap(arr, outdir, title=f"AUC per task ({ts})")
    except Exception as e:
        print("[WARN] heatmap failed:", e)

    print("[INFO] Done.")
    return auc_matrix


# CLI
def parse_overrides(kv_list):
    out = {}
    for item in kv_list or []:
        if "=" in item:
            k, v = item.split("=", 1)
            if v.lower() in ("true", "false"):
                vv = v.lower() == "true"
            else:
                try:
                    vv = int(v)
                except Exception:
                    try:
                        vv = float(v)
                    except Exception:
                        vv = v
            out[k] = vv
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--set", nargs="*")
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)
            if yaml_cfg:
                cfg.update(yaml_cfg)

    overrides = parse_overrides(args.set)
    cfg.update(overrides)

    cfg["DEVICE"] = normalize_device(cfg.get("DEVICE", cfg["DEVICE"]))
    run(cfg)
