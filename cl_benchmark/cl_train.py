# cl_benchmark/cl_train.py  — FIXED VERSION (TRUE INCREMENTAL MEMORY + BWT SAFE)

import os
import random
import argparse
import yaml
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# backbones
from cl_benchmark.backbones.models import (
    SimpleCNN,
    ResNet18,
    ResNet18_pretrained,
)

# CL agents
from cl_benchmark.agents.finetune import FinetuneAgent
from cl_benchmark.agents.replay import ReplayAgent
from cl_benchmark.agents.ewc import EWCAgent
from cl_benchmark.agents.lwf import LwFAgent
from cl_benchmark.agents.gpm import GPMAgent

# datasets
from cl_benchmark.datasets.mvtec_loader import load_mvtec_all_categories
from cl_benchmark.datasets.mvtec_loco_loader import load_mvtec_loco_all_categories

# utils
from cl_benchmark.utils.metrics_io import (
    ensure_dir,
    save_task_output,
    plot_heatmap,
    compute_auc_safe,
)

# =========================
# DEFAULT CONFIG
# =========================

DEFAULT_CFG = {
    "BACKBONE": "ResNet18_pretrained",
    "CL_METHOD": "Finetune",
    "OPTIMIZER": "SGD",
    "SGD_LR": 0.003,
    "ADAM_LR": 0.0005,
    "SGD_MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0,

    "EPOCHS_PER_TASK": 3,
    "BATCH_SIZE": 32,

    # Replay
    "REPLAY_MAX_SAMPLES": 2000,
    "REPLAY_BATCH_SIZE": 32,

    # EWC
    "EWC_LAMBDA": 10.0,

    # LwF
    "LWF_LAMBDA": 1.0,
    "LWF_TEMPERATURE": 2.0,

    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SEED": 42,

    "MVT_ROOT": "data/mvtec",
    "MVT_LOCO_ROOT": "data/mvtec-loco",

    "OUTDIR_ROOT": "results/mvtec+loco/CL",
}

# =========================
# Utilities
# =========================

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


def set_binary_classifier(model, num_outputs=2):
    if getattr(model, "fc_input_features", 0) == 0:
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            x = model.features(dummy)
            x = x.view(x.size(0), -1)
            model.fc_input_features = x.size(1)

    model.num_classes = num_outputs
    new_fc = nn.Linear(model.fc_input_features, num_outputs)
    device = next(model.parameters()).device
    model.classifier = new_fc.to(device)


def get_agent(method: str, model: nn.Module, cfg: dict):
    m = method.lower()
    if m == "finetune":
        return FinetuneAgent(model, cfg)
    if m == "replay":
        return ReplayAgent(model, cfg)
    if m == "ewc":
        return EWCAgent(model, cfg)
    if m == "lwf":
        return LwFAgent(model, cfg)
    if m == "gpm":
        return GPMAgent(model, cfg)
    raise ValueError(f"Unknown CL_METHOD: {method}")

# =========================
# Train / Eval loops
# =========================

def train_one_epoch(model, loader, optimizer, criterion, agent, cfg, device):
    model.train()
    method = cfg["CL_METHOD"].lower()
    gradient_buffer = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            gradient_buffer[name] = []

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        if method == "replay":
            data, target = agent.integrate_replay(data, target)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)

        if method == "ewc":
            loss = loss + agent.penalty()

        if method == "lwf":
            loss = loss + agent.distillation_loss(data, outputs)

        loss.backward()

        if method == "gpm":
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradient_buffer[name].append(
                        param.grad.detach().clone()
                    )
            agent.project_gradients()

        optimizer.step()
    return gradient_buffer


def evaluate(model, dataset, batch_size, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct, total = 0, 0
    labels_all, scores_all = [], []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            out = model(data)
            prob = nn.functional.softmax(out, dim=1)
            pred = prob.argmax(dim=1)

            correct += (pred == target).sum().item()
            total += data.size(0)

            if prob.size(1) >= 2:
                scores_all.extend(prob[:, 1].cpu().numpy().tolist())
            else:
                scores_all.extend(prob[:, 0].cpu().numpy().tolist())

            labels_all.extend(target.cpu().numpy().tolist())

    acc = 100.0 * correct / max(total, 1)
    auc = compute_auc_safe(scores_all, labels_all)
    return acc, auc, scores_all, labels_all

# ==========================================================
# FIXED: BUILD MEMORY BANK (PROTOTYPE) FOR A TASK
# ==========================================================

def build_memory_bank(model, dataset, batch_size, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            feats = model.features(data)
            feats = feats.view(feats.size(0), -1)

            # keep only NORMAL samples (label 0)
            normal_feats = feats[target == 0]
            if normal_feats.numel() > 0:
                features.append(normal_feats.cpu().numpy())

    if len(features) == 0:
        return None

    features = np.concatenate(features, axis=0)
    return features.mean(axis=0, keepdims=True)  # single prototype vector

# ==========================================================
# RUN
# ==========================================================

def run(cfg: dict):
    set_seed(cfg["SEED"])
    device = normalize_device(cfg["DEVICE"])
    cfg["DEVICE"] = device

    mvtec_classes, mvtec_train, mvtec_test = load_mvtec_all_categories(cfg["MVT_ROOT"])
    loco_classes, loco_train, loco_test = load_mvtec_loco_all_categories(cfg["MVT_LOCO_ROOT"])

    all_categories = []
    for c in mvtec_classes:
        all_categories.append(("mvtec", c))
    for c in loco_classes:
        all_categories.append(("loco", c))

    random.seed(cfg["SEED"])
    random.shuffle(all_categories)

    print("[INFO] Task order:")
    for i, (t, c) in enumerate(all_categories):
        print(f"  Task {i+1}: {t}/{c}")

    backbone = cfg["BACKBONE"].lower()
    if backbone == "simplecnn":
        model = SimpleCNN(input_channels=3).to(device)
    elif backbone == "resnet18":
        model = ResNet18(input_channels=3).to(device)
    elif backbone == "resnet18_pretrained":
        model = ResNet18_pretrained(input_channels=3).to(device)
    else:
        raise ValueError(f"Unknown BACKBONE: {cfg['BACKBONE']}")

    for p in model.features.parameters():
        p.requires_grad = True

    set_binary_classifier(model, num_outputs=2)
    agent = get_agent(cfg["CL_METHOD"], model, cfg)

    method_tag = cfg["CL_METHOD"].lower()
    backbone_tag = backbone
    outdir = os.path.join(cfg["OUTDIR_ROOT"], f"{method_tag}_{backbone_tag}")
    ensure_dir(outdir)

    criterion = nn.CrossEntropyLoss()
    acc_matrix = []
    auc_matrix = []

    # ===== KEEP TRACK OF ALL SEEN TRAIN DATASETS (CRUCIAL FIX) =====
    seen_train_datasets = []
    seen_task_names = []

    # ===== SEQUENTIAL TASKS =====
    for t_id, (ds_type, cname) in enumerate(all_categories):
        print(f"\n===== Training Task {t_id+1}/{len(all_categories)} ({ds_type}/{cname}) =====")

        if ds_type == "mvtec":
            train_ds = mvtec_train[cname]
            test_ds = mvtec_test[cname]
        else:
            train_ds = loco_train[cname]
            test_ds = loco_test[cname]

        seen_train_datasets.append(train_ds)
        seen_task_names.append((ds_type, cname))

        train_loader = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"], shuffle=True)

        params = [p for p in model.parameters() if p.requires_grad]

        if cfg["OPTIMIZER"].lower() == "sgd":
            optimizer = optim.SGD(
                params,
                lr=cfg["SGD_LR"],
                momentum=cfg["SGD_MOMENTUM"],
                weight_decay=cfg["WEIGHT_DECAY"],
            )
        else:
            optimizer = optim.Adam(
                params, lr=cfg["ADAM_LR"], weight_decay=cfg["WEIGHT_DECAY"]
            )

        if hasattr(agent, "before_task"):
            agent.before_task()

        for _ in range(cfg["EPOCHS_PER_TASK"]):
            train_one_epoch(
                model, train_loader, optimizer, criterion, agent, cfg, device
            )

        method = cfg["CL_METHOD"].lower()
        if method == "replay":
            agent.after_task(train_ds)
        elif method == "ewc":
            fisher_loader = DataLoader(
                train_ds,
                batch_size=min(64, len(train_ds)),
                shuffle=True,
            )
            agent.after_task(fisher_loader, device)
        elif method == "lwf":
            agent.after_task()
        elif method == "gpm":
            agent.after_task(train_loader, device)

        # ===== EVALUATE ALL SEEN TASKS =====
        row_acc, row_auc = [], []
        for eval_id in range(t_id + 1):
            e_type, e_name = all_categories[eval_id]

            if e_type == "mvtec":
                eval_ds = mvtec_test[e_name]
            else:
                eval_ds = loco_test[e_name]

            acc, auc, scores, labels = evaluate(
                model, eval_ds, cfg["BATCH_SIZE"], device
            )

            row_acc.append(acc)
            row_auc.append(auc)

            save_task_output(outdir, t_id + 1, eval_id + 1, scores, labels)

            print(f"  Task {eval_id+1:2d} ({e_type}/{e_name}): ACC={acc:.2f}%  AUC={auc:.4f}")

        acc_matrix.append(row_acc)
        auc_matrix.append(row_auc)

        # ===== 🔥 CRUCIAL FIX: SAVE MEMORY FOR ALL SEEN TASKS (NOT JUST CURRENT) =====
        for mem_id, (m_type, m_name) in enumerate(seen_task_names):
            mem_vec = build_memory_bank(
                model,
                seen_train_datasets[mem_id],
                cfg["BATCH_SIZE"],
                device,
            )

            if mem_vec is not None:
                mem_path = os.path.join(
                    outdir,
                    f"memory_task_{t_id+1}_{m_type}_{m_name}.npz"
                )
                np.savez(mem_path, memory=mem_vec)
                print(f"[INFO] Saved memory -> {mem_path}")

        torch.save(
            model.state_dict(),
            os.path.join(outdir, f"model_after_task_{t_id+1}.pth"),
        )

    # ===== SAVE MATRICES =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_arr = np.zeros((len(acc_matrix), len(acc_matrix)))
    auc_arr = np.zeros_like(acc_arr)

    for i, row in enumerate(acc_matrix):
        acc_arr[i, : len(row)] = row
    for i, row in enumerate(auc_matrix):
        auc_arr[i, : len(row)] = row

    np.save(os.path.join(outdir, f"acc_matrix_{timestamp}.npy"), acc_arr)
    np.save(os.path.join(outdir, f"auc_matrix_{timestamp}.npy"), auc_arr)

    try:
        plot_heatmap(acc_arr, outdir, title=f"ACC ({cfg['CL_METHOD']})")
        plot_heatmap(auc_arr * 100.0, outdir, title=f"AUCx100 ({cfg['CL_METHOD']})")
    except Exception as e:
        print("[WARN] heatmap failed:", e)

    final_acc = np.mean(acc_matrix[-1])
    final_auc = np.mean(auc_matrix[-1])

    print(
        f"\n[SUMMARY] {cfg['CL_METHOD']} + {cfg['BACKBONE']}: "
        f"Final mean ACC={final_acc:.2f}% | Final mean AUC={final_auc:.4f}"
    )
    return final_acc, final_auc

# ==========================================================
# CLI
# ==========================================================

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
