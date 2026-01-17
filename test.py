# test.py — Incremental Evaluation + Correct Backward Transfer (BWT)
# >>> UPDATED VERSION (SAVES RESULTS INTO mem_dir) <<<

import os
import argparse
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score

# =====================================================
# 1. ResNet18 Pretrained Feature Extractor
# =====================================================

class ResNet18_Pretrained_Features(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        net = models.resnet18(weights=weights)

        self.features = torch.nn.Sequential(*list(net.children())[:-1]).to(device)
        self.out_dim = net.fc.in_features

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

# =====================================================
# 2. Image Preprocessing
# =====================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# =====================================================
# 3. Safe Image Loader
# =====================================================

def load_images(root):
    if not os.path.isdir(root):
        print(f"[WARN] Test folder does not exist: {root}")
        return None, None

    imgs, labels = [], []

    for sub, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img = Image.open(os.path.join(sub, f)).convert("RGB")
                imgs.append(TEST_TRANSFORM(img))

                lbl = 0 if "good" in sub.lower() or "normal" in sub.lower() else 1
                labels.append(lbl)

    if len(imgs) == 0:
        print(f"[WARN] No images found in: {root}")
        return None, None

    return torch.stack(imgs), torch.tensor(labels)

# =====================================================
# 4. Cosine-Distance Anomaly Score
# =====================================================

def anomaly_score(x, memory_bank):
    x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    mem = memory_bank / (memory_bank.norm(dim=1, keepdim=True) + 1e-8)

    sim = x @ mem.T
    score = 1 - sim.max(dim=1)[0]
    return score.cpu().numpy()

# =====================================================
# 5. MAIN TESTING PIPELINE (TRUE INCREMENTAL + BWT)
# =====================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Evaluating with memory banks from: {args.mem_dir}")

    # Ensure results go INTO the same folder as your trained model
    os.makedirs(args.mem_dir, exist_ok=True)

    TASK_ORDER = [
        ("loco", "splicing_connectors"),
        ("mvtec", "hazelnut"),
        ("mvtec", "zipper"),
        ("mvtec", "grid"),
        ("mvtec", "screw"),
        ("mvtec", "wood"),
        ("loco", "breakfast_box"),
        ("loco", "screw_bag"),
        ("mvtec", "leather"),
        ("mvtec", "transistor"),
        ("loco", "pushpins"),
        ("mvtec", "tile"),
        ("mvtec", "cable"),
        ("mvtec", "toothbrush"),
        ("mvtec", "capsule"),
        ("loco", "juice_bottle"),
        ("mvtec", "metal_nut"),
        ("mvtec", "pill"),
        ("mvtec", "bottle"),
        ("mvtec", "carpet"),
    ]

    num_tasks = len(TASK_ORDER)

    # Initialize matrices
    acc_matrix = np.full((num_tasks, num_tasks), np.nan)
    auc_matrix = np.full((num_tasks, num_tasks), np.nan)

    # Feature extractor
    model = ResNet18_Pretrained_Features(device).eval()

    # =====================================================
    # TRUE INCREMENTAL EVALUATION LOOP
    # =====================================================

    for current_task in range(1, num_tasks + 1):
        print(f"\n========== Evaluating AFTER training Task {current_task} ==========")

        for prev_task in range(1, current_task + 1):

            ds_type, cname = TASK_ORDER[prev_task - 1]

            mem_file = os.path.join(
                args.mem_dir,
                f"memory_task_{current_task}_{ds_type}_{cname}.npz"
            )

            if not os.path.exists(mem_file):
                print(f"[ERROR] Missing memory file: {mem_file}")
                continue

            mem_data = np.load(mem_file)
            if "memory" not in mem_data:
                print(f"[ERROR] File missing 'memory' key: {mem_file}")
                continue

            memory_bank = torch.tensor(mem_data["memory"]).float().to(device)

            # dataset path
            if ds_type == "mvtec":
                test_root = f"data/mvtec/{cname}/test"
            else:
                test_root = f"data/mvtec-loco/{cname}/test"

            imgs, labels = load_images(test_root)
            if imgs is None:
                continue

            imgs = imgs.to(device)

            with torch.no_grad():
                feats = model(imgs)

            scores = anomaly_score(feats, memory_bank)

            # ---- AUC ----
            try:
                auc = roc_auc_score(labels.numpy(), scores)
            except Exception:
                auc = float('nan')

            # ---- ACCURACY ----
            preds = (scores > np.median(scores)).astype(int)
            acc = accuracy_score(labels.numpy(), preds) * 100.0

            acc_matrix[current_task - 1, prev_task - 1] = acc
            auc_matrix[current_task - 1, prev_task - 1] = auc

            print(f"After T{current_task} -> Eval T{prev_task:02d} "
                  f"({ds_type}/{cname:20s}) | ACC = {acc:.2f} | AUC = {auc:.4f}")

    # =====================================================
    # TRUE BACKWARD TRANSFER (BWT) — CORRECT VERSION
    # =====================================================

    print("\n===== TRUE BACKWARD TRANSFER (BWT) =====")

    # Initial accuracy when each task was first learned
    initial_acc = np.array([acc_matrix[t, t] for t in range(num_tasks)])

    # Re-evaluate all old tasks using FINAL model (Task 20 memory)
    print("\n[INFO] Re-evaluating all previous tasks using FINAL model (Task 20) ...")

    final_model_acc = []

    for t in range(num_tasks):
        ds_type, cname = TASK_ORDER[t]

        mem_file = os.path.join(
            args.mem_dir,
            f"memory_task_{num_tasks}_{ds_type}_{cname}.npz"
        )

        if not os.path.exists(mem_file):
            print(f"[WARN] Missing final memory for task {t+1}, using matrix value instead.")
            final_model_acc.append(acc_matrix[-1, t])
            continue

        mem_data = np.load(mem_file)
        memory_bank = torch.tensor(mem_data["memory"]).float().to(device)

        if ds_type == "mvtec":
            test_root = f"data/mvtec/{cname}/test"
        else:
            test_root = f"data/mvtec-loco/{cname}/test"

        imgs, labels = load_images(test_root)
        if imgs is None:
            final_model_acc.append(acc_matrix[-1, t])
            continue

        imgs = imgs.to(device)

        with torch.no_grad():
            feats = model(imgs)

        scores = anomaly_score(feats, memory_bank)

        preds = (scores > np.median(scores)).astype(int)
        acc = accuracy_score(labels.numpy(), preds) * 100.0

        final_model_acc.append(acc)

    final_model_acc = np.array(final_model_acc)

    # Compute BWT
    bwt_scores = []
    for t in range(num_tasks - 1):  # last task has no BWT
        bwt = final_model_acc[t] - initial_acc[t]
        bwt_scores.append(bwt)
        print(f"Task {t+1}: ΔACC = {bwt:.2f}")

    overall_bwt = np.nanmean(bwt_scores)
    print(f"\nOverall BWT = {overall_bwt:.2f}")

    # =====================================================
    # SAVE RESULTS (NOW INSIDE mem_dir ✅)
    # =====================================================

    np.save(os.path.join(args.mem_dir, "acc_matrix.npy"), acc_matrix)
    np.save(os.path.join(args.mem_dir, "auc_matrix.npy"), auc_matrix)
    np.save(os.path.join(args.mem_dir, "bwt_scores.npy"), np.array(bwt_scores))

    print("\n=== SAVED RESULTS (IN MODEL FOLDER) ===")
    print(f"Accuracy matrix -> {os.path.join(args.mem_dir, 'acc_matrix.npy')}")
    print(f"AUC matrix      -> {os.path.join(args.mem_dir, 'auc_matrix.npy')}")
    print(f"BWT scores      -> {os.path.join(args.mem_dir, 'bwt_scores.npy')}")

# =====================================================
# CLI
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
