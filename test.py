# test.py — Final stable version for Memory-Based Continual Anomaly Detection

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score


# 1. ResNet18 Pretrained Feature Extractor

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

# 2. Image Preprocessing

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# 3. Safe Image Loader
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

# 4. Cosine-Distance Anomaly Score
def anomaly_score(x, memory_bank):
    x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    mem = memory_bank / (memory_bank.norm(dim=1, keepdim=True) + 1e-8)

    sim = x @ mem.T
    score = 1 - sim.max(dim=1)[0]
    return score.cpu().numpy()


# 5. MAIN TESTING PIPELINE
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---------- MUST MATCH TRAINING ORDER ----------
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

    # Feature extractor
    model = ResNet18_Pretrained_Features(device).eval()

    results = []

    print(f"[INFO] Evaluating with memory banks from: {args.mem_dir}")

    # Evaluate each class using correct memory + correct dataset path
    for task_id, (ds_type, cname) in enumerate(TASK_ORDER, start=1):

        # correct file name for memory
        mem_file = os.path.join(
            args.mem_dir,
            f"memory_task_{task_id}_{ds_type}_{cname}.npz"
        )

        if not os.path.exists(mem_file):
            print(f"[ERROR] Missing memory file: {mem_file}")
            results.append((f"{ds_type}/{cname}", float('nan')))
            continue

        mem_data = np.load(mem_file)
        if "memory" not in mem_data:
            print(f"[ERROR] File missing 'memory' key: {mem_file} -> Keys: {list(mem_data.keys())}")
            results.append((f"{ds_type}/{cname}", float('nan')))
            continue

        memory_bank = torch.tensor(mem_data["memory"]).float().to(device)

        # correct dataset root
        if ds_type == "mvtec":
            test_root = f"data/mvtec/{cname}/test"
        else:
            test_root = f"data/mvtec-loco/{cname}/test"

        imgs, labels = load_images(test_root)
        if imgs is None:
            results.append((f"{ds_type}/{cname}", float('nan')))
            continue

        imgs = imgs.to(device)

        # extract features
        with torch.no_grad():
            feats = model(imgs)

        scores = anomaly_score(feats, memory_bank)

        try:
            auc = roc_auc_score(labels.numpy(), scores)
        except Exception:
            auc = float('nan')

        results.append((f"{ds_type}/{cname}", auc))
        print(f"{ds_type}/{cname:25s} AUC = {auc:.4f}")

    # SUMMARY
    print("\n=== SUMMARY ===")
    for i, (name, auc) in enumerate(results):
        print(f"Class {i:02d} - {name}: AUC = {auc:.4f}")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
