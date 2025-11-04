import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argument import get_args
from datasets.mvtec_loader import get_mvtec_tasks
from models.vit_dne import ViT_DNE


# ===========================================================
# Helper: best threshold per task
# ===========================================================
def find_best_threshold(scores, labels, percentiles=(1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)):
    best_acc, best_thr = 0.0, scores.median().item()
    for p in percentiles:
        thr = np.percentile(scores.numpy(), p)
        preds = (scores.numpy() > thr).astype(np.int64)
        acc = (preds == labels.numpy()).mean() * 100
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr, best_acc


# ===========================================================
# Compute accuracy for one task
# ===========================================================
def compute_task_accuracy(model, dataloader, device, task_idx):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            scores = model.anomaly_scores([(imgs, labels)], memory_idx=task_idx)
            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

    if len(all_scores) == 0:
        return 0.0

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    thr, acc = find_best_threshold(scores, labels)
    return float(acc)


def evaluate_all_tasks(model, test_tasks, upto_task, device):
    task_accuracies = []
    for i in range(upto_task + 1):
        acc = compute_task_accuracy(model, test_tasks[i], device, task_idx=i)
        task_accuracies.append(acc)
        print(f"  Accuracy on Task {i+1}: {acc:.2f}%")
    avg_acc = np.mean(task_accuracies)
    print(f"  → Average accuracy up to Task {upto_task+1}: {avg_acc:.2f}%")
    return task_accuracies, avg_acc


# ===========================================================
# Backward Transfer & Heatmap
# ===========================================================
def compute_backward_transfer(acc_matrix):
    n_tasks = len(acc_matrix)
    if n_tasks < 2:
        return 0.0
    final_accs = acc_matrix[-1]
    initial_accs = [row[i] for i, row in enumerate(acc_matrix) if i < len(final_accs)]
    diffs = [final_accs[i] - initial_accs[i] for i in range(len(initial_accs) - 1)]
    return np.mean(diffs)


def plot_accuracy_heatmap(acc_matrix, save_path="./results/accuracy_matrix.png"):
    max_len = max(len(row) for row in acc_matrix)
    padded = np.full((len(acc_matrix), max_len), np.nan)
    for i, row in enumerate(acc_matrix):
        padded[i, :len(row)] = row

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=0.9)
    sns.heatmap(padded, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5,
                cbar_kws={'label': 'Accuracy (%)'},
                xticklabels=[f"T{i+1}" for i in range(max_len)],
                yticklabels=[f"After T{i+1}" for i in range(len(acc_matrix))])
    plt.xlabel("Task Evaluated")
    plt.ylabel("After Training Task")
    plt.title("Continual Learning Accuracy Matrix (ViT + DNE)")
    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualization] Accuracy matrix heatmap saved to: {save_path}")


# ===========================================================
# Main
# ===========================================================
def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    print("\n[1] Loading MVTec dataset...")
    train_tasks, test_tasks = get_mvtec_tasks(args.data_dir, image_size=224, batch_size=8)

    print(f"\n[2] Preparing model (ViT + DNE)...")
    model = ViT_DNE(device=str(device), freeze_backbone=True, pretrained_backbone=True)

    print(f"\n[3] Starting continual training across {len(train_tasks)} tasks...\n")
    accuracy_matrix = []

    for task_id, (train_loader, test_loader) in enumerate(zip(train_tasks, test_tasks)):
        print("=" * 60)
        print(f"🔹 Task {task_id+1} — Training category {task_id+1}/{len(train_tasks)}")
        print("=" * 60)

        # Fit Gaussian for this task
        print("[Fit] Learning DNE memory from normal samples...")
        model.fit_task(train_loader, max_batches=None)

        # Evaluate all tasks so far
        print("[Eval] Evaluating across all seen tasks...")
        task_accuracies, avg_acc = evaluate_all_tasks(model, test_tasks, task_id, device)
        accuracy_matrix.append(task_accuracies)

        print(f"✅ Finished Task {task_id+1}. Average accuracy: {avg_acc:.2f}%\n")

    # Final metrics
    print("\n================== FINAL RESULTS ==================")
    for i, row in enumerate(accuracy_matrix):
        row_str = " | ".join([f"T{j+1}: {acc:.2f}%" for j, acc in enumerate(row)])
        print(f"After Task {i+1}: [ {row_str} ]")

    avg_final = np.mean([r[-1] for r in accuracy_matrix])
    print(f"\nFinal Average Accuracy: {avg_final:.2f}%")

    bwt_value = compute_backward_transfer(accuracy_matrix)
    print(f"Backward Transfer (Forgetting): {bwt_value:.2f}%")

    plot_accuracy_heatmap(accuracy_matrix)

    if args.save_checkpoint:
        torch.save(model.state_dict(), f"{args.save_path}/vit_dne_final.pth")
        print(f"\n✅ Model saved to {args.save_path}/vit_dne_final.pth")


if __name__ == "__main__":
    main()
