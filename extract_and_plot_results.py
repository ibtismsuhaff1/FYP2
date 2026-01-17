import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================
METHOD = "finetune_resnet18_pretrained"
RESULTS_ROOT = "results/mvtec+loco/CL"
OUT_PLOT_DIR = "results/plots"

os.makedirs(OUT_PLOT_DIR, exist_ok=True)

# =========================
# Load latest accuracy & AUC matrices
# =========================
method_dir = os.path.join(RESULTS_ROOT, METHOD)

acc_files = sorted([f for f in os.listdir(method_dir) if f.startswith("acc_matrix")])
auc_files = sorted([f for f in os.listdir(method_dir) if f.startswith("auc_matrix")])

acc_matrix = np.load(os.path.join(method_dir, acc_files[-1]))
auc_matrix = np.load(os.path.join(method_dir, auc_files[-1]))

num_tasks = acc_matrix.shape[0]
tasks = np.arange(1, num_tasks + 1)

# =========================
# 1) Task-wise Accuracy (after final task)  (YOUR ORIGINAL - KEPT)
# =========================
final_task_accuracy = acc_matrix[-1, :]

plt.figure()
plt.plot(tasks, final_task_accuracy, marker='o')
plt.xlabel("Task Number")
plt.ylabel("Accuracy (%)")
plt.title(f"Task-wise Accuracy ({METHOD})")
plt.grid(True)

plt.savefig(f"{OUT_PLOT_DIR}/{METHOD}_taskwise_accuracy.png",
            dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 2) Average Accuracy per Task (YOUR ORIGINAL - KEPT)
# =========================
avg_accuracy_per_task = np.mean(acc_matrix, axis=1)

plt.figure()
plt.plot(tasks, avg_accuracy_per_task, marker='o')
plt.xlabel("Task Number")
plt.ylabel("Average Accuracy (%)")
plt.title(f"Average Accuracy Across Tasks ({METHOD})")
plt.grid(True)

plt.savefig(f"{OUT_PLOT_DIR}/{METHOD}_average_accuracy.png",
            dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 3) Task-wise AUC (after final task) (YOUR ORIGINAL - KEPT)
# =========================
final_task_auc = auc_matrix[-1, :]

plt.figure()
plt.plot(tasks, final_task_auc, marker='o')
plt.xlabel("Task Number")
plt.ylabel("AUC")
plt.title(f"Task-wise AUC ({METHOD})")
plt.grid(True)

plt.savefig(f"{OUT_PLOT_DIR}/{METHOD}_taskwise_auc.png",
            dpi=300, bbox_inches="tight")
plt.close()

# ==========================================================
# ✅ NEW PART: BACKWARD TRANSFER (BWT) CALCULATION
# ==========================================================

def compute_bwt(acc_matrix):
    """
    Compute Backward Transfer (BWT)

    BWT = average over previous tasks of:
    (final accuracy on task i - accuracy right after training task i)
    """
    T = acc_matrix.shape[0]
    bwt_values = []

    for i in range(T - 1):  # exclude last task
        initial_acc = acc_matrix[i, i]
        final_acc = acc_matrix[T - 1, i]
        bwt_values.append(final_acc - initial_acc)

    return np.mean(bwt_values), np.array(bwt_values)

bwt_avg, bwt_per_task = compute_bwt(acc_matrix)

# Save BWT for record
np.save(f"{OUT_PLOT_DIR}/{METHOD}_bwt.npy", bwt_per_task)

# ==========================================================
# ✅ NEW PART: FINAL AVERAGE ACCURACY (Single Value)
# ==========================================================

final_average_accuracy = np.mean(acc_matrix[-1, :])

with open(f"{OUT_PLOT_DIR}/{METHOD}_metrics.txt", "w") as f:
    f.write(f"Final Average Accuracy: {final_average_accuracy:.2f}\n")
    f.write(f"Backward Transfer (BWT): {bwt_avg:.2f}\n")

# ==========================================================
# ✅ NEW PART: BAR CHART 1 — AVERAGE ACCURACY (VERTICAL)
# ==========================================================

plt.figure()
plt.bar([METHOD], [final_average_accuracy])
plt.ylabel("Average Accuracy (%)")
plt.title("Final Average Accuracy")
plt.grid(axis="y")

plt.savefig(f"{OUT_PLOT_DIR}/{METHOD}_avg_accuracy_bar.png",
            dpi=300, bbox_inches="tight")
plt.close()

# ==========================================================
# ✅ NEW PART: BAR CHART 2 — BWT (VERTICAL)
# ==========================================================

plt.figure()
plt.bar([METHOD], [bwt_avg])
plt.ylabel("Backward Transfer (BWT)")
plt.title("Backward Transfer (BWT)")
plt.grid(axis="y")

plt.savefig(f"{OUT_PLOT_DIR}/{METHOD}_bwt_bar.png",
            dpi=300, bbox_inches="tight")
plt.close()

print("Updated graphs generated from accuracy & AUC matrices")
print(f"Final Average Accuracy: {final_average_accuracy:.2f}")
print(f"Backward Transfer (BWT): {bwt_avg:.2f}")
