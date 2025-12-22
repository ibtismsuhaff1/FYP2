import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration
METHOD = "finetune_resnet18_pretrained" 
RESULTS_ROOT = "results/mvtec+loco/CL"
OUT_PLOT_DIR = "results/plots"

os.makedirs(OUT_PLOT_DIR, exist_ok=True)

# Load latest accuracy & AUC matrices
method_dir = os.path.join(RESULTS_ROOT, METHOD)

acc_files = sorted([f for f in os.listdir(method_dir) if f.startswith("acc_matrix")])
auc_files = sorted([f for f in os.listdir(method_dir) if f.startswith("auc_matrix")])

acc_matrix = np.load(os.path.join(method_dir, acc_files[-1]))
auc_matrix = np.load(os.path.join(method_dir, auc_files[-1]))

num_tasks = acc_matrix.shape[0]
tasks = np.arange(1, num_tasks + 1)

# Task-wise Accuracy (after final task)
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

# Average Accuracy per Task
avg_accuracy = np.mean(acc_matrix, axis=1)

plt.figure()
plt.plot(tasks, avg_accuracy, marker='o')
plt.xlabel("Task Number")
plt.ylabel("Average Accuracy (%)")
plt.title(f"Average Accuracy Across Tasks ({METHOD})")
plt.grid(True)

plt.savefig(f"{OUT_PLOT_DIR}/{METHOD}_average_accuracy.png",
            dpi=300, bbox_inches="tight")
plt.close()


# Task-wise AUC (after final task)

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

print("Real graphs generated from accuracy & AUC matrices")
