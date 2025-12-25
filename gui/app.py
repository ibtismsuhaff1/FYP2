import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Continual Anomaly Detection Benchmark", layout="wide")

# -------------------------
# Configuration
# -------------------------
RESULTS_ROOT = Path("results/mvtec+loco/CL")
METHODS = {
    "Finetune": "finetune_resnet18_pretrained",
    "Replay": "replay_resnet18_pretrained",
    "EWC": "ewc_resnet18_pretrained",
    "LwF": "lwf_resnet18_pretrained",
    "GPM": "gpm_resnet18_pretrained",
}

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Benchmark Settings")

method_name = st.sidebar.selectbox("Continual Learning Method", list(METHODS.keys()))
metric = st.sidebar.radio("Metric", ["Accuracy", "AUC"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Backbone:** ResNet18 (Pretrained)")
st.sidebar.markdown("**Dataset:** MVTec + LOCO")
st.sidebar.markdown("**Tasks:** 20")

# -------------------------
# Load Data
# -------------------------
method_dir = RESULTS_ROOT / METHODS[method_name]

def load_latest_matrix(prefix):
    files = sorted(method_dir.glob(f"{prefix}_matrix_*.npy"))
    if not files:
        return None
    return np.load(files[-1])

acc_matrix = load_latest_matrix("acc")
auc_matrix = load_latest_matrix("auc")

matrix = acc_matrix if metric == "Accuracy" else auc_matrix

# -------------------------
# Main UI
# -------------------------
st.title("Continual Anomaly Detection Benchmark")

if matrix is None:
    st.error("No result matrices found for this method.")
    st.stop()

# ---- Task-wise Curve ----
st.subheader(f"Task-wise {metric}")

taskwise = np.diag(matrix)

fig, ax = plt.subplots()
ax.plot(range(1, len(taskwise) + 1), taskwise, marker="o")
ax.set_xlabel("Task Number")
ax.set_ylabel(metric)
ax.set_title(f"{method_name} - Task-wise {metric}")
ax.grid(True)
st.pyplot(fig)

# ---- Heatmap ----
st.subheader(f"{metric} Heatmap")

fig, ax = plt.subplots()
im = ax.imshow(matrix, cmap="viridis")
ax.set_xlabel("Evaluated Task")
ax.set_ylabel("Trained Task")
fig.colorbar(im, ax=ax)
st.pyplot(fig)

# ---- Summary ----
st.subheader("Summary Statistics")

avg_score = np.mean(taskwise)
forgetting = taskwise[0] - taskwise[-1]

col1, col2 = st.columns(2)
col1.metric("Average " + metric, f"{avg_score:.3f}")
col2.metric("Forgetting (Task 1 → Last)", f"{forgetting:.3f}")
