import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# =========================
# STREAMLIT PAGE SETUP
# =========================
st.set_page_config(
    page_title="Continual Anomaly Detection Benchmark",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to make GUI prettier
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f4f6f9;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .big-title {
        font-size: 28px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# CONFIGURATION (YOUR PATH)
# =========================
RESULTS_ROOT = Path("results/mvtec+loco/CL")

METHODS = {
    "Finetune": "finetune_resnet18_pretrained",
    "Replay": "replay_resnet18_pretrained",
    "EWC": "ewc_resnet18_pretrained",
    "LwF": "lwf_resnet18_pretrained",
    "GPM": "gpm_resnet18_pretrained",
}

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Benchmark Settings")

method_name = st.sidebar.selectbox(
    "Continual Learning Method",
    list(METHODS.keys()),
    help="Select which CL method to visualize",
)

metric = st.sidebar.radio("Metric", ["Accuracy", "AUC"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Backbone:** ResNet18 (Pretrained)")
st.sidebar.markdown("**Dataset:** MVTec + LOCO")
st.sidebar.markdown("**Tasks:** 20")

# =========================
# LOAD FILES (FIXED FOR YOUR OUTPUT NAMES)
# =========================
def load_matrix(method_key, prefix):
    method_dir = RESULTS_ROOT / METHODS[method_key]

    file_path = method_dir / f"{prefix}_matrix.npy"

    if not file_path.exists():
        return None

    return np.load(file_path)

def load_bwt(method_key):
    method_dir = RESULTS_ROOT / METHODS[method_key]
    file_path = method_dir / "bwt_scores.npy"

    if not file_path.exists():
        return None

    return np.load(file_path)

# Load selected method
acc_matrix = load_matrix(method_name, "acc")
auc_matrix = load_matrix(method_name, "auc")
bwt_scores = load_bwt(method_name)

matrix = acc_matrix if metric == "Accuracy" else auc_matrix

# =========================
# MAIN UI
# =========================
st.markdown('<div class="big-title">📊 Continual Anomaly Detection Benchmark</div>',
            unsafe_allow_html=True)

if matrix is None:
    st.error("❌ No result matrices found for this method.")
    st.stop()

# Compute key stats for SELECTED method
taskwise = np.diag(matrix)
final_avg_acc = float(np.mean(taskwise))
forgetting = taskwise[0] - taskwise[-1]

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📈 Curves", "🔥 Heatmap", "📊 Summary & Bars"])

# =========================
# TAB 1 — TASK-WISE CURVE
# =========================
with tab1:
    st.subheader(f"Task-wise {metric} ({method_name})")

    fig, ax = plt.subplots()
    ax.plot(range(1, len(taskwise) + 1), taskwise, marker="o")
    ax.set_xlabel("Task Number")
    ax.set_ylabel(metric)
    ax.set_title(f"{method_name} — Task-wise {metric}")
    ax.grid(True)
    st.pyplot(fig)

# =========================
# TAB 2 — HEATMAP
# =========================
with tab2:
    st.subheader(f"{metric} Heatmap ({method_name})")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xlabel("Evaluated Task")
    ax.set_ylabel("Trained Task")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

# =========================
# TAB 3 — SUMMARY + BARS
# =========================
with tab3:
    st.subheader("📌 Summary (Selected Method)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Final Average Accuracy", f"{final_avg_acc:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Forgetting (T1 → Last)", f"{forgetting:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if bwt_scores is not None:
            st.metric("Overall BWT", f"{np.mean(bwt_scores):.2f}")
        else:
            st.metric("Overall BWT", "Not Found")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ======================================================
    # COMPUTE METRICS FOR ALL METHODS (COMPARISON)
    # ======================================================
    avg_acc_all = {}
    bwt_all = {}

    for m in METHODS.keys():
        acc_mat = load_matrix(m, "acc")
        bwt = load_bwt(m)

        if acc_mat is not None:
            avg_acc_all[m] = float(np.mean(np.diag(acc_mat)))
        else:
            avg_acc_all[m] = np.nan

        if bwt is not None:
            bwt_all[m] = float(np.mean(bwt))
        else:
            bwt_all[m] = np.nan

    methods = list(avg_acc_all.keys())
    avg_vals = [avg_acc_all[m] for m in methods]
    bwt_vals = [bwt_all[m] for m in methods]

    # =========================
    # BAR CHART 1 — AVG ACC (ALL METHODS)
    # =========================
    st.subheader("📊 Final Average Accuracy — All Methods")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(methods, avg_vals)
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Final Average Accuracy (All Methods)")
    ax.set_xticklabels(methods, rotation=20)
    ax.grid(axis="y")
    st.pyplot(fig)

    # =========================
    # BAR CHART 2 — BWT (ALL METHODS)
    # =========================
    st.subheader("📊 Backward Transfer (BWT) — All Methods")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(methods, bwt_vals)
    ax.set_ylabel("Backward Transfer (BWT)")
    ax.set_title("Backward Transfer (BWT) (All Methods)")
    ax.set_xticklabels(methods, rotation=20)
    ax.grid(axis="y")
    st.pyplot(fig)

    # =========================
    # NEW: VERTICAL TASK-WISE BWT BAR CHART (WHAT YOUR DR WANTS)
    # =========================
    if bwt_scores is not None:
        st.subheader(f"📊 Task-wise BWT (Vertical Bar) — {method_name}")

        tasks = np.arange(1, len(bwt_scores) + 1)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(tasks, bwt_scores)
        ax.set_xlabel("Task Number")
        ax.set_ylabel("Δ Accuracy (BWT)")
        ax.set_title(f"Task-wise BWT — {method_name}")
        ax.axhline(0, linestyle="--")
        ax.grid(axis="y")
        st.pyplot(fig)

    # =========================
    # TABLE
    # =========================
    st.markdown("### 📋 Numeric Comparison Table")

    df = pd.DataFrame({
        "Method": methods,
        "Final Avg Accuracy": avg_vals,
        "BWT": bwt_vals
    })

    st.dataframe(df)
