# cl_benchmark/utils/metrics_io.py
import os
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from sklearn.metrics import roc_auc_score, roc_curve, auc


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_task_output(outdir, train_task_id, eval_task_id, scores, labels):
    ensure_dir(outdir)
    fname = f"task_{train_task_id}_eval_{eval_task_id}.npz"
    np.savez_compressed(
        os.path.join(outdir, fname), scores=np.array(scores), labels=np.array(labels)
    )
    return os.path.join(outdir, fname)


def save_accuracy_matrix(matrix, outdir, tag="acc"):
    ensure_dir(outdir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(outdir, f"{tag}_matrix_{ts}.npy"), matrix)
    np.savetxt(
        os.path.join(outdir, f"{tag}_matrix_{ts}.csv"),
        matrix,
        fmt="%.4f",
        delimiter=",",
    )
    return outdir


def plot_heatmap(matrix, outdir, title="Accuracy Matrix"):
    ensure_dir(outdir)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.array(matrix), vmin=0, vmax=100)
    ax.set_title(title)
    ax.set_xlabel("Evaluated task")
    ax.set_ylabel("Trained task")
    plt.colorbar(im, ax=ax)
    path = os.path.join(
        outdir, f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def compute_auc_safe(scores, labels):
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return roc_auc_score(labels, scores)
    except Exception:
        return float("nan")

def save_prediction_viz(img_tensor, pred_label, anomaly_score, outpath):
    """
    Save a single prediction visualization.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    img = img_tensor.cpu().detach()
    save_image(img, outpath.replace(".png", "_img.png"))

    # Write metadata on a black canvas
    canvas = np.zeros((200, 600, 3), dtype=np.uint8)
    text = f"Pred: {pred_label}  Score: {anomaly_score:.4f}"
    cv2.putText(canvas, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    cv2.imwrite(outpath, canvas)