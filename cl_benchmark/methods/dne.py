# cl_benchmark/methods/dne.py
import os
import numpy as np
import torch
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score
import joblib


class DNEDetector:
    """
    Simple DNE-style detector:
    - Fit: given a set of 'good' images, extract patch features, compute mean + diag-cov for each patch location.
      We store per-task memory: (patch_means [N_patches, D], patch_precisions [N_patches, D]) or diag variances.
    - Score-image: compute per-patch Mahalanobis distance to stored mean; aggregate (max or percentile) -> image score.
    """

    def __init__(self, feature_extractor, device="cpu", reduce_dim=None):
        """
        feature_extractor: callable tensor->(B, N_patches, D)
        reduce_dim: optional int to reduce features with PCA (not implemented here). Keep None for simplicity.
        """
        self.fe = feature_extractor
        self.device = device
        self.reduce_dim = reduce_dim
        self.memory = None
        self.n_patches = None
        self.feat_dim = None

    def fit(self, dataloader):
        """
        dataloader yields images (only GOOD images).
        Build memory statistics (mean and diag variance) per patch location.
        """
        self.fe.eval()
        feats_list = []
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                patches = self.fe(imgs)  # (B, N, D)
                feats_list.append(patches.cpu().numpy())

        all_feats = np.concatenate(feats_list, axis=0)  # (N_images, N_patches, D)
        N_images, N_patches, D = all_feats.shape
        self.n_patches = N_patches
        self.feat_dim = D

        # compute mean and diagonal covariance per patch location
        patch_means = np.mean(all_feats, axis=0)  # (N_patches, D)
        patch_vars = np.var(all_feats, axis=0) + 1e-6  # (N_patches, D) (add eps)

        # store inverse-variance (precision) for Mahalanobis (diagonal)
        patch_precisions = 1.0 / patch_vars

        self.memory = {
            "patch_means": patch_means.astype(np.float32),
            "patch_precisions": patch_precisions.astype(np.float32),
        }

    def score_batch(self, imgs):
        """
        imgs: torch tensor (B, C, H, W)
        returns: image_scores (B,), maps (B, N_patches)
        """
        self.fe.eval()
        with torch.no_grad():
            patches = self.fe(imgs.to(self.device)).cpu().numpy()  # (B, N, D)

        B, N, D = patches.shape
        means = self.memory["patch_means"][None, :, :]  # (1, N, D)
        prec = self.memory["patch_precisions"][None, :, :]  # (1, N, D)

        diff = patches - means  # (B, N, D)
        m_dist_sq = np.sum((diff**2) * prec, axis=2)  # (B, N)  (diagonal Mahalanobis)

        # image score: we use max patch distance (common in patch-based AD) or percentile
        img_scores = m_dist_sq.max(axis=1)
        return img_scores, m_dist_sq

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.memory, path)

    def load(self, path):
        self.memory = joblib.load(path)

    @staticmethod
    def compute_image_auc(y_true, scores):
        # y_true: 1 if anomaly (defect), 0 if good
        return roc_auc_score(y_true, scores)
