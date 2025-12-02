import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# 1) ViT Backbone (TorchVision)
# ------------------------------------------------------------
def _load_vit_b_16(pretrained: bool = True):
    """Load Vision Transformer (ViT-B/16) safely."""
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    if pretrained:
        try:
            return vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        except Exception:
            return vit_b_16(weights=None)
    return vit_b_16(weights=None)


# ------------------------------------------------------------
# 2) Gaussian Memory (DNE)
# ------------------------------------------------------------
class DNEMemory(nn.Module):
    """Simple Gaussian model (mean + precision) for normal embeddings."""

    def __init__(
        self,
        feat_dim: int,
        eps: float = 1e-4,
        shrink: float = 1e-2,
        device: str = "cpu",
    ):
        super().__init__()
        self.register_buffer("mean", torch.zeros(feat_dim))
        self.register_buffer("prec", torch.eye(feat_dim))
        self.eps = eps
        self.shrink = shrink
        self.device = device
        self.fitted = False

    @torch.no_grad()
    def fit(self, feats: torch.Tensor):
        """Fit Gaussian to normal embeddings."""
        if feats.numel() == 0:
            self.fitted = False
            return
        mu = feats.mean(dim=0)
        xc = feats - mu
        cov = (xc.T @ xc) / max(1, feats.size(0) - 1)
        cov = cov + self.shrink * torch.eye(cov.size(0), device=cov.device)
        try:
            prec = torch.linalg.inv(cov)
        except RuntimeError:
            jitter = self.eps * torch.eye(cov.size(0), device=cov.device)
            prec = torch.linalg.inv(cov + jitter)
        self.mean.copy_(mu)
        self.prec.copy_(prec)
        self.fitted = True

    @torch.no_grad()
    def score(self, feats: torch.Tensor):
        """Mahalanobis distance (higher = more anomalous)."""
        if not self.fitted:
            return feats.norm(dim=1)
        xc = feats - self.mean
        temp = xc @ self.prec
        dist2 = (temp * xc).sum(dim=1).clamp_min(0)
        return torch.sqrt(dist2 + self.eps)


# ------------------------------------------------------------
# 3) ViT + DNE with Per-Task Memories
# ------------------------------------------------------------
class ViT_DNE(nn.Module):
    """ViT-B/16 backbone + per-task Gaussian memories (DNE)."""

    def __init__(
        self,
        device: str = "cpu",
        freeze_backbone: bool = True,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = _load_vit_b_16(pretrained=pretrained_backbone)
        self.feat_dim = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.memories = []  # one DNEMemory per task
        self.device = device
        self.to(device)

    def forward_features(self, x):
        feats = self.backbone(x)
        return F.normalize(feats, p=2, dim=1)

    # -----------------------------
    # Fit one Gaussian per task
    # -----------------------------
    @torch.no_grad()
    def fit_task(self, dataloader, max_batches: int = None):
        self.eval()
        normal_feats = []

        for b_idx, (imgs, labels) in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break
            mask = labels == 0
            if mask.sum().item() == 0:
                continue
            imgs = imgs[mask].to(self.device, non_blocking=True)
            with (
                torch.amp.autocast("cuda", enabled=False)
                if torch.cuda.is_available()
                else torch.autocast("cpu", enabled=False)
            ):
                feats = self.forward_features(imgs)
            normal_feats.append(feats.cpu())

        if len(normal_feats) == 0:
            mem = DNEMemory(self.feat_dim, device=self.device)
            mem.fitted = False
            self.memories.append(mem)
            return

        feats_all = torch.cat(normal_feats, dim=0).to(self.device)
        mem = DNEMemory(self.feat_dim, device=self.device)
        mem.fit(feats_all)
        self.memories.append(mem)

    # -----------------------------
    # Compute anomaly scores
    # -----------------------------
    @torch.no_grad()
    def anomaly_scores(self, dataloader, memory_idx=None, max_batches: int = None):
        self.eval()
        all_scores = []
        for b_idx, (imgs, _) in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break
            imgs = imgs.to(self.device, non_blocking=True)
            with (
                torch.amp.autocast("cuda", enabled=False)
                if torch.cuda.is_available()
                else torch.autocast("cpu", enabled=False)
            ):
                feats = self.forward_features(imgs)

            if memory_idx is not None and memory_idx < len(self.memories):
                mem = self.memories[memory_idx].to(
                    self.device
                )  # ✅ ensure correct device
                scores = mem.score(feats.to(self.device))
            else:
                if len(self.memories) == 0:
                    cores = feats.norm(dim=1)
                else:
                    dists = []
                    for mem in self.memories:
                        mem = mem.to(self.device)  # ✅ ensure correct device
                        dists.append(mem.score(feats.to(self.device)).unsqueeze(1))
                    scores = torch.cat(dists, dim=1).min(dim=1).values
            all_scores.append(scores.cpu())

        return torch.cat(all_scores, dim=0) if len(all_scores) > 0 else torch.empty(0)
