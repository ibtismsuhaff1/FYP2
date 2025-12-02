# cl_benchmark/agents/sgp.py
import torch
import numpy as np


class SGPAgent:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.masks = {}  # param_name -> important directions (small matrix)
        self.alpha = cfg.get("SGP_ALPHA", 0.9)

    def estimate_importance(self, dataloader, device):
        # simple importance via magnitude of activations for classifier weights
        self.model.eval()
        total = None
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device)
                feats = self.model.features(data)
                feats = feats.view(feats.size(0), -1)
                mag = (feats**2).sum(dim=0).cpu().numpy()
                if total is None:
                    total = mag
                else:
                    total += mag
                break
        if total is None:
            return
        # select top dims
        thr = np.percentile(total, (1.0 - self.alpha) * 100)
        mask = total >= thr
        self.masks["features"] = mask  # boolean vector

    def project_gradients(self):
        # zero-out gradient components corresponding to important dims
        if "features" not in self.masks:
            return
        mask = self.masks["features"]
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.data.view(-1)
            if g.numel() == len(mask):
                g_np = g.cpu().numpy()
                g_np[mask] = 0.0
                p.grad.data.copy_(
                    torch.from_numpy(g_np).view_as(p.grad.data).to(p.grad.data.device)
                )
