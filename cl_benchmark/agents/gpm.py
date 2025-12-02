# cl_benchmark/agents/gpm.py
import torch
import numpy as np


class GPMAgent:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.memory = {}  # layer_name -> numpy array of principal vectors
        self.proj_threshold = cfg.get("GPM_THRESHOLD", 0.9)

    def register_activations(self, dataloader, device):
        # Collect features per module (we'll collect final feature vectors only for simplicity)
        self.model.eval()
        feats = []
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device)
                x = self.model.features(data)
                x = x.view(x.size(0), -1)
                feats.append(x.cpu().numpy())
                if len(feats) >= 10:
                    break
        if not feats:
            return
        X = np.concatenate(feats, axis=0)  # (N, D)
        # center
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        # keep top-k explaining threshold
        var = (S**2) / (S**2).sum()
        cum = np.cumsum(var)
        k = np.searchsorted(cum, self.proj_threshold) + 1
        principal = Vt[:k, :].T  # D x k
        self.memory["features"] = principal  # store principal directions

    def project_gradients(self):
        # Project gradients of classifier parameters to be orthogonal to memory subspace
        if "features" not in self.memory:
            return
        P = self.memory["features"]  # D x k
        # For each parameter in classifier, we flatten and project grad if shape fits
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.data.view(-1)
            # If gradient length equals feature dim, project
            if g.numel() == P.shape[0]:
                g_np = g.cpu().numpy()
                # project onto subspace spanned by P and subtract
                coeffs = P.T.dot(g_np)
                recon = P.dot(coeffs)
                new = g_np - recon
                p.grad.data.copy_(
                    torch.from_numpy(new).view_as(p.grad.data).to(p.grad.data.device)
                )
