import torch
from collections import defaultdict


class GPMAgent:
    """
    Gradient Projection Memory (GPM)
    Saha et al., ICLR 2021

    Framework-compatible implementation.
    """

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = next(model.parameters()).device

        self.threshold = cfg.get("gpm_threshold", 0.97)
        self.task_id = 0

        self.basis = {}  # layer_name -> orthogonal basis
        self.gradients = defaultdict(list)  # temp storage

    # Called BEFORE each task
    def before_task(self):
        self.gradients.clear()

    # Called DURING backward pass
    def collect_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.gradients[name].append(param.grad.detach().clone())

    # Called AFTER each task
    def after_task(self, train_loader=None, device=None):
        print(f"[GPM] Building subspace for task {self.task_id}")

        for name, grads in self.gradients.items():
            if len(grads) == 0:
                continue

            G = torch.stack(grads).to(self.device)
            G = G.view(G.size(0), -1)

            # SVD
            U, S, _ = torch.linalg.svd(G, full_matrices=False)

            energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
            r = int((energy < self.threshold).sum()) + 1
            basis_new = U[:, :r]

            if name in self.basis:
                B = torch.cat([self.basis[name], basis_new], dim=1)
                U2, _, _ = torch.linalg.svd(B, full_matrices=False)
                self.basis[name] = U2[:, :r]
            else:
                self.basis[name] = basis_new

        self.task_id += 1

    # Gradient projection
    def project_gradients(self):
        if not self.basis:
            return

        for name, param in self.model.named_parameters():
            if param.grad is None or name not in self.basis:
                continue

            g = param.grad.view(-1, 1)
            B = self.basis[name].to(g.device)
            proj = B @ (B.t() @ g)
            param.grad.copy_((g - proj).view_as(param.grad))
