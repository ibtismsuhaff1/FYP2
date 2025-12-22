# cl_benchmark/agents/gpm.py

import torch
import numpy as np


class GPMAgent:
    """
    Full Gradient Projection Memory (GPM) implementation.

    Reference:
    Saha et al., "Gradient Projection Memory for Continual Learning", ICLR 2021

    This implementation:
    - Collects gradients after each task
    - Computes an orthogonal basis using SVD
    - Projects future gradients to prevent catastrophic forgetting
    """

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

        self.device = next(model.parameters()).device
        self.threshold = cfg.get("gpm_threshold", 0.97)

        self.task_id = 0
        self.basis = {}   # layer_name -> orthogonal basis (torch.Tensor)

    # ---------------------------------------------------
    # 1. Called BEFORE training each task
    # ---------------------------------------------------
    def before_task(self):
        pass

    # ---------------------------------------------------
    # 2. Called AFTER finishing training a task
    # ---------------------------------------------------
    def after_task(self, gradient_list):
        """
        gradient_list: list of gradients collected during training
        """
        print(f"[GPM] Extracting subspace for task {self.task_id}")

        for name, grad_mat in gradient_list.items():
            grad_mat = grad_mat.to(self.device)

            # Flatten gradients: [num_samples, num_params]
            grad_mat = grad_mat.view(grad_mat.size(0), -1)

            # Compute SVD
            U, S, Vh = torch.linalg.svd(grad_mat, full_matrices=False)

            # Energy threshold
            energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
            r = torch.sum(energy < self.threshold).item() + 1

            basis_new = U[:, :r]

            if name in self.basis:
                # Merge with existing basis
                B = torch.cat([self.basis[name], basis_new], dim=1)
                U2, _, _ = torch.linalg.svd(B, full_matrices=False)
                self.basis[name] = U2[:, :r]
            else:
                self.basis[name] = basis_new

        self.task_id += 1

    # ---------------------------------------------------
    # 3. Called DURING backward pass
    # ---------------------------------------------------
    def project_gradients(self):
        """
        Projects gradients to be orthogonal to stored subspaces
        """
        if not self.basis:
            return

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if name not in self.basis:
                continue

            grad = param.grad.view(-1, 1)
            B = self.basis[name].to(grad.device)

            # Projection: g = g - BB^T g
            proj = B @ (B.t() @ grad)
            param.grad.copy_((grad - proj).view_as(param.grad))
