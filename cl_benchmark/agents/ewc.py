# cl_benchmark/agents/ewc.py
import torch


class EWCAgent:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.fisher = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.lambda_ = cfg.get("EWC_LAMBDA", 1000.0)

    def estimate_fisher(self, dataloader, device):
        self.model.eval()
        for n in self.fisher:
            self.fisher[n].zero_()
        import torch.nn.functional as F

        count = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            self.model.zero_grad()
            outputs = self.model(data)
            loss = F.cross_entropy(outputs, target)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.data.clone().detach() ** 2
            count += 1
            if count >= 20:  # limit passes to save time
                break
        for n in self.fisher:
            self.fisher[n] = self.fisher[n] / max(1, count)

    def penalty(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                loss = loss + (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return 0.5 * self.lambda_ * loss

    def after_task(self, dataloader, device):
        # update stored parameters and fisher
        self.params = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self.estimate_fisher(dataloader, device)
