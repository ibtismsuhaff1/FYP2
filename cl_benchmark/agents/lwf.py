# cl_benchmark/agents/lwf.py
import torch
import torch.nn.functional as F
import copy


class LwFAgent:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.old_model = None
        self.temperature = cfg.get("LWF_TAU", 2.0)
        self.alpha = cfg.get("LWF_ALPHA", 1.0)

    def before_task(self):
        if self.old_model is not None:
            self.old_model.eval()

    def after_task(self):
        self.old_model = copy.deepcopy(self.model)
        for p in self.old_model.parameters():
            p.requires_grad = False

    def distillation_loss(self, inputs, outputs):
        if self.old_model is None:
            return 0.0
        with torch.no_grad():
            old_logits = self.old_model(inputs)
        T = self.temperature
        p_old = F.log_softmax(old_logits / T, dim=1)
        p_new = F.log_softmax(outputs / T, dim=1)
        # KL divergence style
        kd = F.kl_div(p_new, p_old, reduction="batchmean") * (T * T)
        return self.alpha * kd
