# cl_benchmark/agents/finetune.py
class FinetuneAgent:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def before_task(self, *a, **k):
        pass

    def after_task(self, *a, **k):
        pass

    def modify_loss(self, loss, *a, **k):
        return loss

    def modify_gradients(self, *a, **k):
        return
