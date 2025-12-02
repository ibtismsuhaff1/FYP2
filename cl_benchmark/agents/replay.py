import torch
import numpy as np
import os

class ReplayAgent:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.buffer = {}  # label_id -> tensor samples

    def add_to_buffer(self, dataset):
        """Store replay samples using *remapped* labels, not original dataset labels."""
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        images, labels = next(iter(loader))

        unique_labels = torch.unique(labels)
        cap = self.cfg.get("REPLAY_BUFFER_SIZE_PER_CLASS", 20)

        for lbl in unique_labels:
            mask = labels == lbl
            samples = images[mask]

            key = int(lbl.item())

            if key in self.buffer:
                merged = torch.cat([self.buffer[key], samples], dim=0)
                self.buffer[key] = merged[:cap].clone().detach()
            else:
                self.buffer[key] = samples[:cap].clone().detach()

    def save_replay_buffer(self):
        """Save first 10 samples per class as images."""
        outdir = "results/replay_buffer"
        os.makedirs(outdir, exist_ok=True)

        for lbl, imgs in self.buffer.items():
            cls_dir = os.path.join(outdir, f"class_{lbl}")
            os.makedirs(cls_dir, exist_ok=True)

            for i, img in enumerate(imgs[:10]):
                arr = img.permute(1, 2, 0).cpu().numpy()
                arr = (arr * 255).astype("uint8")
                cv2.imwrite(
                    os.path.join(cls_dir, f"sample_{i}.png"),
                    cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                )

    def get_replay_batch(self, n):
        if not self.buffer or n == 0:
            return None, None

        imgs, labs = [], []
        for lbl, samples in self.buffer.items():
            imgs.append(samples)
            labs += [lbl] * len(samples)

        imgs = torch.cat(imgs, dim=0)
        labs = torch.tensor(labs, dtype=torch.long)

        idx = np.random.choice(len(imgs), min(n, len(imgs)), replace=False)

        return imgs[idx], labs[idx]

    def before_task(self, *a, **k):
        pass

    def after_task(self, train_dataset):
        """Save samples AFTER finishing a task."""
        self.add_to_buffer(train_dataset)
        self.save_replay_buffer()

    def integrate_replay(self, batch_data, batch_labels):
        rimgs, rlabs = self.get_replay_batch(self.cfg.get("REPLAY_BATCH_SIZE", 16))
        if rimgs is None:
            return batch_data, batch_labels

        device = batch_data.device
        return (
            torch.cat([batch_data, rimgs.to(device)], dim=0),
            torch.cat([batch_labels, rlabs.to(device)], dim=0),
        )
import cv2

def save_prediction_viz(img_tensor, pred, score, save_path):
    """Save an annotated prediction image."""
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype("uint8")

    text = f"pred={pred} score={score:.3f}"
    img = cv2.putText(img.copy(), text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255), 2)

    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
