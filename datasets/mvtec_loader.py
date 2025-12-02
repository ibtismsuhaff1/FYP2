import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MVTecDataset(Dataset):
    def __init__(self, root, img_size=224):
        """
        root = path/to/data/mvtec/<category>/<train_or_test>/<class_name>
        """
        self.img_paths = []
        self.labels = []

        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        self.transform = transforms.Compose(transform_list)

        # Walk through all PNG/JPG files
        for subdir, _, files in os.walk(root):
            for f in files:
                if f.endswith((".png", ".jpg", ".jpeg")):
                    self.img_paths.append(os.path.join(subdir, f))

                    # label = 0 for good, 1 for ALL anomaly types
                    label = 0 if "good" in subdir.replace("\\", "/") else 1
                    self.labels.append(label)

        if len(self.img_paths) == 0:
            raise ValueError(f"No images found in {root}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


def load_mvtec_all_categories(root="data/mvtec", img_size=224):
    """
    Automatically detects all 15 MVTec categories.
    Creates separate train/test datasets for each category.
    """
    categories = sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    )

    if len(categories) == 0:
        raise ValueError(f"No MVTec categories found inside: {root}")

    train_sets = {}
    test_sets = {}

    for category in categories:
        train_root = os.path.join(root, category, "train")
        test_root = os.path.join(root, category, "test")

        if not os.path.isdir(train_root):
            raise ValueError(f"MVTec train folder not found: {train_root}")

        if not os.path.isdir(test_root):
            raise ValueError(f"MVTec test folder not found: {test_root}")

        train_sets[category] = MVTecDataset(train_root, img_size=img_size)
        test_sets[category] = MVTecDataset(test_root, img_size=img_size)

    return categories, train_sets, test_sets
