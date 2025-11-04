import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class MVTecDataset(Dataset):
    """
    Loads images for one MVTec AD category.
    Each category = one continual learning task.
    """
    def __init__(self, root, category, split='train', transform=None):
        self.root = root
        self.category = category
        self.split = split
        self.transform = transform

        self.images = []
        self.labels = []  # 0 = normal, 1 = anomaly

        category_path = os.path.join(root, category, split)
        if not os.path.exists(category_path):
            raise FileNotFoundError(f"Category path not found: {category_path}")

        if split == 'train':
            # Training: only normal ("good") samples
            normal_path = os.path.join(category_path, 'good')
            for img_name in os.listdir(normal_path):
                img_path = os.path.join(normal_path, img_name)
                self.images.append(img_path)
                self.labels.append(0)
        elif split == 'test':
            # Testing: normal + anomaly samples
            for defect_type in os.listdir(category_path):
                defect_path = os.path.join(category_path, defect_type)
                for img_name in os.listdir(defect_path):
                    img_path = os.path.join(defect_path, img_name)
                    label = 0 if defect_type == 'good' else 1
                    self.images.append(img_path)
                    self.labels.append(label)
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_mvtec_tasks(data_dir, image_size=224, batch_size=16, seed=42):
    """
    Creates a continual learning task list — one per category.
    Returns train and test DataLoaders for each category.
    """
    random.seed(seed)
    categories = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    print(f"[INFO] Found {len(categories)} categories in MVTec dataset.")
    print("Categories:", categories)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_tasks, test_tasks = [], []

    for category in categories:
        print(f"[TASK] Creating loaders for category: {category}")

        train_dataset = MVTecDataset(data_dir, category, split='train', transform=transform)
        test_dataset = MVTecDataset(data_dir, category, split='test', transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_tasks.append(train_loader)
        test_tasks.append(test_loader)

    print(f"[INFO] Created {len(train_tasks)} continual learning tasks.")
    return train_tasks, test_tasks
