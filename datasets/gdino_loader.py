from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
import torch


class GDinoDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


class GDinoDataloader:
    def __init__(self, image_root, batch_size, num_workers, is_train):
        self.batch_size = batch_size
        self.is_train = is_train
        self.num_workers = num_workers

        if self.is_train:
            self.dataset = GDinoDataset(
                root=image_root,
                transform=None  # 정규화 없이 원본 유지
            )
            self.class_names = self.dataset.classes
            self.class_id_to_name = {i: name for i, name in enumerate(self.class_names)}
        else:
            self.image_paths = sorted([
                os.path.join(image_root, f)
                for f in os.listdir(image_root)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ])
            self.dataset = self.image_paths

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.is_train,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def _collate_fn(self, batch):
        batch_output = []
        if self.is_train:
            for image_pil, label, path in batch:
                image = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).contiguous()
                batch_output.append({
                    "image": image,           # torch.uint8
                    "label": label,
                    "file_name": path
                })
        else:
            for path in batch:
                image = Image.open(path).convert("RGB")
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()
                batch_output.append({
                    "image": image,
                    "file_name": path
                })
        return batch_output


def build_dataloader(image_root, batch_size, num_workers, is_train):
    return GDinoDataloader(image_root, batch_size, num_workers, is_train)