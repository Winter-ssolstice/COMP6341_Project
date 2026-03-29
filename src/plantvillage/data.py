from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(slots=True)
class DataConfig:
    data_dir: str = "Input/color"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42
    mixup_alpha: float = 0.4
    pin_memory: bool = True
    persistent_workers: bool = True


class MixUpCollator:
    """Applies MixUp after batching so the same dataloader works for any model."""

    def __init__(self, num_classes: int, alpha: float = 0.4, enabled: bool = True) -> None:
        self.num_classes = num_classes
        self.alpha = alpha
        self.enabled = enabled and alpha > 0

    def __call__(self, batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        images, targets = torch.utils.data.default_collate(batch)
        if not self.enabled:
            return images, targets

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        perm = torch.randperm(images.size(0))
        mixed_images = lam * images + (1.0 - lam) * images[perm]

        targets_a = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
        targets_b = targets_a[perm]
        mixed_targets = lam * targets_a + (1.0 - lam) * targets_b
        return mixed_images, mixed_targets


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    torch.manual_seed(worker_seed + worker_id)


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    resize_size = int(image_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _split_indices(dataset_size: int, train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[int]]:
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1 so test split is non-empty")

    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    if dataset_size >= 3:
        if train_size == 0:
            train_size = 1
        if val_size == 0:
            val_size = 1
        test_size = dataset_size - train_size - val_size
        if test_size <= 0:
            if train_size >= val_size and train_size > 1:
                train_size -= 1
            elif val_size > 1:
                val_size -= 1
            test_size = dataset_size - train_size - val_size
    if test_size <= 0 or train_size <= 0 or val_size <= 0:
        raise ValueError("Computed empty split; increase dataset size or adjust ratios")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dataset_size, generator=generator).tolist()
    return {
        "train": perm[:train_size],
        "val": perm[train_size : train_size + val_size],
        "test": perm[train_size + val_size :],
    }


def _build_subset(root: str | Path, transform: transforms.Compose, indices: list[int]) -> Subset:
    dataset = datasets.ImageFolder(root=root, transform=transform)
    return Subset(dataset, indices)


def create_dataloaders(
    config: DataConfig,
    split_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    base_dataset = datasets.ImageFolder(root=data_dir)
    available_indices = list(range(len(base_dataset)))

    split_positions = _split_indices(len(available_indices), config.train_ratio, config.val_ratio, config.seed)
    split_indices = {
        split_name: [available_indices[pos] for pos in positions]
        for split_name, positions in split_positions.items()
    }

    train_dataset = _build_subset(data_dir, build_train_transform(config.image_size), split_indices["train"])
    val_dataset = _build_subset(data_dir, build_eval_transform(config.image_size), split_indices["val"])
    test_dataset = _build_subset(data_dir, build_eval_transform(config.image_size), split_indices["test"])

    num_classes = len(base_dataset.classes)
    persistent_workers = config.persistent_workers and config.num_workers > 0
    train_collate_fn = MixUpCollator(num_classes=num_classes, alpha=config.mixup_alpha)
    train_generator = torch.Generator().manual_seed(config.seed)
    eval_generator = torch.Generator().manual_seed(config.seed + 1)
    pin_memory = config.pin_memory and torch.backends.mps.is_available() is False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=train_collate_fn,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        generator=eval_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        generator=eval_generator,
    )

    if split_manifest_path is not None:
        manifest_path = Path(split_manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "full_dataset_size": len(base_dataset),
            "active_dataset_size": len(available_indices),
            "dataset_size": len(base_dataset),
            "classes": base_dataset.classes,
            "class_to_idx": base_dataset.class_to_idx,
            "split_sizes": {name: len(indices) for name, indices in split_indices.items()},
            "split_indices": split_indices,
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    metadata = {
        "class_names": deepcopy(base_dataset.classes),
        "class_to_idx": deepcopy(base_dataset.class_to_idx),
        "num_classes": num_classes,
        "dataset_size": len(available_indices),
        "full_dataset_size": len(base_dataset),
        "split_sizes": {name: len(indices) for name, indices in split_indices.items()},
        "pin_memory": pin_memory,
    }

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "metadata": metadata,
    }
