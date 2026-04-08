from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torchvision import models

from ..common.data import DataConfig, create_dataloaders
from ..common.training import TrainingConfig, evaluate_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reusable PlantVillage baseline.")
    parser.add_argument("--data-dir", default="Input/color", help="Path to ImageFolder dataset root.")
    parser.add_argument(
        "--output-dir",
        default="outputs/part1/baseline_resnet18",
        help="Directory for logs and checkpoints.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--max-train-steps", type=int, default=None, help="Optional cap on train batches per epoch.")
    parser.add_argument("--max-val-steps", type=int, default=None, help="Optional cap on validation batches per epoch.")
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=["resnet18", "resnet34", "efficientnet_b0"],
        help="Backbone for the baseline runner.",
    )
    return parser.parse_args()


def build_model(model_name: str, num_classes: int) -> torch.nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)

    data_config = DataConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        mixup_alpha=args.mixup_alpha,
    )
    split_manifest_path = output_dir / "split_manifest.json"
    loaders = create_dataloaders(data_config, split_manifest_path=split_manifest_path)
    metadata = loaders["metadata"]

    model = build_model(args.model, metadata["num_classes"])
    training_config = TrainingConfig(
        output_dir=str(output_dir),
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
    )

    history = train_model(
        model=model,
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        config=training_config,
        extra_config={
            "data": {
                **asdict(data_config),
                **metadata,
            },
            "model_name": args.model,
        },
    )
    test_metrics = evaluate_model(
        model,
        loaders["test_loader"],
        training_config.device,
    )
    Path(output_dir, "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    print("Final validation:", history[-1])
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
