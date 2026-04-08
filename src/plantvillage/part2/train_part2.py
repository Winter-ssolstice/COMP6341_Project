from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..common.data import DataConfig, create_dataloaders
from ..common.models import ModelConfig, build_model, resolve_pretrained_default
from ..common.results import write_experiment_summary
from ..common.training import TrainingConfig, evaluate_model, train_model

DATASET_VERSION_TO_DIR = {
    "color": "Input/color",
    "grayscale": "Input/grayscale",
    "segmented": "Input/segmented",
    "background_segmented": "Input/segmented",
    "background-segmented": "Input/segmented",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 2 PlantVillage experiments.")
    parser.add_argument("--dataset-version", choices=sorted(DATASET_VERSION_TO_DIR), default="color")
    parser.add_argument("--data-dir", default=None, help="Optional explicit dataset directory override.")
    parser.add_argument("--output-root", default="outputs/part2")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument(
        "--model",
        choices=["resnet50", "efficientnet_b3", "vit_small"],
        required=True,
    )
    parser.add_argument(
        "--strategy",
        choices=["from_scratch", "linear_probing", "full_finetune"],
        required=True,
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Force pretrained weights on when the selected model supports them.",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Force pretrained weights off.",
    )
    parser.set_defaults(pretrained=None)
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    canonical_dataset_version = args.dataset_version.replace("-", "_")
    run_name = f"{canonical_dataset_version}_{args.model}_{args.strategy}"
    return Path(args.output_root) / run_name


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    canonical_dataset_version = args.dataset_version.replace("-", "_")
    data_dir = args.data_dir or DATASET_VERSION_TO_DIR[args.dataset_version]
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = DataConfig(
        data_dir=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        mixup_alpha=args.mixup_alpha,
    )
    loaders = create_dataloaders(data_config, split_manifest_path=output_dir / "split_manifest.json")
    metadata = loaders["metadata"]

    pretrained = resolve_pretrained_default(args.model, args.strategy) if args.pretrained is None else args.pretrained
    model = build_model(
        ModelConfig(
            model_name=args.model,
            num_classes=metadata["num_classes"],
            strategy=args.strategy,
            pretrained=pretrained,
        )
    )

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
            "data": {**asdict(data_config), **metadata},
            "dataset_version": canonical_dataset_version,
            "model_name": args.model,
            "strategy": args.strategy,
            "pretrained": pretrained,
        },
    )
    test_metrics = evaluate_model(
        model=model,
        dataloader=loaders["test_loader"],
        device=training_config.device,
    )
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    summary_row = {
        "dataset_version": canonical_dataset_version,
        "model_name": args.model,
        "strategy": args.strategy,
        "pretrained": pretrained,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_loss": history[-1]["val_loss"],
        "val_top1_accuracy": history[-1]["val_accuracy"],
        "val_macro_f1": history[-1]["val_macro_f1"],
        "test_loss": test_metrics["test_loss"],
        "test_top1_accuracy": test_metrics["test_accuracy"],
        "test_macro_f1": test_metrics["test_macro_f1"],
        "output_dir": str(output_dir),
    }
    write_experiment_summary(Path(args.output_root) / "experiment_runs.csv", summary_row)

    print("Final validation:", history[-1])
    print("Test metrics:", test_metrics)
    print("Summary row:", summary_row)


if __name__ == "__main__":
    main()
