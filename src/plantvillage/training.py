from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn


@dataclass(slots=True)
class TrainingConfig:
    output_dir: str = "outputs/baseline"
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    checkpoint_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log_interval: int = 20
    max_train_steps: int | None = None
    max_val_steps: int | None = None


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def _compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    if targets.ndim == 2:
        targets = targets.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _compute_macro_f1_from_counts(confusion: torch.Tensor) -> float:
    true_positives = confusion.diag()
    predicted_positives = confusion.sum(dim=0)
    actual_positives = confusion.sum(dim=1)

    precision = torch.where(predicted_positives > 0, true_positives / predicted_positives, torch.zeros_like(true_positives))
    recall = torch.where(actual_positives > 0, true_positives / actual_positives, torch.zeros_like(true_positives))
    f1 = torch.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    valid_classes = actual_positives > 0
    if valid_classes.any():
        return f1[valid_classes].mean().item()
    return 0.0


def _setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("plantvillage")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    filename: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        output_dir / filename,
    )


def _save_history(output_dir: Path, history: list[dict[str, float]]) -> None:
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _plot_loss_curves(output_dir: Path, history: list[dict[str, float]]) -> None:
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()


def _run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    logger: logging.Logger,
    log_interval: int,
    max_steps: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    criterion = soft_target_cross_entropy
    confusion: torch.Tensor | None = None

    for batch_idx, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            if targets.ndim == 1:
                loss = torch.nn.functional.cross_entropy(logits, targets)
            else:
                loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        detached_logits = logits.detach()
        detached_targets = targets.detach()
        running_acc += _compute_accuracy(detached_logits, detached_targets) * batch_size
        total_samples += batch_size

        target_indices = detached_targets.argmax(dim=1) if detached_targets.ndim == 2 else detached_targets
        pred_indices = detached_logits.argmax(dim=1)
        num_classes = detached_logits.size(1)
        if confusion is None:
            confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
        flat_indices = target_indices.cpu() * num_classes + pred_indices.cpu()
        confusion += torch.bincount(flat_indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

        if is_train and batch_idx % log_interval == 0:
            logger.info(
                "batch=%s/%s loss=%.4f acc=%.4f",
                batch_idx,
                len(dataloader),
                running_loss / total_samples,
                running_acc / total_samples,
            )

        if max_steps is not None and batch_idx >= max_steps:
            logger.info("stopping early after %s step(s)", max_steps)
            break

    return {
        "loss": running_loss / total_samples,
        "accuracy": running_acc / total_samples,
        "macro_f1": _compute_macro_f1_from_counts(confusion) if confusion is not None else 0.0,
    }


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: TrainingConfig,
    extra_config: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(output_dir)

    device = torch.device(config.device)
    model = model.to(device)
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise ValueError("Model has no trainable parameters. Check the chosen training strategy.")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    run_config = {"training": asdict(config)}
    if extra_config:
        run_config.update(extra_config)
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        logger.info("epoch %s/%s started", epoch, config.epochs)

        train_metrics = _run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            logger=logger,
            log_interval=config.log_interval,
            max_steps=config.max_train_steps,
        )
        val_metrics = _run_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            optimizer=None,
            logger=logger,
            log_interval=config.log_interval,
            max_steps=config.max_val_steps,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(epoch_metrics)
        _save_history(output_dir, history)
        _plot_loss_curves(output_dir, history)

        logger.info(
            "epoch=%s train_loss=%.4f train_acc=%.4f train_f1=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f duration=%.2fs",
            epoch,
            epoch_metrics["train_loss"],
            epoch_metrics["train_accuracy"],
            epoch_metrics["train_macro_f1"],
            epoch_metrics["val_loss"],
            epoch_metrics["val_accuracy"],
            epoch_metrics["val_macro_f1"],
            epoch_metrics["epoch_seconds"],
        )

        if epoch % config.checkpoint_every == 0:
            _save_checkpoint(output_dir, model, optimizer, epoch, epoch_metrics, f"checkpoint_epoch_{epoch}.pt")
        _save_checkpoint(output_dir, model, optimizer, epoch, epoch_metrics, "last.pt")

        if epoch_metrics["val_loss"] < best_val_loss:
            best_val_loss = epoch_metrics["val_loss"]
            _save_checkpoint(output_dir, model, optimizer, epoch, epoch_metrics, "best.pt")

    return history


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str | torch.device,
) -> dict[str, float]:
    model.eval()
    model = model.to(device)
    metrics = _run_epoch(
        model=model,
        dataloader=dataloader,
        device=torch.device(device),
        optimizer=None,
        logger=logging.getLogger("plantvillage"),
        log_interval=10**9,
    )
    return {
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
    }
