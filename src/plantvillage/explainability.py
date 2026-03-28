from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.plantvillage.data import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform
from src.plantvillage.models import ModelConfig, build_model, get_default_device


@dataclass(slots=True)
class DatasetSample:
    image_path: str
    target: int
    relative_path: str
    canonical_key: str


@dataclass(slots=True)
class PredictionRecord:
    dataset_version: str
    image_path: str
    relative_path: str
    true_idx: int
    true_label: str
    pred_idx: int
    pred_label: str
    confidence: float
    is_correct: bool


@dataclass(slots=True)
class RunContext:
    dataset_version: str
    run_dir: Path
    data_dir: Path
    image_size: int
    batch_size: int
    num_workers: int
    class_names: list[str]
    checkpoint_path: Path
    model: torch.nn.Module
    device: str
    test_dataset: "ManifestTestDataset"
    reference_key_count: int
    matched_sample_count: int
    missing_reference_keys: list[str]


@dataclass(slots=True)
class DatasetAnalysis:
    dataset_version: str
    output_dir: Path
    predictions: list[PredictionRecord]
    metrics: dict[str, float]
    representative_rows: list[dict[str, str]]
    missing_incorrect_classes: list[str]
    hardest_classes: list[dict[str, float | int | str]]
    top_confusions: list[dict[str, int | str]]
    figure_count: int


class ManifestTestDataset(Dataset[tuple[torch.Tensor, int, str, str]]):
    def __init__(self, data_dir: str | Path, image_size: int, samples: list[DatasetSample]) -> None:
        self.data_dir = Path(data_dir)
        self.samples = samples
        self.loader = datasets.ImageFolder(root=self.data_dir).loader
        self.transform = build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int, str, str]:
        sample = self.samples[item]
        image = self.loader(sample.image_path)
        tensor = self.transform(image)
        return tensor, sample.target, sample.image_path, sample.relative_path


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def _canonical_relative_key(relative_path: str) -> str:
    path = Path(relative_path)
    stem = path.stem
    if stem.endswith("_final_masked"):
        stem = stem[: -len("_final_masked")]
    return f"{path.parent.as_posix()}/{stem}"


def _build_samples_from_indices(data_dir: str | Path, indices: list[int]) -> list[DatasetSample]:
    data_dir = Path(data_dir)
    base_dataset = datasets.ImageFolder(root=data_dir)
    samples: list[DatasetSample] = []
    for dataset_idx in indices:
        image_path, target = base_dataset.samples[dataset_idx]
        relative_path = Path(image_path).relative_to(data_dir).as_posix()
        samples.append(
            DatasetSample(
                image_path=str(image_path),
                target=int(target),
                relative_path=relative_path,
                canonical_key=_canonical_relative_key(relative_path),
            )
        )
    return samples


def _build_samples_from_reference_keys(
    data_dir: str | Path,
    reference_keys: list[str],
) -> tuple[list[DatasetSample], list[str]]:
    data_dir = Path(data_dir)
    base_dataset = datasets.ImageFolder(root=data_dir)
    sample_by_key: dict[str, DatasetSample] = {}
    for image_path, target in base_dataset.samples:
        relative_path = Path(image_path).relative_to(data_dir).as_posix()
        canonical_key = _canonical_relative_key(relative_path)
        sample_by_key[canonical_key] = DatasetSample(
            image_path=str(image_path),
            target=int(target),
            relative_path=relative_path,
            canonical_key=canonical_key,
        )

    matched_samples: list[DatasetSample] = []
    missing_keys: list[str] = []
    for key in reference_keys:
        sample = sample_by_key.get(key)
        if sample is None:
            missing_keys.append(key)
            continue
        matched_samples.append(sample)
    return matched_samples, missing_keys


def _compute_macro_f1(confusion: torch.Tensor) -> float:
    true_positives = confusion.diag()
    predicted_positives = confusion.sum(dim=0)
    actual_positives = confusion.sum(dim=1)

    precision = torch.where(
        predicted_positives > 0,
        true_positives / predicted_positives,
        torch.zeros_like(true_positives, dtype=torch.float32),
    )
    recall = torch.where(
        actual_positives > 0,
        true_positives / actual_positives,
        torch.zeros_like(true_positives, dtype=torch.float32),
    )
    f1 = torch.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    valid_classes = actual_positives > 0
    if not valid_classes.any():
        return 0.0
    return f1[valid_classes].mean().item()


def _resize_and_crop(image: Image.Image, image_size: int) -> Image.Image:
    resize_size = int(image_size * 256 / 224)
    transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
        ]
    )
    return transform(image)


def _normalize_cropped_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform(image)


def _format_class_name(name: str) -> str:
    return name.replace("___", " / ").replace("_", " ")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    token_count = tensor.size(1) - 1
    spatial_size = int(math.sqrt(token_count))
    if spatial_size * spatial_size != token_count:
        raise ValueError(f"Unexpected ViT token count for CAM reshape: {token_count}")
    result = tensor[:, 1:, :].reshape(tensor.size(0), spatial_size, spatial_size, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def load_run_context(
    run_dir: str | Path,
    *,
    batch_size: int | None = None,
    num_workers: int = 0,
    device: str | None = None,
    reference_keys: list[str] | None = None,
) -> RunContext:
    run_dir = Path(run_dir)
    run_config = _load_json(run_dir / "run_config.json")
    split_manifest = _load_json(run_dir / "split_manifest.json")

    if run_config.get("model_name") != "vit_small" or run_config.get("strategy") != "full_finetune":
        raise ValueError(f"Part 3 expects vit_small + full_finetune, got {run_config.get('model_name')} + {run_config.get('strategy')}")

    data_config = run_config["data"]
    class_names = data_config["class_names"]
    model = build_model(
        ModelConfig(
            model_name="vit_small",
            num_classes=len(class_names),
            strategy="full_finetune",
            pretrained=bool(run_config.get("pretrained", True)),
        )
    )

    resolved_device = device or get_default_device()
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()

    if reference_keys is None:
        test_samples = _build_samples_from_indices(
            data_dir=data_config["data_dir"],
            indices=list(split_manifest["split_indices"]["test"]),
        )
        reference_key_count = len(test_samples)
        missing_reference_keys: list[str] = []
    else:
        test_samples, missing_reference_keys = _build_samples_from_reference_keys(
            data_dir=data_config["data_dir"],
            reference_keys=reference_keys,
        )
        reference_key_count = len(reference_keys)

    test_dataset = ManifestTestDataset(
        data_dir=data_config["data_dir"],
        image_size=int(data_config["image_size"]),
        samples=test_samples,
    )

    return RunContext(
        dataset_version=str(run_config["dataset_version"]),
        run_dir=run_dir,
        data_dir=Path(data_config["data_dir"]),
        image_size=int(data_config["image_size"]),
        batch_size=int(batch_size or data_config["batch_size"]),
        num_workers=int(num_workers),
        class_names=list(class_names),
        checkpoint_path=checkpoint_path,
        model=model,
        device=resolved_device,
        test_dataset=test_dataset,
        reference_key_count=reference_key_count,
        matched_sample_count=len(test_samples),
        missing_reference_keys=missing_reference_keys,
    )


def run_inference(context: RunContext) -> tuple[list[PredictionRecord], dict[str, float], torch.Tensor]:
    dataloader = DataLoader(
        context.test_dataset,
        batch_size=context.batch_size,
        shuffle=False,
        num_workers=context.num_workers,
    )
    confusion = torch.zeros((len(context.class_names), len(context.class_names)), dtype=torch.long)
    predictions: list[PredictionRecord] = []
    loss_sum = 0.0
    total_samples = 0

    with torch.inference_mode():
        for images, targets, image_paths, relative_paths in dataloader:
            images = images.to(context.device)
            targets = targets.to(context.device)

            logits = context.model(images)
            probs = F.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)
            loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
            total_samples += targets.size(0)

            flat_indices = targets.cpu() * len(context.class_names) + preds.cpu()
            confusion += torch.bincount(
                flat_indices,
                minlength=len(context.class_names) * len(context.class_names),
            ).reshape(len(context.class_names), len(context.class_names))

            for idx in range(targets.size(0)):
                true_idx = int(targets[idx].item())
                pred_idx = int(preds[idx].item())
                predictions.append(
                    PredictionRecord(
                        dataset_version=context.dataset_version,
                        image_path=str(image_paths[idx]),
                        relative_path=str(relative_paths[idx]),
                        true_idx=true_idx,
                        true_label=context.class_names[true_idx],
                        pred_idx=pred_idx,
                        pred_label=context.class_names[pred_idx],
                        confidence=float(conf[idx].item()),
                        is_correct=bool(pred_idx == true_idx),
                    )
                )

    metrics = {
        "test_loss": loss_sum / total_samples if total_samples else 0.0,
        "test_accuracy": confusion.diag().sum().item() / total_samples if total_samples else 0.0,
        "test_macro_f1": _compute_macro_f1(confusion.float()),
    }
    return predictions, metrics, confusion


def write_predictions_csv(path: str | Path, predictions: list[PredictionRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_version",
                "image_path",
                "relative_path",
                "true_idx",
                "true_label",
                "pred_idx",
                "pred_label",
                "confidence",
                "is_correct",
            ],
        )
        writer.writeheader()
        for row in predictions:
            writer.writerow(
                {
                    "dataset_version": row.dataset_version,
                    "image_path": row.image_path,
                    "relative_path": row.relative_path,
                    "true_idx": row.true_idx,
                    "true_label": row.true_label,
                    "pred_idx": row.pred_idx,
                    "pred_label": row.pred_label,
                    "confidence": f"{row.confidence:.6f}",
                    "is_correct": row.is_correct,
                }
            )


def write_metrics_json(path: str | Path, metrics: dict[str, float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _get_cam_visuals(context: RunContext, image_path: str, target_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pytorch-grad-cam is required for Part 3. Install dependencies from requirements.txt."
        ) from exc

    image = Image.open(image_path).convert("RGB")
    cropped_image = _resize_and_crop(image, context.image_size)
    input_tensor = _normalize_cropped_image(cropped_image).unsqueeze(0).to(context.device)
    rgb_image = np.asarray(cropped_image, dtype=np.float32) / 255.0
    targets = [ClassifierOutputTarget(int(target_idx))]
    target_layers = [context.model.blocks[-1].norm1]

    with GradCAM(model=context.model, target_layers=target_layers, reshape_transform=_vit_reshape_transform) as cam:
        gradcam_map = cam(input_tensor=input_tensor, targets=targets)[0]
    with GradCAMPlusPlus(model=context.model, target_layers=target_layers, reshape_transform=_vit_reshape_transform) as campp:
        gradcampp_map = campp(input_tensor=input_tensor, targets=targets)[0]

    gradcam_overlay = show_cam_on_image(rgb_image, gradcam_map, use_rgb=True)
    gradcampp_overlay = show_cam_on_image(rgb_image, gradcampp_map, use_rgb=True)
    return rgb_image, gradcam_overlay, gradcampp_overlay


def _save_representative_figure(
    context: RunContext,
    record: PredictionRecord,
    sample_type: str,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_image, gradcam_overlay, gradcampp_overlay = _get_cam_visuals(context, record.image_path, record.pred_idx)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original")
    axes[1].imshow(gradcam_overlay)
    axes[1].set_title("GradCAM")
    axes[2].imshow(gradcampp_overlay)
    axes[2].set_title("GradCAM++")
    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"{context.dataset_version} | {_format_class_name(record.true_label)} | {sample_type} | "
        f"pred={_format_class_name(record.pred_label)} | conf={record.confidence:.3f}",
        fontsize=11,
    )
    figure.text(0.5, 0.02, record.relative_path, ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 0.92))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def select_representative_samples(
    context: RunContext,
    predictions: list[PredictionRecord],
    dataset_output_dir: str | Path,
) -> tuple[list[dict[str, str]], list[str], int]:
    dataset_output_dir = Path(dataset_output_dir)
    representative_rows: list[dict[str, str]] = []
    missing_incorrect_classes: list[str] = []
    figure_count = 0

    for class_idx, class_name in enumerate(context.class_names):
        class_records = [record for record in predictions if record.true_idx == class_idx]
        correct_record = max((record for record in class_records if record.is_correct), key=lambda item: item.confidence, default=None)
        incorrect_record = max((record for record in class_records if not record.is_correct), key=lambda item: item.confidence, default=None)

        for sample_type, record in [("correct", correct_record), ("incorrect", incorrect_record)]:
            if record is None:
                if sample_type == "incorrect":
                    missing_incorrect_classes.append(class_name)
                representative_rows.append(
                    {
                        "dataset_version": context.dataset_version,
                        "class_name": class_name,
                        "sample_type": sample_type,
                        "status": "missing",
                        "relative_path": "",
                        "pred_label": "",
                        "confidence": "",
                        "figure_path": "",
                        "note": "no misclassified sample found" if sample_type == "incorrect" else "no correct sample found",
                    }
                )
                continue

            output_name = f"{sample_type}_{_slugify(class_name)}.png"
            figure_path = dataset_output_dir / f"{sample_type}_samples" / output_name
            _save_representative_figure(context, record, sample_type, figure_path)
            figure_count += 1
            representative_rows.append(
                {
                    "dataset_version": context.dataset_version,
                    "class_name": class_name,
                    "sample_type": sample_type,
                    "status": "selected",
                    "relative_path": record.relative_path,
                    "pred_label": record.pred_label,
                    "confidence": f"{record.confidence:.4f}",
                    "figure_path": figure_path.relative_to(dataset_output_dir.parent).as_posix(),
                    "note": "",
                }
            )

    summary_path = dataset_output_dir / "representative_samples.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(representative_rows[0].keys()))
        writer.writeheader()
        writer.writerows(representative_rows)

    return representative_rows, missing_incorrect_classes, figure_count


def _compute_hardest_classes(confusion: torch.Tensor, class_names: list[str], limit: int = 5) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for class_idx, class_name in enumerate(class_names):
        support = int(confusion[class_idx].sum().item())
        if support == 0:
            continue
        recall = float(confusion[class_idx, class_idx].item() / support)
        rows.append(
            {
                "class_name": class_name,
                "support": support,
                "recall": recall,
            }
        )
    rows.sort(key=lambda row: (float(row["recall"]), -int(row["support"])))
    return rows[:limit]


def _compute_top_confusions(confusion: torch.Tensor, class_names: list[str], limit: int = 5) -> list[dict[str, int | str]]:
    pairs: list[dict[str, int | str]] = []
    for true_idx, true_name in enumerate(class_names):
        for pred_idx, pred_name in enumerate(class_names):
            if true_idx == pred_idx:
                continue
            count = int(confusion[true_idx, pred_idx].item())
            if count <= 0:
                continue
            pairs.append(
                {
                    "true_label": true_name,
                    "pred_label": pred_name,
                    "count": count,
                }
            )
    pairs.sort(key=lambda item: int(item["count"]), reverse=True)
    return pairs[:limit]


def analyze_run(
    context: RunContext,
    *,
    output_root: str | Path,
) -> DatasetAnalysis:
    output_dir = Path(output_root) / context.dataset_version
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, metrics, confusion = run_inference(context)
    write_predictions_csv(output_dir / "predictions.csv", predictions)
    write_metrics_json(output_dir / "metrics.json", metrics)
    representative_rows, missing_incorrect_classes, figure_count = select_representative_samples(context, predictions, output_dir)

    return DatasetAnalysis(
        dataset_version=context.dataset_version,
        output_dir=output_dir,
        predictions=predictions,
        metrics=metrics,
        representative_rows=representative_rows,
        missing_incorrect_classes=missing_incorrect_classes,
        hardest_classes=_compute_hardest_classes(confusion, context.class_names),
        top_confusions=_compute_top_confusions(confusion, context.class_names),
        figure_count=figure_count,
    )


def generate_comparison_figures(
    color_context: RunContext,
    color_analysis: DatasetAnalysis,
    segmented_context: RunContext,
    segmented_analysis: DatasetAnalysis,
    output_dir: str | Path,
) -> dict[str, int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    color_by_key = {
        _canonical_relative_key(record.relative_path): record
        for record in color_analysis.predictions
    }
    segmented_by_key = {
        _canonical_relative_key(record.relative_path): record
        for record in segmented_analysis.predictions
    }
    created = 0
    missing = 0

    for row in color_analysis.representative_rows:
        if row["status"] != "selected":
            continue
        relative_path = row["relative_path"]
        canonical_key = _canonical_relative_key(relative_path)
        segmented_record = segmented_by_key.get(canonical_key)
        if segmented_record is None:
            missing += 1
            continue

        color_record = color_by_key[canonical_key]
        color_original, color_gradcam, color_gradcampp = _get_cam_visuals(color_context, color_record.image_path, color_record.pred_idx)
        seg_original, seg_gradcam, seg_gradcampp = _get_cam_visuals(segmented_context, segmented_record.image_path, segmented_record.pred_idx)

        sample_type = row["sample_type"]
        class_slug = _slugify(row["class_name"])
        figure_path = output_dir / f"{sample_type}_{class_slug}.png"

        figure, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes[0, 0].imshow(color_original)
        axes[0, 0].set_title("Color Original")
        axes[0, 1].imshow(color_gradcam)
        axes[0, 1].set_title("Color GradCAM")
        axes[0, 2].imshow(color_gradcampp)
        axes[0, 2].set_title("Color GradCAM++")
        axes[1, 0].imshow(seg_original)
        axes[1, 0].set_title("Segmented Original")
        axes[1, 1].imshow(seg_gradcam)
        axes[1, 1].set_title("Segmented GradCAM")
        axes[1, 2].imshow(seg_gradcampp)
        axes[1, 2].set_title("Segmented GradCAM++")
        for axis in axes.flat:
            axis.axis("off")

        figure.suptitle(
            f"{_format_class_name(color_record.true_label)} | {sample_type} | {relative_path}",
            fontsize=11,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        figure.savefig(figure_path, dpi=200)
        plt.close(figure)
        created += 1

    return {"created": created, "missing_pairs": missing}


def write_analysis_markdown(
    output_root: str | Path,
    analyses: list[DatasetAnalysis],
    contexts: list[RunContext],
    comparison_summary: dict[str, int],
) -> None:
    output_root = Path(output_root)
    context_by_dataset = {context.dataset_version: context for context in contexts}
    metrics_rows = [
        [
            analysis.dataset_version,
            f"{analysis.metrics['test_loss']:.4f}",
            f"{analysis.metrics['test_accuracy']:.4f}",
            f"{analysis.metrics['test_macro_f1']:.4f}",
            str(sum(1 for row in analysis.representative_rows if row["sample_type"] == "correct" and row["status"] == "selected")),
            str(sum(1 for row in analysis.representative_rows if row["sample_type"] == "incorrect" and row["status"] == "selected")),
        ]
        for analysis in analyses
    ]

    lines = [
        "# Part 3 GradCAM Analysis",
        "",
        "Best model fixed for Part 3: `vit_small + full_finetune`.",
        "",
        "## Test Metrics Overview",
        "",
        _markdown_table(
            ["Dataset", "Test Loss", "Top-1 Accuracy", "Macro F1", "Correct Samples", "Incorrect Samples"],
            metrics_rows,
        ),
        "",
        "## Representative Sample Coverage",
        "",
    ]

    for analysis in analyses:
        context = context_by_dataset[analysis.dataset_version]
        lines.extend(
            [
                f"### {analysis.dataset_version}",
                "",
                f"- Predictions: `{analysis.dataset_version}/predictions.csv`",
                f"- Metrics: `{analysis.dataset_version}/metrics.json`",
                f"- Representative samples: `{analysis.dataset_version}/representative_samples.csv`",
                f"- Test samples aligned to color split: {context.matched_sample_count}/{context.reference_key_count}",
                f"- Generated figures: {analysis.figure_count}",
            ]
        )
        if context.missing_reference_keys:
            lines.append(f"- Missing aligned samples in this dataset version: {len(context.missing_reference_keys)}")
        if analysis.missing_incorrect_classes:
            formatted = ", ".join(_format_class_name(name) for name in analysis.missing_incorrect_classes)
            lines.append(f"- Missing incorrect sample(s): {formatted}")
        else:
            lines.append("- Missing incorrect sample(s): none")
        lines.extend(["", f"#### Hardest Classes in {analysis.dataset_version}", ""])
        hardest_rows = [
            [str(index + 1), _format_class_name(str(row["class_name"])), str(row["support"]), f"{float(row['recall']):.4f}"]
            for index, row in enumerate(analysis.hardest_classes)
        ]
        if hardest_rows:
            lines.append(_markdown_table(["Rank", "Class", "Support", "Recall"], hardest_rows))
        lines.extend(["", f"#### Top Confusions in {analysis.dataset_version}", ""])
        confusion_rows = [
            [str(index + 1), _format_class_name(str(row["true_label"])), _format_class_name(str(row["pred_label"])), str(row["count"])]
            for index, row in enumerate(analysis.top_confusions)
        ]
        if confusion_rows:
            lines.append(_markdown_table(["Rank", "True Label", "Predicted Label", "Count"], confusion_rows))
        else:
            lines.append("No misclassifications found.")
        lines.append("")

    color_metrics = next((analysis.metrics for analysis in analyses if analysis.dataset_version == "color"), None)
    segmented_metrics = next((analysis.metrics for analysis in analyses if analysis.dataset_version == "background_segmented"), None)
    lines.extend(
        [
            "## Background Segmentation Comparison",
            "",
            f"- Comparison figures created: {comparison_summary['created']}",
            f"- Color representative samples without segmented match: {comparison_summary['missing_pairs']}",
        ]
    )
    if color_metrics and segmented_metrics:
        accuracy_delta = segmented_metrics["test_accuracy"] - color_metrics["test_accuracy"]
        f1_delta = segmented_metrics["test_macro_f1"] - color_metrics["test_macro_f1"]
        lines.extend(
            [
                f"- Accuracy delta (`background_segmented - color`): {accuracy_delta:+.4f}",
                f"- Macro F1 delta (`background_segmented - color`): {f1_delta:+.4f}",
            ]
        )
        if accuracy_delta > 0 and f1_delta > 0:
            lines.append("- Background segmentation improved overall classification on this split and is a strong candidate for cleaner lesion-focused attention.")
        elif accuracy_delta < 0 and f1_delta < 0:
            lines.append("- Background segmentation reduced overall classification performance on this split, so any localization gain should be weighed against the accuracy drop.")
        else:
            lines.append("- Background segmentation changed localization conditions, but its classification benefit is mixed on this split.")
    lines.extend(
        [
            "",
            "Inspect the comparison figures under `comparisons/` to judge whether attention becomes more concentrated on lesion regions after background removal.",
            "",
            "## Failure Mode Summary",
            "",
        ]
    )

    for analysis in analyses:
        confusion_text = ", ".join(
            f"{_format_class_name(str(item['true_label']))} -> {_format_class_name(str(item['pred_label']))} ({item['count']})"
            for item in analysis.top_confusions[:3]
        )
        hardest_text = ", ".join(
            f"{_format_class_name(str(item['class_name']))} (recall={float(item['recall']):.3f})"
            for item in analysis.hardest_classes[:3]
        )
        lines.extend(
            [
                f"### {analysis.dataset_version}",
                "",
                f"- Hardest classes: {hardest_text or 'n/a'}",
                f"- Most common misclassifications: {confusion_text or 'none'}",
            ]
        )
        error_counts = Counter(record.true_label for record in analysis.predictions if not record.is_correct)
        if error_counts:
            worst_class, worst_count = error_counts.most_common(1)[0]
            lines.append(
                f"- Most error-prone true class: {_format_class_name(worst_class)} ({worst_count} test errors)."
            )
        else:
            lines.append("- No error-prone class identified because the run produced no test errors.")
        lines.append("")

    (output_root / "analysis.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
