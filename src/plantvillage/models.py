from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision import models


@dataclass(slots=True)
class ModelConfig:
    model_name: str
    num_classes: int
    strategy: str = "full_finetune"
    pretrained: bool = True


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _replace_classifier(model: torch.nn.Module, model_name: str, num_classes: int) -> tuple[torch.nn.Module, list[torch.nn.Parameter]]:
    if model_name == "resnet50":
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model, list(model.fc.parameters())
    if model_name == "efficientnet_b3":
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        return model, list(model.classifier[1].parameters())
    raise ValueError(f"Unsupported torchvision model for classifier replacement: {model_name}")


def _freeze_for_linear_probing(model: torch.nn.Module, head_params: list[torch.nn.Parameter]) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in head_params:
        param.requires_grad = True


def _build_torchvision_model(model_name: str, num_classes: int, pretrained: bool) -> tuple[torch.nn.Module, list[torch.nn.Parameter]]:
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        return _replace_classifier(model, model_name, num_classes)
    if model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        return _replace_classifier(model, model_name, num_classes)
    raise ValueError(f"Unsupported torchvision model: {model_name}")


def _build_vit_small(num_classes: int, pretrained: bool) -> torch.nn.Module:
    try:
        import timm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "timm is required for ViT-Small. Install dependencies from requirements.txt."
        ) from exc

    return timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=num_classes)


def build_model(config: ModelConfig) -> torch.nn.Module:
    model_name = config.model_name
    if model_name in {"resnet50", "efficientnet_b3"}:
        model, head_params = _build_torchvision_model(model_name, config.num_classes, config.pretrained)
    elif model_name == "vit_small":
        model = _build_vit_small(config.num_classes, config.pretrained)
        head_params = list(model.get_classifier().parameters())
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if config.strategy == "linear_probing":
        _freeze_for_linear_probing(model, head_params)
    elif config.strategy in {"full_finetune", "from_scratch"}:
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported strategy: {config.strategy}")

    return model


def resolve_pretrained_default(model_name: str, strategy: str) -> bool:
    if model_name == "resnet50" and strategy == "from_scratch":
        return False
    if model_name == "efficientnet_b3":
        return True
    if model_name == "vit_small":
        return True
    return False
