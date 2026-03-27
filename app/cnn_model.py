from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def _safe_import_torchvision():
    try:
        import torch
        from torchvision import models, transforms
        return torch, models, transforms, None
    except Exception as exc:
        return None, None, None, exc


def _grayscale_entropy(gray_array: np.ndarray) -> float:
    if gray_array.size == 0:
        return 0.0

    values = (gray_array * 255.0).clip(0, 255).astype(np.uint8).ravel()
    hist = np.bincount(values, minlength=256).astype(np.float64)
    hist_sum = hist.sum()

    if hist_sum == 0:
        return 0.0

    hist /= hist_sum
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def _edge_density(gray_array: np.ndarray) -> float:
    if gray_array.ndim != 2 or gray_array.size == 0:
        return 0.0

    gx = np.abs(np.diff(gray_array, axis=1))
    gy = np.abs(np.diff(gray_array, axis=0))

    gx_mean = float(gx.mean()) if gx.size else 0.0
    gy_mean = float(gy.mean()) if gy.size else 0.0
    return (gx_mean + gy_mean) / 2.0


def _block_variance(gray_array: np.ndarray, block_size: int = 16) -> float:
    if gray_array.ndim != 2 or gray_array.size == 0:
        return 0.0

    height, width = gray_array.shape
    usable_h = (height // block_size) * block_size
    usable_w = (width // block_size) * block_size

    if usable_h == 0 or usable_w == 0:
        return 0.0

    cropped = gray_array[:usable_h, :usable_w]
    blocks = cropped.reshape(
        usable_h // block_size,
        block_size,
        usable_w // block_size,
        block_size,
    ).transpose(0, 2, 1, 3)

    block_means = blocks.mean(axis=(2, 3))
    return float(block_means.var())


def _label_from_score(score: int) -> str:
    if score >= 75:
        return "Strong visual anomaly"
    if score >= 50:
        return "Moderate visual anomaly"
    return "Low visual anomaly"


def _score_from_metrics(
    image_entropy: float,
    edge_density: float,
    block_variance: float,
    activation_std: float,
) -> tuple[int, int]:
    score = 0.0
    strong_signal_count = 0

    if image_entropy >= 7.3:
        score += 35.0
        strong_signal_count += 1
    elif image_entropy >= 7.0:
        score += 22.0
    elif image_entropy >= 6.8:
        score += 10.0

    if edge_density >= 0.18:
        score += 25.0
        strong_signal_count += 1
    elif edge_density >= 0.14:
        score += 15.0
    elif edge_density >= 0.10:
        score += 8.0

    if block_variance >= 0.020:
        score += 18.0
        strong_signal_count += 1
    elif block_variance >= 0.012:
        score += 10.0

    if activation_std >= 1.20:
        score += 12.0
        strong_signal_count += 1
    elif activation_std >= 0.95:
        score += 7.0

    return int(round(min(100.0, score))), strong_signal_count


def _build_pretrained_resnet18():
    torch, models, transforms, import_error = _safe_import_torchvision()
    if import_error is not None:
        raise RuntimeError(
            "PyTorch/torchvision is not installed. Install torch and torchvision first."
        ) from import_error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        weights_enum = getattr(models, "ResNet18_Weights", None)

        if weights_enum is not None:
            weights = weights_enum.DEFAULT
            model = models.resnet18(weights=weights)
            preprocess = weights.transforms()
            weights_name = str(weights)
        else:
            model = models.resnet18(pretrained=True)
            preprocess = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            weights_name = "legacy-imagenet-pretrained"
    except Exception as exc:
        raise RuntimeError(
            "Could not load official pretrained ResNet18 weights. "
            "If this is the first run, connect once to the internet so torchvision can cache them."
        ) from exc

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval().to(device)

    return torch, feature_extractor, preprocess, device, weights_name


def analyze_image_with_pretrained_cnn(image_path: str) -> dict[str, Any]:
    try:
        torch, feature_extractor, preprocess, device, weights_name = _build_pretrained_resnet18()
    except Exception as exc:
        return {
            "available": False,
            "status": "cnn_unavailable",
            "model_name": "resnet18_feature_extractor",
            "weights": None,
            "pretrained": True,
            "malware_specific": False,
            "visual_score": None,
            "visual_label": None,
            "natural_image_confidence": None,
            "image_entropy": None,
            "edge_density": None,
            "block_variance": None,
            "activation_mean": None,
            "activation_std": None,
            "strong_signal_count": 0,
            "reasons": [],
            "error": str(exc),
        }

    try:
        gray_img = Image.open(image_path).convert("L")
        rgb_img = gray_img.convert("RGB")
    except Exception as exc:
        return {
            "available": False,
            "status": "image_load_failed",
            "model_name": "resnet18_feature_extractor",
            "weights": weights_name,
            "pretrained": True,
            "malware_specific": False,
            "visual_score": None,
            "visual_label": None,
            "natural_image_confidence": None,
            "image_entropy": None,
            "edge_density": None,
            "block_variance": None,
            "activation_mean": None,
            "activation_std": None,
            "strong_signal_count": 0,
            "reasons": [],
            "error": f"Could not load grayscale image: {exc}",
        }

    tensor = preprocess(rgb_img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = feature_extractor(tensor).flatten(1)

    gray_array = np.asarray(gray_img, dtype=np.float32) / 255.0

    image_entropy = _grayscale_entropy(gray_array)
    edge_density = _edge_density(gray_array)
    block_variance = _block_variance(gray_array)
    activation_mean = float(embedding.abs().mean().item())
    activation_std = float(embedding.std().item())

    visual_score, strong_signal_count = _score_from_metrics(
        image_entropy=image_entropy,
        edge_density=edge_density,
        block_variance=block_variance,
        activation_std=activation_std,
    )

    reasons: list[str] = []

    if image_entropy >= 7.3:
        reasons.append("The byte image has very high grayscale entropy, which is common in packed or compressed binaries.")
    elif image_entropy >= 7.0:
        reasons.append("The byte image has high grayscale entropy.")
    elif image_entropy >= 6.8:
        reasons.append("The byte image has mildly elevated grayscale entropy.")

    if edge_density >= 0.18:
        reasons.append("The byte image shows dense abrupt transitions.")
    elif edge_density >= 0.14:
        reasons.append("The byte image shows moderately strong local transitions.")

    if block_variance >= 0.020:
        reasons.append("Large block-to-block intensity variation was observed.")
    elif block_variance >= 0.012:
        reasons.append("Moderate block-level variation was observed.")

    if activation_std >= 1.20:
        reasons.append("Deep CNN features are unusually dispersed for this byte-image texture.")

    if not reasons:
        reasons.append("The visual texture did not show a strong anomaly.")

    return {
        "available": True,
        "status": "ok",
        "model_name": "resnet18_feature_extractor",
        "weights": weights_name,
        "pretrained": True,
        "malware_specific": False,
        "visual_score": visual_score,
        "visual_label": _label_from_score(visual_score),
        "natural_image_confidence": None,
        "image_entropy": round(image_entropy, 4),
        "edge_density": round(edge_density, 4),
        "block_variance": round(block_variance, 4),
        "activation_mean": round(activation_mean, 4),
        "activation_std": round(activation_std, 4),
        "strong_signal_count": strong_signal_count,
        "reasons": reasons,
        "error": None,
    }
