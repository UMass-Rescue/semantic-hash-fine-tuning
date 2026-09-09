"""OpenCLIP integration without vendoring its architecture or text-training code."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import Config
from .images import load_rgb

LOG = logging.getLogger(__name__)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable. Set device to 'cpu' or 'auto'.")
    return device


def create_model(cfg: Config):
    import open_clip

    device = resolve_device(cfg.device)
    LOG.info("Loading %s on %s", cfg.model.name, device)
    model, train_transform, eval_transform = open_clip.create_model_and_transforms(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        require_pretrained=True,
        device="cpu",
        precision="fp32",
    )
    model.to(device)
    return model, train_transform, eval_transform


def autocast_context(device: torch.device, precision: str):
    if precision == "amp" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


class ImageDataset(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            with load_rgb(path) as image:
                return self.transform(image)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot decode/preprocess prepared image {path}: {exc}") from exc


def embed_paths(model, transform, paths: list[str], cfg: Config) -> np.ndarray:
    """Return finite, unit-length float32 embeddings in the input path order."""
    if not paths:
        raise ValueError("Cannot embed an empty image list.")
    device = next(model.parameters()).device
    loader = DataLoader(
        ImageDataset(paths, transform),
        batch_size=cfg.embedding_batch_size,
        num_workers=cfg.workers,
        pin_memory=device.type == "cuda",
    )
    batches = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for images in tqdm(loader, desc="Embedding", unit="batch", leave=False):
                # Keep evaluation and target construction in float32 for stable cosine ranking.
                features = model.encode_image(images.to(device), normalize=True).float()
                batches.append(features.cpu().numpy())
    finally:
        model.train(was_training)
    embeddings = np.concatenate(batches)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(embeddings).all() or np.any(norms < 1e-8):
        raise ValueError("Model produced nonfinite or zero-length embeddings.")
    return embeddings / norms


def load_checkpoint(path: Path, cfg: Config) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if checkpoint["config"]["model"] != cfg.to_dict()["model"]:
        raise ValueError(f"Checkpoint model does not match the config: {path}")
    return checkpoint
