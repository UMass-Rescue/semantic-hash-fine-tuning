"""Image-to-fixed-series-target fine-tuning extracted from the OpenCLIP fork."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

from .config import Config
from .data import assert_prepared, flatten_labels
from .images import load_rgb
from .io import atomic_path, read_json, write_json
from .metrics import retrieval_metrics
from .model import autocast_context, create_model, embed_paths, load_checkpoint

LOG = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def series_targets(
    labels: dict, paths: list[str], embeddings: np.ndarray
) -> dict[str, torch.Tensor]:
    """Normalize the mean of normalized pretrained image embeddings once per series."""
    by_path = {path: i for i, path in enumerate(paths)}
    result = {}
    for name, items in labels.items():
        features = torch.from_numpy(embeddings[[by_path[p] for p in items]]).float()
        mean = F.normalize(features, dim=-1).mean(dim=0)
        if not torch.isfinite(mean).all() or mean.norm() < 1e-8:
            raise ValueError(f"Series {name!r} has an invalid mean embedding.")
        result[name] = F.normalize(mean, dim=0).detach()
    return result


class SeriesDataset(Dataset):
    def __init__(self, labels: dict, targets: dict, transform):
        self.paths, self.series = flatten_labels(labels)
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with load_rgb(self.paths[index]) as image:
            tensor = self.transform(image)
        return tensor, self.targets[self.series[index]]


class UniqueSeriesBatchSampler(Sampler[list[int]]):
    """One image per series per batch; sample each image at most once per epoch.

    Favor series with the most remaining images, as in the source sampler.
    Incomplete batches are dropped so every training example has negatives.
    The plan is seeded per epoch and reports its exact length.
    """

    def __init__(self, series: list[str], batch_size: int, seed: int):
        if batch_size < 2 or batch_size > len(set(series)):
            raise ValueError("Batch size must be between 2 and the number of training series.")
        self.series = series
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def _batches(self):
        rng = random.Random(self.seed + self.epoch)
        remaining = {}
        for index, name in enumerate(self.series):
            remaining.setdefault(name, []).append(index)
        for indices in remaining.values():
            rng.shuffle(indices)
        while len(remaining) >= self.batch_size:
            names = list(remaining)
            rng.shuffle(names)
            names.sort(key=lambda name: len(remaining[name]), reverse=True)
            batch = [remaining[name].pop() for name in names[: self.batch_size]]
            yield batch
            remaining = {name: indices for name, indices in remaining.items() if indices}

    def __iter__(self):
        return self._batches()

    def __len__(self):
        return sum(1 for _ in self._batches())


def contrastive_loss(features, targets, logit_scale, logit_bias, kind: str):
    """Single-device OpenCLIP SigLIP or symmetric CLIP loss against fixed targets."""
    logits = logit_scale.float() * features.float() @ targets.float().T
    if kind == "siglip":
        if logit_bias is not None:
            logits = logits + logit_bias.float()
        labels = 2 * torch.eye(len(features), device=features.device) - 1
        return -F.logsigmoid(labels * logits).sum() / len(features)
    if kind == "clip":
        labels = torch.arange(len(features), device=features.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    raise ValueError(f"Unknown loss: {kind}")


def learning_rate(step: int, total_steps: int, warmup: int, base_lr: float) -> float:
    warmup = min(warmup, max(0, total_steps - 1))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizer(model, cfg: Config):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("visual.") or name in ("logit_scale", "logit_bias")
        )
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            excluded = parameter.ndim < 2 or any(
                s in name for s in ("bn", "ln", "bias", "logit_scale")
            )
            (no_decay if excluded else decay).append(parameter)
    is_vit = "vit" in cfg.model.name.lower()
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.training.learning_rate,
        betas=(0.9, 0.98 if is_vit else 0.999),
        eps=1e-6 if is_vit else 1e-8,
    )


def save_checkpoint(path: Path, checkpoint: dict):
    with atomic_path(path) as temporary:
        torch.save(checkpoint, temporary)


def train(cfg: Config, *, variant: str | None = None, resume: bool = False):
    manifest = assert_prepared(cfg)
    variants = [variant] if variant else cfg.variants
    for name in variants:
        if name not in cfg.variants:
            raise ValueError(f"Variant {name!r} is disabled in the configuration.")
        _train_variant(cfg, name, manifest["fingerprint"], resume)


def _train_variant(cfg: Config, variant: str, data_fingerprint: str, resume: bool):
    run_dir = cfg.checkpoint_root / variant
    latest_path = run_dir / "latest.pt"
    if latest_path.exists() and not resume:
        raise ValueError(f"Training already exists in {run_dir}. Use --resume to continue.")
    seed_everything(cfg.seed)
    previous = load_checkpoint(latest_path, cfg) if latest_path.exists() else None
    if previous:
        if previous["data_fingerprint"] != data_fingerprint or previous["config"] != cfg.to_dict():
            raise ValueError(
                "Resume checkpoint does not match the prepared data and configuration."
            )
        if previous["epoch"] >= cfg.training.epochs:
            write_json(run_dir / "history.json", previous["history"])
            LOG.info("%s training is already complete (%d epochs)", variant, previous["epoch"])
            return
    data_dir = cfg.output_dir / "data"
    labels = read_json(data_dir / f"train_{variant}.json")
    targets = series_targets(
        labels,
        read_json(data_dir / "train_paths.json"),
        np.load(data_dir / "train_embeddings.npy", allow_pickle=False),
    )
    model, train_transform, eval_transform = create_model(cfg)
    if cfg.training.loss == "siglip" and getattr(model, "logit_bias", None) is None:
        raise ValueError(
            "SigLIP loss requires a model with logit_bias. "
            "For CLIP models set training.loss='clip'."
        )
    if cfg.training.grad_checkpointing:
        model.set_grad_checkpointing(True)
    optimizer = make_optimizer(model, cfg)
    device = next(model.parameters()).device
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and cfg.training.precision == "amp"
    )
    dataset = SeriesDataset(labels, targets, train_transform)
    batch_size = min(cfg.training.batch_size, len(labels))
    if batch_size != cfg.training.batch_size:
        LOG.warning(
            "%s batch size reduced from %d to %d series",
            variant,
            cfg.training.batch_size,
            batch_size,
        )
    sampler = UniqueSeriesBatchSampler(dataset.series, batch_size, cfg.seed)
    steps_per_epoch = len(sampler)
    generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    used_images = steps_per_epoch * batch_size
    LOG.info(
        "%s: %d batches/epoch, %d/%d training images used each epoch",
        variant,
        steps_per_epoch,
        used_images,
        len(dataset),
    )
    val_paths, val_series = flatten_labels(read_json(data_dir / "val_series.json"))
    validation_ks = sorted(set([1, *cfg.evaluation.k]))
    baseline = retrieval_metrics(
        np.load(data_dir / "val_embeddings.npy", allow_pickle=False),
        val_series,
        validation_ks,
        cfg.evaluation.query_chunk_size,
    )
    history = [{"epoch": 0, "train_loss": None, "metrics": baseline}]
    start_epoch, best_score, best_epoch = 0, -1.0, 0
    if previous:
        model.load_state_dict(previous["state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        start_epoch, best_score, best_epoch = (
            previous["epoch"],
            previous["best_score"],
            previous["best_epoch"],
        )
        history = previous["history"]
        del previous
    total_steps = steps_per_epoch * cfg.training.epochs
    for epoch in range(start_epoch, cfg.training.epochs):
        # Epoch seeds also reproduce augmentation and dropout when resuming an epoch boundary.
        seed_everything(cfg.seed + epoch)
        generator.manual_seed(cfg.seed + epoch)
        sampler.epoch = epoch
        model.train()
        loss_sum = 0.0
        progress = tqdm(
            loader, desc=f"{variant} epoch {epoch + 1}/{cfg.training.epochs}", unit="batch"
        )
        for batch_index, (images, batch_targets) in enumerate(progress):
            step = epoch * steps_per_epoch + batch_index
            lr = learning_rate(
                step, total_steps, cfg.training.warmup_steps, cfg.training.learning_rate
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, cfg.training.precision):
                features = model.encode_image(images.to(device), normalize=True)
                loss = contrastive_loss(
                    features,
                    batch_targets.to(device),
                    model.logit_scale.exp(),
                    getattr(model, "logit_bias", None),
                    cfg.training.loss,
                )
            if not torch.isfinite(loss):
                raise ValueError(
                    f"Nonfinite loss in {variant} epoch {epoch + 1}; "
                    "try a lower learning rate or fp32."
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))
            loss_sum += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        val_embeddings = embed_paths(model, eval_transform, val_paths, cfg)
        metrics = retrieval_metrics(
            val_embeddings, val_series, validation_ks, cfg.evaluation.query_chunk_size
        )
        score = next(row["accuracy_at_k"] for row in metrics if row["k"] == 1)
        history.append(
            {"epoch": epoch + 1, "train_loss": loss_sum / steps_per_epoch, "metrics": metrics}
        )
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch + 1
        checkpoint = {
            "format_version": 1,
            "epoch": epoch + 1,
            "variant": variant,
            "config": cfg.to_dict(),
            "data_fingerprint": data_fingerprint,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "best_score": best_score,
            "best_epoch": best_epoch,
        }
        if improved:
            save_checkpoint(run_dir / "best.pt", checkpoint)
        if (epoch + 1) % cfg.training.save_every == 0 or epoch + 1 == cfg.training.epochs:
            save_checkpoint(run_dir / f"epoch_{epoch + 1}.pt", checkpoint)
        save_checkpoint(
            latest_path,
            {
                **checkpoint,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
            },
        )
        write_json(run_dir / "history.json", history)
        LOG.info(
            "%s epoch %d: loss %.5f, validation Hits@1 %.4f (best epoch %d)",
            variant,
            epoch + 1,
            loss_sum / steps_per_epoch,
            score,
            best_epoch,
        )
    del model, optimizer, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
