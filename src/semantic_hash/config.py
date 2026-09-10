"""One validated configuration shared by preparation, training, and evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .io import read_json


@dataclass(frozen=True)
class ModelConfig:
    name: str = "hf-hub:timm/ViT-SO400M-16-SigLIP2-384"
    pretrained: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 9
    batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    save_every: int = 3
    loss: str = "siglip"
    precision: str = "amp"
    grad_checkpointing: bool = False


@dataclass(frozen=True)
class ClusteringConfig:
    enabled: bool = True
    distance_threshold: float = 0.3
    min_samples: int = 5


@dataclass(frozen=True)
class EvaluationConfig:
    k: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    query_chunk_size: int = 256


@dataclass(frozen=True)
class Config:
    dataset_name: str
    image_dir: Path
    labels_json: Path
    output_dir: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42
    device: str = "auto"
    workers: int = 4
    embedding_batch_size: int = 32
    split_fractions: list[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])
    deduplicate_images: bool = False
    phash_threshold: int = 4
    checkpoint_dir: Path | None = None
    evaluation_dir: Path | None = None

    @property
    def checkpoint_root(self) -> Path:
        return self.checkpoint_dir or self.output_dir / "checkpoints"

    @property
    def evaluation_root(self) -> Path:
        return self.evaluation_dir or self.output_dir / "evaluation"

    @property
    def variants(self) -> list[str]:
        return ["series", "subseries"] if self.clustering.enabled else ["series"]

    def to_dict(self) -> dict:
        result = asdict(self)
        for key in ("image_dir", "labels_json", "output_dir"):
            result[key] = str(result[key])
        for key in ("checkpoint_dir", "evaluation_dir"):
            value = result.pop(key)
            if value is not None:
                result[key] = str(value)
        return result


def _construct(cls, raw: dict):
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} must be a JSON object.")
    unknown = raw.keys() - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {', '.join(sorted(unknown))}")
    try:
        return cls(**raw)
    except TypeError as exc:
        raise ValueError(f"Invalid {cls.__name__}: {exc}") from exc


def _integer(name: str, value, minimum: int = 1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _number(name: str, value, minimum: float, maximum: float = math.inf):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be a finite number in [{minimum}, {maximum}].")


def load_config(path: Path, *, require_inputs: bool = True) -> Config:
    path = path.expanduser().resolve()
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a JSON object.")
    for key in ("dataset_name", "image_dir", "labels_json", "output_dir"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"Missing required string: {key}")
    for key in ("image_dir", "labels_json", "output_dir"):
        value = Path(raw[key]).expanduser()
        raw[key] = (path.parent / value).resolve()
    for key in ("checkpoint_dir", "evaluation_dir"):
        value = raw.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a nonempty path string or null.")
            raw[key] = (path.parent / Path(value).expanduser()).resolve()
    for key, cls in (
        ("model", ModelConfig),
        ("training", TrainingConfig),
        ("clustering", ClusteringConfig),
        ("evaluation", EvaluationConfig),
    ):
        raw[key] = _construct(cls, raw.get(key, {}))
    cfg = _construct(Config, raw)
    for name, value, minimum in (
        ("seed", cfg.seed, 0),
        ("workers", cfg.workers, 0),
        ("embedding_batch_size", cfg.embedding_batch_size, 1),
        ("training.epochs", cfg.training.epochs, 1),
        ("training.batch_size", cfg.training.batch_size, 2),
        ("training.warmup_steps", cfg.training.warmup_steps, 0),
        ("training.save_every", cfg.training.save_every, 1),
        ("clustering.min_samples", cfg.clustering.min_samples, 2),
        ("evaluation.query_chunk_size", cfg.evaluation.query_chunk_size, 1),
        ("phash_threshold", cfg.phash_threshold, 0),
    ):
        _integer(name, value, minimum)
    _number("training.learning_rate", cfg.training.learning_rate, 1e-12)
    _number("training.weight_decay", cfg.training.weight_decay, 0)
    _number("clustering.distance_threshold", cfg.clustering.distance_threshold, 0, 2)
    if cfg.phash_threshold > 64:
        raise ValueError("phash_threshold must be <= 64.")
    if cfg.training.loss not in ("siglip", "clip"):
        raise ValueError("training.loss must be 'siglip' or 'clip'.")
    if cfg.training.precision not in ("amp", "fp32"):
        raise ValueError("training.precision must be 'amp' or 'fp32'.")
    for name, value in (
        ("clustering.enabled", cfg.clustering.enabled),
        ("deduplicate_images", cfg.deduplicate_images),
        ("training.grad_checkpointing", cfg.training.grad_checkpointing),
    ):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean.")
    if not isinstance(cfg.device, str) or not cfg.device:
        raise ValueError("device must be a nonempty PyTorch device string or 'auto'.")
    if not isinstance(cfg.model.name, str) or not cfg.model.name:
        raise ValueError("model.name must be a nonempty string.")
    if cfg.model.pretrained is not None and (
        not isinstance(cfg.model.pretrained, str) or not cfg.model.pretrained
    ):
        raise ValueError("model.pretrained must be a nonempty string or null.")
    if not cfg.model.name.startswith("hf-hub:") and not cfg.model.pretrained:
        raise ValueError("Set model.pretrained for a model without an hf-hub: identifier.")
    if not isinstance(cfg.evaluation.k, list) or not cfg.evaluation.k:
        raise ValueError("evaluation.k must be a nonempty list of positive integers.")
    for k in cfg.evaluation.k:
        _integer("evaluation.k entry", k)
    if sorted(set(cfg.evaluation.k)) != cfg.evaluation.k:
        raise ValueError("evaluation.k must be sorted and contain no duplicates.")
    if not isinstance(cfg.split_fractions, list) or len(cfg.split_fractions) != 3:
        raise ValueError("split_fractions must contain train, validation, and test fractions.")
    for value in cfg.split_fractions:
        _number("split_fractions entry", value, 1e-12, 1)
    if not math.isclose(sum(cfg.split_fractions), 1.0, abs_tol=1e-9):
        raise ValueError("split_fractions must sum to 1.")
    if require_inputs and not cfg.image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {cfg.image_dir}")
    if require_inputs and not cfg.labels_json.is_file():
        raise ValueError(f"Labels file does not exist: {cfg.labels_json}")
    return cfg
