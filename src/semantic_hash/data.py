"""Decode, de-duplicate, split by parent series, and cluster training labels."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .config import Config
from .images import load_rgb
from .io import file_digest, fingerprint, read_json, write_json
from .model import create_model, embed_paths

LOG = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")


def validate_labels(path: Path, image_dir: Path) -> dict[str, list[str]]:
    raw = read_json(path)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "Labels must be a nonempty JSON object mapping series names to image lists."
        )
    labels, owners = {}, {}
    for name, entries in sorted(raw.items()):
        if not isinstance(name, str) or not name.strip() or not isinstance(entries, list):
            raise ValueError("Each series must have a nonempty name and a list of image paths.")
        paths = set()
        for value in entries:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Series {name!r} contains an invalid image path.")
            resolved = (image_dir / Path(value).expanduser()).resolve()
            if not resolved.is_relative_to(image_dir):
                raise ValueError(f"Labeled image is outside image_dir: {resolved}")
            item = str(resolved)
            if item in owners and owners[item] != name:
                raise ValueError(f"Image occurs in multiple series: {item}")
            owners[item] = name
            paths.add(item)
        labels[name] = sorted(paths)
    return labels


def perceptual_hash(image) -> int:
    """Compute a 64-bit pHash: 32x32 DCT, threshold its 8x8 low frequencies."""
    from scipy.fft import dctn

    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS))
    low = dctn(pixels.astype(np.float64), type=2, norm="ortho")[:8, :8]
    return int.from_bytes(np.packbits(low > np.median(low.ravel()[1:])).tobytes(), "big")


def clean_labels(labels: dict[str, list[str]], cfg: Config) -> tuple[dict, dict]:
    cleaned, exact, hashes = {}, {}, []
    report = {"decode_failures": [], "decode_warnings": [], "duplicates": [], "dropped_series": []}
    for name, paths in tqdm(labels.items(), desc="Checking images", unit="series"):
        kept = []
        for path in paths:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with load_rgb(path) as image:
                        digest = hashlib.sha256(
                            str(image.size).encode() + image.tobytes()
                        ).hexdigest()
                        phash = perceptual_hash(image) if cfg.deduplicate_images else None
            except (
                OSError,
                ValueError,
                SyntaxError,
                EOFError,
                Image.DecompressionBombError,
            ) as exc:
                report["decode_failures"].append({"path": path, "series": name, "error": str(exc)})
                continue
            if caught:
                report["decode_warnings"].append(
                    {"path": path, "warnings": [str(w.message) for w in caught]}
                )
            duplicate_of = exact.get(digest)
            reason = "identical decoded pixels"
            if duplicate_of is None and phash is not None:
                duplicate_of = next(
                    (p for h, p in hashes if (h ^ phash).bit_count() <= cfg.phash_threshold), None
                )
                reason = "perceptual hash"
            if duplicate_of is not None:
                report["duplicates"].append(
                    {"path": path, "series": name, "duplicate_of": duplicate_of, "reason": reason}
                )
                continue
            exact[digest] = path
            if phash is not None:
                hashes.append((phash, path))
            kept.append(path)
        if len(kept) >= 2:
            cleaned[name] = kept
        else:
            report["dropped_series"].append({"series": name, "retained_images": kept})
    return cleaned, report


def split_series(labels: dict, fractions: list[float], seed: int) -> dict[str, dict]:
    """Largest-remainder allocation, keeping at least two parent series in each split."""
    if len(labels) < 6:
        raise ValueError(
            "At least 6 usable series are required (2 per train/validation/test split)."
        )
    raw = np.array(fractions) * len(labels)
    counts = np.floor(raw).astype(int)
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], fractions[i]), reverse=True)
    for index in order[: len(labels) - int(counts.sum())]:
        counts[index] += 1
    for index in range(3):
        while counts[index] < 2:
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[index] += 1
    names = sorted(labels)
    random.Random(seed).shuffle(names)
    result, start = {}, 0
    for split, count in zip(SPLITS, counts, strict=True):
        result[split] = {name: labels[name] for name in names[start : start + count]}
        start += count
    return result


def flatten_labels(labels: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    paths, series = [], []
    for name, items in sorted(labels.items()):
        paths.extend(items)
        series.extend([name] * len(items))
    return paths, series


def cluster_labels(labels: dict, embeddings: np.ndarray, cfg: Config) -> tuple[dict, dict]:
    from sklearn.cluster import AgglomerativeClustering

    paths, _ = flatten_labels(labels)
    by_path = {path: index for index, path in enumerate(paths)}
    clustered, report = {}, {"parents": {}, "dropped_images": []}
    for parent, items in sorted(labels.items()):
        features = embeddings[[by_path[p] for p in items]]
        distance = np.clip(1 - features @ features.T, 0, 2)
        np.fill_diagonal(distance, 0)
        groups = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cfg.clustering.distance_threshold,
            metric="precomputed",
            linkage="average",
        ).fit_predict(distance)
        retained = []
        for group in sorted(set(groups)):
            members = [p for p, g in zip(items, groups, strict=True) if g == group]
            if len(members) < cfg.clustering.min_samples:
                report["dropped_images"].extend(members)
            else:
                retained.append(members)
        for index, members in enumerate(sorted(retained)):
            # JSON encoding makes this pair unambiguous even for unusual parent names.
            name = json.dumps([parent, index], ensure_ascii=False)
            clustered[name] = members
            report["parents"][name] = parent
    return clustered, report


def assert_prepared(cfg: Config) -> dict:
    manifest_path = cfg.output_dir / "data" / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "Data preparation is incomplete. Run 'semhash prepare --config ...' first."
        )
    manifest = read_json(manifest_path)
    if manifest["config"] != cfg.to_dict():
        raise ValueError(
            "Config differs from the prepared run. Use the original config or a new output_dir."
        )
    if manifest["labels_digest"] != file_digest(cfg.labels_json):
        raise ValueError("Input labels changed after preparation. Use a new output_dir.")
    for relative, digest in manifest["artifacts"].items():
        if file_digest(cfg.output_dir / relative) != digest:
            raise ValueError(f"Prepared artifact changed: {relative}. Use a new output_dir.")
    for path, expected in manifest["images"].items():
        stat = Path(path).stat()
        if [stat.st_size, stat.st_mtime_ns] != expected:
            raise ValueError(f"Prepared image changed: {path}. Use a new output_dir.")
    return manifest


def prepare(cfg: Config) -> None:
    data_dir = cfg.output_dir / "data"
    if (data_dir / "manifest.json").is_file():
        assert_prepared(cfg)
        LOG.info("Reusing prepared data: %s", data_dir)
        return
    config_path = cfg.output_dir / "config.json"
    if config_path.exists():
        if read_json(config_path) != cfg.to_dict():
            raise ValueError("Output directory belongs to another config. Choose a new output_dir.")
    elif cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise ValueError(
            "Output directory is nonempty and has no run config. Choose a new output_dir."
        )
    write_json(config_path, cfg.to_dict())
    labels = validate_labels(cfg.labels_json, cfg.image_dir)
    labels, report = clean_labels(labels, cfg)
    write_json(data_dir / "image_audit.json", report)
    LOG.info(
        "Retained %d series; skipped %d decode failures, %d duplicates, and %d small series",
        len(labels),
        len(report["decode_failures"]),
        len(report["duplicates"]),
        len(report["dropped_series"]),
    )
    splits = split_series(labels, cfg.split_fractions, cfg.seed)
    for split, mapping in splits.items():
        write_json(data_dir / f"{split}_series.json", mapping)
        LOG.info("%s: %d series, %d images", split, len(mapping), sum(map(len, mapping.values())))
    model, _, transform = create_model(cfg)
    for split in ("train", "val"):
        paths, _ = flatten_labels(splits[split])
        features = embed_paths(model, transform, paths, cfg)
        np.save(data_dir / f"{split}_embeddings.npy", features, allow_pickle=False)
        write_json(data_dir / f"{split}_paths.json", paths)
        if cfg.clustering.enabled and split == "train":
            mapping, report = cluster_labels(splits[split], features, cfg)
            write_json(data_dir / f"{split}_subseries.json", mapping)
            write_json(data_dir / f"{split}_clustering.json", report)
            if len(mapping) < 2:
                raise ValueError(
                    f"Clustering left fewer than 2 {split} sub-series. Inspect {data_dir}, "
                    "then use a new output_dir with a larger distance_threshold, smaller "
                    "min_samples, or clustering.enabled=false."
                )
            LOG.info("%s: retained %d sub-series", split, len(mapping))
    artifacts = {
        str(path.relative_to(cfg.output_dir)): file_digest(path)
        for path in sorted(data_dir.iterdir())
        if path.is_file()
    }
    paths, _ = flatten_labels(labels)
    images = {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in paths}
    manifest = {
        "format_version": 1,
        "config": cfg.to_dict(),
        "labels_digest": file_digest(cfg.labels_json),
        "artifacts": artifacts,
        "images": images,
    }
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(data_dir / "manifest.json", manifest)
    LOG.info("Prepared data: %s", data_dir)
