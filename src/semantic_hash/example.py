"""Download the UMass image-series example and run a validation-selected experiment."""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import tempfile
from pathlib import Path

from .config import ModelConfig, load_config
from .images import discover_series
from .io import file_digest, read_json, write_json

DATASET_REPO = "https://github.com/UMass-Rescue/image-series-dataset.git"
LOG = logging.getLogger(__name__)


def _git(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "Git is required to download the example dataset. Install git first."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Dataset Git command failed: {exc.stderr.strip()}") from exc
    return result.stdout.strip()


def download_dataset(destination: Path, *, ref: str = "main", repo_url: str = DATASET_REPO) -> dict:
    """Clone once, record the commit and labels, and refuse changes on reuse.

    Build in a temporary sibling directory so an interrupted download can simply
    be retried. Existing downloads stay at their recorded commit on later runs.
    """
    destination = destination.expanduser().resolve()
    metadata_path = destination / "dataset.json"
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata["repository"] != repo_url or metadata["ref"] != ref:
            raise ValueError("Download folder belongs to another dataset/ref. Choose a new folder.")
        repository = destination / "repository"
        if _git("-C", str(repository), "rev-parse", "HEAD") != metadata["commit"]:
            raise ValueError("Downloaded dataset checkout changed. Choose a fresh download folder.")
        if _git(
            "-C", str(repository), "status", "--porcelain", "--untracked-files=all", "--", "series"
        ):
            raise ValueError(
                "Downloaded series images changed. Restore them or use a fresh folder."
            )
        if file_digest(destination / "labels.json") != metadata["labels_sha256"]:
            raise ValueError("Downloaded dataset labels changed. Use a fresh download folder.")
        LOG.info("Reusing dataset commit %s from %s", metadata["commit"], destination)
        return metadata
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(
            f"Download folder is not empty and was not created by this script: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading %s (%s) into %s", repo_url, ref, destination)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-download-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        repository = staging / "repository"
        _git("clone", "--depth", "1", "--branch", ref, "--", repo_url, str(repository))
        discovered = discover_series(repository / "series")
        labels = {
            name: [str(destination / "repository" / Path(p).relative_to(repository)) for p in paths]
            for name, paths in discovered.items()
        }
        if len(labels) < 6:
            raise ValueError("Example dataset has fewer than 6 series folders; cannot split it.")
        write_json(staging / "labels.json", labels)
        metadata = {
            "repository": repo_url,
            "ref": ref,
            "commit": _git("-C", str(repository), "rev-parse", "HEAD"),
            "series_count": len(labels),
            "image_count": sum(map(len, labels.values())),
            "labels_sha256": file_digest(staging / "labels.json"),
        }
        write_json(staging / "dataset.json", metadata)
        staging.replace(destination)
    LOG.info("Downloaded %d images in %d series", metadata["image_count"], metadata["series_count"])
    return metadata


def _check_output_folder(path: Path, run: dict) -> None:
    marker = path / "example_run.json"
    if marker.is_file():
        if read_json(marker) != run:
            raise ValueError(f"Output folder belongs to a different example run: {path}")
    elif path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Use a new or empty output folder: {path}")


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Download image series, fine-tune CLIP, and evaluate its best checkpoint.",
    )
    result.add_argument(
        "--download-dir",
        required=True,
        type=Path,
        help="Destination for the dataset checkout and labels.",
    )
    result.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="Destination for checkpoints and preparation state.",
    )
    result.add_argument(
        "--eval-dir",
        required=True,
        type=Path,
        help="Destination for test results and validation plots.",
    )
    result.add_argument(
        "--ref",
        default="main",
        help="Dataset branch or tag; reused downloads keep their recorded commit.",
    )
    result.add_argument("--epochs", type=_positive_int, default=9)
    result.add_argument("--batch-size", type=_positive_int, default=8)
    result.add_argument("--embedding-batch-size", type=_positive_int, default=32)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument(
        "--device",
        default="auto",
        help="PyTorch device (default: CUDA when available, otherwise CPU).",
    )
    result.add_argument("--model", default=ModelConfig().name)
    result.add_argument(
        "--pretrained", help="Pretrained tag/path for built-in OpenCLIP model names."
    )
    result.add_argument(
        "--loss",
        choices=("siglip", "clip"),
        help="Defaults to siglip for a SigLIP model name, otherwise clip.",
    )
    result.add_argument(
        "--grad-checkpointing", action="store_true", help="Reduce training activation memory."
    )
    result.add_argument(
        "--download-only",
        action="store_true",
        help="Download data and write the run config without loading a model.",
    )
    result.add_argument("--verbose", action="store_true")
    return result


def run_example(args: argparse.Namespace, *, repo_url: str = DATASET_REPO) -> Path:
    download_dir = args.download_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    evaluation_dir = args.eval_dir.expanduser().resolve()
    roots = [download_dir, checkpoint_dir, evaluation_dir]
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError(
                    "The download, checkpoint, and evaluation folders must not overlap."
                )
    config = {
        "dataset_name": "image-series-example",
        "image_dir": str(download_dir / "repository" / "series"),
        "labels_json": str(download_dir / "labels.json"),
        "output_dir": str(checkpoint_dir / "workflow"),
        "checkpoint_dir": str(checkpoint_dir),
        "evaluation_dir": str(evaluation_dir),
        "model": {"name": args.model, "pretrained": args.pretrained},
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "loss": args.loss or ("siglip" if "siglip" in args.model.lower() else "clip"),
            "grad_checkpointing": args.grad_checkpointing,
        },
        "clustering": {"enabled": False},
        "split_fractions": [0.7, 0.15, 0.15],
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "embedding_batch_size": args.embedding_batch_size,
    }
    # Validate before creating user-selected output folders or downloading images.
    with tempfile.TemporaryDirectory(prefix="semhash-config-") as temporary:
        temporary_config = Path(temporary) / "config.json"
        write_json(temporary_config, config)
        cfg = load_config(temporary_config, require_inputs=False)
    run = {"config": cfg.to_dict(), "repository": repo_url, "ref": args.ref}
    for path in (checkpoint_dir, evaluation_dir):
        _check_output_folder(path, run)
    metadata = download_dataset(download_dir, ref=args.ref, repo_url=repo_url)
    for path in (checkpoint_dir, evaluation_dir):
        write_json(path / "example_run.json", run)
    config_path = checkpoint_dir / "example_config.json"
    write_json(config_path, cfg.to_dict())
    LOG.info("Run configuration: %s", config_path)
    if args.download_only:
        LOG.info("Download complete. Repeat without --download-only to fine-tune and evaluate.")
        return config_path

    from .data import prepare
    from .evaluation import evaluate
    from .model import load_checkpoint
    from .training import train

    prepare(cfg)
    train(cfg, variant="series", resume=True)
    best_path = cfg.checkpoint_root / "series" / "best.pt"
    checkpoint = load_checkpoint(best_path, cfg)
    selection = {
        "checkpoint": str(best_path),
        "checkpoint_sha256": file_digest(best_path),
        "epoch": checkpoint["epoch"],
        "criterion": "Highest validation Hits@1; ties keep the earliest trained epoch.",
        "validation_hits_at_1": checkpoint["best_score"],
        "dataset_commit": metadata["commit"],
    }
    del checkpoint
    # Persist the decision before test evaluation; test scores never choose an epoch.
    write_json(evaluation_dir / "selected_checkpoint.json", selection)
    metrics_path = evaluate(cfg, split="test", variant="series")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    splits = {}
    for split in ("train", "val", "test"):
        labels = read_json(cfg.output_dir / "data" / f"{split}_series.json")
        splits[split] = {"series": len(labels), "images": sum(map(len, labels.values()))}
    write_json(
        evaluation_dir / "result.json",
        {
            "selection": selection,
            "splits": splits,
            "metrics_csv": str(metrics_path),
            "test_metrics": metrics,
        },
    )
    LOG.info(
        "Selected epoch %d (validation Hits@1 %.4f)",
        selection["epoch"],
        selection["validation_hits_at_1"],
    )
    LOG.info("Final test results: %s", metrics_path)
    return metrics_path


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s"
    )
    try:
        run_example(args)
    except (OSError, ValueError, RuntimeError) as exc:
        if args.verbose:
            raise
        LOG.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOG.error("Interrupted. Repeat the same command to reuse the download and resume training.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
