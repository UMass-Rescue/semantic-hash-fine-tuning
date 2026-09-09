"""The complete workflow, accessible from any working directory."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="semhash", description="Fine-tune CLIP for image-series retrieval."
    )
    result.add_argument(
        "--verbose", action="store_true", help="Show debug logging and error tracebacks."
    )
    commands = result.add_subparsers(dest="command", required=True)
    labels = commands.add_parser("labels", help="Generate labels from one folder per series.")
    labels.add_argument("--image-dir", required=True, type=Path)
    labels.add_argument("--output", required=True, type=Path)
    for name, help_text in (
        ("prepare", "Check images, split whole series, and compute fixed targets/sub-series."),
        ("train", "Fine-tune the series and sub-series models."),
        ("evaluate", "Compare the baseline and validation-selected checkpoints."),
        ("run", "Prepare, train, and evaluate with one command; resume completed stages."),
        ("embed", "Export normalized embeddings for downstream cosine matching."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path)
        if name in ("train", "evaluate"):
            command.add_argument(
                "--variant", choices=("series", "subseries"), help="Process only this variant."
            )
        if name == "train":
            command.add_argument(
                "--resume", action="store_true", help="Continue from the last completed epoch."
            )
        if name == "evaluate":
            command.add_argument("--split", choices=("val", "test"), default="test")
            command.add_argument(
                "--all-checkpoints",
                action="store_true",
                help="Also evaluate saved epochs (diagnostic).",
            )
            command.add_argument("--baseline-only", action="store_true")
            command.add_argument(
                "--recompute", action="store_true", help="Recompute cached metrics and embeddings."
            )
        if name == "embed":
            command.add_argument(
                "--checkpoint", type=Path, help="A checkpoint produced here; omit for pretrained."
            )
            command.add_argument("--input-dir", required=True, type=Path)
            command.add_argument("--output", required=True, type=Path, help="New output directory.")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s"
    )
    try:
        if args.command == "labels":
            from .images import discover_series
            from .io import write_json

            if args.output.exists():
                raise ValueError(f"Labels output already exists: {args.output}. Choose a new path.")
            mapping = discover_series(args.image_dir)
            write_json(args.output, mapping)
            logging.info("Wrote %d series to %s", len(mapping), args.output)
            return 0
        from .config import load_config

        cfg = load_config(args.config, require_inputs=args.command != "embed")
        if args.command in ("prepare", "run"):
            from .data import prepare

            prepare(cfg)
        if args.command in ("train", "run"):
            from .training import train

            train(
                cfg,
                variant=getattr(args, "variant", None),
                resume=args.command == "run" or args.resume,
            )
        if args.command in ("evaluate", "run"):
            from .evaluation import evaluate

            evaluate(
                cfg,
                split=getattr(args, "split", "test"),
                all_checkpoints=getattr(args, "all_checkpoints", False),
                baseline_only=getattr(args, "baseline_only", False),
                recompute=getattr(args, "recompute", False),
                variant=getattr(args, "variant", None),
            )
        if args.command == "embed":
            export_embeddings(cfg, args)
    except (OSError, ValueError, RuntimeError) as exc:
        if args.verbose:
            raise
        logging.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logging.error(
            "Interrupted. Rerun 'semhash run' or 'semhash train --resume' to continue training."
        )
        return 130
    return 0


def export_embeddings(cfg, args):
    import numpy as np

    from .images import IMAGE_EXTENSIONS
    from .io import write_json
    from .model import create_model, embed_paths, load_checkpoint

    if args.output.exists():
        raise ValueError(f"Embedding output already exists: {args.output}. Choose a new path.")
    paths = sorted(
        str(p.resolve())
        for p in args.input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No images found in {args.input_dir}")
    model, _, transform = create_model(cfg)
    if args.checkpoint:
        model.load_state_dict(load_checkpoint(args.checkpoint, cfg)["state_dict"], strict=True)
    embeddings = embed_paths(model, transform, paths, cfg)
    args.output.mkdir(parents=True)
    np.save(args.output / "embeddings.npy", embeddings, allow_pickle=False)
    write_json(args.output / "image_paths.json", paths)
    write_json(
        args.output / "model.json",
        {
            "model": cfg.to_dict()["model"],
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "normalized": True,
        },
    )
    logging.info("Exported %d embeddings to %s", len(paths), args.output)
