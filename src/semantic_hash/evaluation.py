"""Compare the pretrained baseline and fine-tuned checkpoints on one fixed split."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from .config import Config
from .data import assert_prepared, flatten_labels
from .io import atomic_path, file_digest, fingerprint, read_json, write_json
from .metrics import retrieval_metrics
from .model import create_model, embed_paths, load_checkpoint

LOG = logging.getLogger(__name__)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("Cannot write an empty metrics report.")
    with atomic_path(path) as temporary:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def evaluate(
    cfg: Config,
    *,
    split: str = "test",
    all_checkpoints: bool = False,
    baseline_only: bool = False,
    recompute: bool = False,
    variant: str | None = None,
) -> Path:
    manifest = assert_prepared(cfg)
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be 'val' or 'test'.")
    variants = [variant] if variant else cfg.variants
    if any(name not in cfg.variants for name in variants):
        raise ValueError("Requested evaluation variant is disabled.")
    runs = [("pretrained", "pretrained", None)]
    if not baseline_only:
        for name in variants:
            checkpoint_dir = cfg.output_dir / "checkpoints" / name
            best_path = checkpoint_dir / "best.pt"
            if not best_path.is_file():
                raise ValueError(f"Missing {best_path}. Train first, or use --baseline-only.")
            runs.append((name, "best", best_path))
            if all_checkpoints:
                runs.extend(
                    (name, p.stem, p)
                    for p in sorted(
                        checkpoint_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem[6:])
                    )
                )
    paths, series = flatten_labels(read_json(cfg.output_dir / "data" / f"{split}_series.json"))
    output = cfg.output_dir / "evaluation" / split
    rows = []
    for name, checkpoint_name, checkpoint_path in runs:
        signature = fingerprint(
            {
                "data": manifest["fingerprint"],
                "split": split,
                "checkpoint": file_digest(checkpoint_path) if checkpoint_path else "pretrained",
                "k": cfg.evaluation.k,
                "metric_version": 1,
            }
        )
        cache_path = output / "cache" / f"{name}_{checkpoint_name}.json"
        cached = read_json(cache_path) if cache_path.exists() else None
        if cached and cached["signature"] == signature and not recompute:
            LOG.info("Reusing %s %s %s metrics", split, name, checkpoint_name)
            rows.extend(cached["rows"])
            continue
        checkpoint = load_checkpoint(checkpoint_path, cfg) if checkpoint_path else None
        if checkpoint and checkpoint["data_fingerprint"] != manifest["fingerprint"]:
            raise ValueError(f"Checkpoint belongs to different prepared data: {checkpoint_path}")
        model, _, transform = create_model(cfg)
        if checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=True)
        embeddings = embed_paths(model, transform, paths, cfg)
        del model
        epoch = checkpoint["epoch"] if checkpoint else 0
        result = [
            {
                "dataset": cfg.dataset_name,
                "split": split,
                "variant": name,
                "checkpoint": checkpoint_name,
                "epoch": epoch,
                **metric,
            }
            for metric in retrieval_metrics(
                embeddings, series, cfg.evaluation.k, cfg.evaluation.query_chunk_size
            )
        ]
        del checkpoint
        write_json(cache_path, {"signature": signature, "rows": result})
        rows.extend(result)
    write_csv(output / "metrics.csv", rows)
    write_json(
        output / "protocol.json",
        {
            "split": split,
            "query_count": len(paths),
            "gallery_count": len(paths),
            "gallery": "All labeled images in this split, excluding the query itself.",
            "checkpoint_selection": "Highest validation Hits@1; ties keep the earliest epoch.",
            "precision_recall_averaging": "micro",
            "data_fingerprint": manifest["fingerprint"],
            "all_checkpoints": all_checkpoints,
        },
    )
    make_plots(rows, output / "plots", split)
    validation_rows = []
    for name in variants:
        history_path = cfg.output_dir / "checkpoints" / name / "history.json"
        if history_path.is_file():
            for epoch in read_json(history_path):
                validation_rows.extend(
                    {"variant": name, "epoch": epoch["epoch"], **row}
                    for row in epoch["metrics"]
                    if row["k"] in cfg.evaluation.k
                )
    if validation_rows:
        make_plots(validation_rows, cfg.output_dir / "evaluation" / "val" / "plots", "val")
    LOG.info("Evaluation report: %s", output / "metrics.csv")
    return output / "metrics.csv"


def make_plots(rows: list[dict], output: Path, split: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    variants = sorted({row["variant"] for row in rows})
    ks = sorted({row["k"] for row in rows})
    for metric, label in (("accuracy_at_k", "Hits@k"), ("precision_at_k", "Precision@k")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant in variants:
            for k in ks:
                # The selected checkpoint can also be a saved epoch; plot it once.
                values = {
                    row["epoch"]: row[metric]
                    for row in rows
                    if row["variant"] == variant and row["k"] == k
                }
                epochs = sorted(values)
                if variant == "pretrained":
                    ax.axhline(values[0], linestyle="--", alpha=0.6, label=f"pretrained, k={k}")
                else:
                    ax.plot(
                        epochs, [values[e] for e in epochs], marker="o", label=f"{variant}, k={k}"
                    )
        ax.set(xlabel="Epoch", ylabel=label, title=f"{split}: {label}", ylim=(-0.02, 1.02))
        ax.grid(alpha=0.2)
        ax.legend(fontsize="small", loc="best")
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in variants:
        epoch_values = sorted({r["epoch"] for r in rows if r["variant"] == variant})
        for epoch in epoch_values:
            values = {r["k"]: r for r in rows if r["variant"] == variant and r["epoch"] == epoch}
            curve = [values[k] for k in sorted(values)]
            ax.plot(
                [r["recall_at_k"] for r in curve],
                [r["precision_at_k"] for r in curve],
                marker="o",
                label=f"{variant}, epoch {epoch}",
            )
    ax.set(
        xlabel="Recall@k (micro)",
        ylabel="Precision@k (micro)",
        title=f"{split}: precision vs. recall",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize="small", loc="best")
    fig.tight_layout()
    fig.savefig(output / "precision_recall.png", dpi=160)
    plt.close(fig)
