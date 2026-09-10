import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from semantic_hash.cli import main
from semantic_hash.data import assert_prepared, prepare
from semantic_hash.io import read_json
from semantic_hash.training import train


def test_end_to_end_and_cached_rerun(dataset, tiny_openclip, monkeypatch, tmp_path):
    config_path, cfg, _ = dataset
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    assert main(["run", "--config", str(config_path)]) == 0
    manifest = assert_prepared(cfg)
    assert manifest["fingerprint"]
    with (cfg.output_dir / "evaluation/test/metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert {row["variant"] for row in rows} == {"pretrained", "series", "subseries"}
    for variant in cfg.variants:
        latest = torch.load(cfg.output_dir / f"checkpoints/{variant}/latest.pt", weights_only=True)
        assert latest["epoch"] == 2
        assert len(latest["history"]) == 3
        best = torch.load(cfg.output_dir / f"checkpoints/{variant}/best.pt", weights_only=True)
        trained = latest["history"][1:]
        expected_best = max(trained, key=lambda h: h["metrics"][0]["accuracy_at_k"])
        assert best["epoch"] == expected_best["epoch"]
        initial, _, _ = tiny_openclip()
        assert any(
            not torch.equal(value, initial.state_dict()[key])
            for key, value in latest["state_dict"].items()
            if key.startswith("visual.")
        )
        assert all(
            torch.equal(value, initial.state_dict()[key])
            for key, value in latest["state_dict"].items()
            if not key.startswith("visual.") and not key.startswith("logit_")
        )
    assert list((cfg.output_dir / "evaluation/test/plots").glob("*.png"))
    assert main(["evaluate", "--config", str(config_path), "--all-checkpoints"]) == 0
    with (cfg.output_dir / "evaluation/test/metrics.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 21
    export_dir = tmp_path / "export"
    assert (
        main(
            [
                "embed",
                "--config",
                str(config_path),
                "--checkpoint",
                str(cfg.output_dir / "checkpoints/series/best.pt"),
                "--input-dir",
                str(cfg.image_dir),
                "--output",
                str(export_dir),
            ]
        )
        == 0
    )
    embeddings = np.load(export_dir / "embeddings.npy")
    assert embeddings.shape == (48, 16)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-6)

    def no_model(*args, **kwargs):
        pytest.fail("A completed rerun should reuse preparation, training, and evaluation.")

    monkeypatch.setattr("open_clip.create_model_and_transforms", no_model)
    assert main(["run", "--config", str(config_path)]) == 0
    # Replacing an artifact must invalidate the run, even though metrics are cached.
    (cfg.output_dir / "data/test_series.json").write_text("{}")
    assert main(["evaluate", "--config", str(config_path)]) == 1


def test_single_variant_and_baseline_only(dataset, tiny_openclip):
    import json

    config_path, cfg, _ = dataset
    raw = read_json(config_path)
    raw["clustering"]["enabled"] = False
    config_path.write_text(json.dumps(raw))
    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["evaluate", "--config", str(config_path), "--baseline-only"]) == 0
    assert not (cfg.output_dir / "checkpoints").exists()
    assert main(["run", "--config", str(config_path)]) == 0
    assert not (cfg.output_dir / "checkpoints/subseries").exists()
    # Export remains usable after the training labels are moved elsewhere.
    cfg.labels_json.rename(cfg.labels_json.with_suffix(".moved"))
    assert (
        main(
            [
                "embed",
                "--config",
                str(config_path),
                "--input-dir",
                str(cfg.image_dir),
                "--checkpoint",
                str(cfg.output_dir / "checkpoints/series/best.pt"),
                "--output",
                str(cfg.output_dir / "export"),
            ]
        )
        == 0
    )


def test_separate_checkpoint_and_evaluation_folders(dataset, tiny_openclip):
    import json

    config_path, cfg, _ = dataset
    raw = read_json(config_path)
    raw.update(checkpoint_dir="saved-models", evaluation_dir="final-results")
    raw["clustering"]["enabled"] = False
    config_path.write_text(json.dumps(raw))
    assert main(["run", "--config", str(config_path)]) == 0
    checkpoints = config_path.parent / "saved-models"
    evaluation = config_path.parent / "final-results"
    assert (checkpoints / "series/best.pt").is_file()
    assert (checkpoints / "series/latest.pt").is_file()
    assert (evaluation / "test/metrics.csv").is_file()
    assert (evaluation / "val/plots/accuracy_at_k.png").is_file()
    assert not (cfg.output_dir / "checkpoints").exists()
    assert not (cfg.output_dir / "evaluation").exists()
    assert main(["run", "--config", str(config_path)]) == 0


def test_interrupted_training_resumes_same_result(dataset, tiny_openclip, monkeypatch):
    import semantic_hash.training as training

    _, cfg, _ = dataset
    prepare(cfg)
    train(cfg, variant="series")
    uninterrupted = torch.load(cfg.output_dir / "checkpoints/series/latest.pt", weights_only=True)
    original_save = training.save_checkpoint

    def interrupt_after_first_epoch(path, checkpoint):
        original_save(path, checkpoint)
        if path.name == "latest.pt" and checkpoint["epoch"] == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(training, "save_checkpoint", interrupt_after_first_epoch)
    with pytest.raises(KeyboardInterrupt):
        train(cfg, variant="subseries")
    assert (
        torch.load(cfg.output_dir / "checkpoints/subseries/latest.pt", weights_only=True)["epoch"]
        == 1
    )
    monkeypatch.setattr(training, "save_checkpoint", original_save)
    train(cfg, variant="subseries", resume=True)
    resumed = torch.load(cfg.output_dir / "checkpoints/subseries/latest.pt", weights_only=True)
    assert resumed["epoch"] == uninterrupted["epoch"]
    # Here clustering retains each parent intact, so both runs have the same training data.
    # Compare a separate uninterrupted sub-series rerun to avoid label-sort differences.
    latest_path = cfg.output_dir / "checkpoints/subseries/latest.pt"
    latest_path.rename(latest_path.with_name("interrupted_result.pt"))
    train(cfg, variant="subseries")
    fresh = torch.load(latest_path, weights_only=True)
    for key in fresh["state_dict"]:
        torch.testing.assert_close(
            resumed["state_dict"][key], fresh["state_dict"][key], rtol=0, atol=0
        )
    assert resumed["history"] == fresh["history"]


def test_test_images_are_not_used_for_preparation_or_training(dataset, tiny_openclip, monkeypatch):
    import semantic_hash.model as model_module

    _, cfg, _ = dataset
    seen = set()
    original_getitem = model_module.ImageDataset.__getitem__

    def record(self, index):
        seen.add(self.paths[index])
        return original_getitem(self, index)

    monkeypatch.setattr(model_module.ImageDataset, "__getitem__", record)
    prepare(cfg)
    train(cfg, variant="series")
    held_out = set(sum(read_json(cfg.output_dir / "data/test_series.json").values(), []))
    assert not seen & held_out
    parents = set(read_json(cfg.output_dir / "data/train_series.json"))
    assert (
        set(read_json(cfg.output_dir / "data/train_clustering.json")["parents"].values()) <= parents
    )
    assert not (cfg.output_dir / "data/val_clustering.json").exists()
    # An input edit is caught before any later embedding or training starts.
    first_test_image = Path(next(iter(held_out)))
    first_test_image.write_bytes(b"modified")
    with pytest.raises(ValueError, match="image changed"):
        assert_prepared(cfg)
