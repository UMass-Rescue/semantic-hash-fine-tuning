import csv
import shutil
import subprocess
from pathlib import Path

import pytest

from semantic_hash import example
from semantic_hash.io import read_json


@pytest.fixture
def dataset_repository(dataset, tmp_path):
    _, cfg, _ = dataset
    repository = tmp_path / "source-repo"
    shutil.copytree(cfg.image_dir, repository / "series")
    subprocess.run(["git", "init", "--quiet", "--initial-branch=main", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "series"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Example Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "--quiet",
            "--no-gpg-sign",
            "-m",
            "Add synthetic series",
        ],
        check=True,
    )
    return repository.as_uri()


def example_args(tmp_path, *extra):
    return example.parser().parse_args(
        [
            "--download-dir",
            str(tmp_path / "download"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--eval-dir",
            str(tmp_path / "results"),
            "--epochs",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--batch-size",
            "3",
            *extra,
        ]
    )


def test_download_to_selected_folder_is_pinned_and_reusable(dataset_repository, tmp_path):
    destination = tmp_path / "download"
    destination.mkdir()  # Both a new folder and an existing empty folder are supported.
    metadata = example.download_dataset(destination, repo_url=dataset_repository)
    assert metadata["series_count"] == 12
    assert metadata["image_count"] == 48
    labels = read_json(destination / "labels.json")
    assert all(Path(p).is_file() for paths in labels.values() for p in paths)
    assert all(Path(p).is_relative_to(destination) for paths in labels.values() for p in paths)
    assert example.download_dataset(destination, repo_url=dataset_repository) == metadata
    (destination / "repository/series/untracked.png").write_bytes(b"changed dataset")
    with pytest.raises(ValueError, match="images changed"):
        example.download_dataset(destination, repo_url=dataset_repository)


def test_download_failure_leaves_no_partial_checkout(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("Network unavailable")

    monkeypatch.setattr(example, "_git", fail)
    with pytest.raises(ValueError, match="Network unavailable"):
        example.download_dataset(tmp_path / "download")
    assert not (tmp_path / "download").exists()
    assert not list(tmp_path.glob(".download-download-*"))


def test_example_selects_before_test_evaluation_and_resumes(
    dataset_repository,
    tmp_path,
    tiny_openclip,
    monkeypatch,
):
    import semantic_hash.evaluation as evaluation

    args = example_args(tmp_path, "--download-only")
    config_path = example.run_example(args, repo_url=dataset_repository)
    cfg = read_json(config_path)
    assert cfg["checkpoint_dir"] == str(args.checkpoint_dir)
    assert cfg["evaluation_dir"] == str(args.eval_dir)
    assert cfg["clustering"]["enabled"] is False
    assert not (args.checkpoint_dir / "series").exists()
    original_evaluate = evaluation.evaluate
    calls = []

    def evaluate_after_selection(cfg, **kwargs):
        selection = read_json(args.eval_dir / "selected_checkpoint.json")
        assert Path(selection["checkpoint"]).is_file()
        assert selection["epoch"] >= 1
        assert 0 <= selection["validation_hits_at_1"] <= 1
        calls.append(kwargs)
        return original_evaluate(cfg, **kwargs)

    monkeypatch.setattr(evaluation, "evaluate", evaluate_after_selection)
    args.download_only = False
    metrics_path = example.run_example(args, repo_url=dataset_repository)
    assert calls == [{"split": "test", "variant": "series"}]
    assert metrics_path == args.eval_dir / "test/metrics.csv"
    with metrics_path.open() as handle:
        metrics = list(csv.DictReader(handle))
    assert len(metrics) == 8
    assert {row["variant"] for row in metrics} == {"pretrained", "series"}
    result = read_json(args.eval_dir / "result.json")
    assert sum(split["series"] for split in result["splits"].values()) == 12
    assert (args.checkpoint_dir / "series/best.pt").is_file()
    assert (args.checkpoint_dir / "series/latest.pt").is_file()
    assert not (args.checkpoint_dir / "workflow/checkpoints").exists()
    assert not (args.checkpoint_dir / "workflow/evaluation").exists()
    assert (args.eval_dir / "val/plots/accuracy_at_k.png").is_file()

    def no_model(*args, **kwargs):
        pytest.fail("Completed example runs must resume without creating a model.")

    monkeypatch.setattr("open_clip.create_model_and_transforms", no_model)
    assert example.run_example(args, repo_url=dataset_repository) == metrics_path


def test_example_refuses_unrelated_or_overlapping_outputs(tmp_path):
    args = example_args(tmp_path)
    args.eval_dir = args.download_dir / "results"
    with pytest.raises(ValueError, match="must not overlap"):
        example.run_example(args)
    args = example_args(tmp_path)
    args.checkpoint_dir.mkdir()
    original = args.checkpoint_dir / "keep.txt"
    original.write_text("existing files")
    with pytest.raises(ValueError, match="new or empty"):
        example.run_example(args)
    assert original.read_text() == "existing files"
    assert not args.download_dir.exists()


def test_changed_run_config_is_rejected_before_training(dataset_repository, tmp_path):
    args = example_args(tmp_path, "--download-only")
    config_path = example.run_example(args, repo_url=dataset_repository)
    original = config_path.read_text()
    args.epochs += 1
    with pytest.raises(ValueError, match="different example run"):
        example.run_example(args, repo_url=dataset_repository)
    assert config_path.read_text() == original
