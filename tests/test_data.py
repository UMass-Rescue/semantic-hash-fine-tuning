import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from semantic_hash.config import load_config
from semantic_hash.data import clean_labels, cluster_labels, split_series, validate_labels


def test_paths_resolve_relative_to_config_and_image_root(dataset, tmp_path, monkeypatch):
    path, cfg, labels = dataset
    monkeypatch.chdir(tmp_path.parent)
    assert load_config(path).image_dir == cfg.image_dir
    relative = {
        name: [str(Path(p).relative_to(cfg.image_dir)) for p in paths]
        for name, paths in labels.items()
    }
    cfg.labels_json.write_text(json.dumps(relative))
    assert validate_labels(cfg.labels_json, cfg.image_dir) == labels


@pytest.mark.parametrize(
    "change",
    [
        {"unknown": True},
        {"training": {"batch_size": True}},
        {"training": {"learning_rate": float("nan")}},
        {"evaluation": {"k": [5, 1]}},
        {"split_fractions": [0.8, 0.2]},
        {"clustering": {"enabled": "false"}},
        {"model": {"name": "ViT-B-32"}},
    ],
)
def test_bad_config_fails_early(dataset, change):
    path, _, _ = dataset
    raw = json.loads(path.read_text())
    raw.update(change)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_config(path)


def test_overlap_and_outside_root_rejected(dataset):
    _, cfg, labels = dataset
    labels["1"].append(labels["0"][0])
    cfg.labels_json.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="multiple series"):
        validate_labels(cfg.labels_json, cfg.image_dir)
    cfg.labels_json.write_text(json.dumps({"s": ["../outside.png"]}))
    with pytest.raises(ValueError, match="outside image_dir"):
        validate_labels(cfg.labels_json, cfg.image_dir)


def test_cleaning_audits_corruption_and_exact_duplicates(dataset):
    _, cfg, labels = dataset
    bad = cfg.image_dir / "broken.png"
    bad.write_text("not an image")
    duplicate = cfg.image_dir / "duplicate.png"
    with Image.open(labels["0"][0]) as im:
        im.save(duplicate)
    labels["0"].extend([str(bad), str(duplicate)])
    result, report = clean_labels(labels, cfg)
    assert len(result["0"]) == 4
    assert report["decode_failures"][0]["path"] == str(bad)
    assert report["duplicates"][0]["reason"] == "identical decoded pixels"


def test_whole_series_split_is_reproducible_and_independent_of_json_order(dataset):
    _, cfg, labels = dataset
    result = split_series(labels, cfg.split_fractions, cfg.seed)
    assert result == split_series(
        dict(reversed(list(labels.items()))), cfg.split_fractions, cfg.seed
    )
    assert set().union(*(set(v) for v in result.values())) == set(labels)
    assert sum(map(len, result.values())) == len(labels)
    for mapping in result.values():
        assert len(mapping) >= 2
    assert result != split_series(labels, cfg.split_fractions, cfg.seed + 1)


def test_clustering_preserves_parent_mapping_and_drops_small_clusters(dataset):
    _, cfg, _ = dataset
    cfg = replace(cfg, clustering=replace(cfg.clustering, distance_threshold=0.1))
    labels = {"a": ["a1", "a2", "a3", "a4", "a5"], "a_subseries_0": ["b1", "b2"]}
    features = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [-1, 0], [1, 0], [1, 0]], dtype=np.float32)
    result, report = cluster_labels(labels, features, cfg)
    assert len(result) == 3
    assert report["dropped_images"] == ["a5"]
    assert set(report["parents"].values()) == set(labels)
    assert sorted(sum(result.values(), [])) == ["a1", "a2", "a3", "a4", "b1", "b2"]
