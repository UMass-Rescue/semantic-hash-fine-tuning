import json

import numpy as np
import pytest
import torch
from PIL import Image

from semantic_hash.config import load_config


@pytest.fixture(autouse=True, scope="session")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def dataset(tmp_path):
    images = tmp_path / "images"
    labels = {}
    rng = np.random.default_rng(17)
    for series in range(12):
        folder = images / str(series)
        folder.mkdir(parents=True)
        paths = []
        for index in range(4):
            path = folder / f"{index}.png"
            pixels = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            paths.append(str(path))
        labels[str(series)] = paths
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps(labels))
    raw = {
        "dataset_name": "test",
        "image_dir": "images",
        "labels_json": "labels.json",
        "output_dir": "output",
        "device": "cpu",
        "workers": 0,
        "embedding_batch_size": 8,
        "training": {"epochs": 2, "batch_size": 3, "save_every": 1},
        "clustering": {"distance_threshold": 2.0, "min_samples": 2},
        "evaluation": {"k": [1, 3, 20], "query_chunk_size": 3},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw))
    return config_path, load_config(config_path), labels


@pytest.fixture
def tiny_openclip(monkeypatch):
    """Exercise a real OpenCLIP encoder; substitute only pretrained model acquisition."""
    import open_clip
    from open_clip.model import CLIP, CLIPTextCfg, CLIPVisionCfg
    from torchvision import transforms

    def create(*args, **kwargs):
        with torch.random.fork_rng():
            torch.manual_seed(123)
            model = CLIP(
                embed_dim=16,
                vision_cfg=CLIPVisionCfg(
                    layers=1, width=32, head_width=16, patch_size=8, image_size=16
                ),
                text_cfg=CLIPTextCfg(context_length=4, vocab_size=32, width=32, heads=2, layers=1),
                init_logit_scale=np.log(10),
                init_logit_bias=-10,
            )
        evaluate = transforms.ToTensor()
        train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        return model, train, evaluate

    monkeypatch.setattr(open_clip, "create_model_and_transforms", create)
    return create
