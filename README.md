# Fine-tuning for Semantic Hashing

Fine-tune a CLIP image encoder to recognize **image series**: photos of a similar subject, taken around the same time and place. A semantic hash here is a normalized, floating-point image embedding. Match a query to other images with cosine similarity.

## Quick start

Use Python 3.11. From this repository:

```bash
conda env create -f environment.yml
conda activate semantic-hash-fine-tuning
```

If you already created the environment, activate it and run `python -m pip install -e '.[dev]'`. A regular Python virtual environment also works with the same pip command. The tests run on CPU; practical training of the default model calls for a CUDA GPU. Install a PyTorch build compatible with your GPU before installing this package if your machine requires a particular CUDA version.

Arrange your images in one folder per series:

```text
images/
├── birthday-party/
│   ├── 001.jpg
│   └── 002.jpg
└── afternoon-at-the-park/
    ├── 003.jpg
    └── 004.jpg
```

Generate labels, then copy and edit the four-field config:

```bash
semhash labels --image-dir /absolute/path/to/images --output labels.json
cp finetune_config.example.json finetune_config.json
```

```json
{
  "dataset_name": "my-series",
  "image_dir": "/absolute/path/to/images",
  "labels_json": "labels.json",
  "output_dir": "runs/my-series"
}
```

Run the complete experiment:

```bash
semhash run --config finetune_config.json
```

Rerun that command after interruption. It reuses prepared data, resumes training from the last completed epoch, skips completed training, and reuses matching evaluation results. Keep the same config and inputs; use a new `output_dir` for a different experiment. Run only one process against a given output directory at a time.

The first preparation run downloads pretrained weights. The default experiment uses **`hf-hub:timm/ViT-SO400M-16-SigLIP2-384`**, **9 epochs per variant**, a training batch size of **8**, and a learning rate of **1e-5**. Full model weights and optimizer state require substantial RAM and disk space. See [configuration](docs/configuration.md) for smaller models, memory controls, and all defaults.

## Run the example dataset

The [UMass image-series dataset](https://github.com/UMass-Rescue/image-series-dataset) can be downloaded and used for a complete experiment with one script. After activating the conda environment, run from this repository:

```bash
python scripts/run_example.py \
  --download-dir /path/to/example-images \
  --checkpoint-dir /path/to/example-checkpoints \
  --eval-dir /path/to/example-results
```

The three folders can be new or empty and must be separate (none may contain another). Git must be installed. The script downloads the dataset into `<download-dir>/repository/series`, generates labels, checks images, and splits **whole series** into train/validation/test sets with seed 42 and 70/15/15 proportions. It runs the original-series training variant for 9 epochs, selects the checkpoint with the highest **validation Hits@1**, and evaluates that checkpoint and the pretrained baseline on the held-out **test** images.

The main outputs are:

| Output | Location |
| --- | --- |
| Selected checkpoint | `<checkpoint-dir>/series/best.pt` |
| Resume checkpoint | `<checkpoint-dir>/series/latest.pt` |
| Generated experiment config | `<checkpoint-dir>/example_config.json` |
| Split labels and image audit | `<checkpoint-dir>/workflow/data/` |
| Selected epoch, validation score, and checkpoint hash | `<eval-dir>/selected_checkpoint.json` |
| Final test metrics, including baseline | `<eval-dir>/test/metrics.csv` |
| Precision versus recall curve | `<eval-dir>/test/plots/precision_recall.png` |
| Hits@k versus k | `<eval-dir>/test/plots/hits_at_k.png` |
| Precision@k versus k | `<eval-dir>/test/plots/precision_at_k.png` |
| Validation learning curves | `<eval-dir>/val/plots/` |
| Result summary and split counts | `<eval-dir>/result.json` |

Each test plot compares the pretrained baseline with the selected fine-tuned checkpoint. Points use the configured retrieval depths (1, 5, 10, and 20 by default). The k axes show the actual number of images retrieved; depths capped to the same gallery size are plotted once. The precision/recall curve uses recall on the horizontal axis and precision on the vertical axis. `result.json` includes paths to all three PNG files. Repeating the example command regenerates plots from cached metrics without retraining a completed run.

Repeat the **same command** to reuse the recorded dataset commit and resume unfinished training. A changed configuration requires new checkpoint and evaluation folders; the download folder can be reused. Test results never choose the checkpoint. This example trains on original series only; use the configurable workflow above to compare sub-series training as well.

Add `--download-only` to download images and generate the config without loading a model. To check the process using a smaller CLIP model and fewer epochs:

```bash
python scripts/run_example.py \
  --download-dir /path/to/example-images \
  --checkpoint-dir /path/to/small-model-checkpoints \
  --eval-dir /path/to/small-model-results \
  --model ViT-B-32 --pretrained openai --epochs 3
```

The loss defaults to SigLIP for a SigLIP model name and CLIP otherwise. `--device`, `--batch-size`, `--embedding-batch-size`, `--workers`, `--seed`, and `--grad-checkpointing` are available; see `python scripts/run_example.py --help`. The equivalent installed module, `python -m semantic_hash.example`, works from any directory.

## Input requirements

You can supply your own labels instead of generating them:

```json
{
  "birthday-party": ["birthday-party/001.jpg", "birthday-party/002.jpg"],
  "afternoon-at-the-park": ["afternoon-at-the-park/003.jpg", "afternoon-at-the-park/004.jpg"]
}
```

Config paths are relative to the config file. Image paths inside labels are relative to `image_dir`, or absolute paths beneath it. Only labeled images participate. Paths shared by different series are rejected.

By default, all generated files live beneath `output_dir`. Optional `checkpoint_dir` and `evaluation_dir` config fields place model checkpoints and evaluation outputs elsewhere; their paths also resolve relative to the config file.

Use at least **six usable series**, each with at least two distinct, decodable images. The split reserves at least two series each for training, validation, and testing; larger datasets follow the default 70/15/15 proportions. Useful experiments should have substantially more series than this minimum.

Preparation checks decoding, removes images with identical decoded pixels, and drops series left with fewer than two images. It records failures, warnings, removals, and affected series in `data/image_audit.json`. Optional perceptual de-duplication is available. Source images are never modified.

Sub-series training is enabled by default. It splits each training series into visually similar clusters using the pretrained embeddings, average-linkage cosine distance below 0.3, and at least 5 images per retained cluster. If fewer than two clusters survive, preparation explains how to adjust the settings. For smaller datasets or a single original-series experiment, add:

```json
"clustering": {"enabled": false}
```

## Run individual stages

```bash
semhash prepare  --config finetune_config.json
semhash train    --config finetune_config.json
semhash evaluate --config finetune_config.json
```

Useful options:

```bash
# Resume training, including after a mid-epoch interruption.
semhash train --config finetune_config.json --resume

# Train/evaluate only one enabled variant.
semhash train --config finetune_config.json --variant series
semhash evaluate --config finetune_config.json --variant series

# Evaluate the pretrained baseline before training.
semhash evaluate --config finetune_config.json --baseline-only

# Inspect validation retrieval or all saved epochs.
semhash evaluate --config finetune_config.json --split val --all-checkpoints

# Recompute evaluation instead of using its cache.
semhash evaluate --config finetune_config.json --recompute
```

All commands work outside the repository after installation. `python -m semantic_hash` is equivalent to `semhash`. Use `semhash --help` or `semhash --verbose …` for detailed errors.

## Read the results

```text
runs/my-series/
├── config.json                       # Resolved config and defaults
├── data/                             # Splits, image audit, cached targets, manifest
├── checkpoints/
│   ├── series/
│   │   ├── best.pt                   # Highest validation Hits@1
│   │   ├── latest.pt                 # Last complete epoch + optimizer for resume
│   │   ├── epoch_3.pt                # Also epochs 6 and 9 by default
│   │   └── history.json              # Training loss and validation retrieval per epoch
│   └── subseries/                    # Same files for the independent sub-series run
└── evaluation/
    ├── test/
    │   ├── metrics.csv               # Baseline and selected checkpoints
    │   ├── protocol.json             # Evaluation population and metric definitions
    │   ├── plots/                    # Hits@k, Precision@k, precision vs. recall
    │   └── cache/                    # Metrics with input/checkpoint fingerprints
    └── val/plots/                    # Validation learning curves
```

For each image in a split, evaluation removes that image from the gallery and ranks all other images in the **same split** by cosine similarity. A positive match belongs to the query's original series. All models use exactly the same queries, gallery, and labels.

| CSV metric | Meaning |
| --- | --- |
| `accuracy_at_k` (Hits@k) | Fraction of queries with at least one same-series image in the top k. |
| `precision_at_k` | Same-series retrieved images divided by all retrieved images, pooled over queries. |
| `recall_at_k` | Same-series retrieved images divided by all available same-series positives, pooled over queries. |
| `effective_k` | Actual number retrieved: `min(k, gallery_size - 1)`. |

The CSV includes raw counts and denominators. Precision and recall are **micro-averaged**; larger series contribute more positive pairs to recall. The default k values are 1, 5, 10, and 20. Plots use those k values. Retrieval compares individual images, without aggregating a gallery series into a centroid.

Both variants start from the same pretrained model. Each gets its own best checkpoint, selected by validation Hits@1; a tie keeps the earlier epoch. Test scores do not choose checkpoints. `--all-checkpoints` can also report test learning curves for diagnostics, but choosing an epoch from those curves would use the test set for model selection. Fine-tuning is not assumed to improve retrieval: compare each selected model with the baseline.

## Use the fine-tuned semantic hashes

```bash
semhash embed --config finetune_config.json \
  --checkpoint runs/my-series/checkpoints/series/best.pt \
  --input-dir /path/to/new/images \
  --output runs/new-embeddings
```

This writes aligned `embeddings.npy` and `image_paths.json` files, plus model metadata. Every row is a unit-length float32 vector. Omit `--checkpoint` to export pretrained embeddings. Export uses the model settings from the config; the original training images and labels do not need to remain available.

```python
import numpy as np

embeddings = np.load("runs/new-embeddings/embeddings.npy")
query_index = 0
similarity = embeddings @ embeddings[query_index]
similarity[query_index] = -np.inf
nearest = np.argsort(-similarity)[:5]
```

Embed query and gallery images with the same checkpoint and preprocessing. Checkpoints also contain a full OpenCLIP `state_dict` and can be loaded into the matching model with `open_clip.load_checkpoint`.

## Development

```bash
conda activate semantic-hash-fine-tuning
python -m pytest -q
ruff check .
ruff format --check .
python -m build
```

Tests use a tiny real OpenCLIP architecture and synthetic images, without pretrained downloads or a GPU. They check loss/gradient agreement with OpenCLIP, retrieval counts, split isolation, training, checkpoint selection, interruption/resume, caching, and embedding export. See [testing](docs/testing.md) for a CPU-only conda setup and [provenance](docs/provenance.md) for the extracted components and intentional changes.
