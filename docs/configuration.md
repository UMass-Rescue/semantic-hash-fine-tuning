# Configuration

Start with the four required fields in `finetune_config.example.json`. Extra fields are optional; their defaults are listed below. Unknown keys, invalid numbers, missing inputs, and inconsistent settings fail before a model is downloaded.

| Setting | Default | Purpose |
| --- | --- | --- |
| `checkpoint_dir` | `<output_dir>/checkpoints` | Optional separate root for model checkpoints. |
| `evaluation_dir` | `<output_dir>/evaluation` | Optional separate root for evaluation CSVs, plots, and caches. |
| `model.name` | `hf-hub:timm/ViT-SO400M-16-SigLIP2-384` | OpenCLIP model identifier. |
| `model.pretrained` | `null` | Required pretrained tag/path when using a built-in model name. |
| `seed` | `42` | Parent-series split, training batches, and epoch augmentation seeds. |
| `device` | `auto` | CUDA when available, otherwise CPU; accepts explicit PyTorch devices. |
| `workers` | `4` | Image-loading workers; set to 0 for debugging. |
| `embedding_batch_size` | `32` | Preparation, validation, test, and export inference batch size. |
| `split_fractions` | `[0.7, 0.15, 0.15]` | Train/validation/test proportions by original series count. |
| `deduplicate_images` | `false` | Also remove perceptually similar images globally before splitting. |
| `phash_threshold` | `4` | Maximum Hamming distance between 64-bit pHashes (0–64). |
| `clustering.enabled` | `true` | Run both original-series and clustered-sub-series experiments. |
| `clustering.distance_threshold` | `0.3` | Average-linkage cosine distance at which training clusters stop merging. |
| `clustering.min_samples` | `5` | Minimum images in a retained training cluster. |
| `training.epochs` | `9` | Epochs per variant; any positive integer. |
| `training.batch_size` | `8` | Unique series per training batch; minimum 2. |
| `training.learning_rate` | `1e-5` | AdamW peak learning rate. |
| `training.weight_decay` | `0.1` | Weight decay, excluding gains, biases, and normalization parameters. |
| `training.warmup_steps` | `100` | Linear warmup followed by cosine decay. |
| `training.save_every` | `3` | Keep numbered checkpoints at this interval and at the final epoch. |
| `training.loss` | `siglip` | `siglip` or symmetric cross-entropy `clip`. |
| `training.precision` | `amp` | CUDA float16 autocast with gradient scaling; CPU uses float32. |
| `training.grad_checkpointing` | `false` | Recompute activations to reduce training memory. |
| `evaluation.k` | `[1, 5, 10, 20]` | Sorted unique positive retrieval depths. |
| `evaluation.query_chunk_size` | `256` | Queries ranked together; reduce to lower evaluation RAM usage. |

All config-relative paths are anchored to the config file's directory. Relative paths in labels are anchored to `image_dir`. Tildes are expanded. Resolved absolute paths and defaults are written to the run's `config.json`.

## Training details

The target for a series is the normalized mean of its normalized pretrained image embeddings. These targets remain fixed throughout training and resume. A training batch contains one image from each of several different series, so other batch targets act as negatives. The visual encoder and logit scale/bias are trainable; the text tower is frozen and is not called. The original-series and sub-series runs initialize independently from the pretrained model.

Batch size is reduced, with a warning, if the training split has fewer series than requested. The sampler prioritizes series with the most remaining images and samples each image at most once per epoch. Incomplete batches are dropped, matching upstream; highly unbalanced series can leave images unused in an epoch. The startup log reports the number of images actually used. Epoch seeds change which images are sampled.

For ViT models, AdamW uses beta values `(0.9, 0.98)` and epsilon `1e-6`, as upstream. Other model names use `(0.9, 0.999)` and `1e-8`. Warmup is capped at `total_steps - 1` for very short runs. Validation computes image retrieval after every epoch against original validation-series labels for both variants.

## A smaller CLIP experiment

Merge these fields into your config:

```json
{
  "model": {"name": "ViT-B-32", "pretrained": "openai"},
  "training": {"loss": "clip", "epochs": 3, "batch_size": 8},
  "clustering": {"enabled": false}
}
```

Use `training.loss="clip"` for a CLIP model; SigLIP training requires a model with a logit bias. The implementation always requires pretrained weights and does not silently train from random initialization.

## Memory and reproducibility

Reduce `training.batch_size` for training GPU memory, and `embedding_batch_size` for inference GPU memory. Enable gradient checkpointing if needed. The full text tower is retained for OpenCLIP checkpoint compatibility, though it is frozen. Optimizer state and several full-model snapshots can consume many gigabytes on disk.

Retrieval keeps a chunk-by-gallery similarity matrix in RAM, with float32 cosine values and stable index-order tie-breaking. It still performs all query/gallery comparisons. Clustering computes one square distance matrix per training parent series. Perceptual de-duplication compares against previously retained hashes and can be slow on very large collections.

Preparation saves artifact SHA-256 hashes, the input-label hash, and retained-image sizes and modification times. Later stages reject changed configs, prepared artifacts, or images with changed size/mtime. Image content changes that preserve both size and mtime are not detected. Keep source images immutable during an experiment. Dependency versions and GPU kernels can affect results across machines; the CPU tests verify exact resume equivalence on one machine. Use the same dependency versions and pretrained weights when reproducing an experiment.

`latest.pt` includes optimizer/scaler state and history. Resume starts at the next complete epoch and restores the same frozen targets from preparation. A mid-epoch interruption repeats that epoch. Checkpoints and JSON outputs are replaced atomically. There is no multi-process locking or distributed-training support.

The evaluation cache includes the prepared-data fingerprint, checkpoint content hash, split, k values, and metric version. It caches aggregate metrics, not image embeddings or distances. `--recompute` forces new embeddings and metrics; otherwise plots and CSVs can be regenerated from matching cached results.

Final test evaluation writes three separate files under `test/plots/`: `precision_recall.png` (precision versus recall), `hits_at_k.png` (Hits@k versus k), and `precision_at_k.png` (Precision@k versus k). Each model/checkpoint epoch has its own curve. Points come from `evaluation.k`; the k axes use `effective_k` so capped duplicate depths appear once. Validation learning plots under `val/plots/` show metrics across training epochs.

Older runs may contain `test/plots/accuracy_at_k.png`, an epoch-based plot with one point per k when only the selected epoch was evaluated. Rerunning evaluation replaces it with `hits_at_k.png` and removes the obsolete file; cached metrics can be reused without retraining.
