# Tests in a dedicated conda environment

The environment used for local verification is named `semantic-hash-fine-tuning`. Activate it with:

```bash
conda activate semantic-hash-fine-tuning
python -m pytest -q
ruff check .
ruff format --check .
python -m build
```

To create a CPU-only test environment from scratch on Linux:

```bash
conda create -n semantic-hash-fine-tuning python=3.11 pip -y
conda activate semantic-hash-fine-tuning
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

`environment.yml` is the general setup recipe; pip resolves the default PyTorch build there. The CPU commands above install PyTorch first so the project reuses it. Use a CUDA-enabled build instead for GPU fine-tuning.

The test suite covers preparation, disjoint parent splits, clustering, audit manifests, cosine ranking/counts, target construction, unique-series batches, both losses and their gradients against OpenCLIP, training updates, frozen text parameters, checkpoint saving, exact CPU resume, cache reuse/invalidation, report generation, and embedding export. Integration tests use synthetic images and a tiny real OpenCLIP vision transformer. Only pretrained-model acquisition is replaced, so no model download or GPU is needed.

These checks validate the software workflow. They do not establish fine-tuning quality on a real series dataset, exercise CUDA mixed precision, or verify downloading the full default SigLIP2 model. Those require a separate experiment with real data and suitable hardware.
