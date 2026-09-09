"""Consistent RGB decoding for preflight, training, and inference."""

from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as source:
        source.load()
        return ImageOps.exif_transpose(source).convert("RGB")


def discover_series(image_dir: Path) -> dict[str, list[str]]:
    """Each immediate child folder is a series; images within it are recursive."""
    image_dir = image_dir.expanduser().resolve()
    if not image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    labels = {}
    for directory in sorted(image_dir.iterdir()):
        if directory.is_dir():
            paths = sorted(
                str(p.resolve())
                for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if paths:
                labels[directory.name] = paths
    if not labels:
        raise ValueError(f"No series folders containing images found under {image_dir}")
    return labels
