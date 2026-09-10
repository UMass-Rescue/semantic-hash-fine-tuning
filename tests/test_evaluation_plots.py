from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from semantic_hash.evaluation import make_retrieval_plots
from semantic_hash.metrics import retrieval_metrics


def test_retrieval_plot_axes_use_k_and_recall_not_epoch(tmp_path, monkeypatch):
    features = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
    metrics = retrieval_metrics(features, ["a", "a", "b", "b"], [1, 2, 3, 20])
    rows = [
        {"variant": variant, "epoch": epoch, "checkpoint": checkpoint, **metric}
        for variant, epoch, checkpoint in (
            ("pretrained", 0, "pretrained"),
            ("series", 7, "best"),
            ("series", 7, "epoch_7"),
        )
        for metric in metrics
    ]
    plots = {}
    original_save = Figure.savefig

    def capture(figure, path, **kwargs):
        plots[Path(path).name] = figure.axes[0]
        original_save(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    legacy_plot = tmp_path / "accuracy_at_k.png"
    legacy_plot.write_bytes(b"obsolete epoch-based plot")
    paths = make_retrieval_plots(rows, tmp_path)
    assert not legacy_plot.exists()
    assert {path.name for path in paths.values()} == {
        "precision_recall.png",
        "hits_at_k.png",
        "precision_at_k.png",
    }
    assert all(path.stat().st_size > 0 for path in paths.values())
    for plot in plots.values():
        # Repeated saved/selected epochs must not become duplicate legend entries.
        assert len(plot.lines) == 2
    for filename in ("hits_at_k.png", "precision_at_k.png"):
        assert plots[filename].get_xlabel() == "k (images retrieved)"
        for line in plots[filename].lines:
            # k=20 is capped at 3 and should not add another point at depth 3.
            np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
    for line in plots["hits_at_k.png"].lines:
        np.testing.assert_array_equal(line.get_ydata(), [1, 1, 1])
    for line in plots["precision_at_k.png"].lines:
        np.testing.assert_allclose(line.get_ydata(), [1, 0.5, 1 / 3])
    assert plots["precision_recall.png"].get_xlabel() == "Recall (micro)"
    for line in plots["precision_recall.png"].lines:
        np.testing.assert_array_equal(line.get_xdata(), [1, 1, 1])
        np.testing.assert_allclose(line.get_ydata(), [1, 0.5, 1 / 3])
