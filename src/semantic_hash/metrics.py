"""Leave-one-image-out retrieval, following the upstream Hits/P/R definitions."""

from __future__ import annotations

from collections import Counter

import numpy as np


def retrieval_metrics(
    embeddings: np.ndarray,
    series: list[str],
    ks: list[int],
    chunk_size: int = 256,
) -> list[dict]:
    """Rank all other labeled images by cosine similarity; aggregate query counts.

    Self is excluded. Precision and recall are micro-averaged, matching upstream.
    Memory is O(chunk_size * N + N * D), with no persisted NxN matrix. Ties use
    input index order. For small galleries, effective_k is capped at N - 1.
    """
    features = np.asarray(embeddings, dtype=np.float32)
    if features.ndim != 2 or len(features) != len(series) or len(series) < 2:
        raise ValueError("Expected N x D embeddings and N series labels, with N >= 2.")
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("k values must be positive integers.")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or np.any(norms < 1e-8):
        raise ValueError("Embeddings must be finite and nonzero.")
    features = features / norms
    counts = Counter(series)
    if any(count < 2 for count in counts.values()):
        raise ValueError("Each query must have at least one other image in its series.")
    labels = np.array(series)
    effective_ks = [min(k, len(series) - 1) for k in ks]
    hits = np.zeros(len(ks), dtype=np.int64)
    matched_queries = np.zeros(len(ks), dtype=np.int64)
    for start in range(0, len(series), chunk_size):
        stop = min(start + chunk_size, len(series))
        similarity = features[start:stop] @ features.T
        similarity[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        ranked = np.argsort(-similarity, axis=1, kind="stable")[:, : max(effective_ks)]
        positive = labels[ranked] == labels[start:stop, None]
        cumulative = np.cumsum(positive, axis=1)
        for index, k in enumerate(effective_ks):
            found = cumulative[:, k - 1]
            hits[index] += found.sum()
            matched_queries[index] += (found > 0).sum()
    recall_denom = sum(counts[s] - 1 for s in series)
    return [
        {
            "k": k,
            "effective_k": effective,
            "queries": len(series),
            "matched_queries": int(matched),
            "hits": int(hit),
            "precision_denom": len(series) * effective,
            "recall_denom": recall_denom,
            "accuracy_at_k": float(matched / len(series)),
            "precision_at_k": float(hit / (len(series) * effective)),
            "recall_at_k": float(hit / recall_denom),
        }
        for k, effective, hit, matched in zip(ks, effective_ks, hits, matched_queries, strict=True)
    ]
