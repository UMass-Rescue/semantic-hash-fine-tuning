import numpy as np
import pytest

from semantic_hash.metrics import retrieval_metrics


def test_hand_computed_retrieval_excludes_self_and_caps_k():
    features = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
    rows = retrieval_metrics(features, ["a", "a", "b", "b"], [1, 2, 20], chunk_size=1)
    assert [r["accuracy_at_k"] for r in rows] == [1, 1, 1]
    assert [r["precision_at_k"] for r in rows] == pytest.approx([1, 0.5, 1 / 3])
    assert [r["recall_at_k"] for r in rows] == [1, 1, 1]
    assert rows[-1]["effective_k"] == 3
    assert rows[-1]["hits"] == 4
    assert rows[-1]["recall_denom"] == 4


def test_chunked_metrics_match_independent_brute_force_and_micro_recall():
    features = np.random.default_rng(1).normal(size=(9, 7))
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    labels = ["a"] * 4 + ["b"] * 3 + ["c"] * 2
    for chunk_size in (1, 4, 100):
        rows = retrieval_metrics(features, labels, [1, 3, 8], chunk_size)
        for row in rows:
            total, matched = 0, 0
            for query in range(len(labels)):
                others = [i for i in range(len(labels)) if i != query]
                ranked = sorted(others, key=lambda i: (-np.dot(features[query], features[i]), i))
                hits = sum(labels[i] == labels[query] for i in ranked[: row["k"]])
                total += hits
                matched += hits > 0
            assert row["hits"] == total
            assert row["matched_queries"] == matched
            assert row["recall_denom"] == 20
            assert row["recall_at_k"] == total / 20


def test_ties_are_deterministic_and_self_is_never_a_hit():
    rows = retrieval_metrics(np.ones((4, 2)), ["a", "b", "a", "b"], [1])
    assert rows[0]["matched_queries"] == 1


@pytest.mark.parametrize(
    "features, labels",
    [
        (np.zeros((2, 3)), ["a", "a"]),
        (np.ones((2, 3)), ["a", "b"]),
        (np.full((2, 3), np.nan), ["a", "a"]),
    ],
)
def test_invalid_embeddings_and_singletons_fail(features, labels):
    with pytest.raises(ValueError):
        retrieval_metrics(features, labels, [1])
