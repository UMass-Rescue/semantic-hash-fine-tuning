import numpy as np
import pytest
import torch
import torch.nn.functional as F

from semantic_hash.training import UniqueSeriesBatchSampler, contrastive_loss, series_targets


@pytest.mark.parametrize("sizes,batch_size", [([10, 2, 2], 2), ([3, 3, 3, 3], 3), ([1, 1], 2)])
def test_batches_have_unique_series_and_no_repeated_images(sizes, batch_size):
    labels = [str(i) for i, size in enumerate(sizes) for _ in range(size)]
    sampler = UniqueSeriesBatchSampler(labels, batch_size, 3)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert batches == list(sampler)
    indices = sum(batches, [])
    assert len(indices) == len(set(indices))
    assert all(len({labels[i] for i in b}) == batch_size for b in batches)
    sampler.epoch = 1
    assert len(list(sampler)) == len(sampler)


@pytest.mark.parametrize("kind", ["siglip", "clip"])
def test_loss_and_gradients_match_openclip(kind):
    from open_clip.loss import ClipLoss, SigLipLoss

    torch.manual_seed(2)
    features = F.normalize(torch.randn(4, 7), dim=-1).requires_grad_()
    targets = F.normalize(torch.randn(4, 7), dim=-1)
    scale = torch.tensor(2.0, requires_grad=True)
    bias = torch.tensor(-1.0, requires_grad=True)
    actual = contrastive_loss(features, targets, scale, bias, kind)
    expected = (
        SigLipLoss()(features, targets, scale, bias)
        if kind == "siglip"
        else ClipLoss()(features, targets, scale)
    )
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, features, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, features)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_fixed_targets_are_normalized_means():
    targets = series_targets(
        {"s": ["a", "b"]}, ["a", "b"], np.array([[2, 0], [0, 3]], dtype=np.float32)
    )
    torch.testing.assert_close(targets["s"], torch.tensor([2**-0.5, 2**-0.5]))
    assert not targets["s"].requires_grad
