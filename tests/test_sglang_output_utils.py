import pytest

torch = pytest.importorskip("torch")

from parallax.sglang.output_utils import gather_sampled_token_probs


def test_gather_sampled_token_probs_accepts_1d_token_ids():
    logits = torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], dtype=torch.float32)
    token_ids = torch.tensor([2, 0], dtype=torch.int64)

    gathered = gather_sampled_token_probs(logits, token_ids)

    assert torch.equal(gathered, torch.tensor([0.3, 0.9], dtype=torch.float32))


def test_gather_sampled_token_probs_accepts_2d_token_ids():
    logits = torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], dtype=torch.float32)
    token_ids = torch.tensor([[1], [2]], dtype=torch.int64)

    gathered = gather_sampled_token_probs(logits, token_ids)

    assert torch.equal(gathered, torch.tensor([0.2, 0.7], dtype=torch.float32))


def test_gather_sampled_token_probs_rejects_incompatible_ranks():
    logits = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    token_ids = torch.tensor([[[1]]], dtype=torch.int64)

    with pytest.raises(ValueError, match="same rank"):
        gather_sampled_token_probs(logits, token_ids)
