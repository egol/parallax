"""
Helpers for normalizing SGLang output tensors.
"""

from __future__ import annotations

import torch


def gather_sampled_token_probs(
    next_token_logits: torch.Tensor,
    next_token_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Gather the probability/logit value for each sampled token.

    SGLang can return sampled token ids as either a 1D tensor of shape [batch]
    or a 2D tensor of shape [batch, 1]. Normalize both before gathering.
    """
    gather_index = next_token_ids.to(dtype=torch.long)
    if gather_index.ndim == next_token_logits.ndim - 1:
        gather_index = gather_index.unsqueeze(-1)

    if gather_index.ndim != next_token_logits.ndim:
        raise ValueError(
            "sampled token ids must have the same rank as next-token logits "
            "after normalization"
        )

    gathered = torch.gather(next_token_logits, next_token_logits.ndim - 1, gather_index)
    return gathered.squeeze(-1)
