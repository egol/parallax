from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("sglang")

import torch

from parallax.sglang.batch_info import select_batch


class _FakeReq:
    return_logprob = False
    stream = False
    grammar = False


class _FakeBatch:
    def __init__(self):
        self.device = "cpu"
        self.reqs = [_FakeReq(), _FakeReq()]
        self.req_to_token_pool = object()
        self.token_to_kv_pool_allocator = object()
        self.tree_cache = object()
        self.model_config = SimpleNamespace(is_encoder_decoder=False, vocab_size=32)
        self.multimodal_inputs = None
        self.seq_lens_cpu = torch.tensor([3, 5], dtype=torch.int64)
        self.req_pool_indices = torch.tensor([10, 20], dtype=torch.int64)
        self.seq_lens = torch.tensor([3, 5], dtype=torch.int64)
        self.orig_seq_lens = torch.tensor([3, 5], dtype=torch.int64)
        self.out_cache_loc = torch.tensor([30, 40], dtype=torch.int64)
        self.output_ids = torch.tensor([7, 8], dtype=torch.int64)
        self.return_logprob = False
        self.top_logprobs_nums = None
        self.token_ids_logprobs = None

    def copy(self):
        return SimpleNamespace(device="cuda")


def test_select_batch_preserves_origin_device_for_sampling_info():
    origin_batch = _FakeBatch()

    with patch("parallax.sglang.batch_info.SGLSamplingBatchInfo.from_schedule_batch") as mocked:
        mocked.return_value = object()

        selected = select_batch(origin_batch, [1])

    assert selected.device == "cpu"
    assert selected.req_pool_indices.device.type == "cpu"
    assert selected.seq_lens.device.type == "cpu"
    assert mocked.call_args[0][0].device == "cpu"
