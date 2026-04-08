import importlib
import sys
import types
from types import SimpleNamespace

import torch


def _install_vllm_stubs():
    if "vllm" in sys.modules:
        return

    def ensure_module(name: str):
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        return module

    vllm = ensure_module("vllm")
    vllm.sequence = ensure_module("vllm.sequence")
    vllm.sampling_params = ensure_module("vllm.sampling_params")
    vllm.v1 = ensure_module("vllm.v1")
    vllm.v1.core = ensure_module("vllm.v1.core")
    vllm.v1.core.sched = ensure_module("vllm.v1.core.sched")
    vllm.v1.core.sched.output = ensure_module("vllm.v1.core.sched.output")
    vllm.v1.request = ensure_module("vllm.v1.request")

    class Dummy:
        def __init__(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class DummySamplingParams(Dummy):
        pass

    class DummyStructuredOutputsParams(Dummy):
        pass

    class DummyIntermediateTensors(dict):
        pass

    class DummyCachedRequestData(Dummy):
        @classmethod
        def make_empty(cls):
            return cls()

    class DummyNewRequestData(Dummy):
        pass

    class DummySchedulerOutput(Dummy):
        pass

    class DummyVLLMRequest(Dummy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.output_ids = []
            self.num_computed_tokens = 0

        def append_output_token_ids(self, output_ids):
            self.output_ids.extend(output_ids)

    vllm.sampling_params.SamplingParams = DummySamplingParams
    vllm.sampling_params.StructuredOutputsParams = DummyStructuredOutputsParams
    vllm.sequence.IntermediateTensors = DummyIntermediateTensors
    vllm.v1.core.sched.output.CachedRequestData = DummyCachedRequestData
    vllm.v1.core.sched.output.NewRequestData = DummyNewRequestData
    vllm.v1.core.sched.output.SchedulerOutput = DummySchedulerOutput
    vllm.v1.request.Request = DummyVLLMRequest


_install_vllm_stubs()

batch_info_module = importlib.import_module("parallax.vllm.batch_info")

form_vllm_batch_decode = batch_info_module.form_vllm_batch_decode

from parallax.server.request import IntermediateRequest, RequestStatus
from parallax.server.sampling.sampling_params import SamplingParams


class _DummyBlocks:
    def get_block_ids(self, allow_none: bool = False):
        return ([1],)


class _DummyKVCacheManager:
    num_kv_cache_groups = 1

    def allocate_slots(self, request, num_new_tokens, num_new_computed_tokens):
        assert num_new_tokens == 1
        return _DummyBlocks()


def test_form_vllm_batch_decode_uses_intermediate_request_output_history():
    model_runner = SimpleNamespace(
        kv_cache_manager=_DummyKVCacheManager(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
        requests={},
        block_hasher=None,
        default_lora_req=None,
    )
    request = IntermediateRequest(
        request_id="decode-history",
        current_position=3,
        status=RequestStatus.DECODING,
        input_ids=[1, 2],
        output_ids=[7],
        hidden_states=torch.zeros(1, 4),
        sampling_params=SamplingParams(max_new_tokens=4),
    )

    scheduler_output = form_vllm_batch_decode(
        batched_requests=[request],
        model_runner=model_runner,
        scheduler=None,
    )

    cached = scheduler_output.scheduled_cached_reqs
    assert cached.all_token_ids["decode-history"] == [7]
    assert cached.new_token_ids == [[7]]
    assert cached.num_output_tokens == [1]
    assert cached.num_computed_tokens == [2]
