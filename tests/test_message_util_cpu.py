"""
CPU-focused tests for message serialization helpers.
"""

import pytest

torch = pytest.importorskip("torch")

from parallax.p2p.message_util import (
    bytes_to_tensor,
    proto_to_request,
    request_to_proto,
    tensor_to_bytes,
)
from parallax.server.request import IntermediateRequest, RequestStatus
from parallax.server.sampling.sampling_params import SamplingParams


def test_cpu_tensor_round_trip():
    original = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    serialized = tensor_to_bytes(original, device="cpu")
    restored = bytes_to_tensor(serialized, device="cpu")

    assert torch.equal(restored, original)
    assert restored.device.type == "cpu"


def test_cpu_request_round_trip():
    original = IntermediateRequest(
        request_id="cpu-request",
        input_ids=[1, 2, 3],
        current_position=3,
        status=RequestStatus.PREFILLING,
        hidden_states=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=4),
        routing_table=["node-a"],
        lora_path=None,
    )

    proto = request_to_proto([original], device="cpu")
    restored = proto_to_request(proto, device="cpu")

    assert len(restored) == 1
    restored_request = restored[0]
    assert restored_request.request_id == original.request_id
    assert restored_request.status == original.status
    assert restored_request.input_ids == original.input_ids
    assert restored_request.routing_table == original.routing_table
    assert torch.equal(restored_request.hidden_states, original.hidden_states)
