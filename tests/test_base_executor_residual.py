import torch

from parallax.server.executor.base_executor import BaseExecutor
from parallax.server.request import InitialRequest, IntermediateRequest, RequestStatus
from parallax.server.sampling.sampling_params import SamplingParams


def _make_executor(*, is_first_peer: bool, is_last_peer: bool) -> BaseExecutor:
    executor = BaseExecutor.__new__(BaseExecutor)
    executor.is_first_peer = is_first_peer
    executor.is_last_peer = is_last_peer
    return executor


def test_prepare_next_batch_requests_preserves_prefill_residual_slices():
    executor = _make_executor(is_first_peer=True, is_last_peer=False)
    requests = [
        InitialRequest(
            request_id="req-1",
            input_ids=[1, 2],
            sampling_params=SamplingParams(max_new_tokens=4),
        ),
        InitialRequest(
            request_id="req-2",
            input_ids=[3],
            sampling_params=SamplingParams(max_new_tokens=4),
        ),
    ]

    hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    residual_states = torch.arange(100, 112, dtype=torch.float32).reshape(3, 4)

    next_requests = executor.prepare_next_batch_requests(
        requests=requests,
        batch_output={
            "hidden_states": hidden_states,
            "residual_states": residual_states,
            "probs": None,
        },
        context_lengths=torch.tensor([2, 1]),
    )

    assert len(next_requests) == 2
    assert torch.equal(next_requests[0].hidden_states, hidden_states[:2, :])
    assert torch.equal(next_requests[0].residual_states, residual_states[:2, :])
    assert torch.equal(next_requests[1].hidden_states, hidden_states[2:3, :])
    assert torch.equal(next_requests[1].residual_states, residual_states[2:3, :])


def test_prepare_next_batch_requests_preserves_decode_residual_slices():
    executor = _make_executor(is_first_peer=False, is_last_peer=False)
    requests = [
        IntermediateRequest(
            request_id="req-1",
            current_position=4,
            status=RequestStatus.DECODING,
            input_ids=[1, 2],
            hidden_states=torch.zeros(1, 4),
            residual_states=torch.zeros(1, 4),
            sampling_params=SamplingParams(max_new_tokens=4),
        ),
        IntermediateRequest(
            request_id="req-2",
            current_position=5,
            status=RequestStatus.DECODING,
            input_ids=[3, 4],
            hidden_states=torch.zeros(1, 4),
            residual_states=torch.zeros(1, 4),
            sampling_params=SamplingParams(max_new_tokens=4),
        ),
    ]

    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    residual_states = torch.arange(20, 28, dtype=torch.float32).reshape(2, 4)

    next_requests = executor.prepare_next_batch_requests(
        requests=requests,
        batch_output={
            "hidden_states": hidden_states,
            "residual_states": residual_states,
            "probs": None,
        },
        context_lengths=torch.tensor([4, 5]),
    )

    assert len(next_requests) == 2
    assert torch.equal(next_requests[0].hidden_states, hidden_states[0:1, :])
    assert torch.equal(next_requests[0].residual_states, residual_states[0:1, :])
    assert torch.equal(next_requests[1].hidden_states, hidden_states[1:2, :])
    assert torch.equal(next_requests[1].residual_states, residual_states[1:2, :])
