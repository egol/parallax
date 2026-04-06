from parallax.launch import _p2p_startup_failure_reason
from parallax.utils.shared_state import SharedState


def test_p2p_startup_failure_reason_prefers_runtime_failure():
    shared_state = SharedState.create()
    shared_state.update_runtime_state(
        status="error",
        init_stage="failed",
        init_detail="P2P server startup failed.",
        failure_reason="Failed to initialize Lattica: Address already in use",
        fatal_error=True,
    )

    assert (
        _p2p_startup_failure_reason(shared_state)
        == "Failed to initialize Lattica: Address already in use"
    )
