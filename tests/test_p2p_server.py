from parallax.p2p import server as p2p_server
from parallax.p2p.server import GradientServer
from parallax.utils.shared_state import SharedState


def build_server():
    return GradientServer(
        recv_from_peer_addr="ipc:///tmp/test-recv",
        send_to_peer_addr="ipc:///tmp/test-send",
        scheduler_addr="12D3KooWTestScheduler",
        http_port=3100,
    )


def test_host_worker_retries_join_without_timeout(monkeypatch):
    monkeypatch.delenv("PARALLAX_JOIN_RETRY_FOREVER", raising=False)
    monkeypatch.setenv("PARALLAX_ROLE", "host")
    server = build_server()

    assert server._join_retry_forever() is True
    assert server._join_attempt_is_fatal(server._max_join_attempts) is False
    assert server._join_attempt_is_fatal(server._max_join_attempts + 5) is False


class FakeBlockServers:
    def __init__(self, value):
        self.value = value


class FakeLattica:
    def __init__(self):
        self.stores = []

    def peer_id(self):
        return "self-peer"

    def get_all_peers(self):
        return ["scheduler-peer"]

    def get(self, key):
        if key == "gradient_announce":
            return FakeBlockServers(
                {
                    "worker-a": {"value": {"block_start_index": 20, "block_end_index": 31}},
                    "worker-b": {"value": {"block_start_index": 31, "block_end_index": 40}},
                }
            )
        return None

    def store(self, **kwargs):
        self.stores.append(kwargs)


def test_candidate_peer_discovery_includes_announced_workers(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "compute-node")
    server = build_server()
    server.lattica = FakeLattica()
    server.scheduler_peer_id = "scheduler-peer"

    assert server._discover_candidate_peer_ids() == [
        "scheduler-peer",
        "worker-a",
        "worker-b",
    ]


def test_scheduler_mode_still_announces_current_range(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "compute-node")
    server = build_server()
    server.lattica = FakeLattica()
    server.block_start_index = 20
    server.block_end_index = 31

    server._announce_current_range()

    assert len(server.lattica.stores) == 1
    record = server.lattica.stores[0]
    assert record["key"] == "gradient_announce"
    assert record["subkey"] == "self-peer"
    assert record["value"] == {"block_start_index": 20, "block_end_index": 31}


def test_participant_worker_still_fails_after_retry_budget(monkeypatch):
    monkeypatch.delenv("PARALLAX_JOIN_RETRY_FOREVER", raising=False)
    monkeypatch.setenv("PARALLAX_ROLE", "compute-node")
    server = build_server()

    assert server._join_retry_forever() is False
    assert server._join_attempt_is_fatal(server._max_join_attempts - 1) is False
    assert server._join_attempt_is_fatal(server._max_join_attempts) is True


def test_join_retry_override_can_force_participant_to_wait(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "compute-node")
    monkeypatch.setenv("PARALLAX_JOIN_RETRY_FOREVER", "true")
    server = build_server()

    assert server._join_retry_forever() is True
    assert server._join_attempt_is_fatal(server._max_join_attempts) is False


class FakeConnectionHandler:
    def __init__(self):
        self.calls = []

    def get_stub(self, peer_id):
        self.calls.append(peer_id)
        return {"peer_id": peer_id}


def test_scheduler_peer_uses_scheduler_rpc_stub(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "compute-node")
    server = build_server()
    server.scheduler_peer_id = "scheduler-peer"
    server.scheduler_stub = object()
    server.connection_handler = FakeConnectionHandler()

    assert server.get_stub("scheduler-peer") is server.scheduler_stub
    assert server.connection_handler.calls == []


class FakeRttLattica(FakeLattica):
    def get_peer_rtt(self, peer_id):
        return None


def test_get_node_info_handles_missing_rtt_without_type_error(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "compute-node")
    monkeypatch.setattr("parallax.p2p.server.detect_node_hardware", lambda peer_id: {})
    monkeypatch.setattr("parallax.p2p.server.time.sleep", lambda *_args, **_kwargs: None)
    server = build_server()
    server.lattica = FakeRttLattica()
    server.scheduler_peer_id = "scheduler-peer"

    info = server.get_node_info(is_update=True)

    assert info["rtt_to_nodes"]["scheduler-peer"] == 100


class FakeBuilder:
    def __init__(self, error=None):
        self.error = error
        self.closed = False

    def with_listen_addrs(self, _addrs):
        return self

    def with_key_path(self, _path):
        return self

    def with_bootstraps(self, _bootstraps):
        return self

    def with_relay_servers(self, _relay_servers):
        return self

    def with_dcutr(self, _enabled):
        return self

    def with_protocol(self, _protocol):
        return self

    def with_external_addrs(self, _addrs):
        return self

    def build(self):
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True


def test_build_lattica_handles_addr_in_use_without_crashing(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "host")
    builder = FakeBuilder(RuntimeError('message: "Address already in use"'))
    monkeypatch.setattr(p2p_server.Lattica, "builder", lambda: builder)
    server = GradientServer(
        recv_from_peer_addr="ipc:///tmp/test-recv",
        send_to_peer_addr="ipc:///tmp/test-send",
        scheduler_addr=None,
        host_maddrs=["/ip4/0.0.0.0/tcp/4100"],
    )

    assert server.build_lattica() is False
    assert builder.closed is True
    assert server.lattica is None
    assert "Address already in use" in str(server._last_build_error)


def test_build_lattica_retry_waits_and_recovers(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "host")
    server = build_server()
    attempts = []
    sleeps = []

    def fake_build():
        attempts.append(True)
        if len(attempts) == 1:
            server._last_build_error = RuntimeError("Address already in use")
            return False
        server._last_build_error = None
        return True

    monkeypatch.setattr(server, "build_lattica", fake_build)
    monkeypatch.setattr(p2p_server.time, "sleep", lambda seconds: sleeps.append(seconds))

    server._build_lattica_with_retry()

    assert len(attempts) == 2
    assert sleeps == [3]


def test_shutdown_preserves_failure_when_lattica_never_initialized(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "host")
    server = build_server()
    server._shared_state = SharedState.create()
    server._shared_state.update_runtime_state(
        status="error",
        init_stage="failed",
        init_detail="P2P server startup failed.",
        failure_reason="Address already in use",
        fatal_error=True,
    )

    server.shutdown(preserve_runtime_failure=True)

    runtime_state = server._shared_state.get_runtime_state()
    assert runtime_state["init_stage"] == "failed"
    assert runtime_state["failure_reason"] == "Address already in use"
