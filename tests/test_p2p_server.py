from parallax.p2p import server as p2p_server
from parallax.p2p.proto import forward_pb2
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


def test_send_notify_skips_when_layer_allocation_is_unknown(caplog):
    request = forward_pb2.ForwardRequest()
    req = request.reqs.add()
    req.rid = "req-1"
    req.output_length = 3

    p2p_server.send_notify(None, None, None, request, "started")

    assert "Skip started notification because layer allocation is not available yet" in caplog.text


def test_sync_to_shared_state_refreshes_connection_handler_layer_allocation(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "compute-node")
    server = build_server()
    server.block_start_index = 20
    server.block_end_index = 31
    server.connection_handler = type(
        "FakeHandler",
        (),
        {"block_start_index": None, "block_end_index": None},
    )()

    server._sync_to_shared_state()

    assert server.connection_handler.block_start_index == 20
    assert server.connection_handler.block_end_index == 31


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


class RecordingAbortFuture:
    def __init__(self):
        self.timeouts = []

    def result(self, timeout=None):
        self.timeouts.append(timeout)
        return None


class ExplodingAbortFuture:
    def result(self, timeout=None):
        raise AssertionError("worker abort dispatch should not wait on rpc_abort results")


class RecordingAbortStub:
    def __init__(self, future=None):
        self.requests = []
        self.future = future or RecordingAbortFuture()

    def rpc_abort(self, request):
        self.requests.append([req.rid for req in request.reqs])
        return self.future


def test_scheduler_peer_uses_scheduler_rpc_stub(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "compute-node")
    server = build_server()
    server.scheduler_peer_id = "scheduler-peer"
    server.scheduler_stub = object()
    server.connection_handler = FakeConnectionHandler()

    assert server.get_stub("scheduler-peer") is server.scheduler_stub
    assert server.connection_handler.calls == []


def test_dispatch_abort_request_does_not_wait_for_scheduler_rpc(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "compute-node")
    server = build_server()
    server.scheduler_peer_id = "scheduler-peer"
    stub = RecordingAbortStub(future=ExplodingAbortFuture())
    monkeypatch.setattr(server, "get_stub", lambda _peer_id: stub)

    request = forward_pb2.AbortRequest()
    request.reqs.extend([_build_abort_req("req-1", ["scheduler-peer"])])

    server._dispatch_abort_request("scheduler-peer", request)

    assert stub.requests == [["req-1"]]
    assert isinstance(stub.future, ExplodingAbortFuture)


def test_dispatch_abort_request_does_not_wait_for_non_scheduler_peer(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "compute-node")
    server = build_server()
    server.scheduler_peer_id = "scheduler-peer"
    stub = RecordingAbortStub()
    monkeypatch.setattr(server, "get_stub", lambda _peer_id: stub)

    request = forward_pb2.AbortRequest()
    request.reqs.extend([_build_abort_req("req-1", ["worker-a"])])

    server._dispatch_abort_request("worker-a", request)

    assert stub.requests == [["req-1"]]
    assert stub.future.timeouts == []


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


class CountingRttLattica(FakeLattica):
    def __init__(self):
        super().__init__()
        self.rtt_calls = []

    def get_peer_rtt(self, peer_id):
        self.rtt_calls.append(peer_id)
        if peer_id == "scheduler-peer":
            return 0.012
        return None


def test_get_node_info_update_uses_cached_peers_without_discovery(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "compute-node")
    monkeypatch.setattr("parallax.p2p.server.detect_node_hardware", lambda peer_id: {})
    server = build_server()
    server.lattica = CountingRttLattica()
    server.scheduler_peer_id = "scheduler-peer"
    server.rtts = {"existing-peer": 42}
    server.rtt_last_update = 0

    def fail_discovery():
        raise AssertionError("update heartbeat should not perform peer discovery")

    monkeypatch.setattr(server, "_discover_candidate_peer_ids", fail_discovery)

    info = server.get_node_info(is_update=True)

    assert info["rtt_to_nodes"]["existing-peer"] == 42
    assert info["rtt_to_nodes"]["scheduler-peer"] == 12.0
    assert server.lattica.rtt_calls == ["existing-peer", "scheduler-peer"]


def test_get_node_info_initial_join_falls_back_to_explicit_scheduler_peer(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "host")
    monkeypatch.setattr("parallax.p2p.server.detect_node_hardware", lambda peer_id: {})
    monkeypatch.setattr("parallax.p2p.server.time.sleep", lambda *_args, **_kwargs: None)
    server = build_server()
    server.lattica = FakeRttLattica()
    server.scheduler_peer_id = "scheduler-peer"
    server.rtt_last_update = 0
    monkeypatch.setattr(server, "_discover_candidate_peer_ids", lambda: [])

    info = server.get_node_info(is_update=False)

    assert info["node_id"] == "self-peer"
    assert info["rtt_to_nodes"]["scheduler-peer"] == 100


def test_get_node_info_initial_join_does_not_spend_30_seconds_on_scheduler_rtt(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "host")
    monkeypatch.setattr("parallax.p2p.server.detect_node_hardware", lambda peer_id: {})
    monkeypatch.setattr("parallax.p2p.server.time.sleep", lambda *_args, **_kwargs: None)
    server = build_server()
    server.lattica = CountingRttLattica()
    server.scheduler_peer_id = "scheduler-peer"
    server.rtt_last_update = 0
    monkeypatch.setattr(server, "_discover_candidate_peer_ids", lambda: ["scheduler-peer"])

    info = server.get_node_info(is_update=False)

    assert info["rtt_to_nodes"]["scheduler-peer"] == 12.0
    assert server.lattica.rtt_calls == ["scheduler-peer"]


def test_scheduler_node_join_can_use_local_http_fallback(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "host")
    server = build_server()
    server._local_scheduler_http_url = "http://127.0.0.1:3301"
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        return {"data": {"start_layer": 0, "end_layer": 20}}

    monkeypatch.setattr(server, "_post_local_scheduler_http", fake_post)

    response = server._scheduler_node_join({"node_id": "self-peer"})

    assert response == {"start_layer": 0, "end_layer": 20}
    assert calls == [("/internal/node_join", {"node_id": "self-peer"})]


def test_scheduler_node_update_can_use_local_http_fallback(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "host")
    server = build_server()
    server._local_scheduler_http_url = "http://127.0.0.1:3301"
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        return {"layer_allocation": {"start_layer": 0}, "refit_request": {"version": 1}}

    monkeypatch.setattr(server, "_post_local_scheduler_http", fake_post)

    layer_allocation, refit_request = server._scheduler_node_update({"node_id": "self-peer"})

    assert layer_allocation == {"start_layer": 0}
    assert refit_request == {"version": 1}
    assert calls == [("/internal/node_update", {"node_id": "self-peer"})]


def test_scheduler_node_leave_can_use_local_http_fallback(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "host")
    server = build_server()
    server._local_scheduler_http_url = "http://127.0.0.1:3301"
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        return {"data": {}}

    monkeypatch.setattr(server, "_post_local_scheduler_http", fake_post)

    server._scheduler_node_leave({"node_id": "self-peer"})

    assert calls == [("/internal/node_leave", {"node_id": "self-peer"})]


def _build_abort_req(rid, routing_table):
    req = forward_pb2.Req()
    req.rid = rid
    req.routing_table.extend(routing_table)
    return req


def test_group_abort_requests_routes_centralized_aborts_via_scheduler(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "compute-node")
    server = build_server()
    server.scheduler_peer_id = "scheduler-peer"

    grouped = server._group_abort_requests(
        [
            _build_abort_req("req-1", ["worker-a", "scheduler-peer", "worker-b"]),
            _build_abort_req("req-2", ["worker-c", "scheduler-peer", "worker-d"]),
        ]
    )

    assert list(grouped) == ["scheduler-peer"]
    assert [req.rid for req in grouped["scheduler-peer"]] == ["req-1", "req-2"]


def test_group_abort_requests_broadcasts_without_centralized_scheduler(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROLE", "compute-node")
    server = build_server()
    server.scheduler_peer_id = None

    grouped = server._group_abort_requests(
        [_build_abort_req("req-1", ["worker-a", "worker-b", "worker-c"])]
    )

    assert {peer_id: [req.rid for req in reqs] for peer_id, reqs in grouped.items()} == {
        "worker-a": ["req-1"],
        "worker-b": ["req-1"],
        "worker-c": ["req-1"],
    }


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
