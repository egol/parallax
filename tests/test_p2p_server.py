from parallax.p2p.server import GradientServer


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
