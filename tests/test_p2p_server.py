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
