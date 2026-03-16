from unittest.mock import patch

from backend.server.scheduler_manage import SchedulerManage


def test_run_reuses_completion_handler_registration():
    with patch("backend.server.scheduler_manage.threading.Thread.start", autospec=True):
        manager = SchedulerManage()

    manager.lattica = object()

    with (
        patch.object(manager, "bootstrap_network") as bootstrap_network,
        patch.object(manager, "_start_scheduler") as start_scheduler,
        patch(
            "backend.server.scheduler_manage.TransformerConnectionHandler",
            side_effect=[object(), object()],
        ) as connection_handler_cls,
    ):
        manager.run("Qwen/Qwen3-0.6B", 1, True)
        first_handler = manager.completion_handler

        manager.run("Qwen/Qwen3-0.6B", 1, True)

    assert manager.completion_handler is first_handler
    assert connection_handler_cls.call_count == 1
    assert bootstrap_network.call_count == 2
    assert start_scheduler.call_count == 2
