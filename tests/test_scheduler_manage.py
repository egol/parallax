from types import SimpleNamespace
from unittest.mock import patch

from backend.server.scheduler_manage import SchedulerManage


def test_start_lattica_reuses_existing_connection_handler():
    with patch("backend.server.scheduler_manage.threading.Thread.start", autospec=True):
        manager = SchedulerManage()

    manager.lattica = object()
    manager.scheduler = object()
    manager.network_signature = manager._network_signature_for([], [])
    manager.connection_handler = SimpleNamespace(scheduler=None, scheduler_manage=None)

    with patch("backend.server.scheduler_manage.RPCConnectionHandler") as connection_handler_cls:
        manager._start_lattica([], [])

    assert manager.connection_handler.scheduler is manager.scheduler
    assert manager.connection_handler.scheduler_manage is manager
    connection_handler_cls.assert_not_called()


def test_cluster_runtime_ignores_standby_workers_before_model_start():
    with patch("backend.server.scheduler_manage.threading.Thread.start", autospec=True):
        manager = SchedulerManage()

    runtime = manager._aggregate_cluster_runtime(
        [
            {
                "node_id": "worker-1",
                "runtime": {
                    "status": "joining",
                    "model_name": "",
                    "init_stage": "idle",
                    "init_detail": (
                        "Connected to scheduler discovery; waiting for model start or "
                        "layer allocation."
                    ),
                    "updated_at": "2026-03-17T23:25:09Z",
                },
            }
        ],
        discovered_memory_gb=24.0,
        registered_memory_gb=0.0,
    )

    assert runtime == {"model_init": None}


def test_cluster_runtime_tracks_real_model_init_once_model_selected():
    with patch("backend.server.scheduler_manage.threading.Thread.start", autospec=True):
        manager = SchedulerManage()

    manager.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    runtime = manager._aggregate_cluster_runtime(
        [
            {
                "node_id": "worker-1",
                "runtime": {
                    "status": "initializing",
                    "model_name": "",
                    "init_stage": "allocating",
                    "init_detail": "Received layer allocation [0, 24); initializing executors.",
                    "updated_at": "2026-03-17T23:26:10Z",
                },
            }
        ],
        discovered_memory_gb=24.0,
        registered_memory_gb=24.0,
    )

    assert runtime["model_init"] is not None
    assert runtime["model_init"]["model_name"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert runtime["model_init"]["stage"] == "allocating"
