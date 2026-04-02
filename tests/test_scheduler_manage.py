import os
from datetime import datetime, timezone
from unittest.mock import patch

from backend.server.scheduler_manage import SchedulerManage


def test_run_restarts_scheduler_without_restarting_network_bootstrap_logic():
    with patch("backend.server.scheduler_manage.threading.Thread.start", autospec=True):
        manager = SchedulerManage()

    manager.lattica = object()

    with (
        patch.object(manager, "bootstrap_network") as bootstrap_network,
        patch.object(manager, "_start_scheduler") as start_scheduler,
    ):
        manager.run("Qwen/Qwen3-0.6B", 1, True)
        manager.run("Qwen/Qwen3-0.6B", 1, True)

    assert bootstrap_network.call_count == 2
    assert start_scheduler.call_count == 2


def test_aggregate_cluster_runtime_reports_fresh_ready_runtime():
    with patch("backend.server.scheduler_manage.threading.Thread.start", autospec=True):
        manager = SchedulerManage()

    updated_at = "2026-04-02T00:00:10Z"
    now_ts = datetime(2026, 4, 2, 0, 0, 20, tzinfo=timezone.utc).timestamp()

    runtime = {
        "status": "ready",
        "model_name": "Qwen/Qwen3-0.6B",
        "init_stage": "ready",
        "init_detail": "Ready to serve",
        "downloaded_files": 1,
        "total_files": 1,
        "current_file": "model.safetensors",
        "updated_at": updated_at,
        "failure_reason": "",
    }
    nodes = [{"node_id": "worker-1", "runtime": runtime}]

    with patch("backend.server.scheduler_manage.time.time", return_value=now_ts):
        aggregated = manager._aggregate_cluster_runtime(
            nodes,
            discovered_memory_gb=11.0,
            registered_memory_gb=11.0,
        )

    assert aggregated["model_init"]["stage"] == "ready"
    assert aggregated["runtime_stale"] is False
    assert aggregated["cluster_runtime_updated_at"] == updated_at


def test_aggregate_cluster_runtime_marks_old_runtime_updates_stale():
    with patch("backend.server.scheduler_manage.threading.Thread.start", autospec=True):
        manager = SchedulerManage()

    updated_at = "2026-04-02T00:00:00Z"
    now_ts = datetime(2026, 4, 2, 0, 0, 30, tzinfo=timezone.utc).timestamp()
    runtime = {
        "status": "ready",
        "model_name": "Qwen/Qwen3-0.6B",
        "init_stage": "ready",
        "init_detail": "Ready to serve",
        "downloaded_files": 1,
        "total_files": 1,
        "current_file": "model.safetensors",
        "updated_at": updated_at,
        "failure_reason": "",
    }
    nodes = [{"node_id": "worker-1", "runtime": runtime}]

    with (
        patch.dict(os.environ, {"PARALLAX_CLUSTER_RUNTIME_STALE_THRESHOLD_SEC": "15"}),
        patch("backend.server.scheduler_manage.time.time", return_value=now_ts),
    ):
        aggregated = manager._aggregate_cluster_runtime(
            nodes,
            discovered_memory_gb=11.0,
            registered_memory_gb=11.0,
        )

    assert aggregated["model_init"]["stage"] == "ready"
    assert aggregated["runtime_stale"] is True
    assert aggregated["cluster_runtime_updated_at"] == updated_at
