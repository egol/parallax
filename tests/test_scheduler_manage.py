import os
from datetime import datetime, timezone
from unittest.mock import patch

from backend.server.constants import NODE_STATUS_AVAILABLE, NODE_STATUS_WAITING
from backend.server.scheduler_manage import SchedulerManage
from scheduling.scheduler import Scheduler

from .scheduler_tests.test_utils import build_model_info, build_node, set_rtt_from_coords


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
        "cached_files": 1,
        "ready_bytes": 1234,
        "total_bytes": 1234,
        "cached_bytes": 1234,
        "current_file": "model.safetensors",
        "current_file_bytes": 1234,
        "current_file_total_bytes": 1234,
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
    assert aggregated["model_init"]["ready_bytes"] == 1234
    assert aggregated["model_init"]["total_bytes"] == 1234
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
        "cached_files": 1,
        "ready_bytes": 1234,
        "total_bytes": 1234,
        "cached_bytes": 1234,
        "current_file": "model.safetensors",
        "current_file_bytes": 1234,
        "current_file_total_bytes": 1234,
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


def test_aggregate_cluster_runtime_prefers_ready_pipeline_nodes_over_extra_downloads():
    with patch("backend.server.scheduler_manage.threading.Thread.start", autospec=True):
        manager = SchedulerManage()

    serving_updated_at = "2026-04-02T00:00:10Z"
    extra_updated_at = "2026-04-02T00:00:18Z"
    now_ts = datetime(2026, 4, 2, 0, 0, 20, tzinfo=timezone.utc).timestamp()
    nodes = [
        {
            "node_id": "serving-1",
            "runtime": {
                "status": "ready",
                "model_name": "Qwen/Qwen3-0.6B",
                "init_stage": "ready",
                "init_detail": "Ready to serve",
                "downloaded_files": 1,
                "total_files": 1,
                "cached_files": 1,
                "ready_bytes": 1234,
                "total_bytes": 1234,
                "cached_bytes": 1234,
                "current_file": "model.safetensors",
                "current_file_bytes": 1234,
                "current_file_total_bytes": 1234,
                "updated_at": serving_updated_at,
                "failure_reason": "",
            },
        },
        {
            "node_id": "standby-2",
            "runtime": {
                "status": "initializing",
                "model_name": "Qwen/Qwen3-0.6B",
                "init_stage": "downloading",
                "init_detail": "Downloading extra capacity",
                "downloaded_files": 2,
                "total_files": 4,
                "cached_files": 1,
                "ready_bytes": 2048,
                "total_bytes": 4096,
                "cached_bytes": 1024,
                "current_file": "model-00002-of-00008.safetensors",
                "current_file_bytes": 1024,
                "current_file_total_bytes": 2048,
                "updated_at": extra_updated_at,
                "failure_reason": "",
            },
        },
    ]

    with patch("backend.server.scheduler_manage.time.time", return_value=now_ts):
        aggregated = manager._aggregate_cluster_runtime(
            nodes,
            discovered_memory_gb=22.0,
            registered_memory_gb=11.0,
            ready_pipeline_node_ids={"serving-1"},
        )

    assert aggregated["model_init"]["stage"] == "ready"
    assert aggregated["model_init"]["detail"] == "Ready to serve"
    assert aggregated["model_init"]["source_node_id"] == "serving-1"
    assert aggregated["model_init"]["ready_bytes"] == 1234
    assert aggregated["runtime_stale"] is False
    assert aggregated["cluster_runtime_updated_at"] == serving_updated_at


def test_aggregate_cluster_runtime_uses_all_runtime_nodes_without_ready_pipeline_scope():
    with patch("backend.server.scheduler_manage.threading.Thread.start", autospec=True):
        manager = SchedulerManage()

    nodes = [
        {
            "node_id": "serving-1",
            "runtime": {
                "status": "ready",
                "model_name": "Qwen/Qwen3-0.6B",
                "init_stage": "ready",
                "init_detail": "Ready to serve",
                "downloaded_files": 1,
                "total_files": 1,
                "cached_files": 1,
                "ready_bytes": 1234,
                "total_bytes": 1234,
                "cached_bytes": 1234,
                "current_file": "model.safetensors",
                "current_file_bytes": 1234,
                "current_file_total_bytes": 1234,
                "updated_at": "2026-04-02T00:00:10Z",
                "failure_reason": "",
            },
        },
        {
            "node_id": "standby-2",
            "runtime": {
                "status": "initializing",
                "model_name": "Qwen/Qwen3-0.6B",
                "init_stage": "downloading",
                "init_detail": "Downloading extra capacity",
                "downloaded_files": 2,
                "total_files": 4,
                "cached_files": 1,
                "ready_bytes": 2048,
                "total_bytes": 4096,
                "cached_bytes": 1024,
                "current_file": "model-00002-of-00008.safetensors",
                "current_file_bytes": 1024,
                "current_file_total_bytes": 2048,
                "updated_at": "2026-04-02T00:00:18Z",
                "failure_reason": "",
            },
        },
    ]

    aggregated = manager._aggregate_cluster_runtime(
        nodes,
        discovered_memory_gb=22.0,
        registered_memory_gb=11.0,
    )

    assert aggregated["model_init"]["stage"] == "downloading"
    assert aggregated["model_init"]["detail"] == "Downloading extra capacity"
    assert aggregated["model_init"]["source_node_id"] == "standby-2"


def test_get_schedule_status_waits_for_routable_rr_pipeline():
    with patch("backend.server.scheduler_manage.threading.Thread.start", autospec=True):
        manager = SchedulerManage()

    model = build_model_info(36)
    n1 = build_node("a100-0", model, tflops=312.0, mem_gb=80.0, x=0, y=0)
    n2 = build_node("a100-1", model, tflops=312.0, mem_gb=80.0, x=1, y=0)
    n3 = build_node("a100-2", model, tflops=312.0, mem_gb=80.0, x=2, y=0)
    for node in (n1, n2, n3):
        node.rtt_to_nodes = {}

    scheduler = Scheduler(
        model, [n1, n2, n3], strategy="greedy", routing_strategy="rr", min_nodes_bootstrapping=3
    )
    assert scheduler.bootstrap() is False
    manager.scheduler = scheduler

    assert manager.get_schedule_status() == NODE_STATUS_WAITING

    set_rtt_from_coords([n1, n2, n3])
    for node in (n1, n2, n3):
        scheduler.enqueue_node_update(node.node_id, new_rtt_to_nodes=node.rtt_to_nodes)
    scheduler._process_node_updates()  # type: ignore[attr-defined]

    assert manager.get_schedule_status() == NODE_STATUS_AVAILABLE


def test_get_schedule_status_returns_waiting_without_routable_pipeline():
    with patch("backend.server.scheduler_manage.threading.Thread.start", autospec=True):
        manager = SchedulerManage()

    assert manager.get_schedule_status() == NODE_STATUS_WAITING
