import logging

from fastapi.testclient import TestClient

from backend import main as backend_main
from parallax.utils import selective_download


class _SchedulerManageStub:
    def __init__(self, run_impl):
        self.run_impl = run_impl
        self.run_calls = []
        self.scheduler = object()
        self.connection_handler = None
        self.registered_nodes = []

    def is_running(self):
        return False

    def run(self, model_name, init_nodes_num, network_mode):
        self.run_calls.append((model_name, init_nodes_num, network_mode))
        self.run_impl(model_name, init_nodes_num, network_mode)

    def get_cluster_status(self):
        return {
            "data": {
                "status": "waiting",
                "topology": {
                    "nodes": [],
                    "totals": {
                        "ready_pipelines": 0,
                        "registered_workers": 0,
                        "discovered_workers": 0,
                    },
                },
            }
        }

    def register_discovered_node(self, message):
        self.registered_nodes.append(message)


class _ConnectionHandlerStub:
    def node_join(self, message):
        return {"joined": message.get("node_id")}

    def node_update(self, message):
        return {"start_layer": 0, "node_id": message.get("node_id")}, {"version": 1}

    def node_leave(self, message):
        return {"left": message.get("node_id")}


def test_scheduler_init_triggers_snapshot_download_progress(monkeypatch, tmp_path, caplog):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    downloaded_marker = tmp_path / "downloaded.marker"
    metadata_calls = []
    snapshot_calls = []
    weight_download_calls = []

    def fake_download_metadata_only(
        repo_id,
        cache_dir=None,
        force_download=False,
        local_files_only=False,
        **_kwargs,
    ):
        metadata_calls.append(
            {
                "repo_id": repo_id,
                "cache_dir": cache_dir,
                "force_download": force_download,
                "local_files_only": local_files_only,
            }
        )
        return metadata_dir

    def fake_determine_needed_weight_files_for_download(model_path, start_layer, end_layer):
        return []

    def fake_snapshot_download(repo_id, cache_dir=None, force_download=False, local_files_only=False):
        snapshot_calls.append(
            {
                "repo_id": repo_id,
                "cache_dir": cache_dir,
                "force_download": force_download,
                "local_files_only": local_files_only,
            }
        )
        downloaded_marker.write_text("downloaded", encoding="utf-8")
        return str(metadata_dir)

    def fake_list_remote_weight_files(_repo_id):
        return ["model.safetensors"]

    def fake_download_weight_files(
        repo_id,
        weight_files,
        model_path=None,
        cache_dir=None,
        force_download=False,
        local_files_only=False,
        shared_state=None,
        is_remote=True,
    ):
        weight_download_calls.append(
            {
                "repo_id": repo_id,
                "weight_files": list(weight_files),
                "model_path": model_path,
                "cache_dir": cache_dir,
                "force_download": force_download,
                "local_files_only": local_files_only,
                "shared_state": shared_state,
                "is_remote": is_remote,
            }
        )
        downloaded_marker.write_text("downloaded", encoding="utf-8")

    def run_impl(model_name, init_nodes_num, network_mode):
        selective_download.selective_model_download(
            repo_id=model_name,
            start_layer=0,
            end_layer=1,
            cache_dir=str(tmp_path),
            force_download=True,
        )

    async def fake_validate_scheduler_init_request(_request_data):
        return "Qwen/Qwen3-0.6B-FP8", 1, "centralized"

    scheduler_manage = _SchedulerManageStub(run_impl)
    monkeypatch.setattr(
        selective_download,
        "download_metadata_only",
        fake_download_metadata_only,
    )
    monkeypatch.setattr(
        selective_download,
        "determine_needed_weight_files_for_download",
        fake_determine_needed_weight_files_for_download,
    )
    monkeypatch.setattr(
        selective_download,
        "_list_remote_weight_files",
        fake_list_remote_weight_files,
    )
    monkeypatch.setattr(
        selective_download,
        "_download_weight_files",
        fake_download_weight_files,
    )
    monkeypatch.setattr(selective_download, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(backend_main, "scheduler_manage", scheduler_manage)
    monkeypatch.setattr(
        backend_main,
        "_validate_scheduler_init_request",
        fake_validate_scheduler_init_request,
    )

    caplog.set_level(logging.INFO)
    with TestClient(backend_main.app) as client:
        response = client.post(
            "/scheduler/init",
            json={
                "model_name": "Qwen/Qwen3-0.6B-FP8",
                "init_nodes_num": 1,
                "network_mode": "centralized",
            },
        )

    assert response.status_code == 200, response.text
    assert scheduler_manage.run_calls == [("Qwen/Qwen3-0.6B-FP8", 1, "centralized")]
    assert downloaded_marker.exists(), "scheduler init should trigger a snapshot download"
    assert len(metadata_calls) == 1
    assert len(snapshot_calls) == 0
    assert len(weight_download_calls) == 1
    assert weight_download_calls[0]["weight_files"] == ["model.safetensors"]

    log_messages = [record.getMessage() for record in caplog.records]
    assert any(
        "downloading full model snapshot" in message.lower() for message in log_messages
    ), log_messages
    assert any(
        "downloaded full model snapshot" in message.lower() for message in log_messages
    ), log_messages


def test_internal_scheduler_control_plane_endpoints(monkeypatch):
    scheduler_manage = _SchedulerManageStub(lambda *_args: None)
    scheduler_manage.connection_handler = _ConnectionHandlerStub()
    monkeypatch.setattr(backend_main, "scheduler_manage", scheduler_manage)

    with TestClient(backend_main.app) as client:
        join_response = client.post("/internal/node_join", json={"node_id": "node-a"})
        update_response = client.post("/internal/node_update", json={"node_id": "node-a"})
        leave_response = client.post("/internal/node_leave", json={"node_id": "node-a"})

    assert join_response.status_code == 200
    assert join_response.json() == {"data": {"joined": "node-a"}}
    assert update_response.status_code == 200
    assert update_response.json() == {
        "layer_allocation": {"start_layer": 0, "node_id": "node-a"},
        "refit_request": {"version": 1},
    }
    assert leave_response.status_code == 200
    assert leave_response.json() == {"data": {"left": "node-a"}}


def test_internal_scheduler_control_plane_endpoints_return_quickly_without_scheduler(monkeypatch):
    scheduler_manage = _SchedulerManageStub(lambda *_args: None)
    scheduler_manage.scheduler = None
    scheduler_manage.connection_handler = _ConnectionHandlerStub()
    monkeypatch.setattr(backend_main, "scheduler_manage", scheduler_manage)

    with TestClient(backend_main.app) as client:
        join_response = client.post("/internal/node_join", json={"node_id": "node-a"})
        update_response = client.post("/internal/node_update", json={"node_id": "node-a"})

    assert join_response.status_code == 200
    assert join_response.json() == {"data": {}}
    assert update_response.status_code == 200
    assert update_response.json() == {"layer_allocation": {}, "refit_request": {}}
    assert scheduler_manage.registered_nodes == [{"node_id": "node-a"}, {"node_id": "node-a"}]
