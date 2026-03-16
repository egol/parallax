import logging

from fastapi.testclient import TestClient

from backend import main as backend_main
from parallax.utils import selective_download


class _SchedulerManageStub:
    def __init__(self, run_impl):
        self.run_impl = run_impl
        self.run_calls = []

    def is_running(self):
        return False

    def run(self, model_name, init_nodes_num, is_local_network):
        self.run_calls.append((model_name, init_nodes_num, is_local_network))
        self.run_impl(model_name, init_nodes_num, is_local_network)

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


def test_scheduler_init_triggers_snapshot_download_progress(monkeypatch, tmp_path, caplog):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    downloaded_marker = tmp_path / "downloaded.marker"
    snapshot_calls = []

    def fake_download_metadata_only(repo_id, cache_dir=None, force_download=False, local_files_only=False):
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

    def run_impl(model_name, init_nodes_num, is_local_network):
        selective_download.selective_model_download(
            repo_id=model_name,
            start_layer=0,
            end_layer=1,
            cache_dir=str(tmp_path),
            force_download=True,
        )

    async def fake_validate_scheduler_init_request(_request_data):
        return "Qwen/Qwen3-0.6B-FP8", 1, True

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
                "is_local_network": True,
            },
        )

    assert response.status_code == 200, response.text
    assert scheduler_manage.run_calls == [("Qwen/Qwen3-0.6B-FP8", 1, True)]
    assert downloaded_marker.exists(), "scheduler init should trigger a snapshot download"
    assert len(snapshot_calls) == 1

    log_messages = [record.getMessage() for record in caplog.records]
    assert any(
        "downloading full model snapshot" in message.lower() for message in log_messages
    ), log_messages
    assert any(
        "downloaded full model snapshot" in message.lower() for message in log_messages
    ), log_messages
