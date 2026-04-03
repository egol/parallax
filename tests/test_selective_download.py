from pathlib import Path

from parallax.utils import selective_download
from parallax.utils.shared_state import SharedState


def _make_shared_state(model_name: str = "Qwen/Qwen3-0.6B") -> SharedState:
    return SharedState(
        {
            "status": "initializing",
            "model_name": model_name,
            "runtime_state": {
                "status": "initializing",
                "model_name": model_name,
                "init_stage": "idle",
                "init_detail": "",
                "downloaded_files": None,
                "total_files": None,
                "cached_files": None,
                "ready_bytes": None,
                "total_bytes": None,
                "cached_bytes": None,
                "current_file": "",
                "current_file_bytes": None,
                "current_file_total_bytes": None,
                "failure_reason": "",
                "updated_at": "",
            },
        }
    )


def test_download_weight_files_tracks_cached_and_downloaded_bytes(monkeypatch, tmp_path):
    model_path = tmp_path / "metadata"
    model_path.mkdir()
    cached_blob = tmp_path / "cache" / "model-00001-of-00002.safetensors"
    cached_blob.parent.mkdir(parents=True)
    cached_blob.write_bytes(b"a" * 10)
    shared_state = _make_shared_state()
    calls = []

    monkeypatch.setattr(
        selective_download,
        "_resolve_weight_file_sizes",
        lambda repo_id, weight_files: {
            "model-00001-of-00002.safetensors": 10,
            "model-00002-of-00002.safetensors": 20,
        },
    )
    monkeypatch.setattr(
        selective_download,
        "try_to_load_from_cache",
        lambda repo_id, filename, cache_dir=None, revision=None, repo_type=None: (
            str(cached_blob) if filename == "model-00001-of-00002.safetensors" else None
        ),
    )

    def fake_hf_hub_download(
        repo_id,
        filename,
        *,
        cache_dir=None,
        force_download=False,
        local_files_only=False,
        tqdm_class=None,
        **kwargs,
    ):
        calls.append(
            {
                "filename": filename,
                "local_files_only": local_files_only,
            }
        )
        target = model_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if local_files_only:
            target.write_bytes(cached_blob.read_bytes())
            return str(target)

        progress = tqdm_class(total=20, initial=0, desc=filename, disable=False, name="test")
        progress.update(5)
        progress.update(15)
        progress.close()
        target.write_bytes(b"b" * 20)
        return str(target)

    monkeypatch.setattr(selective_download, "hf_hub_download", fake_hf_hub_download)

    selective_download._download_weight_files(
        "Qwen/Qwen3-0.6B",
        [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        model_path=model_path,
        cache_dir=str(tmp_path / "hf-cache"),
        force_download=False,
        local_files_only=False,
        shared_state=shared_state,
        is_remote=True,
    )

    runtime = shared_state.get_runtime_state()
    assert runtime["downloaded_files"] == 2
    assert runtime["total_files"] == 2
    assert runtime["cached_files"] == 1
    assert runtime["ready_bytes"] == 30
    assert runtime["total_bytes"] == 30
    assert runtime["cached_bytes"] == 10
    assert runtime["current_file"] == "model-00002-of-00002.safetensors"
    assert runtime["current_file_bytes"] == 20
    assert runtime["current_file_total_bytes"] == 20
    assert calls == [
        {
            "filename": "model-00001-of-00002.safetensors",
            "local_files_only": True,
        },
        {
            "filename": "model-00002-of-00002.safetensors",
            "local_files_only": False,
        },
    ]


def test_download_weight_files_marks_local_weights_complete_without_network(monkeypatch, tmp_path):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    (model_path / "model-00001-of-00002.safetensors").write_bytes(b"a" * 10)
    (model_path / "model-00002-of-00002.safetensors").write_bytes(b"b" * 20)
    shared_state = _make_shared_state(str(model_path))

    monkeypatch.setattr(
        selective_download,
        "hf_hub_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    selective_download._download_weight_files(
        str(model_path),
        [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        model_path=model_path,
        cache_dir=None,
        force_download=False,
        local_files_only=True,
        shared_state=shared_state,
        is_remote=False,
    )

    runtime = shared_state.get_runtime_state()
    assert runtime["downloaded_files"] == 2
    assert runtime["total_files"] == 2
    assert runtime["cached_files"] == 2
    assert runtime["ready_bytes"] == 30
    assert runtime["total_bytes"] == 30
    assert runtime["cached_bytes"] == 30


def test_download_weight_files_falls_back_to_file_counts_when_sizes_are_unknown(
    monkeypatch, tmp_path
):
    model_path = tmp_path / "metadata"
    model_path.mkdir()
    shared_state = _make_shared_state()

    monkeypatch.setattr(
        selective_download,
        "_resolve_weight_file_sizes",
        lambda repo_id, weight_files: {"model-00001-of-00001.safetensors": None},
    )
    monkeypatch.setattr(
        selective_download,
        "try_to_load_from_cache",
        lambda repo_id, filename, cache_dir=None, revision=None, repo_type=None: None,
    )

    def fake_hf_hub_download(
        repo_id,
        filename,
        *,
        cache_dir=None,
        force_download=False,
        local_files_only=False,
        tqdm_class=None,
        **kwargs,
    ):
        progress = tqdm_class(total=7, initial=0, desc=filename, disable=False, name="test")
        progress.update(7)
        progress.close()
        target = model_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 7)
        return str(target)

    monkeypatch.setattr(selective_download, "hf_hub_download", fake_hf_hub_download)

    selective_download._download_weight_files(
        "Qwen/Qwen3-0.6B",
        ["model-00001-of-00001.safetensors"],
        model_path=model_path,
        cache_dir=str(tmp_path / "hf-cache"),
        force_download=False,
        local_files_only=False,
        shared_state=shared_state,
        is_remote=True,
    )

    runtime = shared_state.get_runtime_state()
    assert runtime["downloaded_files"] == 1
    assert runtime["total_files"] == 1
    assert runtime["ready_bytes"] is None
    assert runtime["total_bytes"] is None
    assert runtime["cached_bytes"] is None
    assert runtime["current_file"] == "model-00001-of-00001.safetensors"
    assert runtime["current_file_bytes"] == 7
    assert runtime["current_file_total_bytes"] == 7
