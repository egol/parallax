import inspect
import logging
import os
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download, snapshot_download, try_to_load_from_cache

from parallax.utils.shared_state import SharedState

logger = logging.getLogger(__name__)
from parallax.utils.weight_filter_utils import (
    determine_needed_weight_files_for_download,
)

# Monkey patch HfApi.repo_info to add short timeout for faster failure on network issues
# This prevents snapshot_download from hanging silently when Hugging Face Hub is unreachable
_original_repo_info = HfApi.repo_info
_REPO_INFO_TIMEOUT = float(os.environ.get("PARALLAX_HF_REPO_INFO_TIMEOUT", "5.0"))


def _repo_info_with_timeout(self, repo_id, repo_type=None, revision=None, **kwargs):
    """Wrapper for HfApi.repo_info that injects a short timeout if not provided."""
    if "timeout" not in kwargs:
        kwargs["timeout"] = _REPO_INFO_TIMEOUT
        logger.debug(f"Injecting timeout={_REPO_INFO_TIMEOUT}s for repo_info call to {repo_id}")
    return _original_repo_info(
        self, repo_id=repo_id, repo_type=repo_type, revision=revision, **kwargs
    )


# Only apply monkey patch if repo_info accepts timeout parameter
_repo_info_signature = inspect.signature(_original_repo_info)
if "timeout" in _repo_info_signature.parameters or "kwargs" in str(_repo_info_signature):
    HfApi.repo_info = _repo_info_with_timeout
    logger.debug(f"Applied monkey patch to HfApi.repo_info with timeout={_REPO_INFO_TIMEOUT}s")
else:
    logger.warning(
        "HfApi.repo_info does not accept 'timeout' parameter - monkey patch skipped. "
        "Network timeout issues may still occur."
    )

EXCLUDE_WEIGHT_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "pytorch_model*.bin",
    "model*.safetensors",
    "weight*.safetensors",
]
_DOWNLOAD_PROGRESS_THROTTLE_SEC = float(
    os.environ.get("PARALLAX_DOWNLOAD_PROGRESS_THROTTLE_SEC", "0.25")
)


@dataclass
class WeightFileManifest:
    repo_path: str
    file_name: str
    local_path: Path
    total_bytes: Optional[int]
    available_before: bool
    needs_materialization: bool


class _DownloadProgressTracker:
    def __init__(
        self,
        shared_state: Optional[SharedState],
        *,
        total_files: int,
        total_bytes: Optional[int],
        cached_files: int,
        cached_bytes: Optional[int],
    ) -> None:
        self.shared_state = shared_state
        self.total_files = total_files
        self.total_bytes = total_bytes
        self.cached_files = cached_files
        self.cached_bytes = cached_bytes
        self.completed_files = cached_files
        self.completed_bytes = cached_bytes
        self.current_file = ""
        self.current_file_bytes: Optional[int] = None
        self.current_file_total_bytes: Optional[int] = None
        self.failure_reason = ""
        self._last_emit_at = 0.0

    def ready_bytes(self) -> Optional[int]:
        if self.total_bytes is None or self.completed_bytes is None:
            return None
        if self.current_file_bytes is None:
            return self.completed_bytes
        return min(self.total_bytes, self.completed_bytes + self.current_file_bytes)

    def emit(
        self,
        *,
        init_stage: str,
        init_detail: str,
        current_file: Optional[str] = None,
        current_file_bytes: Optional[int] = None,
        current_file_total_bytes: Optional[int] = None,
        failure_reason: Optional[str] = None,
        force: bool = False,
    ) -> None:
        if self.shared_state is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_emit_at) < _DOWNLOAD_PROGRESS_THROTTLE_SEC:
            return
        self._last_emit_at = now
        _update_runtime_state(
            self.shared_state,
            init_stage=init_stage,
            init_detail=init_detail,
            downloaded_files=self.completed_files,
            total_files=self.total_files,
            cached_files=self.cached_files,
            ready_bytes=self.ready_bytes(),
            total_bytes=self.total_bytes,
            cached_bytes=self.cached_bytes,
            current_file=current_file if current_file is not None else self.current_file,
            current_file_bytes=(
                current_file_bytes
                if current_file_bytes is not None
                else self.current_file_bytes
            ),
            current_file_total_bytes=(
                current_file_total_bytes
                if current_file_total_bytes is not None
                else self.current_file_total_bytes
            ),
            failure_reason=(
                failure_reason if failure_reason is not None else self.failure_reason
            ),
        )

    def mark_cached_files(self) -> None:
        if self.total_files <= 0:
            return
        if self.cached_files >= self.total_files:
            detail = "Required weight files are already available locally."
        elif self.cached_files > 0:
            detail = (
                f"Found {self.cached_files}/{self.total_files} required weight files already "
                "available locally."
            )
        else:
            detail = f"Downloading {self.total_files} required weight files."
        self.emit(init_stage="downloading", init_detail=detail, force=True)

    def start_file(self, file_name: str, total_bytes: Optional[int]) -> None:
        self.current_file = file_name
        self.current_file_bytes = 0
        self.current_file_total_bytes = total_bytes
        self.failure_reason = ""
        self.emit(
            init_stage="downloading",
            init_detail=f"Downloading required weight file {file_name}.",
            force=True,
        )

    def update_file(self, file_bytes: int, file_total_bytes: Optional[int], *, force: bool = False):
        self.current_file_bytes = max(file_bytes, 0)
        if file_total_bytes is not None:
            self.current_file_total_bytes = file_total_bytes
        self.emit(
            init_stage="downloading",
            init_detail=f"Downloading required weight file {self.current_file}.",
            force=force,
        )

    def finish_file(self, file_name: str, file_total_bytes: Optional[int]) -> None:
        completed_file_bytes = (
            file_total_bytes
            if file_total_bytes is not None
            else self.current_file_total_bytes
        )
        self.completed_files += 1
        if completed_file_bytes is not None:
            if self.completed_bytes is None:
                self.completed_bytes = 0
            self.completed_bytes += completed_file_bytes
        self.current_file = file_name
        self.current_file_total_bytes = completed_file_bytes
        self.current_file_bytes = completed_file_bytes
        self.emit(
            init_stage="downloading",
            init_detail=f"Downloaded required weight file {file_name}.",
            force=True,
        )

    def fail_file(self, file_name: str, reason: str) -> None:
        self.failure_reason = reason
        self.current_file = file_name
        self.emit(
            init_stage="failed",
            init_detail=f"Failed to download required weight file {file_name}.",
            failure_reason=reason,
            force=True,
        )


class _RuntimeProgressBar:
    def __init__(
        self,
        *,
        tracker: _DownloadProgressTracker,
        file_name: str,
        initial: int = 0,
        total: Optional[int] = None,
        **_: object,
    ) -> None:
        self.tracker = tracker
        self.file_name = file_name
        self.n = int(initial or 0)
        self.total = int(total) if total is not None else None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def update(self, n: int = 1) -> None:
        self.n += int(n)
        self.tracker.update_file(self.n, self.total)

    def close(self) -> None:
        self.tracker.update_file(self.n, self.total, force=True)


def _coerce_shared_state(shared_state: Optional[object]) -> Optional[SharedState]:
    if shared_state is None:
        return None
    if isinstance(shared_state, SharedState):
        return shared_state
    return SharedState(shared_state)


def _update_runtime_state(
    shared_state: Optional[SharedState],
    *,
    init_stage: str,
    init_detail: str,
    downloaded_files=None,
    total_files=None,
    cached_files=None,
    ready_bytes=None,
    total_bytes=None,
    cached_bytes=None,
    current_file: Optional[str] = None,
    current_file_bytes=None,
    current_file_total_bytes=None,
    failure_reason: Optional[str] = None,
) -> None:
    if shared_state is None:
        return
    payload = {
        "status": shared_state.get_status(),
        "model_name": shared_state.get("model_name"),
        "init_stage": init_stage,
        "init_detail": init_detail,
    }
    if downloaded_files is not None:
        payload["downloaded_files"] = downloaded_files
    if total_files is not None:
        payload["total_files"] = total_files
    if cached_files is not None:
        payload["cached_files"] = cached_files
    if ready_bytes is not None:
        payload["ready_bytes"] = ready_bytes
    if total_bytes is not None:
        payload["total_bytes"] = total_bytes
    if cached_bytes is not None:
        payload["cached_bytes"] = cached_bytes
    if current_file is not None:
        payload["current_file"] = current_file
    if current_file_bytes is not None:
        payload["current_file_bytes"] = current_file_bytes
    if current_file_total_bytes is not None:
        payload["current_file_total_bytes"] = current_file_total_bytes
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason
    shared_state.update_runtime_state(**payload)


def _is_weight_file(path: str) -> bool:
    return any(fnmatch(path, pattern) for pattern in EXCLUDE_WEIGHT_PATTERNS)


def _list_remote_weight_files(repo_id: str) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id)
    return sorted(path for path in files if _is_weight_file(path))


def _list_local_weight_files(model_path: Path) -> list[str]:
    weight_files: list[str] = []
    for path in model_path.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(model_path).as_posix()
        if _is_weight_file(relative_path):
            weight_files.append(relative_path)
    return sorted(weight_files)


def _resolve_weight_file_sizes(repo_id: str, weight_files: list[str]) -> dict[str, Optional[int]]:
    if not weight_files:
        return {}
    api = HfApi()
    sizes: dict[str, Optional[int]] = {path: None for path in weight_files}
    try:
        repo_files = api.get_paths_info(repo_id=repo_id, paths=weight_files)
    except Exception as error:
        logger.warning("Failed to resolve remote weight file sizes for %s: %s", repo_id, error)
        return sizes

    for repo_file in repo_files:
        path = getattr(repo_file, "path", None)
        if not isinstance(path, str) or path not in sizes:
            continue
        size = getattr(repo_file, "size", None)
        if size is None:
            lfs = getattr(repo_file, "lfs", None) or {}
            if isinstance(lfs, dict):
                size = lfs.get("size")
        if size is not None:
            sizes[path] = int(size)
    return sizes


def _build_weight_file_manifests(
    repo_id: str,
    weight_files: list[str],
    *,
    model_path: Path,
    cache_dir: Optional[str],
    force_download: bool,
    is_remote: bool,
) -> list[WeightFileManifest]:
    remote_sizes = _resolve_weight_file_sizes(repo_id, weight_files) if is_remote else {}
    manifests: list[WeightFileManifest] = []
    for weight_file in weight_files:
        local_path = model_path / weight_file
        available_before = local_path.exists()
        available_size_path = local_path if available_before else None
        needs_materialization = False

        if not available_before and is_remote and not force_download:
            cached_path = try_to_load_from_cache(
                repo_id=repo_id,
                filename=weight_file,
                cache_dir=cache_dir,
            )
            if isinstance(cached_path, str) and os.path.isfile(cached_path):
                available_before = True
                available_size_path = Path(cached_path)
                needs_materialization = True

        total_bytes = remote_sizes.get(weight_file)
        if total_bytes is None and available_size_path is not None:
            try:
                total_bytes = int(available_size_path.stat().st_size)
            except OSError:
                total_bytes = None

        manifests.append(
            WeightFileManifest(
                repo_path=weight_file,
                file_name=Path(weight_file).name,
                local_path=local_path,
                total_bytes=total_bytes,
                available_before=available_before,
                needs_materialization=needs_materialization,
            )
        )
    return manifests


def _download_weight_files(
    repo_id: str,
    weight_files: list[str],
    *,
    model_path: Path,
    cache_dir: Optional[str],
    force_download: bool,
    local_files_only: bool,
    shared_state: Optional[SharedState],
    is_remote: bool,
) -> None:
    manifests = _build_weight_file_manifests(
        repo_id,
        weight_files,
        model_path=model_path,
        cache_dir=cache_dir,
        force_download=force_download,
        is_remote=is_remote,
    )
    total_weight_files = len(manifests)
    total_weight_bytes = (
        sum(int(manifest.total_bytes) for manifest in manifests if manifest.total_bytes is not None)
        if manifests and all(manifest.total_bytes is not None for manifest in manifests)
        else None
    )
    cached_manifests = [manifest for manifest in manifests if manifest.available_before]
    cached_files = len(cached_manifests)
    cached_bytes = (
        sum(int(manifest.total_bytes) for manifest in cached_manifests if manifest.total_bytes is not None)
        if total_weight_bytes is not None
        else None
    )
    progress_tracker = _DownloadProgressTracker(
        shared_state,
        total_files=total_weight_files,
        total_bytes=total_weight_bytes,
        cached_files=cached_files,
        cached_bytes=cached_bytes,
    )
    logger.info(f"Downloading {total_weight_files} weight files")
    progress_tracker.mark_cached_files()

    if total_weight_files > 0 and cached_files == total_weight_files:
        return

    for index, manifest in enumerate(manifests, start=1):
        if manifest.available_before:
            if manifest.needs_materialization:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=manifest.repo_path,
                    cache_dir=cache_dir,
                    force_download=False,
                    local_files_only=True,
                )
            logger.info(
                "Weight file %d/%d already available locally: %s",
                index,
                total_weight_files,
                manifest.repo_path,
            )
            continue

        progress_tracker.start_file(manifest.file_name, manifest.total_bytes)
        logger.info(
            "Downloading weight file %d/%d: %s", index, total_weight_files, manifest.repo_path
        )
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=manifest.repo_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                tqdm_class=lambda **kwargs: _RuntimeProgressBar(
                    tracker=progress_tracker,
                    file_name=manifest.file_name,
                    **kwargs,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to download {manifest.repo_path} for {repo_id}: {e}")
            logger.error(
                "This node cannot reach Hugging Face Hub to download weight files. "
                "Please check network connectivity or pre-download the model."
            )
            progress_tracker.fail_file(manifest.file_name, str(e))
            raise
        progress_tracker.finish_file(manifest.file_name, manifest.total_bytes)
        logger.info(
            "Downloaded weight file %d/%d: %s", index, total_weight_files, manifest.repo_path
        )


def download_metadata_only(
    repo_id: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    shared_state: Optional[object] = None,
) -> Path:
    shared_state = _coerce_shared_state(shared_state)
    # If a local path is provided, return it directly without contacting HF Hub
    local_path = Path(repo_id)
    if local_path.exists():
        return local_path

    _update_runtime_state(
        shared_state,
        init_stage="resolving-metadata",
        init_detail=f"Resolving model metadata for {repo_id}.",
    )
    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        ignore_patterns=EXCLUDE_WEIGHT_PATTERNS,
        force_download=force_download,
        local_files_only=local_files_only,
    )
    _update_runtime_state(
        shared_state,
        init_stage="resolving-metadata",
        init_detail=f"Resolved model metadata for {repo_id}.",
    )
    return Path(path)


def selective_model_download(
    repo_id: str,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    shared_state: Optional[object] = None,
) -> Path:
    shared_state = _coerce_shared_state(shared_state)
    # Handle local model directory
    local_path = Path(repo_id)
    if local_path.exists():
        model_path = local_path
        logger.debug(f"Using local model path: {model_path}")
        is_remote = False
    else:
        logger.info(f"Resolving model metadata for {repo_id}")
        model_path = download_metadata_only(
            repo_id=repo_id,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            shared_state=shared_state,
        )
        logger.info(f"Resolved model metadata for {repo_id}")
        is_remote = True

    if start_layer is not None and end_layer is not None:
        logger.debug(f"Determining required weight files for layers [{start_layer}, {end_layer})")

        needed_weight_files = determine_needed_weight_files_for_download(
            model_path=model_path,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        if is_remote:
            if not needed_weight_files:
                logger.info(
                    "Could not determine specific weight files for layers "
                    f"[{start_layer}, {end_layer}); downloading full model snapshot"
                )
                weight_files = _list_remote_weight_files(repo_id)
                _download_weight_files(
                    repo_id,
                    weight_files,
                    model_path=model_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    shared_state=shared_state,
                    is_remote=is_remote,
                )
                logger.info("Downloaded full model snapshot")
            else:
                _download_weight_files(
                    repo_id,
                    needed_weight_files,
                    model_path=model_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    shared_state=shared_state,
                    is_remote=is_remote,
                )
                logger.debug(f"Downloaded weight files for layers [{start_layer}, {end_layer})")
        else:
            if not needed_weight_files:
                needed_weight_files = _list_local_weight_files(model_path)
            _download_weight_files(
                repo_id,
                needed_weight_files,
                model_path=model_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                shared_state=shared_state,
                is_remote=is_remote,
            )
            logger.debug("Local model path detected; required weight files already available")
    else:
        # No layer range specified
        weight_files = _list_remote_weight_files(repo_id) if is_remote else _list_local_weight_files(model_path)
        _download_weight_files(
            repo_id,
            weight_files,
            model_path=model_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            shared_state=shared_state,
            is_remote=is_remote,
        )
        if is_remote:
            logger.info("Downloaded full model snapshot")
        else:
            logger.debug("No layer range specified and using local path; required weights already available")

    return model_path


def get_model_path_with_selective_download(
    model_path_or_repo: str,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    local_files_only: bool = False,
    shared_state: Optional[object] = None,
) -> Path:
    return selective_model_download(
        repo_id=model_path_or_repo,
        start_layer=start_layer,
        end_layer=end_layer,
        local_files_only=local_files_only,
        shared_state=shared_state,
    )
