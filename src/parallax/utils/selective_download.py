import inspect
import logging
import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

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
    current_file: Optional[str] = None,
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
    if current_file is not None:
        payload["current_file"] = current_file
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason
    shared_state.update_runtime_state(**payload)


def _is_weight_file(path: str) -> bool:
    return any(fnmatch(path, pattern) for pattern in EXCLUDE_WEIGHT_PATTERNS)


def _list_remote_weight_files(repo_id: str) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id)
    return sorted(path for path in files if _is_weight_file(path))


def _download_weight_files(
    repo_id: str,
    weight_files: list[str],
    *,
    model_path: Path,
    cache_dir: Optional[str],
    force_download: bool,
    local_files_only: bool,
    shared_state: Optional[SharedState],
) -> None:
    total_weight_files = len(weight_files)
    logger.info(f"Downloading {total_weight_files} weight files")
    for index, weight_file in enumerate(weight_files, start=1):
        weight_file_path = model_path / weight_file
        file_name = Path(weight_file).name
        downloaded_before = index - 1
        if weight_file_path.exists():
            _update_runtime_state(
                shared_state,
                init_stage="downloading",
                init_detail=f"Downloaded weight file {index}/{total_weight_files}: {file_name}.",
                downloaded_files=index,
                total_files=total_weight_files,
                current_file=file_name,
            )
            logger.info(f"Downloaded weight file {index}/{total_weight_files}: {weight_file}")
            continue

        _update_runtime_state(
            shared_state,
            init_stage="downloading",
            init_detail=f"Downloading weight file {index}/{total_weight_files}: {file_name}.",
            downloaded_files=downloaded_before,
            total_files=total_weight_files,
            current_file=file_name,
        )
        logger.info(f"Downloading weight file {index}/{total_weight_files}: {weight_file}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=weight_file,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )
        except Exception as e:
            logger.error(f"Failed to download {weight_file} for {repo_id}: {e}")
            logger.error(
                "This node cannot reach Hugging Face Hub to download weight files. "
                "Please check network connectivity or pre-download the model."
            )
            _update_runtime_state(
                shared_state,
                init_stage="failed",
                init_detail=f"Failed to download weight file {file_name}.",
                downloaded_files=downloaded_before,
                total_files=total_weight_files,
                current_file=file_name,
                failure_reason=str(e),
            )
            raise
        _update_runtime_state(
            shared_state,
            init_stage="downloading",
            init_detail=f"Downloaded weight file {index}/{total_weight_files}: {file_name}.",
            downloaded_files=index,
            total_files=total_weight_files,
            current_file=file_name,
        )
        logger.info(f"Downloaded weight file {index}/{total_weight_files}: {weight_file}")


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
                )
                logger.debug(f"Downloaded weight files for layers [{start_layer}, {end_layer})")
        else:
            # Local path: skip any downloads
            logger.debug("Local model path detected; skipping remote weight downloads")
    else:
        # No layer range specified
        if is_remote:
            weight_files = _list_remote_weight_files(repo_id)
            _download_weight_files(
                repo_id,
                weight_files,
                model_path=model_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                shared_state=shared_state,
            )
            logger.info("Downloaded full model snapshot")
        else:
            logger.debug("No layer range specified and using local path; nothing to download")

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
