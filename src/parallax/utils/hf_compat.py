from __future__ import annotations

import json
import subprocess
import sys
from functools import lru_cache
from typing import Any, Dict, Optional

from transformers import AutoConfig, AutoTokenizer

_mlx_convert_chat = None
_mlx_process_message_content = None
_BPEStreamingDetokenizer = None
_NaiveStreamingDetokenizer = None
_SPMStreamingDetokenizer = None
_mlx_is_bpe_decoder = None
_mlx_is_spm_decoder = None
_mlx_is_spm_decoder_no_space = None
_mlx_load_tokenizer = None
_mlx_load_config = None


@lru_cache(maxsize=1)
def _mlx_lm_import_is_safe() -> bool:
    """Probe mlx-lm in a subprocess so parent Python cannot abort on MLX init."""
    if sys.platform != "darwin":
        return False
    probe = [
        sys.executable,
        "-c",
        "from mlx_lm.utils import load_config; print('ok')",
    ]
    try:
        result = subprocess.run(
            probe,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return False
    return result.returncode == 0


@lru_cache(maxsize=1)
def _load_mlx_lm_symbols() -> bool:
    global _mlx_convert_chat
    global _mlx_process_message_content
    global _BPEStreamingDetokenizer
    global _NaiveStreamingDetokenizer
    global _SPMStreamingDetokenizer
    global _mlx_is_bpe_decoder
    global _mlx_is_spm_decoder
    global _mlx_is_spm_decoder_no_space
    global _mlx_load_tokenizer
    global _mlx_load_config

    if not _mlx_lm_import_is_safe():
        return False
    try:
        from mlx_lm.server import convert_chat as mlx_convert_chat
        from mlx_lm.server import process_message_content as mlx_process_message_content
        from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer as mlx_bpe_detokenizer
        from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer as mlx_naive_detokenizer
        from mlx_lm.tokenizer_utils import SPMStreamingDetokenizer as mlx_spm_detokenizer
        from mlx_lm.tokenizer_utils import _is_bpe_decoder as mlx_is_bpe_decoder
        from mlx_lm.tokenizer_utils import _is_spm_decoder as mlx_is_spm_decoder
        from mlx_lm.tokenizer_utils import _is_spm_decoder_no_space as mlx_is_spm_decoder_no_space
        from mlx_lm.tokenizer_utils import load as mlx_load_tokenizer
        from mlx_lm.utils import load_config as mlx_load_config
    except Exception:
        return False

    _mlx_convert_chat = mlx_convert_chat
    _mlx_process_message_content = mlx_process_message_content
    _BPEStreamingDetokenizer = mlx_bpe_detokenizer
    _NaiveStreamingDetokenizer = mlx_naive_detokenizer
    _SPMStreamingDetokenizer = mlx_spm_detokenizer
    _mlx_is_bpe_decoder = mlx_is_bpe_decoder
    _mlx_is_spm_decoder = mlx_is_spm_decoder
    _mlx_is_spm_decoder_no_space = mlx_is_spm_decoder_no_space
    _mlx_load_tokenizer = mlx_load_tokenizer
    _mlx_load_config = mlx_load_config
    return True


class StreamingDetokenizer:
    def __init__(self, tokenizer, tokenmap=None):
        self._tokenizer = tokenizer
        self.tokenmap = tokenmap
        self.reset()

    def reset(self):
        self._token_ids = []
        self._decoded_text = ""
        self.last_segment = ""

    def add_token(self, token_id: int):
        self._token_ids.append(token_id)
        decoded = self._tokenizer.decode(self._token_ids, skip_special_tokens=False)
        if decoded.startswith(self._decoded_text):
            self.last_segment = decoded[len(self._decoded_text) :]
        else:
            self.last_segment = self._tokenizer.decode([token_id], skip_special_tokens=False)
        self._decoded_text = decoded


class NaiveStreamingDetokenizer(
    StreamingDetokenizer
):
    def __new__(cls, tokenizer, tokenmap=None):
        if cls is NaiveStreamingDetokenizer and _load_mlx_lm_symbols() and _NaiveStreamingDetokenizer:
            return _NaiveStreamingDetokenizer(tokenizer, tokenmap)
        return super().__new__(cls)


class BPEStreamingDetokenizer(
    StreamingDetokenizer
):
    def __new__(cls, tokenizer, tokenmap=None):
        if cls is BPEStreamingDetokenizer and _load_mlx_lm_symbols() and _BPEStreamingDetokenizer:
            return _BPEStreamingDetokenizer(tokenizer, tokenmap)
        return super().__new__(cls)


class SPMStreamingDetokenizer(
    StreamingDetokenizer
):
    def __new__(cls, tokenizer, tokenmap=None):
        if cls is SPMStreamingDetokenizer and _load_mlx_lm_symbols() and _SPMStreamingDetokenizer:
            return _SPMStreamingDetokenizer(tokenizer, tokenmap)
        return super().__new__(cls)


def _decoder_type(decoder: Dict[str, Any] | None) -> str:
    if not isinstance(decoder, dict):
        return ""
    return str(decoder.get("type") or decoder.get("type_name") or "").lower()


def is_bpe_decoder(decoder: Dict[str, Any]) -> bool:
    _load_mlx_lm_symbols()
    if _mlx_is_bpe_decoder is not None:
        return _mlx_is_bpe_decoder(decoder)
    return _decoder_type(decoder) == "bytelevel"


def is_spm_decoder(decoder: Dict[str, Any]) -> bool:
    _load_mlx_lm_symbols()
    if _mlx_is_spm_decoder is not None:
        return _mlx_is_spm_decoder(decoder)
    return _decoder_type(decoder) == "sequence"


def is_spm_decoder_no_space(decoder: Dict[str, Any]) -> bool:
    _load_mlx_lm_symbols()
    if _mlx_is_spm_decoder_no_space is not None:
        return _mlx_is_spm_decoder_no_space(decoder)
    return False


def load_config(model_path) -> Dict[str, Any]:
    _load_mlx_lm_symbols()
    if _mlx_load_config is not None:
        return _mlx_load_config(model_path)
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    return config.to_dict()


def load_tokenizer(model_path, tokenizer_config_extra=None, trust_remote_code=True, **kwargs):
    _load_mlx_lm_symbols()
    tokenizer_config_extra = dict(tokenizer_config_extra or {})
    kwargs = dict(kwargs)
    kwargs.pop("eos_token_ids", None)

    if trust_remote_code:
        tokenizer_config_extra.setdefault("trust_remote_code", True)

    if _mlx_load_tokenizer is not None:
        return _mlx_load_tokenizer(
            model_path,
            tokenizer_config_extra=tokenizer_config_extra,
            **kwargs,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        **tokenizer_config_extra,
        **kwargs,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def process_message_content(messages) -> None:
    _load_mlx_lm_symbols()
    if _mlx_process_message_content is not None:
        _mlx_process_message_content(messages)
        return

    for message in messages:
        if isinstance(message, dict) and "content" in message:
            message["content"] = _normalize_message_content(message.get("content"))


def convert_chat(messages, role_mapping: Optional[Dict[str, str]] = None) -> str:
    _load_mlx_lm_symbols()
    if _mlx_convert_chat is not None:
        return _mlx_convert_chat(messages, role_mapping)

    lines = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        if role_mapping is not None:
            role = str(role_mapping.get(role, role))
        content = _normalize_message_content(message.get("content"))
        lines.append(f"{role}: {content}".rstrip())
    lines.append("assistant:")
    return "\n".join(lines)
