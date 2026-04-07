"""
CPU-only import shims for SGLang dependencies.

SGLang's non-AMX CPU fallback imports a few vLLM modules for fused helper
layers even when it later runs plain PyTorch code on CPU. The local Parallax
CPU localnet image does not install full vLLM because that drags GPU-oriented
dependencies into the build. These shims provide the minimal APIs SGLang needs
to import and run its native CPU fallback path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cpu_engine_requested() -> bool:
    return os.getenv("SGLANG_USE_CPU_ENGINE", "0") == "1"


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


class _SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x.shape[-1] // 2
        return F.silu(x[..., :hidden]) * x[..., hidden:]


class _GeluAndMul(nn.Module):
    def __init__(self, approximate: str = "tanh") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x.shape[-1] // 2
        return F.gelu(x[..., :hidden], approximate=self.approximate) * x[..., hidden:]


class _RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        cast_x_before_out_mul: bool = False,
        fp32_residual: bool = False,
        weight_dtype: Optional[torch.dtype] = None,
        override_orig_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        self.cast_x_before_out_mul = cast_x_before_out_mul
        self.fp32_residual = fp32_residual
        self.override_orig_dtype = override_orig_dtype

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = self.override_orig_dtype or x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.clone() if self.fp32_residual else x.to(orig_dtype)

        x_var = x
        if self.variance_size_override is not None:
            x_var = x[..., : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        if self.cast_x_before_out_mul:
            x = self.weight * x.to(orig_dtype)
        else:
            x = (x * self.weight).to(orig_dtype)
        return x if residual is None else (x, residual)


class _GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)


def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def _rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> None:
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query_view = query.view(num_tokens, -1, head_size)
    query.copy_(
        _apply_rotary_emb(query_view, cos, sin, is_neox_style).reshape(query_shape)
    )

    key_shape = key.shape
    key_view = key.view(num_tokens, -1, head_size)
    key.copy_(_apply_rotary_emb(key_view, cos, sin, is_neox_style).reshape(key_shape))


def _build_paged_extend_indices(
    *,
    page_size: int,
    free_pages: list[int],
    prefix_lens: list[int],
    seq_lens: list[int],
    last_locs: list[int],
) -> tuple[list[int], int]:
    out_indices: list[int] = []
    page_cursor = 0

    for pre_len, seq_len, last_loc in zip(prefix_lens, seq_lens, last_locs):
        extend_len = seq_len - pre_len
        if extend_len <= 0:
            continue

        aligned_end = ((pre_len + page_size - 1) // page_size) * page_size
        num_part1 = max(0, min(seq_len, aligned_end) - pre_len)
        if num_part1 > 0:
            out_indices.extend(range(last_loc + 1, last_loc + 1 + num_part1))

        remaining = extend_len - num_part1
        while remaining > 0:
            page = free_pages[page_cursor]
            page_cursor += 1
            take = min(page_size, remaining)
            start = page * page_size
            out_indices.extend(range(start, start + take))
            remaining -= take

    return out_indices, page_cursor


def _build_paged_decode_indices(
    *,
    page_size: int,
    free_pages: list[int],
    seq_lens: list[int],
    last_locs: list[int],
) -> tuple[list[int], int]:
    out_indices: list[int] = []
    page_cursor = 0

    for seq_len, last_loc in zip(seq_lens, last_locs):
        if seq_len % page_size == 0:
            page = free_pages[page_cursor]
            page_cursor += 1
            out_indices.append(page * page_size)
        else:
            out_indices.append(last_loc + 1)

    return out_indices, page_cursor


def _patch_cpu_paged_allocator() -> None:
    from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

    if getattr(PagedTokenToKVPoolAllocator, "_parallax_cpu_patched", False):
        return

    def _alloc_extend_cpu(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        if self.debug_mode:
            assert torch.all((last_loc + 1) % self.page_size == prefix_lens % self.page_size)

        prefix_lens_list = [int(x) for x in prefix_lens_cpu.tolist()]
        seq_lens_list = [int(x) for x in seq_lens_cpu.tolist()]
        last_loc_list = [int(x) for x in last_loc.tolist()]
        num_new_pages = sum(
            ((seq_len + self.page_size - 1) // self.page_size)
            - ((pre_len + self.page_size - 1) // self.page_size)
            for pre_len, seq_len in zip(prefix_lens_list, seq_lens_list)
        )

        if self.need_sort and num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_new_pages > len(self.free_pages):
            return None

        free_pages = [int(x) for x in self.free_pages[:num_new_pages].tolist()]
        out_list, pages_used = _build_paged_extend_indices(
            page_size=self.page_size,
            free_pages=free_pages,
            prefix_lens=prefix_lens_list,
            seq_lens=seq_lens_list,
            last_locs=last_loc_list,
        )
        if len(out_list) != extend_num_tokens:
            raise RuntimeError(
                f"CPU paged allocator produced {len(out_list)} indices for {extend_num_tokens} tokens"
            )

        self.free_pages = self.free_pages[pages_used:]
        return torch.tensor(out_list, dtype=torch.int64, device=self.device)

    def _alloc_decode_cpu(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all((last_loc + 2) % self.page_size == seq_lens % self.page_size)

        seq_lens_list = [int(x) for x in seq_lens_cpu.tolist()]
        last_loc_list = [int(x) for x in last_loc.tolist()]
        num_new_pages = sum(1 for seq_len in seq_lens_list if seq_len % self.page_size == 0)

        if self.need_sort and num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_new_pages > len(self.free_pages):
            return None

        free_pages = [int(x) for x in self.free_pages[:num_new_pages].tolist()]
        out_list, pages_used = _build_paged_decode_indices(
            page_size=self.page_size,
            free_pages=free_pages,
            seq_lens=seq_lens_list,
            last_locs=last_loc_list,
        )

        self.free_pages = self.free_pages[pages_used:]
        return torch.tensor(out_list, dtype=torch.int64, device=self.device)

    PagedTokenToKVPoolAllocator.alloc_extend = _alloc_extend_cpu
    PagedTokenToKVPoolAllocator.alloc_decode = _alloc_decode_cpu
    PagedTokenToKVPoolAllocator._parallax_cpu_patched = True


def _parse_lscpu_topology_with_blank_numa_fallback(output: str) -> list[tuple[int, int, int, int]]:
    cpu_info: list[tuple[int, int, int, int]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise RuntimeError(f"Unexpected lscpu topology line: {raw_line!r}")
        cpu_str, core_str, socket_str, node_str = parts
        if not cpu_str or not core_str:
            raise RuntimeError(f"Incomplete lscpu topology line: {raw_line!r}")
        if not socket_str:
            socket_str = "0"
        if not node_str:
            # Docker on macOS/WSL commonly leaves NUMA blank for a single-node CPU topology.
            node_str = socket_str or "0"
        cpu_info.append((int(cpu_str), int(core_str), int(socket_str), int(node_str)))
    return cpu_info


def _patch_cpu_lscpu_topology() -> None:
    from sglang.srt.utils import common as sgl_common

    if getattr(sgl_common, "_parallax_cpu_topology_patched", False):
        return

    def _parse_lscpu_topology():
        try:
            output = subprocess.check_output(
                ["lscpu", "-p=CPU,Core,Socket,Node"], text=True
            )
        except Exception as exc:  # pragma: no cover - mirrors upstream error path
            raise RuntimeError(f"Unexpected error running 'lscpu': {exc}") from exc
        return _parse_lscpu_topology_with_blank_numa_fallback(output)

    sgl_common.parse_lscpu_topology = _parse_lscpu_topology
    sgl_common._parallax_cpu_topology_patched = True


def install_cpu_import_shims() -> None:
    if not _cpu_engine_requested():
        return
    if getattr(install_cpu_import_shims, "_installed", False):
        return

    vllm_module = _ensure_module("vllm")
    model_executor_module = _ensure_module("vllm.model_executor")
    layers_module = _ensure_module("vllm.model_executor.layers")
    activation_module = _ensure_module("vllm.model_executor.layers.activation")
    activation_module.SiluAndMul = _SiluAndMul
    activation_module.GeluAndMul = _GeluAndMul

    layernorm_module = _ensure_module("vllm.model_executor.layers.layernorm")
    layernorm_module.RMSNorm = _RMSNorm
    layernorm_module.GemmaRMSNorm = _GemmaRMSNorm

    custom_ops_module = _ensure_module("vllm._custom_ops")
    custom_ops_module.rotary_embedding = _rotary_embedding

    vllm_module.model_executor = model_executor_module
    vllm_module._custom_ops = custom_ops_module
    model_executor_module.layers = layers_module
    layers_module.activation = activation_module
    layers_module.layernorm = layernorm_module
    _patch_cpu_lscpu_topology()
    _patch_cpu_paged_allocator()

    install_cpu_import_shims._installed = True
