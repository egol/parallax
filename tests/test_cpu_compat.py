import pytest

pytest.importorskip("torch")

from parallax.sglang.cpu_compat import _build_paged_decode_indices, _build_paged_extend_indices


def test_build_paged_extend_indices_mixes_partial_and_new_pages():
    out_indices, pages_used = _build_paged_extend_indices(
        page_size=4,
        free_pages=[10, 11, 12],
        prefix_lens=[2, 4],
        seq_lens=[5, 6],
        last_locs=[5, 19],
    )

    assert out_indices == [6, 7, 40, 44, 45]
    assert pages_used == 2


def test_build_paged_decode_indices_crosses_page_boundaries_only_when_needed():
    out_indices, pages_used = _build_paged_decode_indices(
        page_size=4,
        free_pages=[7, 8],
        seq_lens=[4, 5, 8],
        last_locs=[19, 21, 35],
    )

    assert out_indices == [28, 22, 32]
    assert pages_used == 2
