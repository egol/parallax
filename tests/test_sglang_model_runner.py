from types import SimpleNamespace

from parallax.sglang.model_config_utils import should_keep_tied_word_embeddings
from parallax.sglang.monkey_patch_utils.qwen2_model import (
    should_load_tied_lm_head_from_embed,
    should_use_local_embed_for_lm_head,
)


def test_keep_tied_embeddings_for_full_model_shard():
    hf_config = SimpleNamespace(num_hidden_layers=24, tie_word_embeddings=True)

    assert should_keep_tied_word_embeddings(hf_config, 0, 24) is True


def test_disable_tied_embeddings_for_non_last_split_shard():
    hf_config = SimpleNamespace(num_hidden_layers=24, tie_word_embeddings=True)

    assert should_keep_tied_word_embeddings(hf_config, 0, 12) is False


def test_keep_tied_embeddings_for_last_split_shard():
    hf_config = SimpleNamespace(num_hidden_layers=24, tie_word_embeddings=True)

    assert should_keep_tied_word_embeddings(hf_config, 12, 24) is True


def test_disable_tied_embeddings_when_model_is_not_tied():
    hf_config = SimpleNamespace(num_hidden_layers=24, tie_word_embeddings=False)

    assert should_keep_tied_word_embeddings(hf_config, 0, 24) is False


def test_qwen2_lm_head_reuses_local_embed_only_for_full_single_shard():
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)

    assert should_use_local_embed_for_lm_head(pp_group, True) is True


def test_qwen2_lm_head_does_not_reuse_missing_embed_on_split_last_shard():
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=True)

    assert should_use_local_embed_for_lm_head(pp_group, True) is False


def test_qwen2_lm_head_does_not_reuse_embed_when_weights_are_not_tied():
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)

    assert should_use_local_embed_for_lm_head(pp_group, False) is False


def test_qwen2_split_last_shard_loads_tied_lm_head_from_embed_weights():
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=True)

    assert should_load_tied_lm_head_from_embed(pp_group, True) is True


def test_qwen2_full_single_shard_does_not_double_load_tied_lm_head():
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)

    assert should_load_tied_lm_head_from_embed(pp_group, True) is False
