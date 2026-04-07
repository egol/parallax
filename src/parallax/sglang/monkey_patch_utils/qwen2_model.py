import logging


logger = logging.getLogger(__name__)


def should_use_local_embed_for_lm_head(pp_group, tie_word_embeddings: bool) -> bool:
    """Reuse local embed tokens for lm_head only on a full single-shard model.

    Parallax keeps sglang's PP world size at 1 and redefines first/last rank from
    the local layer range. On a split last shard, `world_size == 1` is still true
    but `embed_tokens` is a PPMissingLayer, so blindly reusing it as lm_head is
    incorrect.
    """

    return bool(tie_word_embeddings and pp_group.is_first_rank and pp_group.is_last_rank)


def should_load_tied_lm_head_from_embed(pp_group, tie_word_embeddings: bool) -> bool:
    """Load lm_head from embed weights on split last shards with tied embeddings."""

    return bool(tie_word_embeddings and pp_group.is_last_rank and not pp_group.is_first_rank)


def apply_qwen2_monkey_patch():
    """Patch sglang's Qwen2 lm_head construction for decentralized layer splits."""

    from torch import nn
    from sglang.srt.layers.utils import PPMissingLayer
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
    from sglang.srt.utils import add_prefix

    try:
        import sglang.srt.models.qwen2 as m
    except Exception as e:  # pragma: no cover - optional dependency in local host env
        logger.warning("Failed to import sglang.srt.models.qwen2 for monkey patch: %s", e)
        return

    def _pp_qwen2_for_causal_init(self, config, quant_config=None, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.pp_group = m.get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = m.Qwen2Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        if self.pp_group.is_last_rank:
            if should_use_local_embed_for_lm_head(self.pp_group, config.tie_word_embeddings):
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = m.LogitsProcessor(config)
        self.pooler = m.Pooler(pooling_type=m.PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False

    def _pp_qwen2_load_weights(self, weights):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = m.get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (layer_id < self.model.start_layer or layer_id >= self.model.end_layer)
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if self.config.tie_word_embeddings:
                if should_use_local_embed_for_lm_head(
                    self.pp_group, self.config.tie_word_embeddings
                ) and "lm_head.weight" in name:
                    continue

                if should_load_tied_lm_head_from_embed(
                    self.pp_group, self.config.tie_word_embeddings
                ) and name == "model.embed_tokens.weight":
                    name = "lm_head.weight"
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", m.default_weight_loader)
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning("Parameter %s not found in params_dict", name)

    m.Qwen2ForCausalLM.__init__ = _pp_qwen2_for_causal_init
    m.Qwen2ForCausalLM.load_weights = _pp_qwen2_load_weights
