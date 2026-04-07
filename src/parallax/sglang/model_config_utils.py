def should_keep_tied_word_embeddings(hf_config, start_layer: int, end_layer: int) -> bool:
    num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
    tie_word_embeddings = bool(getattr(hf_config, "tie_word_embeddings", False))
    if not tie_word_embeddings:
        return False
    if num_hidden_layers is None:
        return False
    # The last shard still needs tied output weights to produce correct logits
    # for models such as Qwen2/Qwen2.5 where lm_head is tied to embeddings.
    return end_layer == num_hidden_layers
