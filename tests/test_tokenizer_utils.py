from parallax.utils.tokenizer_utils import (
    ParallaxBPEStreamingDetokenizer,
    ParallaxSPMStreamingDetokenizer,
)


class _FakeTokenizer:
    clean_up_tokenization_spaces = False

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(97 + token_id) for token_id in token_ids)


def test_bpe_detokenizer_initializes_without_mlx_helpers():
    detokenizer = ParallaxBPEStreamingDetokenizer(_FakeTokenizer(), ["a", "b", "c"])

    detokenizer.add_token(0)

    assert detokenizer.last_segment


def test_spm_detokenizer_initializes_without_mlx_helpers():
    detokenizer = ParallaxSPMStreamingDetokenizer(_FakeTokenizer(), [b"a", b"b", b"c"])

    detokenizer.add_token(1)

    assert detokenizer.last_segment
