import pytest

from parallax.p2p.message_util import proto_to_request, request_to_proto
from parallax.server.executor.mlx_executor import MLXExecutor
from parallax.server.request_prep import build_prompt_input_ids
from parallax.server.request import InitialRequest
from parallax.server.sampling.sampling_params import SamplingParams

MLX_MODEL_REPO = "mlx-community/Qwen3-0.6B-bf16"
PROMPT_CASES = [
    (
        "sensitive",
        "Complete this sentence in one short sentence: The capital of China is",
        16,
    ),
    (
        "long",
        "Write a detailed paragraph about why cities build public parks, including social, health, and environmental benefits.",
        32,
    ),
]
CHAT_REQUEST_CASES = [
    (
        "chat_long_answer",
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise but clear assistant.",
                },
                {
                    "role": "user",
                    "content": (
                        "Write three detailed paragraphs explaining why cities build public parks. "
                        "Cover social connection, public health, biodiversity, heat reduction, "
                        "stormwater management, and neighborhood identity."
                    ),
                },
            ],
            "max_tokens": 96,
            "sampling_params": {"temperature": 0.0, "top_k": 1, "top_p": 1.0},
        },
    )
]
LAYER_SPLIT = (10, 18, 28)


def create_mlx_executor(start_layer: int, end_layer: int) -> MLXExecutor:
    return MLXExecutor(
        model_repo=MLX_MODEL_REPO,
        start_layer=start_layer,
        end_layer=end_layer,
        kv_cache_memory_fraction=0.1,
        dtype="bfloat16",
        device="mlx",
    )


def run_stage(executor, requests, batch_type, is_last_peer):
    executor.handle_input_requests(requests)
    executor.scheduler.admit_requests()
    input_batch = executor.scheduler.form_batch()
    prepared_batch = executor.prepare_batch_inputs(input_batch)
    assert prepared_batch is not None, "Failed to prepare batch inputs"
    batch_data = prepared_batch[batch_type]
    batch_output = executor.process_batch(batch_data, return_decoded_tokens=is_last_peer)
    output_reqs = executor.prepare_next_batch_requests(
        requests=batch_data["requests"],
        batch_output=batch_output,
        context_lengths=batch_data.get("context_lengths"),
    )
    return output_reqs, batch_output


def run_single_shard(executor, prompt: str, max_new_tokens: int) -> list[int]:
    return run_single_shard_from_input_ids(
        executor, executor.tokenizer.encode(prompt), max_new_tokens
    )


def run_single_shard_from_input_ids(
    executor, input_ids: list[int], max_new_tokens: int
) -> list[int]:
    greedy_sampling = SamplingParams(temperature=0.0, top_k=1)
    request = InitialRequest(
        request_id="single",
        input_ids=input_ids,
        sampling_params=greedy_sampling,
        max_new_tokens=max_new_tokens,
    )

    executor.handle_input_requests([request])
    generated_ids = []
    for batch_type in ["prefill_batch"] + ["decode_batch"] * (max_new_tokens - 1):
        executor.scheduler.admit_requests()
        input_batch = executor.scheduler.form_batch()
        prepared_batch = executor.prepare_batch_inputs(input_batch)
        assert prepared_batch is not None, "Failed to prepare batch inputs"
        batch_data = prepared_batch[batch_type]
        batch_output = executor.process_batch(batch_data, return_decoded_tokens=True)
        output_reqs = executor.prepare_next_batch_requests(
            requests=batch_data["requests"],
            batch_output=batch_output,
            context_lengths=batch_data.get("context_lengths"),
        )
        generated_ids.append(int(batch_output["hidden_states"][0].item()))
        executor.handle_input_requests(output_reqs)

    return generated_ids


def run_three_stage_split(executor_peer1, executor_peer2, executor_peer3, prompt: str, max_new_tokens: int):
    return run_three_stage_split_from_input_ids(
        executor_peer1,
        executor_peer2,
        executor_peer3,
        executor_peer1.tokenizer.encode(prompt),
        max_new_tokens,
    )


def run_three_stage_split_from_input_ids(
    executor_peer1,
    executor_peer2,
    executor_peer3,
    input_ids: list[int],
    max_new_tokens: int,
):
    greedy_sampling = SamplingParams(temperature=0.0, top_k=1)
    initial_request = InitialRequest(
        request_id="split",
        input_ids=input_ids,
        sampling_params=greedy_sampling,
        max_new_tokens=max_new_tokens,
    )

    prefill_reqs_out1, _ = run_stage(executor_peer1, [initial_request], "prefill_batch", False)
    prefill_proto_p1 = request_to_proto(prefill_reqs_out1, device="mlx")
    prefill_reqs_in2 = proto_to_request(prefill_proto_p1, device="mlx")
    prefill_reqs_out2, _ = run_stage(executor_peer2, prefill_reqs_in2, "prefill_batch", False)
    prefill_proto_p2 = request_to_proto(prefill_reqs_out2, device="mlx")
    prefill_reqs_in3 = proto_to_request(prefill_proto_p2, device="mlx")
    prefill_reqs_out3, gen_tokens = run_stage(
        executor_peer3, prefill_reqs_in3, "prefill_batch", True
    )
    first_rank_proto = request_to_proto(prefill_reqs_out3, device="mlx")
    generated_ids = [int(gen_tokens["hidden_states"][0].item())]

    for _ in range(max_new_tokens - 1):
        decode_reqs_in1 = proto_to_request(first_rank_proto, device="mlx")
        decode_reqs_out1, _ = run_stage(executor_peer1, decode_reqs_in1, "decode_batch", False)
        decode_proto_p1 = request_to_proto(decode_reqs_out1, device="mlx")

        decode_reqs_in2 = proto_to_request(decode_proto_p1, device="mlx")
        decode_reqs_out2, _ = run_stage(executor_peer2, decode_reqs_in2, "decode_batch", False)
        decode_proto_p2 = request_to_proto(decode_reqs_out2, device="mlx")

        decode_reqs_in3 = proto_to_request(decode_proto_p2, device="mlx")
        decode_reqs_out3, next_gen_tokens = run_stage(
            executor_peer3, decode_reqs_in3, "decode_batch", True
        )
        first_rank_proto = request_to_proto(decode_reqs_out3, device="mlx")
        generated_ids.append(int(next_gen_tokens["hidden_states"][0].item()))

    return generated_ids


@pytest.mark.mlx
def test_mlx_three_stage_split_matches_single_shard_token_ids():
    for case_name, prompt, max_new_tokens in PROMPT_CASES:
        single_executor = create_mlx_executor(0, LAYER_SPLIT[2])
        executor_peer1 = create_mlx_executor(0, LAYER_SPLIT[0])
        executor_peer2 = create_mlx_executor(LAYER_SPLIT[0], LAYER_SPLIT[1])
        executor_peer3 = create_mlx_executor(LAYER_SPLIT[1], LAYER_SPLIT[2])
        try:
            single_ids = run_single_shard(single_executor, prompt, max_new_tokens)
            split_ids = run_three_stage_split(
                executor_peer1,
                executor_peer2,
                executor_peer3,
                prompt,
                max_new_tokens,
            )
            assert split_ids == single_ids, (
                f"MLX split output diverged for case {case_name!r}: "
                f"split_ids={split_ids}, single_ids={single_ids}, "
                f"split_text={executor_peer1.tokenizer.decode(split_ids)!r}, "
                f"single_text={single_executor.tokenizer.decode(single_ids)!r}"
            )
        finally:
            single_executor.shutdown()
            executor_peer1.shutdown()
            executor_peer2.shutdown()
            executor_peer3.shutdown()


@pytest.mark.mlx
def test_mlx_three_stage_split_matches_single_shard_for_chat_requests():
    for case_name, raw_request in CHAT_REQUEST_CASES:
        single_executor = create_mlx_executor(0, LAYER_SPLIT[2])
        executor_peer1 = create_mlx_executor(0, LAYER_SPLIT[0])
        executor_peer2 = create_mlx_executor(LAYER_SPLIT[0], LAYER_SPLIT[1])
        executor_peer3 = create_mlx_executor(LAYER_SPLIT[1], LAYER_SPLIT[2])
        try:
            input_ids = build_prompt_input_ids(executor_peer1.tokenizer, raw_request)
            max_new_tokens = raw_request["max_tokens"]
            single_ids = run_single_shard_from_input_ids(
                single_executor, input_ids, max_new_tokens
            )
            split_ids = run_three_stage_split_from_input_ids(
                executor_peer1,
                executor_peer2,
                executor_peer3,
                input_ids,
                max_new_tokens,
            )
            assert split_ids == single_ids, (
                f"MLX split chat output diverged for case {case_name!r}: "
                f"split_ids={split_ids}, single_ids={single_ids}, "
                f"split_text={executor_peer1.tokenizer.decode(split_ids)!r}, "
                f"single_text={single_executor.tokenizer.decode(single_ids)!r}"
            )
        finally:
            single_executor.shutdown()
            executor_peer1.shutdown()
            executor_peer2.shutdown()
            executor_peer3.shutdown()
