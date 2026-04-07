#!/usr/bin/env python3

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from parallax.server.request_prep import build_prompt_input_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run unsplit real-model baselines for centralized split smoke cases."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases-json", required=True)
    args = parser.parse_args()

    cases = json.loads(args.cases_json)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for case in cases:
        request_payload = dict(case["request"])
        input_ids = build_prompt_input_ids(tokenizer, request_payload)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=int(request_payload.get("max_tokens", 16)),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][len(input_ids) :].tolist()
        results.append(
            {
                "name": case["name"],
                "generated_token_ids": generated_ids,
                "completion_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
            }
        )

    print(json.dumps(results))


if __name__ == "__main__":
    main()
