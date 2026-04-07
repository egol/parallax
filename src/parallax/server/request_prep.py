from __future__ import annotations

import copy
from typing import Any, Dict, List

from parallax.utils.hf_compat import convert_chat, process_message_content


def build_prompt_input_ids(tokenizer, raw_request: Dict[str, Any]) -> List[int]:
    """Build input token ids from a chat/completions-style request payload.

    This is the shared prompt-preparation path for both live executors and
    split-vs-unsplit test baselines so they cannot silently diverge.
    """

    if "messages" not in raw_request:
        raise ValueError("Request did not contain messages")

    messages = copy.deepcopy(raw_request["messages"])
    if getattr(tokenizer, "chat_template", None):
        process_message_content(messages)
        chat_template_kwargs = dict(raw_request.get("chat_template_kwargs") or {})
        if (
            "extra_body" in raw_request
            and isinstance(raw_request["extra_body"], dict)
            and "chat_template_kwargs" in raw_request["extra_body"]
        ):
            chat_template_kwargs.update(raw_request["extra_body"]["chat_template_kwargs"])

        prompt = tokenizer.apply_chat_template(
            messages,
            raw_request.get("tools") or None,
            tokenize=True,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
    else:
        prompt = convert_chat(messages, raw_request.get("role_mapping"))
        prompt = tokenizer.encode(prompt)

    return list(prompt)
