from parallax.server.request_prep import build_prompt_input_ids


class DummyChatTokenizer:
    chat_template = "template"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            }
        )
        return [11, 12, 13]


class DummyPlainTokenizer:
    chat_template = None

    def encode(self, prompt):
        assert prompt == "converted-prompt"
        return [21, 22]


def test_build_prompt_input_ids_chat_template_merges_kwargs(monkeypatch):
    processed = []

    def fake_process_message_content(messages):
        processed.append(messages)

    monkeypatch.setattr("parallax.server.request_prep.process_message_content", fake_process_message_content)

    tokenizer = DummyChatTokenizer()
    result = build_prompt_input_ids(
        tokenizer,
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "noop"}}],
            "chat_template_kwargs": {"foo": "base", "shared": "request"},
            "extra_body": {"chat_template_kwargs": {"bar": "extra", "shared": "extra"}},
        },
    )

    assert result == [11, 12, 13]
    assert processed
    assert tokenizer.calls == [
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "noop"}}],
            "tokenize": True,
            "add_generation_prompt": True,
            "kwargs": {"foo": "base", "shared": "extra", "bar": "extra"},
        }
    ]


def test_build_prompt_input_ids_falls_back_to_convert_chat(monkeypatch):
    monkeypatch.setattr(
        "parallax.server.request_prep.convert_chat",
        lambda messages, role_mapping: "converted-prompt",
    )

    tokenizer = DummyPlainTokenizer()
    result = build_prompt_input_ids(
        tokenizer,
        {
            "messages": [{"role": "user", "content": "hello"}],
            "role_mapping": {"user": "speaker"},
        },
    )

    assert result == [21, 22]
