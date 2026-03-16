from __future__ import annotations

import os
import multiprocessing
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def _test_runtime_mode() -> str:
    return os.getenv("PARALLAX_TEST_RUNTIME", "synthetic").strip().lower()


def _latest_user_message(request_data: dict[str, Any]) -> str:
    messages = request_data.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                return " ".join(part for part in parts if part)
    return "hello"


def _completion_payload(model_name: str, request_data: dict[str, Any]) -> dict[str, Any]:
    prompt = _latest_user_message(request_data)
    content = f"[parallax-test-mode:{model_name}] {prompt}"
    return {
        "id": f"chatcmpl-test-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": max(len(prompt.split()), 1),
            "completion_tokens": max(len(content.split()), 1),
            "total_tokens": max(len(prompt.split()), 1) + max(len(content.split()), 1),
        },
    }


class _TransformersResponder:
    def __init__(self, model_name: str):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - exercised in Docker harness
            raise RuntimeError(
                "Transformers test runtime requires both torch and transformers"
            ) from exc

        self.model_name = model_name
        self.torch = torch
        torch.set_num_threads(max(int(os.getenv("PARALLAX_TEST_TORCH_THREADS", "1")), 1))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def _build_prompt(self, request_data: dict[str, Any]) -> str:
        messages = request_data.get("messages") or []
        lines = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                content = " ".join(part for part in parts if part)
            lines.append(f"{role}: {content}".strip())
        lines.append("assistant:")
        return "\n".join(lines)

    def _generate_text(self, request_data: dict[str, Any]) -> str:
        prompt = self._build_prompt(request_data)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        max_new_tokens = max(int(request_data.get("max_tokens") or 24), 1)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return text or "[empty-generation]"

    def completion_payload(self, request_data: dict[str, Any]) -> dict[str, Any]:
        content = self._generate_text(request_data)
        prompt = _latest_user_message(request_data)
        return {
            "id": f"chatcmpl-test-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": max(len(prompt.split()), 1),
                "completion_tokens": max(len(content.split()), 1),
                "total_tokens": max(len(prompt.split()), 1) + max(len(content.split()), 1),
            },
        }


def _build_test_app(model_name: str) -> FastAPI:
    app = FastAPI()
    responder = None
    runtime_mode = _test_runtime_mode()
    if runtime_mode == "transformers":
        responder = _TransformersResponder(model_name)

    @app.post("/v1/chat/completions")
    async def chat_completions(raw_request: Request):
        request_data = await raw_request.json()
        payload = (
            responder.completion_payload(request_data)
            if responder is not None
            else _completion_payload(model_name, request_data)
        )
        if request_data.get("stream", False):

            async def stream():
                delta = payload["choices"][0]["message"]["content"]
                yield (
                    "data: "
                    + JSONResponse(
                        content={
                            "id": payload["id"],
                            "object": "chat.completion.chunk",
                            "created": payload["created"],
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": delta},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    ).body.decode()
                    + "\n\n"
                )
                yield (
                    "data: "
                    + JSONResponse(
                        content={
                            "id": payload["id"],
                            "object": "chat.completion.chunk",
                            "created": payload["created"],
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    ).body.decode()
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        return JSONResponse(content=payload)

    @app.get("/health")
    async def health():
        return {"ok": True, "model": model_name, "runtime": runtime_mode}

    return app


def _run_test_http_server(host: str, port: int, model_name: str):
    app = _build_test_app(model_name)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def launch_test_http_server(args) -> multiprocessing.Process:
    process = multiprocessing.Process(
        target=_run_test_http_server,
        args=(args.host, args.port, args.model_path or "parallax-test-model"),
    )
    process.start()
    return process


def stop_test_http_server(process: multiprocessing.Process | None):
    if process is None:
        return
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
