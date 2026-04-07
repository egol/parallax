import asyncio
from http import HTTPStatus
from types import SimpleNamespace

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch might be unavailable in CI
    import importlib.machinery
    import sys
    import types

    torch_stub = types.ModuleType("torch")
    torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)

    class _DeviceStatus:
        @staticmethod
        def is_available():
            return False

    torch_stub.cuda = _DeviceStatus()
    torch_stub.mps = _DeviceStatus()
    torch_stub.float16 = "float16"
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.float32 = "float32"
    sys.modules.setdefault("torch", torch_stub)

from parallax.server.http_server import HTTPHandler, HTTPRequestInfo


def test_http_handler_marks_non_stream_error():
    async def scenario():
        handler = HTTPHandler.__new__(HTTPHandler)
        handler.processing_requests = {}

        rid = "req-non-stream"
        request_info = HTTPRequestInfo(id=rid, stream=False)
        handler.processing_requests[rid] = request_info

        await handler._handle_executor_error(
            rid,
            {
                "error": "Invalid template",
                "error_type": "TemplateError",
                "status_code": HTTPStatus.BAD_REQUEST.value,
            },
        )
        return request_info

    request_info = asyncio.run(scenario())

    assert request_info.is_finish is True
    assert request_info.finish_reason == "error"
    assert request_info.error_message == "Invalid template"
    assert request_info.error_type == "TemplateError"
    assert request_info.error_status == HTTPStatus.BAD_REQUEST


def test_http_handler_stream_error_pushes_queue_event():
    async def scenario():
        handler = HTTPHandler.__new__(HTTPHandler)
        handler.processing_requests = {}

        rid = "req-stream"
        request_info = HTTPRequestInfo(id=rid, stream=True)
        request_info.token_queue = asyncio.Queue()
        handler.processing_requests[rid] = request_info

        await handler._handle_executor_error(
            rid,
            {
                "error": "Executor failure",
                "error_type": "InternalServerError",
                "status_code": HTTPStatus.INTERNAL_SERVER_ERROR.value,
            },
        )

        error_chunk = await request_info.token_queue.get()
        sentinel = await request_info.token_queue.get()
        return request_info, error_chunk, sentinel

    request_info, error_chunk, sentinel = asyncio.run(scenario())

    assert error_chunk["type"] == "error"
    assert error_chunk["payload"]["message"] == "Executor failure"
    assert error_chunk["payload"]["type"] == "InternalServerError"
    assert error_chunk["payload"]["code"] == HTTPStatus.INTERNAL_SERVER_ERROR.value
    assert sentinel is None


def test_http_handler_preserves_final_token_on_length_finish():
    async def scenario():
        handler = HTTPHandler.__new__(HTTPHandler)
        handler.processing_requests = {}
        handler.recv_from_executor = SimpleNamespace(
            recv_pyobj=_single_message_recv(
                {
                    "rid": "req-length",
                    "prompt_tokens": 4,
                    "next_token_id": 3925,
                    "length": True,
                }
            )
        )

        request_info = HTTPRequestInfo(id="req-length", stream=False)
        request_info.detokenizer = _FakeDetokenizer("OK")
        handler.processing_requests["req-length"] = request_info

        task = asyncio.create_task(handler._handle_loop())
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return request_info

    request_info = asyncio.run(scenario())

    assert request_info.text == "OK"
    assert request_info.completion_tokens == 1
    assert request_info.finish_reason == "length"
    assert request_info.is_finish is True


def test_http_handler_preserves_final_token_on_eos_finish():
    async def scenario():
        handler = HTTPHandler.__new__(HTTPHandler)
        handler.processing_requests = {}
        handler.recv_from_executor = SimpleNamespace(
            recv_pyobj=_single_message_recv(
                {
                    "rid": "req-eos",
                    "prompt_tokens": 4,
                    "next_token_id": 3925,
                    "eos": True,
                }
            )
        )

        request_info = HTTPRequestInfo(id="req-eos", stream=False)
        request_info.detokenizer = _FakeDetokenizer("OK")
        handler.processing_requests["req-eos"] = request_info

        task = asyncio.create_task(handler._handle_loop())
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return request_info

    request_info = asyncio.run(scenario())

    assert request_info.text == "OK"
    assert request_info.completion_tokens == 1
    assert request_info.finish_reason == "stop"
    assert request_info.is_finish is True


def test_http_handler_suppresses_abort_output():
    async def scenario():
        handler = HTTPHandler.__new__(HTTPHandler)
        handler.processing_requests = {}
        handler.recv_from_executor = SimpleNamespace(
            recv_pyobj=_single_message_recv(
                {
                    "rid": "req-abort",
                    "prompt_tokens": 4,
                    "next_token_id": 3925,
                    "abort": True,
                }
            )
        )

        request_info = HTTPRequestInfo(id="req-abort", stream=False)
        request_info.detokenizer = _FakeDetokenizer("OK")
        handler.processing_requests["req-abort"] = request_info

        task = asyncio.create_task(handler._handle_loop())
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return request_info

    request_info = asyncio.run(scenario())

    assert request_info.text == ""
    assert request_info.completion_tokens == 1
    assert request_info.finish_reason == "abort"
    assert request_info.is_finish is True


class _FakeDetokenizer:
    def __init__(self, segment):
        self.last_segment = ""
        self._segment = segment

    def add_token(self, _token_id):
        self.last_segment = self._segment


def _single_message_recv(message):
    state = {"used": False}

    async def _recv_pyobj():
        if state["used"]:
            await asyncio.sleep(3600)
        state["used"] = True
        return message

    return _recv_pyobj
