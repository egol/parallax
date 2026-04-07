import asyncio

from backend.server.request_handler import RequestHandler


class FakeStreamingResponse:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.cancelled = False

    def __iter__(self):
        return iter(self._chunks)

    def cancel(self):
        self.cancelled = True


class FakeStub:
    def __init__(self, response):
        self.response = response

    def chat_completion(self, _request_data):
        return self.response


class FakeSchedulerManage:
    def __init__(self, routing_table):
        self.routing_table = list(routing_table)
        self.released = []

    def get_schedule_status(self):
        return "available"

    def get_routing_table(self, _request_id, _received_ts):
        return list(self.routing_table)

    def release_routing_table(self, routing_table):
        self.released.append(list(routing_table))


def test_non_stream_request_releases_routing_table_and_preserves_status():
    handler = RequestHandler()
    handler.set_scheduler_manage(FakeSchedulerManage(["node-a", "scheduler-peer", "node-b"]))
    handler.get_stub = lambda _node_id: FakeStub(
        FakeStreamingResponse([b'{"error":"busy","status_code":429}'])
    )

    response = asyncio.run(
        handler.v1_chat_completions({"stream": False}, "req-1", 0)
    )

    assert response.status_code == 429
    assert handler.scheduler_manage.released == [["node-a", "scheduler-peer", "node-b"]]


def test_streaming_request_releases_routing_table_on_generator_shutdown():
    handler = RequestHandler()
    handler.set_scheduler_manage(FakeSchedulerManage(["node-a", "scheduler-peer", "node-b"]))
    upstream = FakeStreamingResponse([b"data: chunk-1\n\n", b"data: [DONE]\n\n"])
    handler.get_stub = lambda _node_id: FakeStub(upstream)

    response = asyncio.run(handler.v1_chat_completions({"stream": True}, "req-2", 0))

    async def consume(body_iterator):
        chunks = []
        async for chunk in body_iterator:
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(consume(response.body_iterator))

    assert chunks == [b"data: chunk-1\n\n", b"data: [DONE]\n\n"]
    assert upstream.cancelled is True
    assert handler.scheduler_manage.released == [["node-a", "scheduler-peer", "node-b"]]
