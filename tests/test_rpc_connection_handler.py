import json

import httpx

from backend.server.rpc_connection_handler import RPCConnectionHandler
from parallax.p2p.proto import forward_pb2


class FakeLattica:
    def register_service(self, _service):
        return None

    def peer_id(self):
        return "scheduler-peer"


class FakeFuture:
    def result(self):
        return None


class ExplodingFuture:
    def result(self):
        raise AssertionError("rpc_abort fanout should not wait on downstream results")


class RecordingStub:
    def __init__(self, peer_id, calls, *, abort_future=None):
        self.peer_id = peer_id
        self.calls = calls
        self.abort_future = abort_future or FakeFuture()

    def rpc_pp_forward(self, request):
        self.calls.append(
            {
                "peer_id": self.peer_id,
                "forward_mode": request.forward_mode,
                "routing_tables": [list(req.routing_table) for req in request.reqs],
                "rids": [req.rid for req in request.reqs],
            }
        )
        return FakeFuture()

    def rpc_abort(self, request):
        self.calls.append(
            {
                "peer_id": self.peer_id,
                "rids": [req.rid for req in request.reqs],
            }
        )
        return self.abort_future


class RecordingSchedulerManage:
    def __init__(self):
        self.touched_routing_tables = []

    def touch_routing_table(self, routing_table):
        self.touched_routing_tables.append(list(routing_table))


def test_scheduler_proxy_groups_forward_requests_by_next_peer():
    handler = RPCConnectionHandler(FakeLattica(), scheduler=None, http_port=0)
    recorded_calls = []
    handler._get_transformer_stub = lambda peer_id: RecordingStub(peer_id, recorded_calls)

    request = forward_pb2.ForwardRequest()
    request.forward_mode = 7
    for rid, routing_table in [
        ("req-1", ["worker-a", "scheduler-peer", "worker-b"]),
        ("req-2", ["worker-c", "scheduler-peer", "worker-d"]),
        ("req-3", ["worker-x", "scheduler-peer", "worker-b"]),
    ]:
        req = request.reqs.add()
        req.rid = rid
        req.routing_table.extend(routing_table)

    handler.rpc_pp_forward(request)

    assert recorded_calls == [
        {
            "peer_id": "worker-b",
            "forward_mode": 7,
            "routing_tables": [
                ["scheduler-peer", "worker-b", "worker-a"],
                ["scheduler-peer", "worker-b", "worker-x"],
            ],
            "rids": ["req-1", "req-3"],
        },
        {
            "peer_id": "worker-d",
            "forward_mode": 7,
            "routing_tables": [["scheduler-peer", "worker-d", "worker-c"]],
            "rids": ["req-2"],
        },
    ]


def test_scheduler_proxy_touches_routing_tables_before_forwarding():
    scheduler_manage = RecordingSchedulerManage()
    handler = RPCConnectionHandler(
        FakeLattica(), scheduler=None, http_port=0, scheduler_manage=scheduler_manage
    )
    recorded_calls = []
    handler._get_transformer_stub = lambda peer_id: RecordingStub(peer_id, recorded_calls)

    request = forward_pb2.ForwardRequest()
    request.forward_mode = 3
    for rid, routing_table in [
        ("req-1", ["worker-a", "scheduler-peer", "worker-b"]),
        ("req-2", ["worker-c", "scheduler-peer", "worker-d"]),
    ]:
        req = request.reqs.add()
        req.rid = rid
        req.routing_table.extend(routing_table)

    handler.rpc_pp_forward(request)

    assert scheduler_manage.touched_routing_tables == [
        ["worker-a", "scheduler-peer", "worker-b"],
        ["worker-c", "scheduler-peer", "worker-d"],
    ]
    assert [call["peer_id"] for call in recorded_calls] == ["worker-b", "worker-d"]


def test_scheduler_proxy_touches_routing_tables_before_abort():
    scheduler_manage = RecordingSchedulerManage()
    handler = RPCConnectionHandler(
        FakeLattica(), scheduler=None, http_port=0, scheduler_manage=scheduler_manage
    )
    recorded_calls = []
    handler._get_transformer_stub = lambda peer_id: RecordingStub(peer_id, recorded_calls)

    request = forward_pb2.AbortRequest()
    for rid, routing_table in [
        ("req-1", ["worker-a", "scheduler-peer", "worker-b"]),
        ("req-2", ["worker-c", "scheduler-peer", "worker-d"]),
    ]:
        req = request.reqs.add()
        req.rid = rid
        req.routing_table.extend(routing_table)

    handler.rpc_abort(request)

    assert scheduler_manage.touched_routing_tables == [
        ["worker-a", "scheduler-peer", "worker-b"],
        ["worker-c", "scheduler-peer", "worker-d"],
    ]
    assert {call["peer_id"] for call in recorded_calls} == {
        "worker-a",
        "worker-b",
        "worker-c",
        "worker-d",
    }


def test_scheduler_proxy_abort_fanout_does_not_wait_for_downstream_results():
    handler = RPCConnectionHandler(FakeLattica(), scheduler=None, http_port=0)
    recorded_calls = []
    handler._get_transformer_stub = lambda peer_id: RecordingStub(
        peer_id, recorded_calls, abort_future=ExplodingFuture()
    )

    request = forward_pb2.AbortRequest()
    req = request.reqs.add()
    req.rid = "req-1"
    req.routing_table.extend(["worker-a", "scheduler-peer", "worker-b"])

    handler.rpc_abort(request)

    assert {call["peer_id"] for call in recorded_calls} == {"worker-a", "worker-b"}


def test_scheduler_proxy_targets_transformer_service_name():
    handler = RPCConnectionHandler(FakeLattica(), scheduler=None, http_port=0)

    stub = handler._get_transformer_stub("worker-peer")

    assert stub.peer_id == "worker-peer"
    assert stub.service_name == "TransformerConnectionHandler"


def test_chat_completion_wraps_http_status_error_as_json_payload(monkeypatch):
    request = {"stream": False}
    response = httpx.Response(429, text='{"error":"busy"}')
    error = httpx.HTTPStatusError(
        "busy", request=httpx.Request("POST", "http://x"), response=response
    )

    class RaisingClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *_args, **_kwargs):
            raise error

    monkeypatch.setattr(
        "backend.server.rpc_connection_handler.httpx.Client",
        lambda **_kwargs: RaisingClient(),
    )

    handler = RPCConnectionHandler(FakeLattica(), scheduler=None, http_port=3001)
    payload = b"".join(handler.chat_completion(request))

    assert json.loads(payload.decode()) == {"error": "busy", "status_code": 429}
