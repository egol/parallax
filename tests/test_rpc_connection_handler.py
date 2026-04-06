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


class RecordingStub:
    def __init__(self, peer_id, calls):
        self.peer_id = peer_id
        self.calls = calls

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


def test_scheduler_proxy_targets_transformer_service_name():
    handler = RPCConnectionHandler(FakeLattica(), scheduler=None, http_port=0)

    stub = handler._get_transformer_stub("worker-peer")

    assert stub.peer_id == "worker-peer"
    assert stub.service_name == "TransformerConnectionHandler"
