from backend.server.rpc_connection_handler import (
    RPCConnectionHandler,
    _NODE_JOIN_INITIAL_ALLOCATION_WAIT_SEC,
)


class FakeLattica:
    def register_service(self, _service):
        return None

    def peer_id(self):
        return "scheduler-peer"


class FakeNode:
    def __init__(self, node_id):
        self.node_id = node_id


class FakeScheduler:
    def __init__(self):
        self.joined_nodes = []

    def enqueue_join(self, node):
        self.joined_nodes.append(node.node_id)


class FakeSchedulerManage:
    def __init__(self, scheduler, allocation):
        self.scheduler = scheduler
        self.allocation = allocation
        self.registered = []
        self.ready_waits = []
        self.allocation_waits = []

    def register_discovered_node(self, message):
        self.registered.append(message)

    def wait_for_scheduler_ready(self, timeout):
        self.ready_waits.append(timeout)
        return True

    def wait_for_node_allocation(self, node_id, wait_seconds):
        self.allocation_waits.append((node_id, wait_seconds))
        return self.allocation


def test_node_join_returns_quickly_when_allocation_is_not_ready():
    scheduler = FakeScheduler()
    scheduler_manage = FakeSchedulerManage(scheduler, allocation={})
    handler = RPCConnectionHandler(
        FakeLattica(), scheduler=None, http_port=0, scheduler_manage=scheduler_manage
    )
    handler.build_node = lambda message: FakeNode(message["node_id"])

    response = handler.node_join({"node_id": "node-1", "hardware": {}})

    assert response == {}
    assert scheduler_manage.ready_waits == [30]
    assert scheduler_manage.allocation_waits == [
        ("node-1", _NODE_JOIN_INITIAL_ALLOCATION_WAIT_SEC)
    ]
    assert scheduler.joined_nodes == ["node-1"]


def test_node_join_returns_allocation_when_immediately_available():
    allocation = {
        "node_id": "node-2",
        "model_name": "Qwen/Qwen3-0.6B",
        "start_layer": 0,
        "end_layer": 28,
        "tp_size": 1,
        "enable_weight_refit": False,
        "weight_refit_mode": "disk",
    }
    scheduler = FakeScheduler()
    scheduler_manage = FakeSchedulerManage(scheduler, allocation=allocation)
    handler = RPCConnectionHandler(
        FakeLattica(), scheduler=None, http_port=0, scheduler_manage=scheduler_manage
    )
    handler.build_node = lambda message: FakeNode(message["node_id"])

    response = handler.node_join({"node_id": "node-2", "hardware": {}})

    assert response == allocation
    assert scheduler_manage.allocation_waits == [
        ("node-2", _NODE_JOIN_INITIAL_ALLOCATION_WAIT_SEC)
    ]
    assert scheduler.joined_nodes == ["node-2"]
