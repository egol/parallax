from lattica import ConnectionHandler, Lattica, rpc_method, rpc_stream, rpc_stream_iter
from lattica.connection_handler import ServiceStub

from parallax.p2p.proto import forward_pb2
from parallax_utils.logging_config import get_logger
from scheduling.node import Node, NodeHardwareInfo
from scheduling.scheduler import Scheduler

logger = get_logger(__name__)

import json

import httpx

_NODE_JOIN_SCHEDULER_READY_TIMEOUT_SEC = 30
_NODE_JOIN_INITIAL_ALLOCATION_WAIT_SEC = 5


class RPCConnectionHandler(ConnectionHandler):
    """
    Handles RPC requests from clients, forwarding them to the appropriate TransformerBackend.
    Inherits from hivemind's ConnectionHandler.
    """

    def __init__(
        self,
        lattica: Lattica,
        scheduler: Scheduler,
        http_port: int,
        scheduler_manage=None,
    ):
        # Initialize the base class
        super().__init__(lattica)
        self.lattica = lattica
        self.scheduler = scheduler
        self.http_port = http_port
        self.scheduler_manage = scheduler_manage

    @staticmethod
    def _rotate_routing_table_for_self(routing_table, self_peer_id):
        if not routing_table:
            return list(routing_table)
        entries = list(routing_table)
        try:
            self_index = entries.index(self_peer_id)
        except ValueError as exc:
            raise RuntimeError("Can not find self in the routing table") from exc
        return entries[self_index:] + entries[:self_index]

    def _group_requests_by_next_peer(self, requests):
        grouped_requests = {}
        self_peer_id = self.lattica.peer_id()
        for req in requests:
            rotated = self._rotate_routing_table_for_self(req.routing_table, self_peer_id)
            req.routing_table[:] = rotated
            next_peer_id = rotated[(1 % len(rotated))]
            if next_peer_id == self_peer_id:
                logger.warning("Proxy routing table for %s loops back to self, skipping", req.rid)
                continue
            grouped_requests.setdefault(next_peer_id, []).append(req)
        return grouped_requests

    def _get_transformer_stub(self, peer_id: str) -> ServiceStub:
        return ServiceStub(self, peer_id, "TransformerConnectionHandler")

    def _touch_request_routes(self, requests) -> None:
        if self.scheduler_manage is None:
            return
        for req in requests:
            try:
                self.scheduler_manage.touch_routing_table(list(req.routing_table))
            except Exception:
                logger.debug("Failed to touch routing table for %s", req.rid, exc_info=True)

    @rpc_stream
    def rpc_pp_forward(
        self,
        request: forward_pb2.ForwardRequest,
    ) -> forward_pb2.ForwardResponse:
        """Scheduler-host proxy hop for centralized worker routing."""
        try:
            self._touch_request_routes(request.reqs)
            grouped_requests = self._group_requests_by_next_peer(request.reqs)
            for next_peer_id, requests in grouped_requests.items():
                stub = self._get_transformer_stub(next_peer_id)
                new_forward_request = forward_pb2.ForwardRequest()
                new_forward_request.forward_mode = request.forward_mode
                new_forward_request.reqs.extend(requests)
                response = stub.rpc_pp_forward(new_forward_request)
                if hasattr(response, "result"):
                    response.result()
        except Exception as e:
            logger.exception(f"Error in rpc_pp_forward: {e}")
        return forward_pb2.ForwardResponse()

    @rpc_method
    def rpc_abort(
        self,
        request: forward_pb2.AbortRequest,
    ) -> forward_pb2.AbortResponse:
        try:
            self._touch_request_routes(request.reqs)
            grouped_requests = {}
            self_peer_id = self.lattica.peer_id()
            for req in request.reqs:
                for peer_id in req.routing_table:
                    if peer_id == self_peer_id:
                        continue
                    grouped_requests.setdefault(peer_id, []).append(req)

            for peer_id, requests in grouped_requests.items():
                stub = self._get_transformer_stub(peer_id)
                new_abort_request = forward_pb2.AbortRequest()
                new_abort_request.reqs.extend(requests)
                # Abort fanout is best-effort. Waiting for downstream abort
                # acknowledgements can deadlock centralized routing when the
                # origin worker is still inside its upstream abort path.
                stub.rpc_abort(new_abort_request)
        except Exception as e:
            logger.exception(f"Error in rpc_abort: {e}")
        return forward_pb2.AbortResponse()

    @rpc_stream
    def node_join(self, message):
        # node = {
        #     "node_id": "lattica peer id",
        #     "hardware": {
        #         "node_id": "lattica peer id",
        #         "tflops_fp16": 100,
        #         "memory_gb": 100,
        #         "memory_bandwidth_gbps": 100,
        #     },
        #     "kvcache_mem_ratio": 0.3,
        #     "param_mem_ratio": 0.5,
        #     "max_concurrent_requests": 16,
        #     "max_sequence_length": 1024,
        # }
        logger.info(f"receive node_join request: {message}")
        if self.scheduler_manage is not None:
            self.scheduler_manage.register_discovered_node(message)
        if self.scheduler is None:
            if self.scheduler_manage is None or not self.scheduler_manage.wait_for_scheduler_ready(
                timeout=_NODE_JOIN_SCHEDULER_READY_TIMEOUT_SEC
            ):
                logger.warning("Timed out waiting for scheduler readiness during node_join")
                return {}
            self.scheduler = self.scheduler_manage.scheduler
        try:
            node = self.build_node(message)
            self.scheduler.enqueue_join(node)

            # Do not hold node_join open for the full allocation window. Workers can
            # remain in JOINING and pick up layer assignments via heartbeats without
            # tearing down and rebuilding their local P2P listener.
            response = self.wait_layer_allocation(
                node.node_id, wait_seconds=_NODE_JOIN_INITIAL_ALLOCATION_WAIT_SEC
            )
            logger.debug(f"node_join response: {response}")
            return response
        except Exception as e:
            logger.exception(f"node_join error: {e}")
            return {}

    @rpc_method
    def node_leave(self, message):
        logger.debug(f"receive node_leave request: {message}")
        node_id = message.get("node_id")
        if not node_id:
            hardware = message.get("hardware", {})
            if isinstance(hardware, dict):
                node_id = hardware.get("node_id")
        if self.scheduler_manage is not None:
            self.scheduler_manage.unregister_discovered_node(node_id)
        if self.scheduler is None:
            return {}
        try:
            node = self.build_node(message)
            self.scheduler.enqueue_leave(node.node_id)
            return {}
        except Exception as e:
            logger.exception(f"node_leave error: {e}")
            return {}

    @rpc_method
    def node_update(self, message):
        """
        Returns a Tuple[Dict, Dict] where
        first dict contains layer allocation result and
        second dict records weight refit information.
        """
        logger.debug(f"receive node_update request: {message}")
        if self.scheduler_manage is not None:
            self.scheduler_manage.register_discovered_node(message)
        if self.scheduler is None:
            if self.scheduler_manage is None or not self.scheduler_manage.wait_for_scheduler_ready(
                timeout=30
            ):
                return {}, {}
            self.scheduler = self.scheduler_manage.scheduler
        try:
            node = self.build_node(message)
            # Check if node exists in scheduler
            if self.scheduler.get_node(node.node_id) is None:
                # Node not found, automatically join it (e.g., after model switch)
                logger.info(
                    f"Node {node.node_id} not found in scheduler, auto-joining via node_update"
                )
                self.scheduler.enqueue_join(node)
                # Return layer allocation after join
                layer_allocation = self.wait_layer_allocation(node.node_id, wait_seconds=5)
                return layer_allocation, {}

            # Node exists, update its info
            self.scheduler.enqueue_node_update(
                node.node_id,
                current_requests=node.current_requests,
                layer_latency_ms=node.layer_latency_ms,
                new_rtt_to_nodes=node.rtt_to_nodes,
                is_active=node.is_active,
                last_refit_time=node.last_refit_time,
            )
            # Return current layer allocation to node
            layer_allocation = self.get_layer_allocation(node.node_id)
            refit_request = {}
            if self.scheduler.refit_request:
                if node.node_id not in self.scheduler.refit_set and node.is_active:
                    refit_request = self.scheduler.refit_request
                    self.scheduler.refit_set.add(node.node_id)
            return layer_allocation, refit_request
        except Exception as e:
            logger.exception(f"node_update error: {e}")
            return {}, {}

    @rpc_stream_iter
    def chat_completion(
        self,
        request,
    ):
        """Handle chat completion request"""
        logger.debug(f"Chat completion request: {request}, type: {type(request)}")
        try:
            with httpx.Client(timeout=10 * 60, proxy=None, trust_env=False) as client:
                if request.get("stream", False):
                    with client.stream(
                        "POST",
                        f"http://127.0.0.1:{self.http_port}/v1/chat/completions",
                        json=request,
                    ) as response:
                        response.raise_for_status()
                        for chunk in response.iter_bytes():
                            if chunk:
                                yield chunk
                else:
                    response = client.post(
                        f"http://127.0.0.1:{self.http_port}/v1/chat/completions", json=request
                    )
                    response.raise_for_status()
                    yield response.content
        except httpx.HTTPStatusError as e:
            status_code = getattr(e.response, "status_code", 500)
            detail = e.response.text[:1000] if e.response is not None else ""
            logger.exception(
                "Error in chat completion upstream response "
                f"(status={status_code}, "
                f"body={detail!r})"
            )
            if detail:
                try:
                    payload = json.loads(detail)
                except Exception:
                    payload = {"error": detail}
            else:
                payload = {"error": "upstream http error"}
            if isinstance(payload, dict) and "status_code" not in payload:
                payload["status_code"] = int(status_code)
            yield json.dumps(payload).encode()
        except Exception as e:
            logger.exception(f"Error in chat completion: {e}")
            yield json.dumps({"error": "internal server error", "status_code": 500}).encode()

    @rpc_stream_iter
    def cluster_status(self):
        if self.scheduler_manage is None:
            yield json.dumps({"error": "internal server error"}).encode()
            return

        try:
            snapshot = self.scheduler_manage.get_cluster_status()
            yield json.dumps(snapshot, ensure_ascii=False).encode() + b"\n"
            version = snapshot.get("data", {}).get("snapshot_version", 0)
            while True:
                snapshot = self.scheduler_manage.wait_for_cluster_status_version(
                    version, 30.0
                )
                if snapshot is None:
                    continue
                version = snapshot.get("data", {}).get("snapshot_version", version)
                yield json.dumps(snapshot, ensure_ascii=False).encode() + b"\n"
        except Exception as e:
            logger.exception(f"Error in cluster status: {e}")
            yield json.dumps({"error": "internal server error"}).encode()

    def wait_layer_allocation(self, current_node_id, wait_seconds):
        if self.scheduler_manage is not None:
            allocation = self.scheduler_manage.wait_for_node_allocation(
                current_node_id, wait_seconds
            )
            return allocation or {}
        return {}

    def get_layer_allocation(self, current_node_id):
        if self.scheduler_manage is not None:
            return self.scheduler_manage.get_layer_allocation(current_node_id)
        list_node_allocations = self.scheduler.list_node_allocations()
        for node_id, start_layer, end_layer in list_node_allocations:
            if current_node_id == node_id:
                node = self.scheduler.get_node(node_id)
                if node:
                    return {
                        "node_id": node_id,
                        "model_name": (
                            node.model_info.model_name
                            if node.hardware.device != "mlx"
                            else node.model_info.mlx_model_name
                        ),
                        "start_layer": start_layer,
                        "end_layer": end_layer,
                        "tp_size": node.hardware.num_gpus,
                        "enable_weight_refit": self.scheduler.enable_weight_refit,
                        "weight_refit_mode": self.scheduler.weight_refit_mode,
                    }
        return {}

    def build_node(self, node_json: dict):
        node = Node(
            node_id=node_json.get("node_id"),
            hardware=self.build_hardware(node_json.get("hardware")),
            model_info=self.scheduler.model_info,
            kvcache_mem_ratio=node_json.get("kvcache_mem_ratio"),
            param_mem_ratio=node_json.get("param_mem_ratio"),
            max_concurrent_requests=node_json.get("max_concurrent_requests"),
            max_sequence_length=node_json.get("max_sequence_length"),
            is_active=node_json.get("is_active", True),
            manual_layer_assignment=node_json.get("manual_layer_assignment", False),
            last_refit_time=node_json.get("last_refit_time", 0.0),
        )
        if node_json.get("start_layer", None) is not None:
            node.start_layer = node_json.get("start_layer")
        if node_json.get("end_layer", None) is not None:
            node.end_layer = node_json.get("end_layer")
        if node_json.get("current_requests", None) is not None:
            node.current_requests = node_json.get("current_requests")
        if node_json.get("layer_latency_ms", None) is not None:
            node.avg_layer_latency_ms = node_json.get("layer_latency_ms")
        if node_json.get("rtt_to_nodes", None) is not None:
            node.rtt_to_nodes = node_json.get("rtt_to_nodes")
        return node

    def build_hardware(self, hardware_json):
        node_id = hardware_json.get("node_id")
        num_gpus = hardware_json.get("num_gpus")
        tflops_fp16 = hardware_json.get("tflops_fp16")
        gpu_name = hardware_json.get("gpu_name")
        memory_gb = hardware_json.get("memory_gb")
        memory_bandwidth_gbps = hardware_json.get("memory_bandwidth_gbps")
        device = hardware_json.get("device")
        return NodeHardwareInfo(
            node_id=node_id,
            num_gpus=num_gpus,
            tflops_fp16=tflops_fp16,
            gpu_name=gpu_name,
            memory_gb=memory_gb,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            device=device,
        )
