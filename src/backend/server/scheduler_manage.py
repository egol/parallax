import copy
import os
import threading
import time
from typing import Any, Dict, List, Optional

from lattica import Lattica

from backend.server.constants import NODE_STATUS_AVAILABLE, NODE_STATUS_WAITING
from backend.server.rpc_connection_handler import RPCConnectionHandler
from backend.server.static_config import get_model_info, get_node_join_command
from parallax.cli import PUBLIC_INITIAL_PEERS, PUBLIC_RELAY_SERVERS
from parallax_utils.logging_config import get_logger
from scheduling.node import RequestSignal
from scheduling.scheduler import Scheduler

logger = get_logger(__name__)


def resolve_lattica_key_path(default: Optional[str] = None) -> Optional[str]:
    key_path = os.getenv("PARALLAX_LATTICA_KEY_PATH", "").strip()
    if key_path:
        os.makedirs(key_path, exist_ok=True)
        return key_path
    return default


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using default %.1f", name, raw, default)
        return default


class SchedulerManage:
    """
    Coordinates the in-process scheduler and the P2P RPC layer.

    This manager owns the `Scheduler` instance and the Lattica P2P node,
    wiring RPC calls from workers to scheduler events.
    """

    def __init__(
        self,
        initial_peers: List[str] = [],
        relay_servers: List[str] = [],
        dht_prefix: str = "gradient",
        host_maddrs: List[str] = [],
        announce_maddrs: List[str] = [],
        http_port: int = 3001,
        use_hfcache: bool = False,
        enable_weight_refit: bool = False,
        weight_refit_mode: str = "disk",
    ):
        """Initialize the manager with networking bootstrap parameters."""
        self.initial_peers = initial_peers
        self.relay_servers = relay_servers
        self.configured_initial_peers = list(initial_peers)
        self.configured_relay_servers = list(relay_servers)
        self.dht_prefix = dht_prefix
        self.host_maddrs = host_maddrs
        self.announce_maddrs = announce_maddrs
        self.http_port = http_port
        self.use_hfcache = use_hfcache
        self.enable_weight_refit = enable_weight_refit
        self.weight_refit_mode = weight_refit_mode
        self.model_name = None
        self.init_nodes_num = None
        self.scheduler = None
        self.node_id = f"{dht_prefix}_announce"
        self.lattica = None
        self.stubs = {}
        self.is_local_network = False
        self.network_signature = None
        self.discovered_nodes: Dict[str, Dict[str, Any]] = {}
        # A worker's initial node_join blocks until allocation is assigned, and the
        # heartbeat updater does not start until after that returns. Keep discovered
        # nodes alive long enough for scheduler init/bootstrap to finish on first join.
        self.discovered_node_ttl_seconds = 360.0
        self.cluster_status_lock = threading.Lock()
        self.cluster_status_condition = threading.Condition(self.cluster_status_lock)
        self.cluster_status_last_error = ""
        self.cluster_status_snapshot = self._empty_cluster_status_snapshot(
            snapshot_stale=False
        )
        self.cluster_status_generated_at = time.time()
        self.cluster_status_version = 0
        self.scheduler_ready_event = threading.Event()
        self.publish_cluster_status(reason="startup", force=True)

    def run(self, model_name, init_nodes_num, is_local_network=True):
        """
        Start the scheduler and the P2P service for RPC handling.
        If Lattica is already running, it will be reused.
        Nodes will automatically rejoin via their heartbeat (node_update) mechanism.
        """
        logger.debug(
            f"SchedulerManage starting: model_name={model_name}, init_nodes_num={init_nodes_num}"
        )
        self.bootstrap_network(is_local_network)
        self._start_scheduler(model_name, init_nodes_num)
        self.publish_cluster_status(reason="scheduler_run")

    def bootstrap_network(self, is_local_network=True):
        """
        Start only the scheduler's P2P network so workers/chat can discover the
        scheduler peer before a model is initialized.
        """
        self.is_local_network = is_local_network
        initial_peers, relay_servers = self._resolve_network_config(is_local_network)
        self._start_lattica(initial_peers, relay_servers)
        self.publish_cluster_status(reason="network_bootstrap")

    def is_running(self):
        """
        Returns True if the scheduler is running, False otherwise.
        """
        return self.scheduler is not None

    def is_initialized(self):
        return self.scheduler is not None

    def stop(self):
        """
        Stop the scheduler only. Lattica will remain running.
        """
        logger.info("Stopping scheduler...")

        # Stop scheduler if running
        if self.scheduler is not None:
            logger.debug("Stopping scheduler...")
            self.scheduler._stop_event.set()
            # Wait a bit for threads to finish
            time.sleep(0.1)
            self.scheduler = None
            self.scheduler_ready_event.clear()
            if hasattr(self, "connection_handler") and self.connection_handler is not None:
                self.connection_handler.scheduler = None
            logger.debug("Scheduler stopped")

        # Note: We don't close Lattica here to allow model switching without restarting P2P

        logger.info("Scheduler stopped")
        self.publish_cluster_status(reason="scheduler_stopped")

    def get_model_name(self):
        return self.model_name

    def get_init_nodes_num(self):
        return self.init_nodes_num

    def get_is_local_network(self):
        return self.is_local_network

    def get_network_mode(self):
        return "local" if self.is_local_network else "relay"

    def get_peer_id(self):
        if self.lattica is None:
            return None
        return self.lattica.peer_id()

    def weight_refit(self, request_data):
        """
        Trigger weight refit on every nodes.
        """
        if self.scheduler is None:
            return False
        self.scheduler.refit_request = request_data
        self.scheduler.refit_set = set()
        return True

    def get_last_refit_time(self):
        return self.scheduler.update_last_refit_time()

    def need_more_nodes(self):
        return self.scheduler.need_more_nodes() if self.scheduler else False

    def get_cluster_status(self):
        self._prune_discovered_nodes()
        with self.cluster_status_lock:
            snapshot = copy.deepcopy(self.cluster_status_snapshot)
            generated_at = self.cluster_status_generated_at
            last_error = self.cluster_status_last_error
            snapshot_version = self.cluster_status_version

        stale = (time.time() - generated_at) > 3.0
        data = snapshot.get("data", {})
        topology = data.get("topology", {})
        totals = topology.get("totals", {})
        if isinstance(totals, dict):
            totals["snapshot_stale"] = stale
            totals["snapshot_generated_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(generated_at)
            )
        if isinstance(data, dict):
            data["snapshot_version"] = snapshot_version
        data["last_snapshot_error"] = last_error
        return snapshot

    def publish_cluster_status(
        self, *, reason: str = "", force: bool = False, last_error: Optional[str] = None
    ):
        try:
            snapshot = self._build_cluster_status_snapshot()
            with self.cluster_status_condition:
                self.cluster_status_snapshot = snapshot
                self.cluster_status_generated_at = time.time()
                self.cluster_status_last_error = last_error or ""
                self.cluster_status_version += 1
                self.cluster_status_condition.notify_all()
        except Exception as error:
            logger.warning(f"Failed to publish cluster status ({reason}): {error}")
            with self.cluster_status_condition:
                self.cluster_status_last_error = last_error or str(error)
                if force or not self.cluster_status_snapshot:
                    self.cluster_status_snapshot = self._empty_cluster_status_snapshot(
                        snapshot_stale=True
                    )
                self.cluster_status_generated_at = time.time()
                self.cluster_status_version += 1
                self.cluster_status_condition.notify_all()

    def wait_for_cluster_status_version(
        self, previous_version: int, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        self._prune_discovered_nodes()
        with self.cluster_status_condition:
            if self.cluster_status_version <= previous_version:
                self.cluster_status_condition.wait(timeout=timeout)
            if self.cluster_status_version <= previous_version:
                return None
            snapshot = copy.deepcopy(self.cluster_status_snapshot)
            data = snapshot.get("data", {})
            if isinstance(data, dict):
                data["snapshot_version"] = self.cluster_status_version
            return snapshot

    def wait_for_scheduler_ready(self, timeout: Optional[float] = None) -> bool:
        return self.scheduler_ready_event.wait(timeout=timeout)

    def get_layer_allocation(self, node_id: str) -> Dict[str, Any]:
        if self.scheduler is None:
            return {}
        list_node_allocations = self.scheduler.list_node_allocations()
        for allocated_node_id, start_layer, end_layer in list_node_allocations:
            if allocated_node_id != node_id:
                continue
            node = self.scheduler.get_node(node_id)
            if node is None:
                return {}
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

    def wait_for_node_allocation(
        self, node_id: str, wait_seconds: float
    ) -> Optional[Dict[str, Any]]:
        deadline = time.time() + wait_seconds
        last_seen_version = -1
        while time.time() < deadline:
            allocation = self.get_layer_allocation(node_id)
            if allocation:
                return allocation
            remaining = max(0.0, deadline - time.time())
            snapshot = self.wait_for_cluster_status_version(
                last_seen_version, timeout=min(1.0, remaining)
            )
            if snapshot is None:
                continue
            data = snapshot.get("data", {})
            if isinstance(data, dict):
                last_seen_version = data.get("snapshot_version", last_seen_version)
        return None

    def on_scheduler_state_change(self, reason: str):
        self.publish_cluster_status(reason=reason)

    def _get_pipeline_capacity_report(self):
        if self.scheduler is None:
            return None, 0, 0
        return self.scheduler.node_manager.report_pipeline_capacity(ready_only=True)

    def get_topology_snapshot(
        self,
        per_pipeline_min: Optional[Dict[int, tuple[int, int]]],
        total_capacity: int,
        current_capacity: int,
    ) -> Dict[str, Any]:
        discovered_nodes = [
            self.build_discovered_node_info(node)
            for node in self.get_discovered_node_payloads()
        ]
        discovered_memory_gb = sum(
            self._node_memory_gb(node) for node in discovered_nodes
        )
        if self.scheduler is None:
            runtime = self._aggregate_cluster_runtime(
                discovered_nodes,
                discovered_memory_gb=discovered_memory_gb,
                registered_memory_gb=0,
            )
            return {
                "nodes": discovered_nodes,
                "pipelines": [],
                "edges": [],
                "totals": {
                    "registered_pipelines": 0,
                    "ready_pipelines": 0,
                    "total_capacity": 0,
                    "current_capacity": 0,
                    "joined_nodes": len(discovered_nodes),
                    "discovered_workers": len(discovered_nodes),
                    "registered_workers": 0,
                    "discovered_memory_gb": discovered_memory_gb,
                    "registered_memory_gb": 0,
                },
                "runtime": runtime,
            }

        node_manager = self.scheduler.node_manager
        registered_pipelines = node_manager.get_registered_pipelines()
        node_lookup = {node.node_id: node for node in node_manager.nodes}
        pipeline_membership: Dict[str, int] = {}
        stage_lookup: Dict[str, int] = {}
        pipelines: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        for pipeline_id, pipeline in sorted(registered_pipelines.items()):
            pipeline.recompute_capacity()
            min_capacity, remaining_capacity = (
                per_pipeline_min.get(pipeline_id, (pipeline.min_node_capacity, pipeline.min_remaining_capacity))
                if per_pipeline_min is not None
                else (pipeline.min_node_capacity, pipeline.min_remaining_capacity)
            )
            pipelines.append(
                {
                    "id": pipeline_id,
                    "ready": pipeline.is_ready,
                    "min_capacity": int(min_capacity),
                    "remaining_capacity": int(remaining_capacity),
                    "node_ids": list(pipeline.node_ids),
                }
            )
            for stage_index, node in enumerate(pipeline.nodes):
                pipeline_membership[node.node_id] = pipeline_id
                stage_lookup[node.node_id] = stage_index
                if stage_index > 0:
                    previous = pipeline.nodes[stage_index - 1]
                    edges.append(
                        {
                            "id": f"{pipeline_id}:{previous.node_id}->{node.node_id}",
                            "pipeline_id": pipeline_id,
                            "source": previous.node_id,
                            "target": node.node_id,
                        }
                    )

        nodes = [
            self.build_node_info(
                node,
                pipeline_id=pipeline_membership.get(node.node_id),
                stage_index=stage_lookup.get(node.node_id),
            )
            for node in node_manager.nodes
        ]
        registered_memory_gb = sum(self._node_memory_gb(node) for node in nodes)
        discovered_nodes = [
            self.build_discovered_node_info(node)
            for node in self.get_discovered_node_payloads(exclude_ids=set(node_lookup))
        ]
        discovered_memory_gb = sum(
            self._node_memory_gb(node) for node in discovered_nodes
        )
        nodes.extend(discovered_nodes)
        runtime = self._aggregate_cluster_runtime(
            nodes,
            discovered_memory_gb=discovered_memory_gb,
            registered_memory_gb=registered_memory_gb,
        )

        return {
            "nodes": nodes,
            "pipelines": pipelines,
            "edges": edges,
            "totals": {
                "registered_pipelines": len(pipelines),
                "ready_pipelines": sum(1 for pipeline in pipelines if pipeline["ready"]),
                "total_capacity": int(total_capacity),
                "current_capacity": int(current_capacity),
                "joined_nodes": len(nodes),
                "discovered_workers": len(discovered_nodes),
                "registered_workers": len(node_manager.nodes),
                "discovered_memory_gb": discovered_memory_gb,
                "registered_memory_gb": registered_memory_gb,
            },
            "runtime": runtime,
        }

    def get_node_list(self):
        if self.scheduler is None:
            return [
                self.build_discovered_node_info(node)
                for node in self.get_discovered_node_payloads()
            ]

        node_ids = {node.node_id for node in self.scheduler.node_manager.nodes}
        return [self.build_node_info(node) for node in self.scheduler.node_manager.nodes] + [
            self.build_discovered_node_info(node)
            for node in self.get_discovered_node_payloads(exclude_ids=node_ids)
        ]

    def register_discovered_node(self, node_payload: Dict[str, Any]):
        node_id = self._extract_node_id(node_payload)
        if not node_id:
            return
        previous_payload = self.discovered_nodes.get(node_id, {}).get("payload")
        next_payload = {
            "payload": dict(node_payload),
            "last_seen": time.time(),
        }
        changed = previous_payload != next_payload["payload"]
        self.discovered_nodes[node_id] = next_payload
        if changed:
            self.publish_cluster_status(reason=f"discovered_node:{node_id}")

    def unregister_discovered_node(self, node_id: Optional[str]):
        if not node_id:
            return
        if self.discovered_nodes.pop(node_id, None) is not None:
            self.publish_cluster_status(reason=f"discovered_node_removed:{node_id}")

    def get_discovered_node_payloads(self, exclude_ids: Optional[set[str]] = None):
        self._prune_discovered_nodes()
        excluded = exclude_ids or set()
        payloads: List[Dict[str, Any]] = []
        for node_id, entry in self.discovered_nodes.items():
            if node_id in excluded:
                continue
            payload = entry.get("payload")
            if isinstance(payload, dict):
                payloads.append(payload)
        return payloads

    def _prune_discovered_nodes(self):
        cutoff = time.time() - self.discovered_node_ttl_seconds
        stale_ids = [
            node_id
            for node_id, entry in self.discovered_nodes.items()
            if entry.get("last_seen", 0) < cutoff
        ]
        removed_any = False
        for node_id in stale_ids:
            self.discovered_nodes.pop(node_id, None)
            removed_any = True
        if removed_any:
            self.publish_cluster_status(reason="discovered_node_pruned")

    def _extract_node_id(self, node_payload: Dict[str, Any]) -> Optional[str]:
        if not isinstance(node_payload, dict):
            return None
        node_id = node_payload.get("node_id")
        if isinstance(node_id, str) and node_id:
            return node_id
        hardware = node_payload.get("hardware")
        if isinstance(hardware, dict):
            hardware_node_id = hardware.get("node_id")
            if isinstance(hardware_node_id, str) and hardware_node_id:
                return hardware_node_id
        return None

    def _normalize_runtime_stage(self, runtime_payload: Dict[str, Any]) -> str:
        stage = runtime_payload.get("init_stage")
        if isinstance(stage, str) and stage:
            return stage
        status = runtime_payload.get("status")
        if status == "ready":
            return "ready"
        if status in {"failed", "error"}:
            return "failed"
        if status in {"joining", "initializing"}:
            return "allocating"
        return "idle"

    def _runtime_stage_priority(self, stage: str) -> int:
        priorities = {
            "failed": 0,
            "loading-shards": 1,
            "downloading": 2,
            "resolving-metadata": 3,
            "allocating": 4,
            "rejoining-workers": 5,
            "ready": 6,
            "idle": 7,
        }
        return priorities.get(stage, 7)

    def _runtime_snapshot_from_payload(
        self, node_payload: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(node_payload, dict):
            return None
        runtime = node_payload.get("runtime")
        if not isinstance(runtime, dict):
            return None
        return {
            "status": runtime.get("status", "") or "",
            "model_name": runtime.get("model_name", "") or "",
            "init_stage": self._normalize_runtime_stage(runtime),
            "init_detail": runtime.get("init_detail", "") or "",
            "downloaded_files": runtime.get("downloaded_files"),
            "total_files": runtime.get("total_files"),
            "current_file": runtime.get("current_file", "") or "",
            "updated_at": runtime.get("updated_at", "") or "",
            "failure_reason": runtime.get("failure_reason", "") or "",
        }

    def _runtime_snapshot_for_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        payload = self.discovered_nodes.get(node_id, {}).get("payload")
        if not isinstance(payload, dict):
            return None
        return self._runtime_snapshot_from_payload(payload)

    def _aggregate_cluster_runtime(
        self,
        nodes: List[Dict[str, Any]],
        *,
        discovered_memory_gb: float,
        registered_memory_gb: float,
    ) -> Dict[str, Any]:
        runtime_nodes = [
            node
            for node in nodes
            if isinstance(node.get("runtime"), dict)
            and (
                node["runtime"].get("model_name")
                or node["runtime"].get("init_stage")
                or node["runtime"].get("init_detail")
            )
        ]
        if not runtime_nodes:
            return {"model_init": None}

        min_priority = min(
            self._runtime_stage_priority(node["runtime"].get("init_stage", "idle"))
            for node in runtime_nodes
        )
        stage_candidates = [
            node
            for node in runtime_nodes
            if self._runtime_stage_priority(node["runtime"].get("init_stage", "idle"))
            == min_priority
        ]
        stage_node = max(
            stage_candidates,
            key=lambda node: node["runtime"].get("updated_at", ""),
        )
        most_recent_node = max(
            runtime_nodes,
            key=lambda node: node["runtime"].get("updated_at", ""),
        )
        counted_nodes = [
            node
            for node in runtime_nodes
            if node["runtime"].get("downloaded_files") is not None
            and node["runtime"].get("total_files") is not None
        ]
        if counted_nodes and len(counted_nodes) == len(runtime_nodes):
            downloaded_files = sum(
                int(node["runtime"].get("downloaded_files") or 0) for node in counted_nodes
            )
            total_files = sum(
                int(node["runtime"].get("total_files") or 0) for node in counted_nodes
            )
        else:
            downloaded_files = None
            total_files = None

        stage = stage_node["runtime"].get("init_stage", "idle") or "idle"
        updated_at_values = [
            node["runtime"].get("updated_at", "")
            for node in runtime_nodes
            if node["runtime"].get("updated_at")
        ]
        model_name = (
            stage_node["runtime"].get("model_name")
            or most_recent_node["runtime"].get("model_name")
            or self.model_name
            or ""
        )
        return {
            "model_init": {
                "active": stage not in {"idle", "ready", "failed"},
                "stage": stage,
                "model_name": model_name,
                "required_memory_gb": None,
                "discovered_memory_gb": discovered_memory_gb,
                "registered_memory_gb": registered_memory_gb,
                "derived_bootstrap_nodes": None,
                "detail": stage_node["runtime"].get("init_detail", "") or "",
                "started_at": min(updated_at_values) if updated_at_values else "",
                "updated_at": most_recent_node["runtime"].get("updated_at", "") or "",
                "downloaded_files": downloaded_files,
                "total_files": total_files,
                "current_file": most_recent_node["runtime"].get("current_file", "") or "",
                "last_progress_at": most_recent_node["runtime"].get("updated_at", "") or "",
                "failure_reason": stage_node["runtime"].get("failure_reason", "") or "",
                "source_node_id": stage_node.get("node_id", "") or "",
            }
        }

    def build_node_info(
        self,
        node,
        *,
        pipeline_id: Optional[int] = None,
        stage_index: Optional[int] = None,
    ):
        latency_ms = node.layer_latency_ms
        return {
            "node_id": node.node_id,
            "status": NODE_STATUS_AVAILABLE if node.is_active else NODE_STATUS_WAITING,
            "source": "registered",
            "gpu_num": node.hardware.num_gpus,
            "gpu_name": node.hardware.gpu_name,
            "gpu_memory": node.hardware.memory_gb,
            "device": node.hardware.device,
            "start_layer": node.start_layer,
            "end_layer": node.end_layer,
            "current_requests": node.current_requests,
            "max_requests": node.max_requests,
            "assigned_request_count": self.scheduler.node_manager.node_assigned_request_count.get(
                node.node_id, 0
            )
            if self.scheduler
            else 0,
            "pipeline_id": pipeline_id,
            "stage_index": stage_index,
            "is_active": node.is_active,
            "layer_latency_ms": None if latency_ms == float("inf") else latency_ms,
            "runtime": self._runtime_snapshot_for_node(node.node_id),
        }

    def build_discovered_node_info(self, node_payload: Dict[str, Any]):
        hardware = node_payload.get("hardware", {})
        if not isinstance(hardware, dict):
            hardware = {}
        return {
            "node_id": self._extract_node_id(node_payload) or "",
            "status": NODE_STATUS_WAITING,
            "source": "discovered",
            "gpu_num": hardware.get("num_gpus", 0),
            "gpu_name": hardware.get("gpu_name", ""),
            "gpu_memory": hardware.get("memory_gb", 0),
            "device": hardware.get("device", ""),
            "start_layer": node_payload.get("start_layer"),
            "end_layer": node_payload.get("end_layer"),
            "current_requests": node_payload.get("current_requests", 0),
            "max_requests": node_payload.get("max_concurrent_requests", 0),
            "assigned_request_count": 0,
            "pipeline_id": None,
            "stage_index": None,
            "is_active": bool(node_payload.get("is_active", False)),
            "layer_latency_ms": node_payload.get("layer_latency_ms"),
            "runtime": self._runtime_snapshot_from_payload(node_payload),
        }

    def _node_memory_gb(self, node_info: Dict[str, Any]) -> float:
        gpu_num = node_info.get("gpu_num", 0) or 0
        gpu_memory = node_info.get("gpu_memory", 0) or 0
        try:
            return max(float(gpu_num), 0.0) * max(float(gpu_memory), 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _empty_cluster_status_snapshot(self, snapshot_stale: bool) -> Dict[str, Any]:
        generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return {
            "type": "cluster_status",
            "data": {
                "status": NODE_STATUS_WAITING,
                "model_name": self.model_name,
                "init_nodes_num": self.init_nodes_num,
                "initialized": self.is_initialized(),
                "scheduler_peer_id": self.get_peer_id(),
                "network_mode": self.get_network_mode(),
                "node_join_command": get_node_join_command(
                    self.get_peer_id(), self.is_local_network
                ),
                "node_list": [],
                "need_more_nodes": False,
                "max_running_request": 0,
                "last_snapshot_error": self.cluster_status_last_error,
                "topology": {
                    "nodes": [],
                    "pipelines": [],
                    "edges": [],
                    "totals": {
                        "registered_pipelines": 0,
                        "ready_pipelines": 0,
                        "total_capacity": 0,
                        "current_capacity": 0,
                        "joined_nodes": 0,
                        "discovered_workers": 0,
                        "registered_workers": 0,
                        "discovered_memory_gb": 0,
                        "registered_memory_gb": 0,
                        "snapshot_generated_at": generated_at,
                        "snapshot_stale": snapshot_stale,
                    },
                },
                "runtime": {
                    "model_init": None,
                },
            },
        }

    def _build_cluster_status_snapshot(self) -> Dict[str, Any]:
        per_pipeline_min, total_capacity, current_capacity = (
            self._get_pipeline_capacity_report()
        )
        topology = self.get_topology_snapshot(
            per_pipeline_min, total_capacity, current_capacity
        )
        return {
            "type": "cluster_status",
            "data": {
                "status": self.get_schedule_status(),
                "model_name": self.model_name,
                "init_nodes_num": self.init_nodes_num,
                "initialized": self.is_initialized(),
                "scheduler_peer_id": self.get_peer_id(),
                "network_mode": self.get_network_mode(),
                "node_join_command": get_node_join_command(
                    self.get_peer_id(), self.is_local_network
                ),
                "node_list": self.get_node_list(),
                "need_more_nodes": self.need_more_nodes(),
                "max_running_request": total_capacity,
                "last_snapshot_error": "",
                "topology": topology,
                "runtime": topology.get("runtime", {"model_init": None}),
            },
        }

    def _start_scheduler(self, model_name, init_nodes_num):
        """
        Create the scheduler and start its background run loop.
        If scheduler already exists, it will be stopped and recreated.
        Nodes will automatically rejoin via their heartbeat (node_update) mechanism.
        """
        # Stop existing scheduler if running
        if self.scheduler is not None:
            logger.info("Scheduler already running, stopping it first for re-initialization")
            self.stop()

        self.model_name = model_name
        self.init_nodes_num = init_nodes_num
        heartbeat_timeout = _get_env_float("PARALLAX_SCHEDULER_HEARTBEAT_TIMEOUT_SEC", 30.0)
        inactive_heartbeat_timeout = _get_env_float(
            "PARALLAX_SCHEDULER_INACTIVE_HEARTBEAT_TIMEOUT_SEC",
            max(heartbeat_timeout, 300.0),
        )

        model_info = get_model_info(model_name, self.use_hfcache)
        self.scheduler = Scheduler(
            model_info,
            [],
            min_nodes_bootstrapping=init_nodes_num,
            enable_weight_refit=self.enable_weight_refit,
            weight_refit_mode=self.weight_refit_mode,
            heartbeat_timeout=heartbeat_timeout,
            inactive_heartbeat_timeout=inactive_heartbeat_timeout,
            state_change_callback=self.on_scheduler_state_change,
        )
        logger.info(
            "Scheduler heartbeat timeouts configured: ready=%.1fs inactive=%.1fs",
            heartbeat_timeout,
            inactive_heartbeat_timeout,
        )
        self.scheduler_ready_event.set()
        if hasattr(self, "connection_handler") and self.connection_handler is not None:
            self.connection_handler.scheduler = self.scheduler

        # Run the scheduler's event/dispatch loops in background so the process
        # can continue to serve RPCs and HTTP traffic.
        threading.Thread(
            target=self.scheduler.run,
            kwargs={"poll_interval": 0.05},
            name="SchedulerMain",
            daemon=True,
        ).start()
        logger.debug("Scheduler background thread started (poll_interval=0.05)")
        logger.info("Nodes will automatically rejoin via heartbeat (node_update) mechanism")
        self.publish_cluster_status(reason="scheduler_started")

    def _resolve_network_config(self, is_local_network: bool):
        initial_peers = list(self.configured_initial_peers)
        relay_servers = list(self.configured_relay_servers)
        if not is_local_network and not initial_peers and not relay_servers:
            logger.debug("Using public relay servers")
            initial_peers = list(PUBLIC_INITIAL_PEERS)
            relay_servers = list(PUBLIC_RELAY_SERVERS)
        return initial_peers, relay_servers

    def _network_signature_for(
        self, initial_peers: List[str], relay_servers: List[str]
    ) -> tuple:
        return (
            self.is_local_network,
            tuple(initial_peers),
            tuple(relay_servers),
            tuple(self.host_maddrs),
            tuple(self.announce_maddrs),
        )

    def _stop_lattica(self):
        if self.lattica is None:
            return
        try:
            self.lattica.close()
        except Exception as e:
            logger.warning(f"Failed to close Lattica cleanly: {e}")
        self.lattica = None
        self.connection_handler = None
        self.network_signature = None
        self.stubs = {}
        self.discovered_nodes = {}
        self.publish_cluster_status(reason="lattica_stopped")

    def _start_lattica(self, initial_peers: List[str], relay_servers: List[str]):
        """
        Initialize and start the Lattica P2P node used for RPCs.
        If the network config changes, restart Lattica so peer discovery state
        matches the active local/relay mode.
        """
        signature = self._network_signature_for(initial_peers, relay_servers)

        if self.lattica is not None and self.network_signature != signature:
            logger.info("Restarting Lattica because the scheduler network mode changed")
            self._stop_lattica()

        # Reuse existing Lattica if running
        if self.lattica is not None:
            logger.debug("Lattica already running, reusing existing instance")
            # Update connection handler with new scheduler if it exists
            if hasattr(self, "connection_handler") and self.connection_handler is not None:
                self.connection_handler.scheduler = self.scheduler
                logger.debug("Updated connection handler with new scheduler")
            else:
                # Create connection handler if it doesn't exist
                self.connection_handler = RPCConnectionHandler(
                    lattica=self.lattica,
                    scheduler=self.scheduler,
                    http_port=self.http_port,
                    scheduler_manage=self,
                )
                logger.debug("Created connection handler with existing Lattica")
            self.connection_handler.scheduler_manage = self
            self.publish_cluster_status(reason="lattica_reused")
            return

        logger.debug(
            f"Starting Lattica with host_maddrs={self.host_maddrs}, mdns=False, dht_prefix={self.dht_prefix}"
        )
        key_path = resolve_lattica_key_path(".")
        self.lattica = Lattica.builder().with_listen_addrs(self.host_maddrs).with_key_path(
            key_path
        )
        self.network_signature = signature

        if len(relay_servers) > 0:
            logger.info(f"Using relay servers: {relay_servers}")
            self.lattica.with_relay_servers(relay_servers).with_dcutr(True).with_protocol("")

        if len(self.announce_maddrs) > 0:
            logger.info(f"Using announce maddrs: {self.announce_maddrs}")
            self.lattica.with_external_addrs(self.announce_maddrs)

        if len(initial_peers) > 0:
            logger.info(f"Using initial peers: {initial_peers}")
            self.lattica.with_bootstraps(initial_peers)

        self.lattica.build()
        logger.debug("Lattica node built")

        if len(relay_servers) > 0:
            try:
                is_symmetric_nat = self.lattica.is_symmetric_nat()
                if is_symmetric_nat is None:
                    logger.warning("Failed to get is symmetric NAT, skip")
                elif is_symmetric_nat:
                    logger.error(
                        "Your network NAT type is symmetric, relay does not work on this type of NAT, see https://en.wikipedia.org/wiki/Network_address_translation"
                    )
                    exit(1)
            except Exception as e:
                logger.exception(f"Error in is symmetric NAT: {e}")

        store_success = False
        for _ in range(10):
            try:
                if self.lattica.store(
                    "scheduler_peer_id",
                    self.lattica.peer_id(),
                    expiration_time=time.time() + 365 * 24 * 60 * 60,
                ):
                    logger.info(f"Stored scheduler peer id: {self.lattica.peer_id()}")
                    store_success = True
                    break
                logger.warning("Failed to store scheduler peer id, waiting for 10 seconds")
                time.sleep(10)
            except Exception as e:
                logger.error(f"Failed to store scheduler peer id: {e}, waiting for 10 seconds")
                time.sleep(10)

        if not store_success:
            logger.error("Failed to store scheduler peer id, after 10 times")
            exit(1)

        self.connection_handler = RPCConnectionHandler(
            lattica=self.lattica,
            scheduler=self.scheduler,
            http_port=self.http_port,
            scheduler_manage=self,
        )
        logger.debug("RPCConnectionHandler initialized")
        self.publish_cluster_status(reason="lattica_started")

    def get_routing_table(self, request_id, received_ts):
        """Block briefly until the scheduler assigns a routing path for the request.

        Distinguish three states via `RequestSignal.routing_table`:
        - None: not yet decided, keep waiting up to timeout
        - []: decided but no capacity (pipelines full), return immediately
        - [..]: valid routing path, return immediately
        """
        logger.debug(f"Routing table requested for request_id={request_id}")
        request = RequestSignal(request_id, received_ts)
        self.scheduler.receive_request(request)

        # Wait up to 5 seconds, but return immediately if the routing table is set (including an empty list)
        start_time = time.time()
        while request.routing_table is None and (time.time() - start_time) < 5.0:
            time.sleep(0.05)

        # Return the routing_table
        if request.routing_table is None:
            logger.debug(
                f"Routing table not ready after {(time.time() - start_time):.2f}s for request_id={request_id}"
            )
        else:
            logger.debug(
                f"Routing table resolved for request_id={request_id}: {request.routing_table}"
            )
        return request.routing_table

    def get_schedule_status(self):
        """
        Return whether a full pipeline has been allocated across joined nodes.
        """
        if self.scheduler is None:
            logger.debug("SchedulerManage status queried: waiting (scheduler not initialized)")
            return NODE_STATUS_WAITING

        # todo rebalance status
        status = (
            NODE_STATUS_AVAILABLE if self.scheduler.has_full_pipeline() else NODE_STATUS_WAITING
        )
        logger.debug(f"SchedulerManage status queried: {status}")
        return status

    def get_call_url_by_node_id(self, node_id):
        """
        Lookup the HTTP endpoint for a given node id managed by the RPC layer.
        """
        url = self.connection_handler.get_call_url_by_node_id(node_id)
        logger.debug(f"Lookup call_url for node_id={node_id} -> {url}")
        return url
