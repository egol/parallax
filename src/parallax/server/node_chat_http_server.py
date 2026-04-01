import asyncio
import json
import os
import time
from typing import Dict, List, Optional

import fastapi
import uvicorn
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from lattica import Lattica
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import State

from backend.server.rpc_connection_handler import RPCConnectionHandler
from parallax_utils.file_util import get_project_root
from parallax_utils.logging_config import get_logger
from parallax_utils.runtime_compat import bundled_frontend_path, uvicorn_loop_name

logger = get_logger(__name__)

import uuid


def resolve_lattica_key_path(default: Optional[str] = None) -> Optional[str]:
    key_path = os.getenv("PARALLAX_LATTICA_KEY_PATH", "").strip()
    if key_path:
        os.makedirs(key_path, exist_ok=True)
        return key_path
    return default

# Fast API
app = fastapi.FastAPI(
    openapi_url="/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "http://tauri.localhost",
        "https://tauri.localhost",
        "tauri://localhost",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def init_app_states(state: State, node_chat_http_server):
    """Init FastAPI app states, including http handler, etc."""
    state.http_server = node_chat_http_server


async def v1_chat_completions(request_data: Dict, request_id: str, received_ts: int):
    return await app.state.http_server.chat_completion(request_data, request_id, received_ts)


async def get_cluster_status():
    return app.state.http_server.get_cluster_status()


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    """OpenAI v1/chat/complete post function"""
    request_data = await raw_request.json()
    request_id = uuid.uuid4()
    received_ts = time.time()
    return await v1_chat_completions(request_data, request_id, received_ts)


@app.get("/cluster/status")
async def cluster_status():
    return await get_cluster_status()


# Disable caching for index.html
@app.get("/")
async def serve_index():
    response = FileResponse(bundled_frontend_path(get_project_root(), "chat.html"))
    # Disable cache
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# mount the frontend
app.mount(
    "/",
    StaticFiles(directory=str(get_project_root() / "src" / "frontend" / "dist"), html=True),
    name="static",
)


class NodeChatHttpServer:

    def __init__(self, args):
        self.host = args.host
        self.port = args.node_chat_port
        self.tcp_port = args.tcp_port
        self.udp_port = args.udp_port
        self.scheduler_addr = args.scheduler_addr
        self.relay_servers = args.relay_servers
        self.announce_maddrs = args.announce_maddrs
        self.initial_peers = args.initial_peers
        self.host_maddrs = [
            f"/ip4/0.0.0.0/tcp/{self.tcp_port}",
            f"/ip4/0.0.0.0/udp/{self.udp_port}/quic-v1",
        ]
        self.scheduler_peer_id = None
        self.scheduler_stub = None
        self.lattica = None
        self._max_build_attempts = 3

    @staticmethod
    def _parse_scheduler_target(scheduler_addr: Optional[str]) -> tuple[List[str], Optional[str]]:
        if scheduler_addr is None or scheduler_addr == "auto":
            return [], None

        normalized = scheduler_addr.strip()
        if not normalized:
            raise ValueError("scheduler_addr must not be empty")

        if normalized.startswith("/"):
            normalized = normalized.rstrip("/")
            parts = [part for part in normalized.split("/") if part]
            if len(parts) < 2 or parts[-2] != "p2p" or not parts[-1]:
                raise ValueError(
                    "scheduler_addr multiaddr must end with /p2p/<scheduler-peer-id>"
                )
            return [normalized], parts[-1]

        if "/" in normalized:
            raise ValueError("scheduler_addr must be a bare peer id or /p2p multiaddr")
        return [], normalized

    def _close_lattica(self):
        if self.lattica is not None:
            try:
                self.lattica.close()
            except Exception:
                logger.debug("Failed to close Lattica cleanly", exc_info=True)
            finally:
                self.lattica = None
                self.scheduler_stub = None

    def build_lattica(self):
        self.lattica = Lattica.builder().with_listen_addrs(self.host_maddrs)
        key_path = resolve_lattica_key_path()
        if key_path is not None:
            self.lattica.with_key_path(key_path)

        try:
            scheduler_bootstraps, self.scheduler_peer_id = self._parse_scheduler_target(
                self.scheduler_addr
            )
        except ValueError as exc:
            logger.error(f"Invalid scheduler target {self.scheduler_addr!r}: {exc}")
            self._close_lattica()
            return False

        if scheduler_bootstraps:
            logger.info(f"Using scheduler addr: {scheduler_bootstraps[0]}")
            self.lattica.with_bootstraps(scheduler_bootstraps)

        if len(self.relay_servers) > 0:
            logger.info(f"Using relay servers: {self.relay_servers}")
            self.lattica.with_relay_servers(self.relay_servers).with_dcutr(True)
            if self.scheduler_peer_id is not None:
                logger.info(f"Using protocol: /{self.scheduler_peer_id}")
                self.lattica.with_protocol("/" + self.scheduler_peer_id)

        if len(self.announce_maddrs) > 0:
            logger.info(f"Using announce maddrs: {self.announce_maddrs}")
            self.lattica.with_external_addrs(self.announce_maddrs)

        if len(self.initial_peers) > 0:
            logger.info(f"Using initial peers: {self.initial_peers}")
            self.lattica.with_bootstraps(self.initial_peers)

        self.lattica.build()

        if len(self.relay_servers) > 0:
            try:
                is_symmetric_nat = self.lattica.is_symmetric_nat()
                if is_symmetric_nat is None:
                    logger.warning("Failed to get is symmetric NAT, skip")
                elif is_symmetric_nat:
                    logger.error(
                        "Your network NAT type is symmetric, relay does not work on this type of NAT, see https://en.wikipedia.org/wiki/Network_address_translation"
                    )
                    self._close_lattica()
                    return False
            except Exception as e:
                logger.exception(f"Error in is symmetric NAT: {e}, skip")

        if self.scheduler_addr == "auto":
            self.scheduler_peer_id = None
            for _ in range(20):
                try:
                    time.sleep(3)
                    self.scheduler_peer_id = self.lattica.get("scheduler_peer_id")
                    if self.scheduler_peer_id is not None:
                        self.scheduler_peer_id = self.scheduler_peer_id.value
                        logger.info(f"Found scheduler peer id: {self.scheduler_peer_id}")
                        break
                    logger.info(
                        f"Discovering scheduler peer id, {_ + 1} times, you can specify scheduler peer id by -s"
                    )
                except Exception as e:
                    logger.warning(f"Failed to get scheduler addr: {e}, waiting for 3 seconds.")
            if self.scheduler_peer_id is None:
                logger.error("Failed to get scheduler peer id")
                self._close_lattica()
                return False

        # ensure connect to scheduler
        for _ in range(10):
            if self.scheduler_peer_id in self.lattica.get_all_peers():
                return True
            logger.warning("Scheduler peer id not found, waiting for 1 second.")
            time.sleep(1)

        self._close_lattica()
        return False

    async def chat_completion(self, request_data, request_id: str, received_ts: int):
        if self.scheduler_addr is not None:  # central scheduler mode
            try:
                if self.scheduler_stub is None:
                    self.scheduler_stub = RPCConnectionHandler(self.lattica, None, None).get_stub(
                        self.scheduler_peer_id
                    )
                stub = self.scheduler_stub
                is_stream = request_data.get("stream", False)
                try:
                    if is_stream:

                        async def stream_generator():
                            response = stub.chat_completion(request_data)

                            try:
                                iterator = iterate_in_threadpool(response)
                                async for chunk in iterator:
                                    yield chunk
                            finally:
                                response.cancel()

                        resp = StreamingResponse(
                            stream_generator(),
                            media_type="text/event-stream",
                            headers={
                                "X-Content-Type-Options": "nosniff",
                                "Cache-Control": "no-cache",
                            },
                        )
                        logger.debug(f"Streaming response initiated for {request_id}")
                        return resp
                    else:
                        response = stub.chat_completion(request_data)
                        chunks = []
                        try:
                            iterator = iterate_in_threadpool(response)
                            async for chunk in iterator:
                                chunks.append(chunk)
                        finally:
                            response.cancel()

                        content = b"".join(chunks).decode()
                        if not content:
                            raise RuntimeError("Upstream chat completion returned an empty body")
                        logger.debug(f"Non-stream response completed for {request_id}")
                        return JSONResponse(content=json.loads(content))
                except Exception as e:
                    logger.exception(f"Error in _forward_request: {e}")
                    return JSONResponse(
                        content={"error": "Internal server error"},
                        status_code=500,
                    )

            except Exception as e:
                logger.exception(f"Error in chat completion: {e}")
                return JSONResponse(
                    content={"error": "Internal server error"},
                    status_code=500,
                )
        else:
            logger.error("No scheduler address specified")
            return JSONResponse(
                content={"error": "No scheduler address specified"},
                status_code=500,
            )

    def get_cluster_status(self):
        if self.scheduler_addr is not None:  # central scheduler mode
            try:
                if self.scheduler_stub is None:
                    self.scheduler_stub = RPCConnectionHandler(self.lattica, None, None).get_stub(
                        self.scheduler_peer_id
                    )
                stub = self.scheduler_stub
                try:

                    async def stream_status():
                        response = stub.cluster_status()
                        try:
                            iterator = iterate_in_threadpool(response)
                            async for chunk in iterator:
                                yield chunk
                        finally:
                            response.cancel()

                    resp = StreamingResponse(
                        stream_status(),
                        media_type="application/x-ndjson",
                        headers={
                            "X-Content-Type-Options": "nosniff",
                            "Cache-Control": "no-cache",
                        },
                    )
                    return resp
                except Exception as e:
                    logger.exception(f"Error in get cluster status: {e}")
                    return JSONResponse(
                        content={"error": "Internal server error"},
                        status_code=500,
                    )

            except Exception as e:
                logger.exception(f"Error in get cluster status: {e}")
                return JSONResponse(
                    content={"error": "Internal server error"},
                    status_code=500,
                )
        else:
            logger.error("No scheduler address specified")
            return JSONResponse(
                content={"error": "No scheduler address specified"},
                status_code=500,
            )

    async def run_uvicorn(self):
        """
        Since uvicorn.run() uses asyncio.run, we need another wrapper
        to create a uvicorn asyncio task to run multiple tasks.
        """
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            timeout_keep_alive=5,
            loop=uvicorn_loop_name(),
        )
        server = uvicorn.Server(config)
        await server.serve()

    def run(self):
        """
        Launch A FastAPI server that routes requests to the executor.

        Note:
        1. The HTTP server and executor both run in the main process.
        2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
        """
        for attempt in range(1, self._max_build_attempts + 1):
            if self.build_lattica():
                break
            if attempt == self._max_build_attempts:
                raise RuntimeError("Failed to build Lattica for node chat")
            logger.error(
                f"Failed to build lattica, waiting for 10 seconds before retry {attempt + 1}/{self._max_build_attempts}"
            )
            time.sleep(10)
        logger.info("Lattica built successfully")

        asyncio.run(
            init_app_states(
                app.state,
                self,
            )
        )
        asyncio.run(self.run_uvicorn())


def run_node_chat_http_server(args):
    node_chat_http_server = NodeChatHttpServer(args)
    node_chat_http_server.run()
