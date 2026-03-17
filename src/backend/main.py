import asyncio
import json
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.server.request_handler import RequestHandler
from backend.server.scheduler_manage import SchedulerManage
from backend.server.server_args import parse_args
from backend.server.static_config import (
    get_model_info,
    get_model_list,
    get_node_join_command,
    init_model_info_dict_cache,
)
from parallax_utils.ascii_anime import display_parallax_run
from parallax_utils.file_util import get_project_root
from parallax_utils.logging_config import get_logger, set_log_level
from parallax_utils.version_check import check_latest_release

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = get_logger(__name__)

scheduler_manage = None
request_handler = RequestHandler()


def _coerce_init_nodes_num(value):
    try:
        init_nodes_num = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("init_nodes_num must be an integer") from exc
    if init_nodes_num < 1:
        raise ValueError("init_nodes_num must be greater than 0")
    return init_nodes_num


def _coerce_is_local_network(value):
    if value is None:
        if scheduler_manage is not None:
            return scheduler_manage.get_is_local_network()
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError("is_local_network must be a boolean")


async def _validate_scheduler_init_request(request_data):
    model_name = request_data.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name is required")
    model_name = model_name.strip()

    init_nodes_num = _coerce_init_nodes_num(request_data.get("init_nodes_num"))
    is_local_network = _coerce_is_local_network(request_data.get("is_local_network"))

    try:
        model_info = await asyncio.to_thread(
            get_model_info, model_name, scheduler_manage.use_hfcache
        )
    except Exception as exc:
        raise ValueError(f"model_name is invalid or unavailable: {model_name}") from exc
    if model_info is None:
        raise ValueError(f"model_name is invalid or unavailable: {model_name}")

    return model_name, init_nodes_num, is_local_network


@app.post("/weight/refit")
async def weight_refit(raw_request: Request):
    request_data = await raw_request.json()
    status = scheduler_manage.weight_refit(request_data)
    if status:
        return JSONResponse(
            content={
                "type": "weight_refit",
                "data": None,
            },
            status_code=200,
        )
    else:
        return JSONResponse(
            content={
                "type": "weight_refit",
                "data": "Sever not ready",
            },
            status_code=500,
        )


@app.get("/weight/refit/timestamp")
async def weight_refit_timstamp():
    last_refit_time = scheduler_manage.get_last_refit_time()

    return JSONResponse(
        content={
            "latest_timestamp": last_refit_time,
        },
        status_code=200,
    )


@app.get("/model/list")
async def model_list():
    return JSONResponse(
        content={
            "type": "model_list",
            "data": get_model_list(),
        },
        status_code=200,
    )


@app.post("/scheduler/init")
async def scheduler_init(raw_request: Request):
    if scheduler_manage is None:
        return JSONResponse(
            content={
                "type": "scheduler_init",
                "error": "scheduler is not initialized",
            },
            status_code=503,
        )

    request_data = await raw_request.json()
    try:
        model_name, init_nodes_num, is_local_network = await _validate_scheduler_init_request(
            request_data
        )
    except ValueError as exc:
        return JSONResponse(
            content={
                "type": "scheduler_init",
                "error": str(exc),
            },
            status_code=400,
        )

    try:
        # Validate the incoming model config before this boundary so a healthy scheduler
        # is not torn down for malformed requests.
        if scheduler_manage.is_running():
            logger.info(f"Stopping existing scheduler to switch to model: {model_name}")
            scheduler_manage.stop()

        # Start scheduler with new model
        logger.info(
            f"Initializing scheduler with model: {model_name}, init_nodes_num: {init_nodes_num}"
        )
        # `scheduler_manage.run()` performs synchronous Lattica registration that can
        # block for seconds. Run it off the event loop so status/model endpoints stay
        # responsive during allocation and downloads.
        await asyncio.to_thread(
            scheduler_manage.run, model_name, init_nodes_num, is_local_network
        )

        return JSONResponse(
            content={
                "type": "scheduler_init",
                "data": scheduler_manage.get_cluster_status()["data"],
            },
            status_code=200,
        )
    except Exception as e:
        logger.exception(f"Error initializing scheduler: {e}")
        return JSONResponse(
            content={
                "type": "scheduler_init",
                "error": str(e),
            },
            status_code=500,
        )


@app.post("/scheduler/bootstrap")
async def scheduler_bootstrap(raw_request: Request):
    request_data = await raw_request.json()
    is_local_network = request_data.get("is_local_network", True)

    try:
        # `bootstrap_network()` can block while the local P2P node advertises and
        # registers services. Keep the HTTP event loop free so probes can continue.
        await asyncio.to_thread(scheduler_manage.bootstrap_network, is_local_network)
        return JSONResponse(
            content={
                "type": "scheduler_bootstrap",
                "data": scheduler_manage.get_cluster_status()["data"],
            },
            status_code=200,
        )
    except Exception as e:
        logger.exception(f"Error bootstrapping scheduler network: {e}")
        return JSONResponse(
            content={
                "type": "scheduler_bootstrap",
                "error": str(e),
            },
            status_code=500,
        )


@app.get("/node/join/command")
async def node_join_command():
    peer_id = scheduler_manage.get_peer_id()
    is_local_network = scheduler_manage.get_is_local_network()

    return JSONResponse(
        content={
            "type": "node_join_command",
            "data": get_node_join_command(peer_id, is_local_network),
        },
        status_code=200,
    )


@app.get("/cluster/status")
async def cluster_status():
    async def stream_cluster_status():
        snapshot = scheduler_manage.get_cluster_status()
        yield json.dumps(snapshot, ensure_ascii=False) + "\n"
        version = snapshot.get("data", {}).get("snapshot_version", 0)
        while True:
            next_snapshot = await asyncio.to_thread(
                scheduler_manage.wait_for_cluster_status_version, version, 30.0
            )
            if next_snapshot is None:
                continue
            version = next_snapshot.get("data", {}).get("snapshot_version", version)
            yield json.dumps(next_snapshot, ensure_ascii=False) + "\n"

    return StreamingResponse(
        stream_cluster_status(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/cluster/status_json")
async def cluster_status_json() -> JSONResponse:
    if scheduler_manage is None:
        return JSONResponse(content={"error": "Scheduler is not initialized"}, status_code=503)
    return JSONResponse(content=scheduler_manage.get_cluster_status(), status_code=200)


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    request_data = await raw_request.json()
    request_id = uuid.uuid4()
    received_ts = time.time()
    return await request_handler.v1_chat_completions(request_data, request_id, received_ts)


# Disable caching for index.html
@app.get("/")
async def serve_index():
    response = FileResponse(str(get_project_root()) + "/src/frontend/dist/index.html")
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

if __name__ == "__main__":
    args = parse_args()
    set_log_level(args.log_level)
    logger.info(f"args: {args}")

    if args.model_name is None:
        init_model_info_dict_cache(args.use_hfcache)

    if args.log_level != "DEBUG":
        display_parallax_run()

    check_latest_release()

    scheduler_manage = SchedulerManage(
        initial_peers=args.initial_peers,
        relay_servers=args.relay_servers,
        dht_prefix=args.dht_prefix,
        host_maddrs=[
            f"/ip4/0.0.0.0/tcp/{args.tcp_port}",
            f"/ip4/0.0.0.0/udp/{args.udp_port}/quic-v1",
        ],
        announce_maddrs=args.announce_maddrs,
        http_port=args.port,
        use_hfcache=args.use_hfcache,
        enable_weight_refit=args.enable_weight_refit,
        weight_refit_mode=args.weight_refit_mode,
    )

    request_handler.set_scheduler_manage(scheduler_manage)

    model_name = args.model_name
    init_nodes_num = args.init_nodes_num
    is_local_network = args.is_local_network
    if model_name is not None and init_nodes_num is not None:
        scheduler_manage.run(model_name, init_nodes_num, is_local_network)

    host = args.host
    port = args.port

    uvicorn.run(app, host=host, port=port, log_level="info", loop="uvloop")
