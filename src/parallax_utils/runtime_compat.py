from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import uvloop
except Exception:  # pragma: no cover - optional dependency
    uvloop = None


def uvicorn_loop_name() -> str:
    return "uvloop" if uvloop is not None else "asyncio"


def create_event_loop() -> asyncio.AbstractEventLoop:
    try:
        asyncio.get_event_loop().stop()
    except RuntimeError:
        pass

    if uvloop is not None:
        uvloop.install()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def _allocate_loopback_tcp_endpoint() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        _, port = sock.getsockname()
    return f"tcp://127.0.0.1:{port}"


def _allocate_ipc_endpoint(prefix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=f"{prefix}-", suffix=".sock")
    os.close(fd)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    return f"ipc://{path}"


def local_zmq_endpoint(prefix: str) -> str:
    if sys.platform.startswith("win"):
        return _allocate_loopback_tcp_endpoint()
    return _allocate_ipc_endpoint(prefix)


def popen_session_kwargs() -> dict[str, int | bool]:
    if sys.platform.startswith("win"):
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": creationflags}
    return {"start_new_session": True}


def terminate_process_tree(process: subprocess.Popen, *, grace_seconds: float = 5.0) -> None:
    if process.poll() is not None:
        return

    if sys.platform.startswith("win"):
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=grace_seconds)
            return
        except Exception:
            pass

        try:
            process.terminate()
            process.wait(timeout=grace_seconds)
            return
        except Exception:
            pass

        process.kill()
        process.wait()
        return

    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=grace_seconds)
        return
    except Exception:
        pass

    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
        return
    except Exception:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except Exception:
        process.kill()
    process.wait()


def bundled_frontend_path(project_root: Path, entrypoint: str) -> Path:
    return project_root / "src" / "frontend" / "dist" / entrypoint
