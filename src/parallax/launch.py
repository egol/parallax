"""
Launch the Parallax server.

This script is used to launch the Parallax server.
It will start the following services:
    1.Executor each tp_rank as a subprocess.
    2.HTTP server as a subprocess.
    3.P2P server as a subprocess.

Example command:
python src/parallax/launch.py \
    --model-path Qwen/Qwen3-0.6B \
    --max-num-tokens-per-batch 16384 \
    --max-batch-size 128 \
    --start-layer 0 \
    --end-layer 28
"""

import argparse
import multiprocessing
import os
import tempfile
import time

from parallax.p2p.server import ServerState, launch_p2p_server_process, stop_p2p_server
from parallax.server.server_args import parse_args
from parallax.utils.shared_state import SharedState
from parallax_utils.ascii_anime import display_parallax_join
from parallax_utils.logging_config import get_logger, set_log_level
from parallax_utils.version_check import check_latest_release

logger = get_logger("parallax.launch")


def _update_args_from_shared_state(args, shared_state: SharedState, force_update: bool):
    """Update args with layer allocation from shared state"""
    model_info = shared_state.get_model_info()
    args.start_layer = model_info["block_start_index"]
    args.end_layer = model_info["block_end_index"]
    if args.model_path is not None and force_update == False:
        # Use local model path first
        pass
    elif model_info["model_name"]:
        # Update model_path if provided
        args.model_path = model_info["model_name"]
        logger.debug(f"Updated model_path to: {args.model_path}")
    else:
        assert False, "Neither scheduler nor worker provides a valid model path!"
    # Update tp_size if provided, otherwise keep current value
    args.tp_size = model_info["tp_size"] or args.tp_size
    # Update weight refit switch
    args.enable_weight_refit = model_info["enable_weight_refit"] or args.enable_weight_refit
    args.weight_refit_mode = model_info["weight_refit_mode"] or args.weight_refit_mode


def _stop_executor_processes(executor_subprocs):
    """Stop all executor processes"""
    from parallax.server.executor.factory import stop_executor_process

    for executor_process in executor_subprocs:
        if executor_process.is_alive():
            logger.debug(f"Terminating executor process {executor_process.pid}")
            stop_executor_process(executor_process)


def _wait_executors_check_layer_change(shared_state: SharedState, executor_subprocs):
    """Wait for executor processes and check if layer allocation changed.

    Returns:
        True if layer allocation changed (need to reload executors),
        False if all executors exited normally.
    """
    while any(proc.is_alive() for proc in executor_subprocs):
        for proc in executor_subprocs:
            if proc.is_alive():
                proc.join(timeout=1.0)  # Check every second

        if shared_state.get_layer_allocation_changed():
            return True

    # Check race condition: layer allocation changed after all processes exited
    return shared_state.get_layer_allocation_changed()


def _is_test_mode() -> bool:
    return os.getenv("PARALLAX_TEST_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def _launch_http_server_for_mode(args, test_mode: bool):
    if test_mode:
        from parallax.test_mode import launch_test_http_server

        return launch_test_http_server(args)

    from parallax.server.http_server import launch_http_server

    return launch_http_server(args)


def _stop_http_server_for_mode(process, test_mode: bool):
    if process is None:
        return
    if test_mode:
        from parallax.test_mode import stop_test_http_server

        stop_test_http_server(process)
        return

    from parallax.server.http_server import stop_http_server

    stop_http_server(process)


def _launch_executor_processes(args, shared_state: SharedState, conn_refit):
    from parallax.server.executor.factory import run_executor_process

    executor_subprocs = []
    conn_tp_0 = [conn_refit]
    conn_tp_i = []
    for i in range(1, args.tp_size):
        conn1, conn2 = multiprocessing.Pipe()
        conn_tp_0.append(conn1)
        conn_tp_i.append(conn2)

    for tp_rank in range(args.tp_size):
        args_copy = argparse.Namespace(**vars(args))
        args_copy.tp_rank = tp_rank
        proc = multiprocessing.Process(
            target=run_executor_process,
            args=(
                args_copy,
                shared_state.dict,
                conn_tp_0 if tp_rank == 0 else [conn_tp_i[tp_rank - 1]],
            ),
        )
        proc.start()
        executor_subprocs.append(proc)

    return executor_subprocs


def _wait_worker_check_layer_change(p2p_server_process, shared_state: SharedState):
    while p2p_server_process is not None and p2p_server_process.is_alive():
        p2p_server_process.join(timeout=1.0)
        if shared_state.get_layer_allocation_changed():
            return True

    return shared_state.get_layer_allocation_changed()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    p2p_server_process = None
    http_server_process = None
    executor_subprocs = []
    test_mode = False
    # Shared state for layer allocation info (used when P2P server is in subprocess)
    shared_state = SharedState.create()
    shared_state.set_status(ServerState.JOINING.value)
    shared_state.update_runtime_state(
        status=ServerState.JOINING.value,
        init_stage="allocating",
        init_detail="Waiting for scheduler layer allocation.",
        failure_reason="",
    )

    try:
        args = parse_args()
        test_mode = _is_test_mode()
        set_log_level(args.log_level)
        logger.debug(f"args: {args}")
        args.recv_from_peer_addr = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.send_to_peer_addr = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.executor_input_ipc = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.executor_output_ipc = f"ipc://{tempfile.NamedTemporaryFile().name}"
        if args.nccl_port is None:
            from parallax.utils.utils import initialize_nccl_port

            args.nccl_port = initialize_nccl_port()

        # Silence tokenizer warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        logger.debug(f"executor_input_addr: {args.executor_input_ipc}")
        logger.debug(f"executor_output_addr: {args.executor_output_ipc}")
        logger.debug(f"nccl_port: {args.nccl_port}")

        # Pipe for subprocess communication
        conn_main, conn_refit = multiprocessing.Pipe()

        if args.scheduler_addr is None:
            if args.log_level != "DEBUG":
                display_parallax_join(args.model_path)
            check_latest_release()

            from parallax.utils.utils import fetch_model_from_hf

            config = fetch_model_from_hf(args.model_path, local_files_only=args.use_hfcache)
            if args.start_layer is None:
                args.start_layer = 0
            if args.end_layer is None:
                args.end_layer = config.get("num_hidden_layers")

            # only launch http server on head node
            if args.start_layer == 0:
                http_server_process = _launch_http_server_for_mode(args, test_mode)
            # Launch P2P server as subprocess
            if not (args.start_layer == 0 and args.end_layer == config.get("num_hidden_layers")):
                p2p_server_process = launch_p2p_server_process(
                    initial_peers=args.initial_peers,
                    scheduler_addr=args.scheduler_addr,
                    relay_servers=args.relay_servers,
                    pp_start_layer=args.start_layer,
                    pp_end_layer=args.end_layer,
                    hidden_layers=config.get("num_hidden_layers"),
                    tp_size=args.tp_size,
                    dp_size=args.dp_size,
                    tcp_port=args.tcp_port,
                    udp_port=args.udp_port,
                    dht_prefix=args.dht_prefix,
                    announce_maddrs=args.announce_maddrs,
                    http_port=args.port,
                    notify_url=args.notify_url,
                    recv_from_peer_addr=args.recv_from_peer_addr,
                    send_to_peer_addr=args.send_to_peer_addr,
                    model_name=args.model_path,
                    max_batch_size=args.max_batch_size,
                    max_sequence_length=args.max_sequence_length,
                    param_mem_ratio=args.param_mem_ratio,
                    kvcache_mem_ratio=args.kvcache_mem_ratio,
                    shared_state=shared_state.dict,
                    log_level=args.log_level,
                    conn=conn_main,
                )

            if test_mode:
                logger.info(
                    "PARALLAX_TEST_MODE enabled: skipping executor startup for standalone node"
                )
            else:
                executor_subprocs = _launch_executor_processes(args, shared_state, conn_refit)

            time.sleep(2)  # Give executors time to start
            shared_state.set_status(ServerState.READY.value)
            shared_state.update_runtime_state(
                status=ServerState.READY.value,
                model_name=args.model_path,
                init_stage="ready",
                init_detail="Standalone Parallax node is ready.",
                failure_reason="",
            )

            # Wait for all executor processes
            for proc in executor_subprocs:
                proc.join()
        else:
            # Launch P2P server as subprocess (with scheduler)
            # Pass dict to subprocess (multiprocessing requires serializable objects)
            p2p_server_process = launch_p2p_server_process(
                initial_peers=args.initial_peers,
                scheduler_addr=args.scheduler_addr,
                relay_servers=args.relay_servers,
                pp_start_layer=args.start_layer,
                pp_end_layer=args.end_layer,
                hidden_layers=None,
                tp_size=args.tp_size,
                dp_size=args.dp_size,
                tcp_port=args.tcp_port,
                udp_port=args.udp_port,
                dht_prefix=args.dht_prefix,
                announce_maddrs=args.announce_maddrs,
                http_port=args.port,
                notify_url=args.notify_url,
                recv_from_peer_addr=args.recv_from_peer_addr,
                send_to_peer_addr=args.send_to_peer_addr,
                model_name=args.model_path,
                max_batch_size=args.max_batch_size,
                max_sequence_length=args.max_sequence_length,
                param_mem_ratio=args.param_mem_ratio,
                kvcache_mem_ratio=args.kvcache_mem_ratio,
                shared_state=shared_state.dict,  # Pass dict to subprocess
                log_level=args.log_level,
                conn=conn_main,
            )

            # Wait for layer allocation from scheduler (via shared state)
            logger.debug("Waiting for layer allocation from scheduler...")
            max_wait_time = 300  # 5 minutes
            wait_start = time.time()
            while True:
                model_info = shared_state.get_model_info()
                if (
                    model_info["block_start_index"] is not None
                    and model_info["block_end_index"] is not None
                    and model_info["model_name"] is not None
                ):
                    break
                if time.time() - wait_start > max_wait_time:
                    logger.error("Timeout waiting for layer allocation from scheduler")
                    shared_state.update_runtime_state(
                        status=ServerState.JOINING.value,
                        init_stage="failed",
                        init_detail="Timed out waiting for layer allocation from the scheduler.",
                        failure_reason="Failed to get layer allocation from scheduler",
                    )
                    raise RuntimeError("Failed to get layer allocation from scheduler")
                time.sleep(1)

            # Get layer allocation from shared state
            _update_args_from_shared_state(args, shared_state, force_update=False)
            shared_state.update_runtime_state(
                status=ServerState.INITIALIZING.value,
                model_name=args.model_path,
                init_stage="allocating",
                init_detail=(
                    f"Received layer allocation [{args.start_layer}, {args.end_layer}); "
                    "initializing executors."
                ),
                failure_reason="",
            )

            logger.debug(
                f"Start Executor with start_layer: {args.start_layer}, end_layer: {args.end_layer}, "
                f"model: {args.model_path}"
            )

            if args.log_level != "DEBUG":
                display_parallax_join(args.model_path)
            check_latest_release()

            # Main execution loop with layer reallocation support
            while True:
                try:
                    # only launch http server on head node
                    if args.start_layer == 0:
                        http_server_process = _launch_http_server_for_mode(args, test_mode)

                    if test_mode:
                        logger.info(
                            "PARALLAX_TEST_MODE enabled: worker will attach with synthetic "
                            "chat serving and no executor processes"
                        )
                        shared_state.set_status(ServerState.READY.value)
                        shared_state.update_runtime_state(
                            status=ServerState.READY.value,
                            model_name=args.model_path,
                            init_stage="ready",
                            init_detail="Synthetic Parallax test worker is ready.",
                            failure_reason="",
                        )
                        if _wait_worker_check_layer_change(p2p_server_process, shared_state):
                            logger.warning("Layer allocation changed! Reloading test worker...")
                            shared_state.update(
                                _layer_allocation_changed=False,
                                status=ServerState.INITIALIZING.value,
                            )
                            shared_state.update_runtime_state(
                                status=ServerState.INITIALIZING.value,
                                model_name=args.model_path,
                                init_stage="allocating",
                                init_detail="Layer allocation changed; reloading the worker.",
                                downloaded_files=None,
                                total_files=None,
                                current_file="",
                                failure_reason="",
                            )
                            if http_server_process is not None:
                                _stop_http_server_for_mode(http_server_process, test_mode)
                                http_server_process = None
                            _update_args_from_shared_state(args, shared_state, force_update=True)
                            logger.info(
                                f"Reloading test worker with layers [{args.start_layer}, "
                                f"{args.end_layer})"
                            )
                            continue

                        break

                    executor_subprocs = _launch_executor_processes(args, shared_state, conn_refit)

                    # Wait for executors and restart if layer allocation changes
                    if _wait_executors_check_layer_change(shared_state, executor_subprocs):
                        logger.warning("Layer allocation changed! Stopping executors to reload...")
                        # Reset flag and set status to INITIALIZING
                        shared_state.update(
                            _layer_allocation_changed=False,
                            status=ServerState.INITIALIZING.value,
                        )
                        shared_state.update_runtime_state(
                            status=ServerState.INITIALIZING.value,
                            model_name=args.model_path,
                            init_stage="allocating",
                            init_detail="Layer allocation changed; reloading the executor.",
                            downloaded_files=None,
                            total_files=None,
                            current_file="",
                            failure_reason="",
                        )
                        _stop_executor_processes(executor_subprocs)
                        if http_server_process is not None:
                            _stop_http_server_for_mode(http_server_process, test_mode)
                            http_server_process = None
                        _update_args_from_shared_state(args, shared_state, force_update=True)
                        logger.info(
                            f"Reloading executor with layers [{args.start_layer}, {args.end_layer})"
                        )
                        continue

                    # All processes exited normally
                    break
                except KeyboardInterrupt:
                    logger.debug("Received interrupt signal, shutting down...")
                    break
                except Exception as e:
                    logger.exception(f"Executor error: {e}")
                    shared_state.update_runtime_state(
                        status=ServerState.INITIALIZING.value,
                        model_name=args.model_path,
                        init_stage="failed",
                        init_detail="Executor startup failed.",
                        failure_reason=str(e),
                    )
                    # Shutdown all executor processes on error
                    from parallax.server.executor.factory import stop_executor_process

                    for proc in executor_subprocs:
                        if proc.is_alive():
                            stop_executor_process(proc)
                    raise
    except KeyboardInterrupt:
        logger.debug("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.exception(e)
        shared_state.update_runtime_state(
            status=shared_state.get_status(),
            model_name=shared_state.get("model_name"),
            init_stage="failed",
            init_detail="Parallax worker startup failed.",
            failure_reason=str(e),
        )
    finally:
        # Shutdown all processes
        logger.debug("Shutting down all processes...")

        # Shutdown executor subprocesses
        if executor_subprocs:
            from parallax.server.executor.factory import stop_executor_process

            for executor_process in executor_subprocs:
                if executor_process.is_alive():
                    stop_executor_process(executor_process)

        # Shutdown P2P server subprocess
        if p2p_server_process is not None:
            stop_p2p_server(p2p_server_process)

        # Shutdown http server
        if http_server_process is not None:
            _stop_http_server_for_mode(http_server_process, test_mode)

        logger.debug("All processes shut down.")
