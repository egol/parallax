#!/usr/bin/env python3
"""Lightweight benchmark data collector for Parallax localnet testing.

Runs inside Docker containers using only Python stdlib — no numpy,
matplotlib, or other dependencies required.

Sends N chat requests to the local chat proxy, measures timing, and
collects cluster status metrics.  Outputs a single JSON object to stdout.
"""

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone


def fetch_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def send_chat_request(chat_url, model, prompt, stream=True, max_tokens=16, timeout=30):
    """Send a single chat request and measure timing.

    When stream=True, measures time-to-first-token (TTFT) from SSE chunks.
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        chat_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    ttft = None
    status_code = 0
    completion_tokens = 0
    prompt_tokens = 0
    body = ""

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = resp.status
            if stream:
                first_chunk = True
                for line in resp:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if not decoded.startswith("data:"):
                        continue
                    data_str = decoded[5:].strip()
                    if data_str == "[DONE]":
                        break
                    if first_chunk:
                        ttft = (time.perf_counter() - start) * 1000
                        first_chunk = False
                    try:
                        chunk = json.loads(data_str)
                        usage = chunk.get("usage", {})
                        if usage.get("completion_tokens"):
                            completion_tokens = usage["completion_tokens"]
                        if usage.get("prompt_tokens"):
                            prompt_tokens = usage["prompt_tokens"]
                    except json.JSONDecodeError:
                        pass
            else:
                body = resp.read().decode()
                data = json.loads(body)
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
    except Exception as exc:
        end = time.perf_counter()
        return {
            "e2e_ms": (end - start) * 1000,
            "ttft_ms": ttft,
            "status": status_code,
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "error": str(exc),
        }

    end = time.perf_counter()
    return {
        "e2e_ms": (end - start) * 1000,
        "ttft_ms": ttft if ttft is not None else (end - start) * 1000,
        "status": status_code,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "error": None,
    }


def collect_cluster_metrics(scheduler_url):
    """Fetch cluster status and extract per-node RTT and latency metrics."""
    try:
        raw = fetch_json(scheduler_url, timeout=5)
        data = raw.get("data", {})
    except Exception as exc:
        return {"error": str(exc)}

    node_rtts = {}
    node_layer_latencies = {}
    nodes = data.get("node_list", [])

    for node in nodes:
        node_id = node.get("node_id", "unknown")
        short_id = node_id[:12] if len(node_id) > 12 else node_id

        rtt_map = node.get("rtt_to_nodes", {})
        for peer_id, rtt_ms in rtt_map.items():
            peer_short = peer_id[:12] if len(peer_id) > 12 else peer_id
            key = f"{short_id}->{peer_short}"
            node_rtts[key] = rtt_ms

        layer_lat = node.get("layer_latency_ms")
        if layer_lat is not None and layer_lat != float("inf"):
            node_layer_latencies[short_id] = layer_lat

    topology = data.get("topology", {})
    totals = topology.get("totals", {})

    return {
        "num_workers": len(nodes),
        "ready_pipelines": totals.get("ready_pipelines", 0),
        "registered_workers": totals.get("registered_workers", 0),
        "discovered_workers": totals.get("discovered_workers", 0),
        "node_rtts": node_rtts,
        "node_layer_latencies": node_layer_latencies,
        "status": data.get("status"),
        "initialized": data.get("initialized"),
    }


def run_benchmark(args):
    results = []
    errors = 0

    # Warmup: send a few requests to flush the synthetic backend's readiness
    # cycle and P2P heartbeat spikes before measuring.
    for _ in range(args.warmup):
        send_chat_request(
            chat_url=args.chat_url,
            model=args.model,
            prompt="warmup",
            stream=args.stream,
            max_tokens=4,
            timeout=args.request_timeout,
        )
        time.sleep(0.05)

    wall_start = time.perf_counter()
    for i in range(args.num_requests):
        result = send_chat_request(
            chat_url=args.chat_url,
            model=args.model,
            prompt=args.prompt,
            stream=args.stream,
            max_tokens=args.max_tokens,
            timeout=args.request_timeout,
        )
        results.append(result)
        if result.get("error"):
            errors += 1
        if args.delay > 0 and i < args.num_requests - 1:
            time.sleep(args.delay)
    wall_end = time.perf_counter()

    cluster = collect_cluster_metrics(args.scheduler_url)

    successful = [r for r in results if not r.get("error")]
    e2e_values = [r["e2e_ms"] for r in successful]
    ttft_values = [r["ttft_ms"] for r in successful if r.get("ttft_ms") is not None]
    total_tokens = sum(r.get("completion_tokens", 0) for r in successful)

    # Wall-clock duration for throughput (includes inter-request delay)
    duration_s = wall_end - wall_start if results else 0

    def percentile(vals, pct):
        if not vals:
            return 0
        s = sorted(vals)
        idx = int(len(s) * pct / 100)
        return s[min(idx, len(s) - 1)]

    def trimmed_mean(vals, trim_pct=10):
        """Mean after removing the top trim_pct% of values."""
        if not vals:
            return 0
        s = sorted(vals)
        cutoff = max(1, int(len(s) * (1 - trim_pct / 100)))
        trimmed = s[:cutoff]
        return sum(trimmed) / len(trimmed) if trimmed else 0

    summary = {
        "mean_e2e_ms": sum(e2e_values) / len(e2e_values) if e2e_values else 0,
        "trimmed_mean_e2e_ms": trimmed_mean(e2e_values, 10),
        "p50_e2e_ms": percentile(e2e_values, 50),
        "p90_e2e_ms": percentile(e2e_values, 90),
        "p99_e2e_ms": percentile(e2e_values, 99),
        "min_e2e_ms": min(e2e_values) if e2e_values else 0,
        "max_e2e_ms": max(e2e_values) if e2e_values else 0,
        "mean_ttft_ms": sum(ttft_values) / len(ttft_values) if ttft_values else 0,
        "trimmed_mean_ttft_ms": trimmed_mean(ttft_values, 10),
        "p50_ttft_ms": percentile(ttft_values, 50),
        "total_completion_tokens": total_tokens,
        "requests_per_second": len(successful) / duration_s if duration_s > 0 else 0,
        "tokens_per_second": total_tokens / duration_s if duration_s > 0 else 0,
        "success_count": len(successful),
        "error_count": errors,
    }

    # Compute outlier-free throughput: exclude requests above p90 e2e
    # so that random backend spikes don't distort throughput metrics.
    p90_cutoff = percentile(e2e_values, 90)
    trimmed_reqs = [r for r in successful if r["e2e_ms"] <= p90_cutoff * 1.01]
    trimmed_tokens = sum(r.get("completion_tokens", 0) for r in trimmed_reqs)
    trimmed_e2e_sum_s = sum(r["e2e_ms"] for r in trimmed_reqs) / 1000 if trimmed_reqs else 0
    summary["trimmed_requests_per_second"] = (
        len(trimmed_reqs) / trimmed_e2e_sum_s if trimmed_e2e_sum_s > 0 else 0
    )
    summary["trimmed_tokens_per_second"] = (
        trimmed_tokens / trimmed_e2e_sum_s if trimmed_e2e_sum_s > 0 else 0
    )

    # Fetch model name, topology, and node hardware from full cluster status
    model_name = args.model
    topology_desc = ""
    system_info = {}
    try:
        raw = fetch_json(args.scheduler_url, timeout=5)
        data = raw.get("data", {})
        model_name = data.get("model_name", args.model) or args.model
        num_nodes = data.get("init_nodes_num", 0)
        network_mode = data.get("network_mode", "unknown")
        topo = data.get("topology", {})
        totals = topo.get("totals", {})
        pipelines = totals.get("registered_pipelines", 0)
        ready = totals.get("ready_pipelines", 0)
        topology_desc = (
            f"{num_nodes}-node init, {pipelines} pipelines "
            f"({ready} ready), {network_mode} network"
        )

        # Collect per-node hardware info
        nodes = data.get("node_list", [])
        node_details = []
        for n in nodes:
            gpu_name = n.get("gpu_name", "Unknown")
            gpu_mem = n.get("gpu_memory", 0)
            device = n.get("device", "Unknown")
            layers = f"{n.get('start_layer', '?')}-{n.get('end_layer', '?')}"
            node_details.append({
                "gpu": gpu_name,
                "memory_gb": gpu_mem,
                "device": device,
                "layers": layers,
            })

        total_mem = totals.get("registered_memory_gb", 0)
        system_info = {
            "total_memory_gb": total_mem,
            "nodes": node_details,
            "max_concurrent_requests": data.get("max_running_request", 0),
        }
    except Exception:
        pass

    # Collect host-level system info (CPU, RAM) from inside the container
    try:
        import os as _os
        import platform as _plat
        cpu_count = _os.cpu_count() or 0
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_kb = int(line.split()[1])
                    system_info["host_ram_gb"] = round(mem_kb / 1024 / 1024, 1)
                    break
        system_info["cpu_count"] = cpu_count
        system_info["platform"] = _plat.machine()
    except Exception:
        pass

    output = {
        "condition": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "topology": topology_desc,
        "system": system_info,
        "num_requests": args.num_requests,
        "stream": args.stream,
        "summary": summary,
        "requests": results,
        "cluster": cluster,
    }

    json.dump(output, sys.stdout)
    sys.stdout.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Parallax localnet benchmark collector")
    parser.add_argument("--chat-url", default="http://127.0.0.1:3200/v1/chat/completions")
    parser.add_argument("--scheduler-url", default="http://127.0.0.1:3301/cluster/status_json")
    parser.add_argument("--model", default="/parallax/docker/test-models/parallax-smoke")
    parser.add_argument("--prompt", default="Benchmark ping from nettest collector.")
    parser.add_argument("--label", default="baseline", help="Label for this benchmark condition")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between requests (seconds)")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup requests before measuring")
    parser.add_argument("--no-stream", action="store_true", help="Use non-streaming requests")

    args = parser.parse_args()
    args.stream = not args.no_stream

    run_benchmark(args)


if __name__ == "__main__":
    main()
