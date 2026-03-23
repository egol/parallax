#!/usr/bin/env python3
"""Matplotlib plot generator for Parallax localnet benchmark results.

Reads benchmark-results.json produced by run_nettest_benchmark.sh and
generates visualization plots for inference latency, throughput, TTFT,
and node-to-node RTT across different network conditions.

Every plot includes a context footer showing the model, topology,
request count, and date so figures are self-documenting.

Usage:
    python3 plot_nettest.py \
        --input /tmp/parallax-nettest-benchmark/benchmark-results.json \
        --output /tmp/parallax-nettest-benchmark/plots/
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path):
    with open(path) as f:
        return json.load(f)


def build_context_lines(results):
    """Build descriptive subtitle lines from benchmark metadata."""
    first = results[0]
    model = first.get("model", "unknown model")
    topology = first.get("topology", "")
    num_requests = first.get("num_requests", "?")
    num_conditions = len(results)
    ts = first.get("timestamp", "")
    date = ts[:10] if ts else "unknown date"
    stream = "streaming" if first.get("stream") else "non-streaming"
    system = first.get("system", {})

    # Clarify synthetic test models
    if "test-models" in model or "parallax-smoke" in model:
        model = f"{model} (synthetic echo backend, no real inference)"

    # Line 1: model + topology + test params
    parts = [model]
    if topology:
        parts.append(topology)
    parts.append(f"{num_requests} reqs x {num_conditions} conditions ({stream}), {date}")
    line1 = "  |  ".join(parts)

    # Line 2: system info (hardware)
    sys_parts = []
    if system.get("cpu_count"):
        sys_parts.append(f"{system['cpu_count']} vCPUs")
    if system.get("host_ram_gb"):
        sys_parts.append(f"{system['host_ram_gb']} GB RAM")
    if system.get("platform"):
        sys_parts.append(system["platform"])
    total_mem = system.get("total_memory_gb", 0)
    nodes = system.get("nodes", [])
    if nodes:
        gpu = nodes[0].get("gpu", "Unknown")
        device = nodes[0].get("device", "Unknown")
        per_node_mem = nodes[0].get("memory_gb", 0)
        if gpu != "Unknown":
            sys_parts.append(f"{len(nodes)}x {gpu} ({per_node_mem} GB each)")
        else:
            sys_parts.append(f"{len(nodes)} nodes, {device} device, {per_node_mem} GB/node")
        if total_mem:
            sys_parts.append(f"{total_mem:.0f} GB total cluster memory")
    max_req = system.get("max_concurrent_requests", 0)
    if max_req:
        sys_parts.append(f"max {max_req} concurrent reqs")

    line2 = "  |  ".join(sys_parts) if sys_parts else ""
    return line1, line2


def apply_context(fig, title, context):
    """Set suptitle with context subtitle lines and add bottom margin."""
    line1, line2 = context
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.935, line1,
        ha="center", va="top", fontsize=7,
        color="#666666", style="italic",
        transform=fig.transFigure,
    )
    if line2:
        fig.text(
            0.5, 0.91, line2,
            ha="center", va="top", fontsize=7,
            color="#888888",
            transform=fig.transFigure,
        )
    fig.subplots_adjust(top=0.86, bottom=0.18)


def plot_latency_by_condition(results, output_dir, context_line):
    """Bar chart of median E2E latency per network condition."""
    labels = [r["condition"] for r in results]

    # Compute p25, p50, p75 from raw request data for tight error bars
    p25s, p50s, p75s = [], [], []
    for r in results:
        vals = sorted([req["e2e_ms"] for req in r["requests"] if not req.get("error")])
        n = len(vals)
        p25s.append(vals[max(0, n // 4)] if vals else 0)
        p50s.append(vals[n // 2] if vals else 0)
        p75s.append(vals[min(n - 1, n * 3 // 4)] if vals else 0)

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.9), 7))
    x = range(len(labels))

    # Error bars: p25 to p75 (IQR, excludes outlier spikes)
    err_lo = [p - lo for p, lo in zip(p50s, p25s)]
    err_hi = [hi - p for p, hi in zip(p50s, p75s)]
    ax.bar(x, p50s, color="#4C72B0", alpha=0.85,
           yerr=[err_lo, err_hi], capsize=4, error_kw={"linewidth": 1.2, "color": "#2C3E50"},
           label="Median (p50), whiskers: p25–p75")

    for i, v in enumerate(p50s):
        ax.text(i, v + max(p50s) * 0.05, f"{v:.0f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("E2E Latency (ms)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    apply_context(fig, "End-to-End Chat Latency by Network Condition", context_line)
    fig.savefig(os.path.join(output_dir, "latency_by_condition.png"), dpi=150)
    plt.close(fig)
    print("  latency_by_condition.png")


def plot_latency_distribution(results, output_dir, context_line):
    """Box plot of E2E latency distributions."""
    labels = [r["condition"] for r in results]
    data = []
    for r in results:
        e2e_values = [req["e2e_ms"] for req in r["requests"] if not req.get("error")]
        data.append(e2e_values if e2e_values else [0])

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.9), 7))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)

    colors = plt.cm.viridis([i / max(1, len(labels) - 1) for i in range(len(labels))])
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("E2E Latency (ms)")
    ax.grid(axis="y", alpha=0.3)
    apply_context(fig, "Latency Distribution by Network Condition", context_line)
    fig.savefig(os.path.join(output_dir, "latency_distribution.png"), dpi=150)
    plt.close(fig)
    print("  latency_distribution.png")


def plot_ttft_by_condition(results, output_dir, context_line):
    """Bar chart of Time to First Token (median)."""
    labels = [r["condition"] for r in results]
    ttfts = [r["summary"].get("p50_ttft_ms", r["summary"]["mean_ttft_ms"]) for r in results]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.9), 7))
    ax.bar(range(len(labels)), ttfts, color="#27AE60", alpha=0.85)

    for i, v in enumerate(ttfts):
        ax.text(i, v + max(ttfts) * 0.02, f"{v:.0f}", ha="center", fontsize=8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("TTFT (ms)")
    ax.grid(axis="y", alpha=0.3)
    apply_context(fig, "Time to First Token by Network Condition", context_line)
    fig.savefig(os.path.join(output_dir, "ttft_by_condition.png"), dpi=150)
    plt.close(fig)
    print("  ttft_by_condition.png")


def plot_throughput_by_condition(results, output_dir, context_line):
    """Bar chart of requests/second and tokens/second (outlier-trimmed)."""
    labels = [r["condition"] for r in results]
    rps = [r["summary"].get("trimmed_requests_per_second", r["summary"]["requests_per_second"]) for r in results]
    tps = [r["summary"].get("trimmed_tokens_per_second", r["summary"]["tokens_per_second"]) for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, len(labels) * 1.2), 7))

    ax1.bar(range(len(labels)), rps, color="#8E44AD", alpha=0.85)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Requests / second")
    ax1.set_title("Request Throughput", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(range(len(labels)), tps, color="#E67E22", alpha=0.85)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Tokens / second")
    ax2.set_title("Token Throughput", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    apply_context(fig, "Throughput by Network Condition", context_line)
    fig.savefig(os.path.join(output_dir, "throughput_by_condition.png"), dpi=150)
    plt.close(fig)
    print("  throughput_by_condition.png")


def plot_node_rtt_heatmap(results, output_dir, context_line):
    """Heatmap of node-to-node RTT across conditions."""
    all_keys = set()
    for r in results:
        rtts = r.get("cluster", {}).get("node_rtts", {})
        all_keys.update(rtts.keys())

    if not all_keys:
        print("  node_rtt_heatmap.png — skipped (no RTT data)")
        return

    sorted_keys = sorted(all_keys)
    labels = [r["condition"] for r in results]

    matrix = []
    for r in results:
        rtts = r.get("cluster", {}).get("node_rtts", {})
        row = [rtts.get(k, float("nan")) for k in sorted_keys]
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(max(8, len(sorted_keys) * 1.5), max(7, len(labels) * 0.5)))

    valid = [v for row in matrix for v in row if v == v]
    vmin = min(valid) if valid else 0
    vmax = max(valid) if valid else 1

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(sorted_keys)))
    ax.set_xticklabels(sorted_keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Node Pair")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("RTT (ms)")

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val == val:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if val > (vmax - vmin) * 0.6 + vmin else "black")

    apply_context(fig, "Node-to-Node RTT (ms) by Network Condition", context_line)
    fig.savefig(os.path.join(output_dir, "node_rtt_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  node_rtt_heatmap.png")


def plot_summary_table(results, output_dir, context_line):
    """Summary table as a plot for quick reference."""
    headers = ["Condition", "Median E2E", "p90 E2E", "Median TTFT", "Req/s*", "Tok/s*", "Err"]
    rows = []
    for r in results:
        s = r["summary"]
        rows.append([
            r["condition"],
            f"{s['p50_e2e_ms']:.0f} ms",
            f"{s.get('p90_e2e_ms', s['p99_e2e_ms']):.0f} ms",
            f"{s.get('p50_ttft_ms', s['mean_ttft_ms']):.0f} ms",
            f"{s.get('trimmed_requests_per_second', s['requests_per_second']):.2f}",
            f"{s.get('trimmed_tokens_per_second', s['tokens_per_second']):.1f}",
            str(s["error_count"]),
        ])

    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.4 + 2)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#34495E")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    for i in range(len(rows)):
        color = "#ECF0F1" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(color)

    fig.text(
        0.5, 0.06, "* Throughput excludes requests above p90 E2E to remove backend heartbeat outliers.",
        ha="center", va="bottom", fontsize=7, color="#999999",
    )
    apply_context(fig, "Benchmark Summary", context_line)
    fig.savefig(os.path.join(output_dir, "summary_table.png"), dpi=150)
    plt.close(fig)
    print("  summary_table.png")


def main():
    parser = argparse.ArgumentParser(description="Plot Parallax nettest benchmark results")
    parser.add_argument("--input", required=True, help="Path to benchmark-results.json")
    parser.add_argument("--output", required=True, help="Directory for output plots")
    args = parser.parse_args()

    results = load_results(args.input)
    if not results:
        print("No benchmark results found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    context = build_context_lines(results)
    print(f"Context L1: {context[0]}")
    if context[1]:
        print(f"Context L2: {context[1]}")
    print(f"Generating plots from {len(results)} conditions...")

    plot_latency_by_condition(results, args.output, context)
    plot_latency_distribution(results, args.output, context)
    plot_ttft_by_condition(results, args.output, context)
    plot_throughput_by_condition(results, args.output, context)
    plot_node_rtt_heatmap(results, args.output, context)
    plot_summary_table(results, args.output, context)
    print(f"Done — {len(os.listdir(args.output))} plots in {args.output}")


if __name__ == "__main__":
    main()
