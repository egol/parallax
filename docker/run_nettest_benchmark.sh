#!/usr/bin/env bash
# Full benchmark runner for Parallax localnet network conditions.
#
# Uses a real (tiny) model so benchmark results reflect actual inference
# latency across the distributed pipeline, not just P2P overhead.
#
# Usage:
#   ./parallax/docker/run_nettest_benchmark.sh
#
# Environment:
#   PARALLAX_NETTEST_MODEL        — model name (default: sshleifer/tiny-gpt2)
#   PARALLAX_NETTEST_NUM_REQUESTS — requests per condition (default: 10)
#   PARALLAX_LOCALNET_KEEP_RUNNING — set to 1 to keep containers after test
#   PARALLAX_LOCALNET_ARTIFACT_DIR — override output directory
#   PARALLAX_LOCALNET_RESET_VOLUMES — set to 1 to clear HF cache
#
# Output:
#   $ARTIFACT_DIR/benchmark-results.json  — raw benchmark data
#   $ARTIFACT_DIR/plots/                  — matplotlib plots (if available)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.localnet.yml"
KEEP_RUNNING="${PARALLAX_LOCALNET_KEEP_RUNNING:-0}"
RESET_VOLUMES="${PARALLAX_LOCALNET_RESET_VOLUMES:-0}"
PARALLAX_LOCALNET_INIT_NODES="${PARALLAX_LOCALNET_INIT_NODES:-3}"
MODEL_PATH="${PARALLAX_NETTEST_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SCHEDULER_LOG="/tmp/parallax-run.log"
CHAT_LOG="/tmp/parallax-chat.log"
ARTIFACT_DIR="${PARALLAX_LOCALNET_ARTIFACT_DIR:-/tmp/parallax-nettest-benchmark}"
NUM_REQUESTS="${PARALLAX_NETTEST_NUM_REQUESTS:-10}"
PROMPT="${PARALLAX_NETTEST_PROMPT:-What color is the sky?}"
PARALLAX_LOCALNET_READY_TIMEOUT="${PARALLAX_LOCALNET_READY_TIMEOUT:-300}"

export PARALLAX_LOCALNET_TEST_MODE="${PARALLAX_LOCALNET_TEST_MODE:-0}"
# Workers use the transformers runtime for real (CPU) inference
export WORKER_ENV="PARALLAX_TEST_RUNTIME=transformers PARALLAX_TEST_OVERRIDE_MEMORY_GB=${PARALLAX_TEST_OVERRIDE_MEMORY_GB:-0.7}"
WORKER_JOIN_ARGS="${WORKER_JOIN_ARGS:---kv-cache-memory-fraction 0.45 --max-total-tokens 4096 --max-sequence-length 1024 --max-batch-size 2 --max-num-tokens-per-batch 512}"

source "$SCRIPT_DIR/lib_localnet.sh"
source "$SCRIPT_DIR/lib_nettest.sh"

# Override cleanup to optionally preserve volumes (for HF cache)
cleanup() {
  if [[ "$KEEP_RUNNING" != "1" ]]; then
    if [[ "$RESET_VOLUMES" == "1" ]]; then
      compose down -v >/dev/null 2>&1 || true
    else
      compose down >/dev/null 2>&1 || true
    fi
  fi
}

trap 'on_exit $?' EXIT

mkdir -p "$ARTIFACT_DIR"

benchmark_condition() {
  local label="$1"
  echo "--- benchmarking: $label ($NUM_REQUESTS requests) ---"
  compose exec -T host python /parallax/docker/bench_collect.py \
    --chat-url http://127.0.0.1:3200/v1/chat/completions \
    --scheduler-url http://127.0.0.1:3301/cluster/status_json \
    --model "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --num-requests "$NUM_REQUESTS" \
    --max-tokens 12 \
    --request-timeout 60 \
    --warmup 3 \
    --no-stream \
    --label "$label" \
    >> "$ARTIFACT_DIR/benchmark-results.jsonl"
}

echo "=== Parallax nettest: benchmark (real model: $MODEL_PATH) ==="

# Boot cluster — uses real model via WORKER_ENV
boot_full_cluster
echo "$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"

# Verify real model produces actual text
echo "--- verifying real model output ---"
chat_response="$(compose exec -T host python -c "import json,time,urllib.request,sys
payload = json.dumps({
    'model': '$MODEL_PATH',
    'messages': [{'role': 'user', 'content': 'Say hello.'}],
    'stream': False,
    'max_tokens': 12,
}).encode()
request = urllib.request.Request(
    'http://127.0.0.1:3200/v1/chat/completions',
    data=payload,
    headers={'Content-Type': 'application/json'},
)
last = None
for _ in range(30):
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            last = response.read().decode()
        data = json.loads(last)
        if 'choices' in data and data['choices'][0]['message']['content'].strip():
            print(last)
            sys.exit(0)
    except Exception as exc:
        last = json.dumps({'error': str(exc)})
    time.sleep(2)
print(last)
sys.exit(1)")"

python3 - <<'PY' "$chat_response" "$MODEL_PATH"
import json
import sys

payload = json.loads(sys.argv[1])
model_name = sys.argv[2]
content = payload["choices"][0]["message"]["content"].strip()
if not content:
    raise SystemExit("real-model response was empty")
if "[parallax-test-mode:" in content:
    raise SystemExit("benchmark unexpectedly fell back to synthetic runtime")
print(f"model response: {content[:100]}")
PY

# Clear any prior results
rm -f "$ARTIFACT_DIR/benchmark-results.jsonl"

# 1. Baseline
benchmark_condition "baseline"

# 2. Latency sweep
for delay in 50 100 200 500; do
  for worker in "${WORKER_SERVICES[@]}"; do
    inject_latency "$worker" "$delay"
  done
  benchmark_condition "latency-${delay}ms"
  for worker in "${WORKER_SERVICES[@]}"; do
    clear_tc "$worker"
  done
done

# 3. Packet loss sweep
for loss in 1 5 10; do
  for worker in "${WORKER_SERVICES[@]}"; do
    inject_packet_loss "$worker" "$loss"
  done
  benchmark_condition "loss-${loss}pct"
  for worker in "${WORKER_SERVICES[@]}"; do
    clear_tc "$worker"
  done
done

# 4. Bandwidth sweep
for bw in 1024 512 256 128; do
  for worker in "${WORKER_SERVICES[@]}"; do
    limit_bandwidth "$worker" "$bw"
  done
  benchmark_condition "bandwidth-${bw}kbit"
  for worker in "${WORKER_SERVICES[@]}"; do
    clear_tc "$worker"
  done
done

# 5. Combined conditions
for worker in "${WORKER_SERVICES[@]}"; do
  inject_conditions "$worker" 100 2
done
benchmark_condition "combined-100ms-2pct"
clear_all_network_conditions

# Convert JSONL to JSON array
python3 -c "
import json, sys
results = []
with open('$ARTIFACT_DIR/benchmark-results.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))
with open('$ARTIFACT_DIR/benchmark-results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Collected {len(results)} benchmark conditions')
"

# Generate plots if matplotlib is available
echo "--- generating plots ---"
if python3 -c "import matplotlib" 2>/dev/null; then
  python3 "$SCRIPT_DIR/plot_nettest.py" \
    --input "$ARTIFACT_DIR/benchmark-results.json" \
    --output "$ARTIFACT_DIR/plots/"
  echo "plots saved to $ARTIFACT_DIR/plots/"
else
  echo "matplotlib not available on host — skipping plot generation"
  echo "install matplotlib and run manually:"
  echo "  python3 $SCRIPT_DIR/plot_nettest.py \\"
  echo "    --input $ARTIFACT_DIR/benchmark-results.json \\"
  echo "    --output $ARTIFACT_DIR/plots/"
fi

compose exec -T host curl -sS http://127.0.0.1:3301/cluster/status_json >"$ARTIFACT_DIR/final-cluster-status.json"
echo "parallax nettest benchmark passed (model: $MODEL_PATH) — results in $ARTIFACT_DIR/"
