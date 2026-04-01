#!/usr/bin/env bash
# Latency injection smoke test for Parallax localnet.
#
# Verifies that the Parallax P2P layer and scheduler tolerate high-latency
# links without breaking cluster formation or chat completions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.localnet.yml"
KEEP_RUNNING="${PARALLAX_LOCALNET_KEEP_RUNNING:-0}"
MODEL_PATH="/parallax/docker/test-models/parallax-smoke"
SCHEDULER_LOG="/tmp/parallax-run.log"
CHAT_LOG="/tmp/parallax-chat.log"
ARTIFACT_DIR="${PARALLAX_LOCALNET_ARTIFACT_DIR:-/tmp/parallax-nettest-latency-smoke}"

source "$SCRIPT_DIR/lib_localnet.sh"
source "$SCRIPT_DIR/lib_nettest.sh"

trap 'on_exit $?' EXIT

echo "=== Parallax nettest: latency smoke ==="

boot_full_cluster
echo "$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"

# Baseline chat
echo "--- baseline chat ---"
baseline="$(request_synthetic_chat)"
validate_synthetic_response "$baseline" "baseline"

# Inject latency
echo "--- injecting latency ---"
inject_latency worker1 200
inject_latency worker2 350 25
inject_latency worker3 500 50

# Chat under latency (extended timeout via retries in request_synthetic_chat)
echo "--- chat under latency ---"
latency_response="$(request_synthetic_chat)"
validate_synthetic_response "$latency_response" "latency"

# Clear and verify recovery
echo "--- clearing latency ---"
clear_tc worker1
clear_tc worker2
clear_tc worker3

echo "--- post-recovery chat ---"
recovered="$(request_synthetic_chat)"
validate_synthetic_response "$recovered" "post-recovery"

compose exec -T host curl -sS http://127.0.0.1:3301/cluster/status_json
echo "parallax nettest latency smoke passed"
