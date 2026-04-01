#!/usr/bin/env bash
# Combined network conditions + chaos smoke test for Parallax localnet.
#
# Applies realistic network degradation (latency + packet loss) and then
# performs a container-level chaos test to verify recovery under degraded
# network conditions.  This is the most production-like scenario.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.localnet.yml"
KEEP_RUNNING="${PARALLAX_LOCALNET_KEEP_RUNNING:-0}"
MODEL_PATH="/parallax/docker/test-models/parallax-smoke"
SCHEDULER_LOG="/tmp/parallax-run.log"
CHAT_LOG="/tmp/parallax-chat.log"
ARTIFACT_DIR="${PARALLAX_LOCALNET_ARTIFACT_DIR:-/tmp/parallax-nettest-combined-chaos}"

source "$SCRIPT_DIR/lib_localnet.sh"
source "$SCRIPT_DIR/lib_nettest.sh"

trap 'on_exit $?' EXIT

echo "=== Parallax nettest: combined chaos ==="

boot_full_cluster
echo "$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"

# Baseline chat
echo "--- baseline chat ---"
baseline="$(request_synthetic_chat)"
validate_synthetic_response "$baseline" "baseline"

# Inject network degradation on both workers
echo "--- injecting network conditions ---"
inject_conditions worker1 100 2
inject_conditions worker2 100 2
inject_conditions worker3 150 3

# Chat under degraded conditions
echo "--- chat under degraded conditions ---"
degraded="$(request_synthetic_chat)"
validate_synthetic_response "$degraded" "degraded-conditions"

# Chaos: stop worker1 while conditions are active
echo "--- stopping worker1 (chaos) ---"
compose stop worker1 >/dev/null
wait_status "data.get('topology', {}).get('totals', {}).get('registered_workers', 0) < 3" 90 >/dev/null

# Restart worker1 — network conditions are lost on container restart,
# so re-inject them after start
echo "--- restarting worker1 ---"
compose start worker1 >/dev/null
sleep 2
inject_conditions worker1 100 2
start_worker_join worker1 3101 4101 5101 "$scheduler_addr"

# Wait for recovery under degraded conditions (extended timeouts)
echo "--- waiting for recovery under degraded conditions ---"
wait_status "data.get('topology', {}).get('totals', {}).get('registered_workers', 0) >= 3" 120 >/dev/null
wait_status "data.get('status') == 'available' and data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) >= 1" 120 >/dev/null
echo "$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"

# Chat after recovery (still under degraded conditions)
echo "--- chat after recovery (still degraded) ---"
recovered_degraded="$(request_synthetic_chat)"
validate_synthetic_response "$recovered_degraded" "recovered-degraded"

# Clear all conditions
echo "--- clearing all network conditions ---"
clear_all_network_conditions

# Final clean chat
echo "--- final clean chat ---"
final="$(request_synthetic_chat)"
validate_synthetic_response "$final" "final-clean"

compose exec -T host curl -sS http://127.0.0.1:3301/cluster/status_json
echo "parallax nettest combined chaos passed"
