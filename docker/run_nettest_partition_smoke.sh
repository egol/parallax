#!/usr/bin/env bash
# Network partition smoke test for Parallax localnet.
#
# Tests partition detection, recovery, and iptables-based selective blocking.
#
# Phase 1: Container-level partition (compose stop) for fast detection
# Phase 2: iptables-based selective block to verify network simulation tools
#
# Note: iptables DROP causes "silent" failure that relies on libp2p keepalive
# timeouts (can take 5+ minutes). Container stop provides immediate detection.
# This test uses container stop for the main assertion and iptables for
# validating the network simulation tooling.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.localnet.yml"
KEEP_RUNNING="${PARALLAX_LOCALNET_KEEP_RUNNING:-0}"
MODEL_PATH="/parallax/docker/test-models/parallax-smoke"
SCHEDULER_LOG="/tmp/parallax-run.log"
CHAT_LOG="/tmp/parallax-chat.log"
ARTIFACT_DIR="${PARALLAX_LOCALNET_ARTIFACT_DIR:-/tmp/parallax-nettest-partition-smoke}"

source "$SCRIPT_DIR/lib_localnet.sh"
source "$SCRIPT_DIR/lib_nettest.sh"

trap 'on_exit $?' EXIT

echo "=== Parallax nettest: partition smoke ==="

boot_full_cluster
split_topology="$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"
echo "$split_topology"

# Baseline chat
echo "--- baseline chat ---"
baseline="$(request_synthetic_chat)"
validate_synthetic_response "$baseline" "baseline"

# Phase 1: Container-level partition (fast detection)
echo "--- phase 1: container stop partition ---"
compose stop worker1 >/dev/null
# Check registered_workers count, not node_list (stale entries linger)
wait_status "data.get('topology', {}).get('totals', {}).get('registered_workers', 0) < 3" 90 >/dev/null
echo "--- scheduler detected worker1 loss ---"

# Restart and recover
compose start worker1 >/dev/null
sleep 2
start_worker_join worker1 3101 4101 5101 "$scheduler_addr"
wait_status "data.get('topology', {}).get('totals', {}).get('registered_workers', 0) >= 3" 120 >/dev/null
wait_status "data.get('status') == 'available' and data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) >= 1" 120 >/dev/null
split_topology="$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"
echo "$split_topology"

echo "--- phase 1 recovery verified ---"
recovered="$(request_synthetic_chat)"
validate_synthetic_response "$recovered" "phase1-recovery"

# Phase 2: iptables selective block (validate network sim tools)
echo "--- phase 2: iptables selective block ---"
block_traffic worker1 host
assert_unreachable worker1 host 3301

# Verify worker2 is unaffected
assert_reachable worker2 host 3301
assert_reachable worker3 host 3301

# Unblock and verify connectivity restored
unblock_traffic worker1 host
sleep 1
assert_reachable worker1 host 3301

echo "--- phase 2 iptables tools verified ---"

# Final chat to confirm everything still works
echo "--- final chat ---"
final="$(request_synthetic_chat)"
validate_synthetic_response "$final" "post-partition-final"

compose exec -T host curl -sS http://127.0.0.1:3301/cluster/status_json
echo "parallax nettest partition smoke passed"
