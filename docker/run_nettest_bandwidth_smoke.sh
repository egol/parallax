#!/usr/bin/env bash
# Bandwidth constraint smoke test for Parallax localnet.
#
# Verifies that the Parallax P2P layer handles slow links without
# protocol timeouts breaking cluster formation or chat completions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.localnet.yml"
KEEP_RUNNING="${PARALLAX_LOCALNET_KEEP_RUNNING:-0}"
MODEL_PATH="/parallax/docker/test-models/parallax-smoke"
SCHEDULER_LOG="/tmp/parallax-run.log"
CHAT_LOG="/tmp/parallax-chat.log"
ARTIFACT_DIR="${PARALLAX_LOCALNET_ARTIFACT_DIR:-/tmp/parallax-nettest-bandwidth-smoke}"

source "$SCRIPT_DIR/lib_localnet.sh"
source "$SCRIPT_DIR/lib_nettest.sh"

trap 'on_exit $?' EXIT

echo "=== Parallax nettest: bandwidth smoke ==="

boot_full_cluster
echo "$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"

# Baseline chat
echo "--- baseline chat ---"
baseline="$(request_synthetic_chat)"
validate_synthetic_response "$baseline" "baseline"

# Limit bandwidth on workers
echo "--- limiting bandwidth ---"
limit_bandwidth worker1 256
limit_bandwidth worker2 256
limit_bandwidth worker3 256

# Chat under bandwidth constraint (synthetic payloads are small)
echo "--- chat under bandwidth constraint ---"
bw_response="$(request_synthetic_chat)"
validate_synthetic_response "$bw_response" "bandwidth-limited"

# Clear and verify
echo "--- clearing bandwidth limits ---"
clear_tc worker1
clear_tc worker2
clear_tc worker3

echo "--- post-recovery chat ---"
recovered="$(request_synthetic_chat)"
validate_synthetic_response "$recovered" "post-bandwidth-recovery"

compose exec -T host curl -sS http://127.0.0.1:3301/cluster/status_json
echo "parallax nettest bandwidth smoke passed"
