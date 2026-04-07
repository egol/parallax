#!/usr/bin/env bash
# Centralized split-proof smoke test for Parallax localnet.
#
# Proves that a 3-stage pipeline can serve end to end in centralized mode even
# when workers cannot reach each other directly and all stage hops proxy
# through the host scheduler.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.localnet.yml"
KEEP_RUNNING="${PARALLAX_LOCALNET_KEEP_RUNNING:-0}"
MODEL_PATH="/parallax/docker/test-models/parallax-smoke"
SCHEDULER_LOG="/tmp/parallax-run.log"
CHAT_LOG="/tmp/parallax-chat.log"
ARTIFACT_DIR="${PARALLAX_LOCALNET_ARTIFACT_DIR:-/tmp/parallax-nettest-centralized-split-smoke}"

source "$SCRIPT_DIR/lib_localnet.sh"
source "$SCRIPT_DIR/lib_nettest.sh"

trap 'on_exit $?' EXIT

echo "=== Parallax nettest: centralized split smoke ==="

teardown_before_boot
compose up -d --build

start_scheduler

echo "--- install worker mesh partition ---"
block_worker_mesh

for worker in "${WORKER_SERVICES[@]}"; do
  assert_reachable "$worker" host 4301
done

start_all_workers "$scheduler_addr"
wait_for_worker_count "$PARALLAX_LOCALNET_WORKER_COUNT" 120

init_model "$PARALLAX_LOCALNET_INIT_NODES"
start_chat_proxy

split_topology="$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"
echo "$split_topology"

echo "--- verify direct worker mesh remains blocked ---"
assert_unreachable worker1 worker2 4102
assert_unreachable worker1 worker3 4103
assert_unreachable worker2 worker1 4101
assert_unreachable worker2 worker3 4103
assert_unreachable worker3 worker1 4101
assert_unreachable worker3 worker2 4102

for worker in "${WORKER_SERVICES[@]}"; do
  assert_reachable "$worker" host 4301
done

echo "--- final centralized chat ---"
response="$(request_synthetic_chat)"
validate_synthetic_response "$response" "centralized-split"

compose exec -T host curl -sS http://127.0.0.1:3301/cluster/status_json
echo "parallax nettest centralized split smoke passed"
