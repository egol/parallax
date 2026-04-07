#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.localnet.yml"
KEEP_RUNNING="${PARALLAX_LOCALNET_KEEP_RUNNING:-0}"
PARALLAX_LOCALNET_INIT_NODES="${PARALLAX_LOCALNET_INIT_NODES:-3}"
MODEL_PATH="/parallax/docker/test-models/parallax-smoke"
SCHEDULER_LOG="/tmp/parallax-run.log"
CHAT_LOG="/tmp/parallax-chat.log"
ARTIFACT_DIR="${PARALLAX_LOCALNET_ARTIFACT_DIR:-/tmp/parallax-localnet-chaos-smoke}"

source "$SCRIPT_DIR/lib_localnet.sh"

trap 'on_exit $?' EXIT

boot_full_cluster
split_topology="$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"
echo "$split_topology"

baseline_response="$(request_synthetic_chat)"
python3 - <<'PY' "$baseline_response"
import json
import sys

payload = json.loads(sys.argv[1])
content = payload["choices"][0]["message"]["content"]
if "[parallax-test-mode:" not in content:
    raise SystemExit("unexpected synthetic baseline payload: " + content)
print(json.dumps(payload))
PY

compose stop worker1 >/dev/null
wait_status "data.get('topology', {}).get('totals', {}).get('registered_workers', 0) < 3 or data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) < 1" 90 >/dev/null

compose start worker1 >/dev/null
sleep 2
start_worker_join worker1 3101 4101 5101 "$scheduler_addr"
wait_status "data.get('topology', {}).get('totals', {}).get('registered_workers', 0) >= 3" 90 >/dev/null
wait_status "data.get('status') == 'available' and data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) >= 1" 90 >/dev/null
split_topology="$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"
echo "$split_topology"

recovered_response="$(request_synthetic_chat)"
python3 - <<'PY' "$recovered_response"
import json
import sys

payload = json.loads(sys.argv[1])
content = payload["choices"][0]["message"]["content"]
if "[parallax-test-mode:" not in content:
    raise SystemExit("unexpected synthetic recovery payload: " + content)
print(json.dumps(payload))
PY

compose exec -T host curl -sS http://127.0.0.1:3301/cluster/status_json
echo "parallax localnet chaos smoke passed"
