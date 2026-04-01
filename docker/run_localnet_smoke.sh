#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.localnet.yml"
KEEP_RUNNING="${PARALLAX_LOCALNET_KEEP_RUNNING:-0}"
PARALLAX_LOCALNET_INIT_NODES="${PARALLAX_LOCALNET_INIT_NODES:-3}"
MODEL_PATH="/parallax/docker/test-models/parallax-smoke"
SCHEDULER_LOG="/tmp/parallax-run.log"
CHAT_LOG="/tmp/parallax-chat.log"

source "$SCRIPT_DIR/lib_localnet.sh"

trap 'on_exit $?' EXIT

boot_full_cluster
split_proof="$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"
echo "$split_proof"

chat_response="$(compose exec -T host python -c "import json,time,urllib.request,sys
payload = json.dumps({
    'model': '$MODEL_PATH',
    'messages': [{'role': 'user', 'content': 'ping from localnet smoke'}],
    'stream': False,
}).encode()
request = urllib.request.Request(
    'http://127.0.0.1:3200/v1/chat/completions',
    data=payload,
    headers={'Content-Type': 'application/json'},
)
last = None
for _ in range(20):
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            last = response.read().decode()
        data = json.loads(last)
        if 'choices' in data:
            print(last)
            sys.exit(0)
    except Exception as exc:
        last = json.dumps({'error': str(exc)})
    time.sleep(1)
print(last)
sys.exit(1)")"

python3 - <<'PY' "$chat_response"
import json
import sys

payload = json.loads(sys.argv[1])
content = payload["choices"][0]["message"]["content"]
if "[parallax-test-mode:" not in content:
    raise SystemExit("unexpected chat payload: " + content)
print(json.dumps(payload))
PY

final_status="$(compose exec -T host curl -sS http://127.0.0.1:3301/cluster/status_json)"
echo "$final_status"
echo "parallax localnet smoke passed"
