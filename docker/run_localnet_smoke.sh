#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.localnet.yml"
KEEP_RUNNING="${PARALLAX_LOCALNET_KEEP_RUNNING:-0}"
MODEL_PATH="/parallax/docker/test-models/parallax-smoke"
SCHEDULER_LOG="/tmp/parallax-run.log"
CHAT_LOG="/tmp/parallax-chat.log"

compose() {
  docker compose -f "$COMPOSE_FILE" "$@"
}

cleanup() {
  if [[ "$KEEP_RUNNING" != "1" ]]; then
    compose down -v >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

wait_url() {
  local service="$1"
  local url="$2"
  local attempts="${3:-30}"
  compose exec -T "$service" python -c "import sys,time,urllib.request; url='$url'
for _ in range($attempts):
    try:
        urllib.request.urlopen(url, timeout=2)
        sys.exit(0)
    except Exception:
        time.sleep(1)
sys.exit(1)"
}

wait_status() {
  local condition="$1"
  local attempts="${2:-45}"
  compose exec -T host python -c "import json,sys,time,urllib.request; url='http://127.0.0.1:3301/cluster/status_json'
for _ in range($attempts):
    data=json.load(urllib.request.urlopen(url, timeout=2))['data']
    if $condition:
        print(json.dumps(data))
        sys.exit(0)
    time.sleep(1)
print(json.dumps(data))
sys.exit(1)"
}

compose down -v >/dev/null 2>&1 || true
compose up -d --build

compose exec -T host sh -lc "nohup parallax run -u --port 3301 --host 0.0.0.0 --tcp-port 4301 --udp-port 5301 --announce-maddrs /dns4/host/tcp/4301 /dns4/host/udp/5301/quic-v1 >$SCHEDULER_LOG 2>&1 &"
wait_url host "http://127.0.0.1:3301/cluster/status_json" 30

compose exec -T host curl -sS -X POST http://127.0.0.1:3301/scheduler/bootstrap \
  -H 'Content-Type: application/json' \
  -d '{"is_local_network":true}' >/dev/null

scheduler_peer_id="$(compose exec -T host sh -lc "grep -m1 'Stored scheduler peer id:' $SCHEDULER_LOG | awk '{print \$NF}'")"
if [[ -z "$scheduler_peer_id" ]]; then
  echo "failed to resolve scheduler peer id" >&2
  exit 1
fi
scheduler_addr="/dns4/host/tcp/4301/p2p/$scheduler_peer_id"

compose exec -T worker1 sh -lc "nohup parallax join -u --scheduler-addr $scheduler_addr --host 0.0.0.0 --port 3101 --tcp-port 4101 --udp-port 5101 >/tmp/parallax-join.log 2>&1 &"
compose exec -T worker2 sh -lc "nohup parallax join -u --scheduler-addr $scheduler_addr --host 0.0.0.0 --port 3102 --tcp-port 4102 --udp-port 5102 >/tmp/parallax-join.log 2>&1 &"

wait_status "len(data.get('node_list', [])) >= 2" 45 >/dev/null

compose exec -T host curl -sS -X POST http://127.0.0.1:3301/scheduler/init \
  -H 'Content-Type: application/json' \
  -d "{\"model_name\":\"$MODEL_PATH\",\"init_nodes_num\":2,\"is_local_network\":true}" >/dev/null

wait_status "data.get('initialized') is True" 60 >/dev/null
wait_status "data.get('status') == 'available' and data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) >= 1" 60 >/dev/null

compose exec -T host sh -lc "nohup parallax chat --scheduler-addr $scheduler_addr --host 0.0.0.0 --node-chat-port 3200 --tcp-port 4200 --udp-port 5200 >$CHAT_LOG 2>&1 &"
wait_url host "http://127.0.0.1:3200/cluster/status" 30

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
