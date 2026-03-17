#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.localnet.yml"
KEEP_RUNNING="${PARALLAX_LOCALNET_KEEP_RUNNING:-0}"
RESET_VOLUMES="${PARALLAX_LOCALNET_RESET_VOLUMES:-0}"
MODEL_NAME="${PARALLAX_LOCALNET_REAL_MODEL:-sshleifer/tiny-gpt2}"
SCHEDULER_LOG="/tmp/parallax-run.log"
CHAT_LOG="/tmp/parallax-chat.log"
ARTIFACT_DIR="${PARALLAX_LOCALNET_ARTIFACT_DIR:-/tmp/parallax-localnet-chaos-real-model}"

compose() {
  docker compose -f "$COMPOSE_FILE" "$@"
}

cleanup() {
  if [[ "$KEEP_RUNNING" != "1" ]]; then
    if [[ "$RESET_VOLUMES" == "1" ]]; then
      compose down -v >/dev/null 2>&1 || true
    else
      compose down >/dev/null 2>&1 || true
    fi
  fi
}

dump_artifacts() {
  mkdir -p "$ARTIFACT_DIR"
  compose ps >"$ARTIFACT_DIR/compose-ps.txt" 2>&1 || true
  compose logs host >"$ARTIFACT_DIR/host.log" 2>&1 || true
  compose logs worker1 >"$ARTIFACT_DIR/worker1.log" 2>&1 || true
  compose logs worker2 >"$ARTIFACT_DIR/worker2.log" 2>&1 || true
  compose exec -T host sh -lc "cat $SCHEDULER_LOG" >"$ARTIFACT_DIR/scheduler.log" 2>&1 || true
  compose exec -T host sh -lc "cat $CHAT_LOG" >"$ARTIFACT_DIR/chat.log" 2>&1 || true
  compose exec -T worker1 sh -lc "cat /tmp/parallax-join.log" >"$ARTIFACT_DIR/worker1-join.log" 2>&1 || true
  compose exec -T worker2 sh -lc "cat /tmp/parallax-join.log" >"$ARTIFACT_DIR/worker2-join.log" 2>&1 || true
  compose exec -T host curl -sS http://127.0.0.1:3301/cluster/status_json >"$ARTIFACT_DIR/cluster-status.json" 2>&1 || true
}

on_exit() {
  local exit_code="$1"
  if [[ "$exit_code" -ne 0 ]]; then
    dump_artifacts
  fi
  cleanup
  exit "$exit_code"
}

trap 'on_exit $?' EXIT

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
  local attempts="${2:-90}"
  compose exec -T host python -c "import json,sys,time,urllib.request; url='http://127.0.0.1:3301/cluster/status_json'
data = {}
for _ in range($attempts):
    data=json.load(urllib.request.urlopen(url, timeout=3)).get('data', {})
    if $condition:
        print(json.dumps(data))
        sys.exit(0)
    time.sleep(1)
print(json.dumps(data))
sys.exit(1)"
}

start_worker_join() {
  local service="$1"
  local port="$2"
  local tcp_port="$3"
  local udp_port="$4"
  local scheduler_addr="$5"
  compose exec -T "$service" sh -lc "PARALLAX_TEST_RUNTIME=transformers nohup parallax join -u --scheduler-addr $scheduler_addr --host 0.0.0.0 --port $port --tcp-port $tcp_port --udp-port $udp_port >/tmp/parallax-join.log 2>&1 &"
}

request_real_model_chat() {
  compose exec -T host python -c "import json,time,urllib.request,sys
payload = json.dumps({
    'model': '$MODEL_NAME',
    'messages': [{'role': 'user', 'content': 'Say the word recover exactly once.'}],
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
        with urllib.request.urlopen(request, timeout=20) as response:
            last = response.read().decode()
        data = json.loads(last)
        if 'choices' in data and data['choices'][0]['message']['content'].strip():
            print(last)
            sys.exit(0)
    except Exception as exc:
        last = json.dumps({'error': str(exc)})
    time.sleep(2)
print(last)
sys.exit(1)"
}

if [[ "$RESET_VOLUMES" == "1" ]]; then
  echo "resetting localnet volumes before the real-model chaos lane"
  compose down -v >/dev/null 2>&1 || true
else
  echo "preserving Docker volumes and Hugging Face caches; set PARALLAX_LOCALNET_RESET_VOLUMES=1 to clear them"
  compose down >/dev/null 2>&1 || true
fi

echo "real-model chaos lane uses a tiny CPU model and can still take time on a cold cache or slow network"
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

start_worker_join worker1 3101 4101 5101 "$scheduler_addr"
start_worker_join worker2 3102 4102 5102 "$scheduler_addr"

wait_status "len(data.get('node_list', [])) >= 2" 60 >/dev/null

compose exec -T host curl -sS -X POST http://127.0.0.1:3301/scheduler/init \
  -H 'Content-Type: application/json' \
  -d "{\"model_name\":\"$MODEL_NAME\",\"init_nodes_num\":2,\"is_local_network\":true}" >/dev/null

wait_status "data.get('initialized') is True" 120 >/dev/null
wait_status "data.get('status') == 'available' and data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) >= 1" 120 >/dev/null

compose exec -T host sh -lc "nohup parallax chat --scheduler-addr $scheduler_addr --host 0.0.0.0 --node-chat-port 3200 --tcp-port 4200 --udp-port 5200 >$CHAT_LOG 2>&1 &"
wait_url host "http://127.0.0.1:3200/cluster/status" 30

baseline_response="$(request_real_model_chat)"
python3 - <<'PY' "$baseline_response" "$MODEL_NAME"
import json
import sys

payload = json.loads(sys.argv[1])
model_name = sys.argv[2]
if payload.get("model") != model_name:
    raise SystemExit(f"unexpected model name: {payload.get('model')}")
content = payload["choices"][0]["message"]["content"].strip()
if not content or "[parallax-test-mode:" in content:
    raise SystemExit("unexpected baseline real-model payload: " + content)
print(json.dumps(payload))
PY

compose stop worker1 >/dev/null
wait_status "len(data.get('node_list', [])) < 2 or data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) < 1" 120 >/dev/null

compose start worker1 >/dev/null
sleep 2
start_worker_join worker1 3101 4101 5101 "$scheduler_addr"
wait_status "len(data.get('node_list', [])) >= 2" 120 >/dev/null
wait_status "data.get('status') == 'available' and data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) >= 1" 120 >/dev/null

compose stop worker2 >/dev/null
wait_status "len(data.get('node_list', [])) < 2 or data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) < 1" 120 >/dev/null

compose start worker2 >/dev/null
sleep 2
start_worker_join worker2 3102 4102 5102 "$scheduler_addr"
wait_status "len(data.get('node_list', [])) >= 2" 120 >/dev/null
wait_status "data.get('status') == 'available' and data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) >= 1" 120 >/dev/null

recovered_response="$(request_real_model_chat)"
python3 - <<'PY' "$recovered_response" "$MODEL_NAME"
import json
import sys

payload = json.loads(sys.argv[1])
model_name = sys.argv[2]
if payload.get("model") != model_name:
    raise SystemExit(f"unexpected model name: {payload.get('model')}")
content = payload["choices"][0]["message"]["content"].strip()
if not content or "[parallax-test-mode:" in content:
    raise SystemExit("unexpected recovery real-model payload: " + content)
print(json.dumps(payload))
PY

compose exec -T host curl -sS http://127.0.0.1:3301/cluster/status_json
echo "parallax localnet real-model chaos lane passed"
