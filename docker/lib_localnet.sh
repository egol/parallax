#!/usr/bin/env bash
# Shared helpers for Parallax localnet test scripts.
#
# Callers must set these variables before sourcing:
#   COMPOSE_FILE     — path to docker-compose.localnet.yml
#   SCHEDULER_LOG    — in-container path for scheduler log
#   CHAT_LOG         — in-container path for chat proxy log
#   MODEL_PATH       — model path or name for synthetic/real tests
#
# Optional:
#   KEEP_RUNNING     — set to "1" to skip teardown on exit
#   ARTIFACT_DIR     — directory for failure artifacts

compose() {
  docker compose -f "$COMPOSE_FILE" "$@"
}

cleanup() {
  if [[ "${KEEP_RUNNING:-0}" != "1" ]]; then
    compose down -v >/dev/null 2>&1 || true
  fi
}

dump_artifacts() {
  local dir="${ARTIFACT_DIR:?ARTIFACT_DIR must be set}"
  mkdir -p "$dir"
  compose ps >"$dir/compose-ps.txt" 2>&1 || true
  compose logs host >"$dir/host.log" 2>&1 || true
  compose logs worker1 >"$dir/worker1.log" 2>&1 || true
  compose logs worker2 >"$dir/worker2.log" 2>&1 || true
  compose exec -T host sh -lc "cat $SCHEDULER_LOG" >"$dir/scheduler.log" 2>&1 || true
  compose exec -T host sh -lc "cat $CHAT_LOG" >"$dir/chat.log" 2>&1 || true
  compose exec -T worker1 sh -lc "cat /tmp/parallax-join.log" >"$dir/worker1-join.log" 2>&1 || true
  compose exec -T worker2 sh -lc "cat /tmp/parallax-join.log" >"$dir/worker2-join.log" 2>&1 || true
  compose exec -T host curl -sS http://127.0.0.1:3301/cluster/status_json >"$dir/cluster-status.json" 2>&1 || true
}

on_exit() {
  local exit_code="$1"
  if [[ "$exit_code" -ne 0 ]] && [[ -n "${ARTIFACT_DIR:-}" ]]; then
    dump_artifacts
  fi
  cleanup
  exit "$exit_code"
}

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
  local attempts="${2:-60}"
  compose exec -T host python -c "import json,sys,time,urllib.request; url='http://127.0.0.1:3301/cluster/status_json'
data = {}
for _ in range($attempts):
    data=json.load(urllib.request.urlopen(url, timeout=2)).get('data', {})
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
  local env_prefix="${WORKER_ENV:-}"
  compose exec -T "$service" sh -lc "${env_prefix} nohup parallax join -u --scheduler-addr $scheduler_addr --host 0.0.0.0 --port $port --tcp-port $tcp_port --udp-port $udp_port >/tmp/parallax-join.log 2>&1 &"
}

request_synthetic_chat() {
  compose exec -T host python -c "import json,time,urllib.request,sys
payload = json.dumps({
    'model': '$MODEL_PATH',
    'messages': [{'role': 'user', 'content': 'ping from nettest'}],
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
sys.exit(1)"
}

validate_synthetic_response() {
  local response="$1"
  local label="${2:-synthetic}"
  python3 - <<PY "$response"
import json
import sys

payload = json.loads(sys.argv[1])
content = payload["choices"][0]["message"]["content"]
if "[parallax-test-mode:" not in content:
    raise SystemExit("unexpected $label payload: " + content)
print(json.dumps(payload))
PY
}

# Composite: start scheduler, bootstrap, capture peer ID, set scheduler_addr.
# Sets global variables: scheduler_peer_id, scheduler_addr
start_scheduler() {
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
}

start_chat_proxy() {
  compose exec -T host sh -lc "nohup parallax chat --scheduler-addr $scheduler_addr --host 0.0.0.0 --node-chat-port 3200 --tcp-port 4200 --udp-port 5200 >$CHAT_LOG 2>&1 &"
  wait_url host "http://127.0.0.1:3200/cluster/status" 30
}

init_model() {
  local init_nodes="${1:-2}"
  compose exec -T host curl -sS -X POST http://127.0.0.1:3301/scheduler/init \
    -H 'Content-Type: application/json' \
    -d "{\"model_name\":\"$MODEL_PATH\",\"init_nodes_num\":$init_nodes,\"is_local_network\":true}" >/dev/null

  wait_status "data.get('initialized') is True" 60 >/dev/null
  wait_status "data.get('status') == 'available' and data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) >= 1" 60 >/dev/null
}

# Full cluster bootstrap: scheduler + 2 workers + model init + chat proxy
boot_full_cluster() {
  compose down -v >/dev/null 2>&1 || true
  compose up -d --build

  start_scheduler
  start_worker_join worker1 3101 4101 5101 "$scheduler_addr"
  start_worker_join worker2 3102 4102 5102 "$scheduler_addr"
  wait_status "len(data.get('node_list', [])) >= 2" 60 >/dev/null

  init_model 2
  start_chat_proxy
}
