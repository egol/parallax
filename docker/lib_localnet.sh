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
#   RESET_VOLUMES    — set to "1" to remove volumes during teardown

: "${PARALLAX_LOCALNET_WORKER_COUNT:=3}"
: "${PARALLAX_LOCALNET_INIT_NODES:=${PARALLAX_LOCALNET_WORKER_COUNT}}"
: "${PARALLAX_LOCALNET_READY_TIMEOUT:=60}"

readonly WORKER_SERVICES=(worker1 worker2 worker3)

compose() {
  docker compose -f "$COMPOSE_FILE" "$@"
}

cleanup() {
  if [[ "${KEEP_RUNNING:-0}" != "1" ]]; then
    if [[ "${RESET_VOLUMES:-0}" == "1" ]]; then
      compose down -v >/dev/null 2>&1 || true
    else
      compose down >/dev/null 2>&1 || true
    fi
  fi
}

teardown_before_boot() {
  if [[ "${RESET_VOLUMES:-0}" == "1" ]]; then
    compose down -v >/dev/null 2>&1 || true
  else
    compose down >/dev/null 2>&1 || true
  fi
}

dump_artifacts() {
  local dir="${ARTIFACT_DIR:?ARTIFACT_DIR must be set}"
  mkdir -p "$dir"
  compose ps >"$dir/compose-ps.txt" 2>&1 || true
  compose logs host >"$dir/host.log" 2>&1 || true
  for worker in "${WORKER_SERVICES[@]}"; do
    compose logs "$worker" >"$dir/${worker}.log" 2>&1 || true
  done
  compose exec -T host sh -lc "cat $SCHEDULER_LOG" >"$dir/scheduler.log" 2>&1 || true
  compose exec -T host sh -lc "cat $CHAT_LOG" >"$dir/chat.log" 2>&1 || true
  for worker in "${WORKER_SERVICES[@]}"; do
    compose exec -T "$worker" sh -lc "cat /tmp/parallax-join.log" >"$dir/${worker}-join.log" 2>&1 || true
  done
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
    try:
        data=json.load(urllib.request.urlopen(url, timeout=5)).get('data', {})
        if $condition:
            print(json.dumps(data))
            sys.exit(0)
    except Exception:
        pass
    time.sleep(1)
print(json.dumps(data))
sys.exit(1)"
}

wait_for_worker_count() {
  local expected="${1:-$PARALLAX_LOCALNET_WORKER_COUNT}"
  local attempts="${2:-60}"
  wait_status "data.get('topology', {}).get('totals', {}).get('registered_workers', 0) >= $expected or data.get('topology', {}).get('totals', {}).get('discovered_workers', 0) >= $expected or len(data.get('topology', {}).get('nodes', [])) >= $expected or len(data.get('node_list', [])) >= $expected" "$attempts" >/dev/null
}

start_worker_join() {
  local service="$1"
  local port="$2"
  local tcp_port="$3"
  local udp_port="$4"
  local scheduler_addr="$5"
  local service_key="${service^^}"
  local service_env_var="${service_key}_ENV"
  local service_join_args_var="${service_key}_JOIN_ARGS"
  local env_prefix="${!service_env_var:-${WORKER_ENV:-}}"
  local extra_args="${!service_join_args_var:-${WORKER_JOIN_ARGS:-}}"
  compose exec -T "$service" sh -lc "${env_prefix} nohup parallax join -u --scheduler-addr $scheduler_addr --host 0.0.0.0 --port $port --tcp-port $tcp_port --udp-port $udp_port $extra_args >/tmp/parallax-join.log 2>&1 &"
}

start_all_workers() {
  local scheduler_addr="$1"
  local index=1
  for worker in "${WORKER_SERVICES[@]}"; do
    start_worker_join "$worker" "$((3100 + index))" "$((4100 + index))" "$((5100 + index))" "$scheduler_addr"
    index=$((index + 1))
  done
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

validate_split_topology() {
  local expected_nodes="${1:-$PARALLAX_LOCALNET_INIT_NODES}"
  compose exec -T host python -c "import json,sys,urllib.request
expected = $expected_nodes
data = json.load(urllib.request.urlopen('http://127.0.0.1:3301/cluster/status_json', timeout=5)).get('data', {})
topology = data.get('topology', {})
nodes = topology.get('nodes') or data.get('node_list') or []
pipelines = topology.get('pipelines') or []
totals = topology.get('totals') or {}
if not data.get('initialized'):
    raise SystemExit('cluster is not initialized')
if data.get('init_nodes_num') != expected:
    raise SystemExit(f'expected init_nodes_num={expected}, got {data.get(\"init_nodes_num\")}')
if totals.get('ready_pipelines', 0) < 1:
    raise SystemExit('no ready pipeline is available')
serving_ids = sorted({
    str(node.get('node_id') or '').strip()
    for node in nodes
    if node.get('pipeline_id') is not None and str(node.get('node_id') or '').strip()
})
if len(serving_ids) != expected:
    raise SystemExit(f'expected {expected} serving node ids, got {serving_ids}')
for pipeline in pipelines:
    if not pipeline.get('ready'):
        continue
    pipeline_node_ids = []
    for raw_node_id in pipeline.get('node_ids') or []:
        node_id = str(raw_node_id or '').strip()
        if node_id and node_id not in pipeline_node_ids:
            pipeline_node_ids.append(node_id)
    if len(pipeline_node_ids) != expected:
        continue
    stage_nodes = []
    for node_id in pipeline_node_ids:
        node = next((candidate for candidate in nodes if str(candidate.get('node_id') or '').strip() == node_id), None)
        if node is None:
            stage_nodes = []
            break
        stage_nodes.append(node)
    if len(stage_nodes) != expected:
        continue
    stage_nodes.sort(key=lambda node: node.get('stage_index') if isinstance(node.get('stage_index'), int) else 999)
    expected_stages = list(range(expected))
    if [node.get('stage_index') for node in stage_nodes] != expected_stages:
        continue
    if any(node.get('pipeline_id') != pipeline.get('id') for node in stage_nodes):
        continue
    if any(node.get('start_layer') is None or node.get('end_layer') is None for node in stage_nodes):
        continue
    contiguous = all(stage_nodes[index - 1]['end_layer'] == stage_nodes[index]['start_layer'] for index in range(1, len(stage_nodes)))
    if not contiguous:
        continue
    print(json.dumps({
        'init_nodes_num': data.get('init_nodes_num'),
        'pipeline_id': pipeline.get('id'),
        'ready_pipelines': totals.get('ready_pipelines', 0),
        'serving_node_ids': serving_ids,
        'stages': [
            {
                'node_id': node.get('node_id'),
                'stage_index': node.get('stage_index'),
                'pipeline_id': node.get('pipeline_id'),
                'start_layer': node.get('start_layer'),
                'end_layer': node.get('end_layer'),
                'is_active': node.get('is_active'),
            }
            for node in stage_nodes
        ],
    }))
    sys.exit(0)
raise SystemExit(f'no contiguous ready pipeline spans {expected} workers')
"
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
  local init_nodes="${1:-$PARALLAX_LOCALNET_INIT_NODES}"
  compose exec -T host curl -sS -X POST http://127.0.0.1:3301/scheduler/init \
    -H 'Content-Type: application/json' \
    -d "{\"model_name\":\"$MODEL_PATH\",\"init_nodes_num\":$init_nodes,\"is_local_network\":true}" >/dev/null

  wait_status "data.get('initialized') is True" 60 >/dev/null
  wait_status "data.get('status') == 'available' and data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) >= 1" "$PARALLAX_LOCALNET_READY_TIMEOUT" >/dev/null
}

# Full cluster bootstrap: scheduler + workers + model init + chat proxy
boot_full_cluster() {
  teardown_before_boot
  compose up -d --build

  start_scheduler
  start_all_workers "$scheduler_addr"
  wait_for_worker_count "$PARALLAX_LOCALNET_WORKER_COUNT" 60

  init_model "$PARALLAX_LOCALNET_INIT_NODES"
  start_chat_proxy
}
