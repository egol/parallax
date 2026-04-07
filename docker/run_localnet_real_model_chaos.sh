#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.localnet.yml"
KEEP_RUNNING="${PARALLAX_LOCALNET_KEEP_RUNNING:-0}"
RESET_VOLUMES="${PARALLAX_LOCALNET_RESET_VOLUMES:-0}"
PARALLAX_LOCALNET_INIT_NODES="${PARALLAX_LOCALNET_INIT_NODES:-3}"
MODEL_PATH="${PARALLAX_LOCALNET_REAL_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SCHEDULER_LOG="/tmp/parallax-run.log"
CHAT_LOG="/tmp/parallax-chat.log"
ARTIFACT_DIR="${PARALLAX_LOCALNET_ARTIFACT_DIR:-/tmp/parallax-localnet-chaos-real-model}"
PARALLAX_LOCALNET_READY_TIMEOUT="${PARALLAX_LOCALNET_READY_TIMEOUT:-300}"

export PARALLAX_LOCALNET_TEST_MODE="${PARALLAX_LOCALNET_TEST_MODE:-0}"
export PARALLAX_LOCALNET_INSTALL_SGLANG="${PARALLAX_LOCALNET_INSTALL_SGLANG:-1}"
WORKER_ENV="PARALLAX_TEST_RUNTIME=transformers PARALLAX_TEST_OVERRIDE_MEMORY_GB=${PARALLAX_TEST_OVERRIDE_MEMORY_GB:-0.7}"
WORKER_JOIN_ARGS="${WORKER_JOIN_ARGS:---kv-cache-memory-fraction 0.7 --max-total-tokens 4096 --max-sequence-length 1024 --max-batch-size 2 --max-num-tokens-per-batch 512}"

source "$SCRIPT_DIR/lib_localnet.sh"

trap 'on_exit $?' EXIT

request_real_model_chat() {
  compose exec -T host python -c "import json,time,urllib.request,sys
payload = json.dumps({
    'model': '$MODEL_PATH',
    'messages': [{'role': 'user', 'content': 'Reply with exactly one token: OK'}],
    'stream': False,
    'max_tokens': 4,
    'sampling_params': {
        'temperature': 0.0,
        'top_k': 1,
        'top_p': 1.0,
    },
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
else
  echo "preserving Docker volumes and Hugging Face caches; set PARALLAX_LOCALNET_RESET_VOLUMES=1 to clear them"
fi

echo "real-model chaos lane uses a small CPU model and can still take time on a cold cache or slow network"
boot_full_cluster
split_topology="$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"
echo "$split_topology"

baseline_response="$(request_real_model_chat)"
python3 - <<'PY' "$baseline_response" "$MODEL_PATH"
import json
import sys

payload = json.loads(sys.argv[1])
model_name = sys.argv[2]
if payload.get("model") != model_name:
    raise SystemExit(f"unexpected model name: {payload.get('model')}")
content = payload["choices"][0]["message"]["content"].strip()
normalized = content.strip().strip(".!?,:;\"'").upper()
if not content or "[parallax-test-mode:" in content or normalized != "OK":
    raise SystemExit("unexpected baseline real-model payload: " + content)
print(json.dumps(payload))
PY

compose stop worker1 >/dev/null
wait_status "data.get('topology', {}).get('totals', {}).get('registered_workers', 0) < 3 or data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) < 1" 120 >/dev/null

compose start worker1 >/dev/null
sleep 2
start_worker_join worker1 3101 4101 5101 "$scheduler_addr"
wait_status "data.get('topology', {}).get('totals', {}).get('registered_workers', 0) >= 3" 120 >/dev/null
wait_status "data.get('status') == 'available' and data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) >= 1" 120 >/dev/null
split_topology="$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"
echo "$split_topology"

compose stop worker2 >/dev/null
wait_status "data.get('topology', {}).get('totals', {}).get('registered_workers', 0) < 3 or data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) < 1" 120 >/dev/null

compose start worker2 >/dev/null
sleep 2
start_worker_join worker2 3102 4102 5102 "$scheduler_addr"
wait_status "data.get('topology', {}).get('totals', {}).get('registered_workers', 0) >= 3" 120 >/dev/null
wait_status "data.get('status') == 'available' and data.get('topology', {}).get('totals', {}).get('ready_pipelines', 0) >= 1" 120 >/dev/null
split_topology="$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"
echo "$split_topology"

recovered_response="$(request_real_model_chat)"
python3 - <<'PY' "$recovered_response" "$MODEL_PATH"
import json
import sys

payload = json.loads(sys.argv[1])
model_name = sys.argv[2]
if payload.get("model") != model_name:
    raise SystemExit(f"unexpected model name: {payload.get('model')}")
content = payload["choices"][0]["message"]["content"].strip()
normalized = content.strip().strip(".!?,:;\"'").upper()
if not content or "[parallax-test-mode:" in content or normalized != "OK":
    raise SystemExit("unexpected recovery real-model payload: " + content)
print(json.dumps(payload))
PY

compose exec -T host curl -sS http://127.0.0.1:3301/cluster/status_json
echo "parallax localnet real-model chaos lane passed"
