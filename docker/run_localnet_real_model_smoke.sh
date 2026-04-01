#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.localnet.yml"
KEEP_RUNNING="${PARALLAX_LOCALNET_KEEP_RUNNING:-0}"
RESET_VOLUMES="${PARALLAX_LOCALNET_RESET_VOLUMES:-0}"
PARALLAX_LOCALNET_INIT_NODES="${PARALLAX_LOCALNET_INIT_NODES:-3}"
MODEL_PATH="${PARALLAX_LOCALNET_REAL_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PARALLAX_LOCALNET_READY_TIMEOUT="${PARALLAX_LOCALNET_READY_TIMEOUT:-300}"
SCHEDULER_LOG="/tmp/parallax-run.log"
CHAT_LOG="/tmp/parallax-chat.log"

export PARALLAX_LOCALNET_TEST_MODE="${PARALLAX_LOCALNET_TEST_MODE:-0}"
WORKER_ENV="PARALLAX_TEST_RUNTIME=transformers PARALLAX_TEST_OVERRIDE_MEMORY_GB=${PARALLAX_TEST_OVERRIDE_MEMORY_GB:-0.7}"
WORKER_JOIN_ARGS="${WORKER_JOIN_ARGS:---kv-cache-memory-fraction 0.45 --max-total-tokens 4096 --max-sequence-length 1024 --max-batch-size 2 --max-num-tokens-per-batch 512}"

source "$SCRIPT_DIR/lib_localnet.sh"

trap cleanup EXIT

if [[ "$RESET_VOLUMES" == "1" ]]; then
  echo "resetting localnet volumes before the real-model smoke"
else
  echo "preserving Docker volumes and Hugging Face caches; set PARALLAX_LOCALNET_RESET_VOLUMES=1 to clear them"
fi

echo "real-model smoke uses a small CPU model and can still take time on a cold cache or slow network"
boot_full_cluster
split_proof="$(validate_split_topology "$PARALLAX_LOCALNET_INIT_NODES")"
echo "$split_proof"

chat_response="$(compose exec -T host python -c "import json,time,urllib.request,sys
payload = json.dumps({
    'model': '$MODEL_PATH',
    'messages': [{'role': 'user', 'content': 'Say the word compose exactly once.'}],
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
sys.exit(1)")"

python3 - <<'PY' "$chat_response" "$MODEL_PATH"
import json
import sys

payload = json.loads(sys.argv[1])
model_name = sys.argv[2]
if payload.get("model") != model_name:
    raise SystemExit(f"unexpected model name: {payload.get('model')}")
content = payload["choices"][0]["message"]["content"].strip()
if not content:
    raise SystemExit("actual-model response was empty")
if "[parallax-test-mode:" in content:
    raise SystemExit("actual-model smoke unexpectedly fell back to synthetic runtime")
print(json.dumps(payload))
PY

final_status="$(compose exec -T host curl -sS http://127.0.0.1:3301/cluster/status_json)"
echo "$final_status"
echo "parallax localnet real-model smoke passed"
