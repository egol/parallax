#!/usr/bin/env bash
# Centralized split-proof real-model smoke test for Parallax localnet.
#
# Same topology proof as the synthetic centralized lane, but uses a small real
# CPU model to validate actual generation through the host-routed path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.localnet.yml"
KEEP_RUNNING="${PARALLAX_LOCALNET_KEEP_RUNNING:-0}"
RESET_VOLUMES="${PARALLAX_LOCALNET_RESET_VOLUMES:-0}"
PARALLAX_LOCALNET_INIT_NODES="${PARALLAX_LOCALNET_INIT_NODES:-3}"
PARALLAX_LOCALNET_READY_TIMEOUT="${PARALLAX_LOCALNET_READY_TIMEOUT:-300}"
MODEL_PATH="${PARALLAX_LOCALNET_REAL_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SCHEDULER_LOG="/tmp/parallax-run.log"
CHAT_LOG="/tmp/parallax-chat.log"
ARTIFACT_DIR="${PARALLAX_LOCALNET_ARTIFACT_DIR:-/tmp/parallax-nettest-centralized-split-real-model}"

export PARALLAX_LOCALNET_TEST_MODE="${PARALLAX_LOCALNET_TEST_MODE:-0}"
export PARALLAX_LOCALNET_INSTALL_SGLANG="${PARALLAX_LOCALNET_INSTALL_SGLANG:-1}"
if [[ "$PARALLAX_LOCALNET_TEST_MODE" != "0" ]]; then
  echo "centralized real-model smoke requires PARALLAX_LOCALNET_TEST_MODE=0" >&2
  exit 1
fi
WORKER_ENV="PARALLAX_TEST_OVERRIDE_MEMORY_GB=${PARALLAX_TEST_OVERRIDE_MEMORY_GB:-0.7} PARALLAX_TRACE_REQUESTS=1 PARALLAX_TRACE_REQUESTS_FILE=/tmp/parallax-request-trace.jsonl"
WORKER_JOIN_ARGS="${WORKER_JOIN_ARGS:---kv-cache-memory-fraction 0.7 --max-total-tokens 4096 --max-sequence-length 1024 --max-batch-size 2 --max-num-tokens-per-batch 512}"

source "$SCRIPT_DIR/lib_localnet.sh"
source "$SCRIPT_DIR/lib_nettest.sh"

trap 'on_exit $?' EXIT

if [[ "$RESET_VOLUMES" == "1" ]]; then
  echo "resetting localnet volumes before the centralized real-model smoke"
else
  echo "preserving Docker volumes and Hugging Face caches; set PARALLAX_LOCALNET_RESET_VOLUMES=1 to clear them"
fi

echo "=== Parallax nettest: centralized split real-model smoke ==="

teardown_before_boot
compose up -d --build

PROMPT_MATRIX_JSON="$(python3 - <<'PY'
import json

prompt_cases = [
    {
        "name": "exact_ok",
        "request": {
            "model": "MODEL_PLACEHOLDER",
            "messages": [{"role": "user", "content": "Reply with exactly one token: OK"}],
            "stream": False,
            "max_tokens": 1,
            "sampling_params": {"temperature": 0.0, "top_k": 1, "top_p": 1.0},
        },
    },
    {
        "name": "exact_blue_sky",
        "request": {
            "model": "MODEL_PLACEHOLDER",
            "messages": [{"role": "user", "content": "Reply with exactly two tokens: blue sky"}],
            "stream": False,
            "max_tokens": 2,
            "sampling_params": {"temperature": 0.0, "top_k": 1, "top_p": 1.0},
        },
    },
    {
        "name": "capital_of_china",
        "request": {
            "model": "MODEL_PLACEHOLDER",
            "messages": [
                {"role": "user", "content": "Complete this sentence briefly: The capital of China is"}
            ],
            "stream": False,
            "max_tokens": 4,
            "sampling_params": {"temperature": 0.0, "top_k": 1, "top_p": 1.0},
        },
    },
]

print(json.dumps(prompt_cases))
PY
)"
PROMPT_MATRIX_JSON="${PROMPT_MATRIX_JSON//MODEL_PLACEHOLDER/$MODEL_PATH}"

baseline_file="$(mktemp -t parallax-centralized-baseline.XXXXXX.json)"
compose exec -T host python /parallax/docker/real_model_baseline.py \
  --model "$MODEL_PATH" \
  --cases-json "$PROMPT_MATRIX_JSON" >"$baseline_file"

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

echo "--- differential centralized real-model chat ---"
case_lines="$(python3 - <<'PY' "$PROMPT_MATRIX_JSON"
import base64
import json
import sys

for case in json.loads(sys.argv[1]):
    print(base64.b64encode(json.dumps(case).encode()).decode())
PY
)"

case_index=0
while IFS= read -r case_b64; do
  [[ -n "$case_b64" ]] || continue
  case_json="$(python3 - <<'PY' "$case_b64"
import base64
import sys

print(base64.b64decode(sys.argv[1]).decode())
PY
)"
  case_name="$(python3 - <<'PY' "$case_json"
import json
import sys

print(json.loads(sys.argv[1])["name"])
PY
)"

  echo "--- prompt case: ${case_name} ---"
  clear_worker_request_traces

  chat_response="$(
    compose exec -T host python - "$case_json" <<'PY'
import json
import sys
import time
import urllib.request

payload = json.dumps(json.loads(sys.argv[1])["request"]).encode()
request = urllib.request.Request(
    "http://127.0.0.1:3200/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)

last = None
for _ in range(30):
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            last = response.read().decode()
        data = json.loads(last)
        if "choices" in data and data["choices"][0]["message"]["content"]:
            print(last)
            sys.exit(0)
    except Exception as exc:
        last = json.dumps({"error": str(exc)})
    time.sleep(2)

print(last)
sys.exit(1)
PY
  )"

  compare_json="$(python3 - <<'PY' "$baseline_file" "$case_index" "$chat_response" "$MODEL_PATH"
import json
import sys

baseline = json.load(open(sys.argv[1], "r", encoding="utf-8"))[int(sys.argv[2])]
payload = json.loads(sys.argv[3])
model_name = sys.argv[4]

if payload.get("model") != model_name:
    raise SystemExit(f"unexpected model name: {payload.get('model')}")

content = payload["choices"][0]["message"]["content"]
if "[parallax-test-mode:" in content:
    raise SystemExit("centralized real-model lane fell back to synthetic runtime")

split_text = content.rstrip()
baseline_text = baseline["completion_text"].rstrip()
if not split_text or not baseline_text:
    raise SystemExit(
        f"empty completion for {baseline['name']}: split={split_text!r} baseline={baseline_text!r}"
    )
if split_text != baseline_text:
    raise SystemExit(
        "split output diverged from unsplit baseline for "
        f"{baseline['name']}: split={split_text!r} baseline={baseline_text!r}"
    )

print(
    json.dumps(
        {
            "name": baseline["name"],
            "request_id": payload["id"],
            "split_text": split_text,
            "baseline_text": baseline_text,
            "require_decode": len(baseline["generated_token_ids"]) > 1,
        }
    )
)
PY
)"
  echo "$compare_json"

  trace_dir="$(mktemp -d -t parallax-centralized-traces.XXXXXX)"
  collect_worker_request_traces "$trace_dir"
  python3 - <<'PY' "$trace_dir" "$compare_json"
import json
import pathlib
import sys

trace_dir = pathlib.Path(sys.argv[1])
compare = json.loads(sys.argv[2])
request_id = compare["request_id"]
require_decode = bool(compare["require_decode"])

events = []
for trace_file in sorted(trace_dir.glob("*.jsonl")):
    if not trace_file.exists():
        continue
    for line in trace_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("request_id") == request_id:
            events.append(event)

if not events:
    raise SystemExit(f"no request trace found for {compare['name']} ({request_id})")

roles_by_phase = {}
node_ids = set()
for event in events:
    if event.get("status") != "processed":
        continue
    phase = event.get("phase")
    role = event.get("stage_role")
    if phase and role:
        roles_by_phase.setdefault(phase, set()).add(role)
    if event.get("node_id"):
        node_ids.add(event["node_id"])

expected_roles = {"first", "middle", "last"}
prefill_roles = roles_by_phase.get("prefill", set())
if prefill_roles != expected_roles:
    raise SystemExit(
        f"prefill did not traverse all stages for {compare['name']}: {sorted(prefill_roles)}"
    )

if len(node_ids) != 3:
    raise SystemExit(
        f"expected traces from 3 distinct nodes for {compare['name']}, got {sorted(node_ids)}"
    )

if require_decode:
    decode_roles = roles_by_phase.get("decode", set())
    if decode_roles != expected_roles:
        raise SystemExit(
            f"decode did not traverse all stages for {compare['name']}: {sorted(decode_roles)}"
        )

print(
    json.dumps(
        {
            "name": compare["name"],
            "request_id": request_id,
            "prefill_roles": sorted(prefill_roles),
            "decode_roles": sorted(roles_by_phase.get("decode", set())),
            "node_ids": sorted(node_ids),
        }
    )
)
PY
  rm -rf "$trace_dir"
  case_index=$((case_index + 1))
done <<< "$case_lines"

rm -f "$baseline_file"

compose exec -T host curl -sS http://127.0.0.1:3301/cluster/status_json
echo "parallax nettest centralized split real-model smoke passed"
