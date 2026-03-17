# Parallax Local-Network Harness

This harness is for raw Parallax command validation on one machine using a
dedicated Docker network. It is not the primary Mycelia desktop regression
path.

The default localnet image is intentionally lightweight. It uses
`PARALLAX_TEST_MODE=1` so workers still run the real scheduler, Lattica, chat,
and join control plane, but serve a synthetic chat backend instead of pulling
GPU model-serving dependencies or real weights.

Use it when:

- the single-machine localhost runbook is not enough
- you need to pressure-test scheduler bootstrap and worker attach over a real
  container network
- you want to validate raw `parallax run`, `parallax chat`, and
  `parallax join` before debugging Mycelia orchestration

## Prerequisites

- Docker Engine with `docker compose`
- enough local disk for a Python slim image

## Start The Harness

From the repository root:

```bash
docker compose -f parallax/docker/docker-compose.localnet.yml up -d --build
```

For the fully automated smoke path:

```bash
./parallax/docker/run_localnet_smoke.sh
```

For the abrupt-disconnect synthetic chaos path:

```bash
./parallax/docker/run_localnet_chaos_smoke.sh
```

For the real-model CPU smoke path:

```bash
./parallax/docker/run_localnet_real_model_smoke.sh
```

For the abrupt-disconnect real-model chaos path:

```bash
./parallax/docker/run_localnet_real_model_chaos.sh
```

By default the real-model smoke preserves Docker volumes, including the shared
Hugging Face cache, so repeated runs do not redownload model assets. Set
`PARALLAX_LOCALNET_RESET_VOLUMES=1` if you explicitly want a cold-cache run.

This creates:

- `host`
- `worker1`
- `worker2`
- a shared Hugging Face cache volume
- a dedicated bridge network named by Compose

All three containers idle on `sleep infinity` so you can run the upstream
commands manually.

## Host Commands

Open a shell in the host container:

```bash
docker compose -f parallax/docker/docker-compose.localnet.yml exec host bash
```

Start the scheduler:

```bash
parallax run \
  --host 0.0.0.0 \
  --port 3301 \
  --tcp-port 4301 \
  --udp-port 5301 \
  --announce-maddrs /dns4/host/tcp/4301 /dns4/host/udp/5301/quic-v1 \
  -u
```

In a second host shell, bootstrap the scheduler:

```bash
curl -sS -X POST http://127.0.0.1:3301/scheduler/bootstrap \
  -H 'Content-Type: application/json' \
  -d '{"is_local_network": true}'
```

Record the scheduler peer ID from the host startup logs, then build the
bootstrap address:

```text
/dns4/host/tcp/4301/p2p/<scheduler-peer-id>
```

Start host chat:

```bash
parallax chat \
  --scheduler-addr /dns4/host/tcp/4301/p2p/<scheduler-peer-id> \
  --node-chat-port 3200
```

Optional host compute:

```bash
parallax join \
  --scheduler-addr /dns4/host/tcp/4301/p2p/<scheduler-peer-id> \
  --port 3100 \
  --tcp-port 4100 \
  --udp-port 5100 \
  -u
```

## Worker Commands

Open a shell in `worker1` or `worker2`:

```bash
docker compose -f parallax/docker/docker-compose.localnet.yml exec worker1 bash
```

Start chat:

```bash
parallax chat \
  --scheduler-addr /dns4/host/tcp/4301/p2p/<scheduler-peer-id> \
  --node-chat-port 3201
```

Start compute join:

```bash
parallax join \
  --scheduler-addr /dns4/host/tcp/4301/p2p/<scheduler-peer-id> \
  --port 3101 \
  --tcp-port 4101 \
  --udp-port 5101 \
  -u
```

Repeat on `worker2` with a separate port set such as `3202`, `3102`, `4102`,
and `5102`.

## Small-Model Validation

For the lightweight harness, use the built-in synthetic model fixture:
`/parallax/docker/test-models/parallax-smoke`.

Example scheduler init from the host container:

```bash
  curl -sS -X POST http://127.0.0.1:3301/scheduler/init \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "/parallax/docker/test-models/parallax-smoke",
    "init_nodes_num": 2,
    "is_local_network": true
  }'
```

After init, whichever worker receives layer `0` will start a synthetic
OpenAI-compatible `/v1/chat/completions` server. A `parallax chat` process can
then proxy a real chat request through the scheduler and return a synthetic
completion, which is enough to validate end-to-end orchestration locally.

## Suggested Validation Order

1. confirm `docker compose ... config` is clean
2. start the three containers
3. validate raw scheduler bootstrap on `host`
4. validate `parallax chat` and `parallax join` from at least one worker
5. confirm scheduler status and cluster visibility
6. run the synthetic-model init only after worker capacity is visible
7. send a chat request through `parallax chat` and confirm the synthetic
   completion path works

The repository script `parallax/docker/run_localnet_smoke.sh` automates exactly
that sequence and exits non-zero on any failure.

`parallax/docker/run_localnet_real_model_smoke.sh` runs the same topology, but
switches worker test mode to a tiny CPU Hugging Face model so completions come
from an actual model instead of the synthetic echo backend. It is still a
best-effort local smoke: cold-cache downloads, CPU-only inference, and slower
network conditions can make it materially slower than the synthetic harness.

`parallax/docker/run_localnet_chaos_smoke.sh` adds an abrupt worker container
stop/start during serving and verifies that the scheduler degrades and
recovers without wedging.

`parallax/docker/run_localnet_real_model_chaos.sh` does the same with the tiny
real-model lane, then adds a process-level worker kill/restart so both
container-level and process-level reconnect behavior are covered.

## Shutdown

```bash
docker compose -f parallax/docker/docker-compose.localnet.yml down
```

To also delete the shared caches and other named volumes:

```bash
docker compose -f parallax/docker/docker-compose.localnet.yml down -v
```
