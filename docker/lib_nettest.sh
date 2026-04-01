#!/usr/bin/env bash
# Network simulation functions for Parallax localnet testing.
#
# Provides tc (traffic control) and iptables wrappers for injecting
# latency, packet loss, bandwidth limits, and network partitions into
# Docker containers.
#
# Prerequisites:
#   - Containers must have cap_add: [NET_ADMIN]
#   - iproute2 and iptables must be installed in the container image
#   - lib_localnet.sh must be sourced first (provides compose())

# Resolve the default network interface inside a container.
resolve_interface() {
  local container="$1"
  compose exec -T "$container" sh -c "ip route | awk '/default/ {print \$5}' | head -1"
}

# --- tc (traffic control) functions ---

# Inject latency on all egress traffic.
# Usage: inject_latency <container> <delay_ms> [jitter_ms]
inject_latency() {
  local container="$1" delay_ms="$2" jitter_ms="${3:-0}"
  local iface
  iface="$(resolve_interface "$container")"
  clear_tc "$container" 2>/dev/null || true
  if [[ "$jitter_ms" -gt 0 ]]; then
    compose exec -T "$container" tc qdisc add dev "$iface" root netem delay "${delay_ms}ms" "${jitter_ms}ms"
  else
    compose exec -T "$container" tc qdisc add dev "$iface" root netem delay "${delay_ms}ms"
  fi
  echo "  [tc] ${container}: +${delay_ms}ms latency (jitter=${jitter_ms}ms) on $iface"
}

# Inject packet loss on all egress traffic.
# Usage: inject_packet_loss <container> <loss_pct>
inject_packet_loss() {
  local container="$1" loss_pct="$2"
  local iface
  iface="$(resolve_interface "$container")"
  clear_tc "$container" 2>/dev/null || true
  compose exec -T "$container" tc qdisc add dev "$iface" root netem loss "${loss_pct}%"
  echo "  [tc] ${container}: ${loss_pct}% packet loss on $iface"
}

# Limit bandwidth on all egress traffic.
# Usage: limit_bandwidth <container> <rate_kbit>
limit_bandwidth() {
  local container="$1" rate_kbit="$2"
  local iface
  iface="$(resolve_interface "$container")"
  clear_tc "$container" 2>/dev/null || true
  compose exec -T "$container" tc qdisc add dev "$iface" root tbf rate "${rate_kbit}kbit" burst 32kbit latency 400ms
  echo "  [tc] ${container}: bandwidth limited to ${rate_kbit}kbit on $iface"
}

# Inject combined latency + packet loss.
# Usage: inject_conditions <container> <delay_ms> <loss_pct>
inject_conditions() {
  local container="$1" delay_ms="$2" loss_pct="$3"
  local iface
  iface="$(resolve_interface "$container")"
  clear_tc "$container" 2>/dev/null || true
  compose exec -T "$container" tc qdisc add dev "$iface" root netem delay "${delay_ms}ms" loss "${loss_pct}%"
  echo "  [tc] ${container}: +${delay_ms}ms latency, ${loss_pct}% loss on $iface"
}

# Remove all tc rules from a container.
# Usage: clear_tc <container>
clear_tc() {
  local container="$1"
  local iface
  iface="$(resolve_interface "$container")"
  compose exec -T "$container" tc qdisc del dev "$iface" root 2>/dev/null || true
  echo "  [tc] ${container}: cleared on $iface"
}

# --- iptables (partition) functions ---

# Partition a container from all network traffic.
# Usage: partition_container <container>
partition_container() {
  local container="$1"
  compose exec -T "$container" iptables -A INPUT -j DROP 2>/dev/null || true
  compose exec -T "$container" iptables -A OUTPUT -j DROP 2>/dev/null || true
  echo "  [iptables] ${container}: partitioned (all traffic dropped)"
}

# Heal a partition — flush all iptables rules.
# Usage: heal_partition <container>
heal_partition() {
  local container="$1"
  compose exec -T "$container" iptables -F 2>/dev/null || true
  echo "  [iptables] ${container}: partition healed"
}

# Block traffic from a container to a specific host.
# Usage: block_traffic <container> <target_host>
block_traffic() {
  local container="$1" target="$2"
  local target_ip
  target_ip="$(compose exec -T "$container" getent hosts "$target" | awk '{print $1}')"
  if [[ -n "$target_ip" ]]; then
    compose exec -T "$container" iptables -A OUTPUT -d "$target_ip" -j DROP 2>/dev/null || true
    compose exec -T "$container" iptables -A INPUT -s "$target_ip" -j DROP 2>/dev/null || true
    echo "  [iptables] ${container}: blocked traffic to/from $target ($target_ip)"
  else
    echo "  [iptables] ${container}: could not resolve $target" >&2
    return 1
  fi
}

# Unblock traffic from a container to a specific host.
# Usage: unblock_traffic <container> <target_host>
unblock_traffic() {
  local container="$1" target="$2"
  local target_ip
  target_ip="$(compose exec -T "$container" getent hosts "$target" | awk '{print $1}')"
  if [[ -n "$target_ip" ]]; then
    compose exec -T "$container" iptables -D OUTPUT -d "$target_ip" -j DROP 2>/dev/null || true
    compose exec -T "$container" iptables -D INPUT -s "$target_ip" -j DROP 2>/dev/null || true
    echo "  [iptables] ${container}: unblocked traffic to/from $target ($target_ip)"
  fi
}

# --- Connectivity verification ---

# Assert that a container can reach a target on a given port.
# Usage: assert_reachable <container> <target_host> <target_port>
assert_reachable() {
  local container="$1" target="$2" port="$3"
  if compose exec -T "$container" python -c "
import socket, sys
try:
    s = socket.create_connection(('$target', $port), timeout=5)
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
"; then
    echo "  [check] ${container} -> ${target}:${port}: reachable"
  else
    echo "  [check] ${container} -> ${target}:${port}: UNREACHABLE (expected reachable)" >&2
    return 1
  fi
}

# Assert that a container cannot reach a target on a given port.
# Usage: assert_unreachable <container> <target_host> <target_port>
assert_unreachable() {
  local container="$1" target="$2" port="$3"
  if compose exec -T "$container" python -c "
import socket, sys
try:
    s = socket.create_connection(('$target', $port), timeout=3)
    s.close()
    sys.exit(1)
except Exception:
    sys.exit(0)
"; then
    echo "  [check] ${container} -> ${target}:${port}: unreachable (as expected)"
  else
    echo "  [check] ${container} -> ${target}:${port}: REACHABLE (expected unreachable)" >&2
    return 1
  fi
}

# Clear all network simulation state on all containers.
clear_all_network_conditions() {
  for container in host "${WORKER_SERVICES[@]}"; do
    clear_tc "$container" 2>/dev/null || true
    heal_partition "$container" 2>/dev/null || true
  done
}
