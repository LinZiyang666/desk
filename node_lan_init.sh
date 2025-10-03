#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   sudo IFACE=enp6s0f0 IP_CIDR=10.10.0.2/24 GW=10.10.0.1 /usr/local/sbin/node_lan_init.sh
#
# Behavior:
# - Adds the new IP, adds a backup default route (higher metric), probes gateway.
# - If reachable, switches default route to the new GW (lower metric).
# - A 30s watchdog will roll back if the script didn't complete (marker remains).
#
# Env vars (required):
#   IFACE   : NIC name, e.g., enp6s0f0
#   IP_CIDR : e.g., 10.10.0.2/24
#   GW      : e.g., 10.10.0.1
# Optional:
#   PROBE_IP: default gateway to ping (defaults to $GW)
#   WATCH_SECS: rollback window (default 30)

IFACE=${IFACE:?set IFACE}
IP_CIDR=${IP_CIDR:?set IP_CIDR like 10.10.0.2/24}
GW=${GW:?set GW like 10.10.0.1}
PROBE_IP=${PROBE_IP:-$GW}
WATCH_SECS=${WATCH_SECS:-30}

# Working files
RUN_DIR=/run/node-net-switch
mkdir -p "$RUN_DIR"
MARKER="$RUN_DIR/marker.$$"
BACKUP_RT="$RUN_DIR/backup_route.$$"
ADDED_RT="$RUN_DIR/added_route.$$"
ADDED_IP="$RUN_DIR/added_ip.$$"
LOG="$RUN_DIR/log.$$"

# Capture current default route (if any)
ip route show default 0.0.0.0/0 > "$BACKUP_RT" || true

# Record what we'll add (for rollback)
echo "$IP_CIDR" > "$ADDED_IP"
echo "$GW"      > "$ADDED_RT"

# Create rollback marker
touch "$MARKER"

rollback() {
  echo "[node-rollback] starting rollback ..." | tee -a "$LOG"
  # Delete new default routes (both metric 100 and 500 variants if present)
  ip route del default via "$GW" metric 100 2>/dev/null || true
  ip route del default via "$GW" metric 500 2>/dev/null || true
  ip route del default via "$GW" 2>/dev/null || true

  # Delete the added IP (if still present)
  if ip -4 addr show dev "$IFACE" | grep -q " ${IP_CIDR%/*}/"; then
    ip addr del "$IP_CIDR" dev "$IFACE" 2>/dev/null || true
  fi

  # Restore previous default route if we had one
  if [[ -s "$BACKUP_RT" ]]; then
    # It may contain more than one line; restore the first default
    OLD_DEF_LINE=$(grep -m1 '^default ' "$BACKUP_RT" || true)
    if [[ -n "$OLD_DEF_LINE" ]]; then
      # Attempt to restore exactly; if fails, try a simplified form
      if ! ip route add $OLD_DEF_LINE 2>/dev/null; then
        OLD_VIA=$(echo "$OLD_DEF_LINE" | awk '/via/ {print $3}')
        if [[ -n "${OLD_VIA:-}" ]]; then
          ip route replace default via "$OLD_VIA" 2>/dev/null || true
        fi
      fi
    fi
  else
    echo "[node-rollback] no previous default route snapshot" | tee -a "$LOG"
  fi

  echo "[node-rollback] rollback done." | tee -a "$LOG"
}

# Watchdog: after WATCH_SECS, if marker still exists, perform rollback
(
  sleep "$WATCH_SECS"
  if [[ -f "$MARKER" ]]; then
    rollback
    rm -f "$MARKER" "$BACKUP_RT" "$ADDED_RT" "$ADDED_IP" "$LOG" 2>/dev/null || true
  fi
) & disown

echo "[node-safe] configuring $IFACE ip=$IP_CIDR gw=$GW (watchdog ${WATCH_SECS}s)" | tee -a "$LOG"

# 1) Ensure interface up, add IP if not present
ip link set "$IFACE" up
if ! ip -4 addr show dev "$IFACE" | grep -q " ${IP_CIDR%/*}/"; then
  ip addr add "$IP_CIDR" dev "$IFACE"
fi

# 2) Add a backup default route via new GW with higher metric (doesn't preempt current)
ip route add default via "$GW" metric 500 2>/dev/null || true

# 3) Probe gateway reachability
if ping -c1 -W1 "$PROBE_IP" >/dev/null 2>&1; then
  echo "[node-safe] gateway reachable -> promoting new default route" | tee -a "$LOG"

  # Remember old primary default to optionally keep as backup
  OLD_PRIMARY=$(ip route show default 0.0.0.0/0 | head -n1 || true)
  OLD_PRIMARY_VIA=$(echo "$OLD_PRIMARY" | awk '/via/ {print $3}')

  # 4) Promote new route (lower metric)
  ip route replace default via "$GW" metric 100

  # 5) Optionally keep old default as a lower priority fallback (if different)
  if [[ -n "${OLD_PRIMARY_VIA:-}" && "$OLD_PRIMARY_VIA" != "$GW" ]]; then
    ip route add default via "$OLD_PRIMARY_VIA" metric 400 2>/dev/null || true
  fi

  # Success path: remove marker to cancel rollback
  rm -f "$MARKER"
  echo "[node-safe] switched. Current routes:" | tee -a "$LOG"
  ip route show | tee -a "$LOG"
else
  echo "[node-safe] WARN: gateway not reachable; keeping old default. Watchdog will rollback if this script exits early." | tee -a "$LOG"
  # Do not promote; leave marker so rollback will clean the added IP/route after timeout.
fi

# If we reach here (script completed), and marker still exists (e.g., gateway unreachable),
# we leave it for the watchdog to rollback. Otherwise, everything is OK.
