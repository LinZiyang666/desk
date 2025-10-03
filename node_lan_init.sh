#!/usr/bin/env bash
set -euo pipefail

IFACE=${IFACE:?set IFACE}
GW=${GW:?set GW like 10.10.0.1}
WATCH_SECS=${WATCH_SECS:-30}
NEW_METRIC=${NEW_METRIC:-100}
OLD_METRIC=${OLD_METRIC:-400}

RUN_DIR=/run/node-net-switch
mkdir -p "$RUN_DIR"
MARKER="$RUN_DIR/switch_marker.$$"
BACKUP_RT="$RUN_DIR/backup_default.$$"
LOG="$RUN_DIR/switch_log.$$.log"

touch "$MARKER"
ip route show default 0.0.0.0/0 > "$BACKUP_RT" || true

echo "[stage2] switch default via $GW on $IFACE (watch=$WATCH_SECS s)" | tee -a "$LOG"

rollback() {
  echo "[stage2][rollback] start..." | tee -a "$LOG"
  # 删除可能添加的新默认路由
  ip route del default via "$GW" metric "$NEW_METRIC" 2>/dev/null || true
  ip route del default via "$GW" 2>/dev/null || true

  # 恢复旧默认路由
  if [[ -s "$BACKUP_RT" ]]; then
    OLD_LINE=$(grep -m1 '^default ' "$BACKUP_RT" || true)
    if [[ -n "$OLD_LINE" ]]; then
      if ! ip route replace $OLD_LINE 2>/dev/null; then
        OLD_VIA=$(echo "$OLD_LINE" | awk '/via/ {print $3}')
        if [[ -n "${OLD_VIA:-}" ]]; then
          ip route replace default via "$OLD_VIA" metric 100 2>/dev/null || true
        fi
      fi
    fi
  fi
  echo "[stage2][rollback] done." | tee -a "$LOG"
}

# 看门狗
(
  sleep "$WATCH_SECS"
  if [[ -f "$MARKER" ]]; then
    rollback
    rm -f "$MARKER" "$BACKUP_RT" "$LOG" 2>/dev/null || true
  fi
) & disown

# 当前主默认（切换前）
OLD_PRIMARY=$(ip route show default 0.0.0.0/0 | head -n1 || true)
OLD_VIA=$(echo "$OLD_PRIMARY" | awk '/via/ {print $3}')

# 先把“旧默认”降级为后备（仍保留）
if [[ -n "${OLD_VIA:-}" && "$OLD_VIA" != "$GW" ]]; then
  ip route replace default via "$OLD_VIA" metric "$OLD_METRIC" || true
fi

# 添加“新默认路由”（优先级更高，metric 小）
ip route replace default via "$GW" metric "$NEW_METRIC"

echo "[stage2] switched (tentative). Routes now:" | tee -a "$LOG"
ip route show | tee -a "$LOG"

echo
echo ">>> If everything is OK, CONFIRM to cancel rollback:"
echo "    rm -f $MARKER"
echo ">>> If you lose connectivity, wait ${WATCH_SECS}s for auto-rollback."
