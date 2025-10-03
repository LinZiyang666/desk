#!/usr/bin/env bash
set -euo pipefail

IFACE=${IFACE:-enp6s0f0}
IP_CIDR=${IP_CIDR:?e.g., 10.10.0.2/24}
GW=${GW:?e.g., 10.10.0.1}
TEST_IP=${TEST_IP:-10.10.0.1}   # 先测网关可达性

echo "[node-safe] iface=$IFACE ip=$IP_CIDR gw=$GW"

# 1) 不清 IP；仅确保接口 up，并添加新 IP（若已存在则忽略）
ip link set "$IFACE" up
ip -4 addr show dev "$IFACE" | grep -q " ${IP_CIDR%/*}\/" || ip addr add "$IP_CIDR" dev "$IFACE"

# 2) 新增一条“备选默认路由”，设置较大 metric（不会立刻抢主）
#    允许多条默认路由并存，metric 小者优先
set +e
ip route add default via "$GW" metric 500 2>/dev/null
set -e

# 3) 健康检查（连通网关）
if ping -c1 -W1 "$TEST_IP" >/dev/null 2>&1; then
  echo "[node-safe] gateway reachable, switching default route priority..."
  # 3a) 读取当前主默认路由（如有）
  OLD_DEF=$(ip route show default 0.0.0.0/0 | head -n1 || true)

  # 3b) 让新网关成为主路由（较小 metric）
  ip route replace default via "$GW" metric 100

  # 3c) 如果存在旧默认路由，把它降为备选（大 metric）
  if [[ -n "$OLD_DEF" ]]; then
    # 提取旧网关 IP（可能没有 via，就跳过）
    OLD_VIA=$(echo "$OLD_DEF" | awk '/via/ {print $3}')
    if [[ -n "${OLD_VIA:-}" && "$OLD_VIA" != "$GW" ]]; then
      ip route add default via "$OLD_VIA" metric 400 2>/dev/null || true
    fi
  fi
  echo "[node-safe] switched. Current routes:"
else
  echo "[node-safe] WARN: gateway NOT reachable; kept old default route."
  echo "You can retry later: ip route replace default via $GW metric 100"
fi

ip -4 addr show dev "$IFACE"
ip route show
