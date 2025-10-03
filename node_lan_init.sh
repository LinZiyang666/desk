#!/usr/bin/env bash
set -euo pipefail

IFACE=${IFACE:-enp6s0f0}
IP_CIDR=${IP_CIDR:?e.g., 10.10.0.2/24}
GW=${GW:?e.g., 10.10.0.1}

echo "[node] configuring ${IFACE} ip=${IP_CIDR} gw=${GW}"

# 清理旧地址（只清当前接口的 IPv4）
ip addr flush dev "$IFACE" scope global || true
ip link set "$IFACE" up

# 配置 IP
ip addr add "$IP_CIDR" dev "$IFACE"

# 设置默认路由
ip route replace default via "$GW" dev "$IFACE"

echo "[node] done."
ip -4 addr show dev "$IFACE"
ip route show
