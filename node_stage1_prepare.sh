#!/usr/bin/env bash
set -euo pipefail

# 必填：统一用 export 传入
IFACE=${IFACE:?set IFACE}
IP_CIDR=${IP_CIDR:?set IP_CIDR like 10.10.0.2/24}
GW=${GW:?set GW like 10.10.0.1}
TABLE=${TABLE:-100}           # 策略路由表号
PRIORITY=${PRIORITY:-1000}    # ip rule 优先级（越小越先匹配）

NEW_IP="${IP_CIDR%/*}"

echo "[stage1] iface=$IFACE add $IP_CIDR; policy route: from $NEW_IP -> table $TABLE via $GW"

# 1) 挂新 IP（不 flush，不动旧配置）
ip link set "$IFACE" up
ip -4 addr show dev "$IFACE" | grep -q " ${NEW_IP}/" || ip addr add "$IP_CIDR" dev "$IFACE"

# 2) 添加策略路由（只让“源自 NEW_IP 的流量”用表100）
#    表100默认路由 -> 新网关
ip route replace default via "$GW" dev "$IFACE" table "$TABLE"

#    添加从 NEW_IP 走表100 的规则
#    先删再加，避免重复
ip rule del from "$NEW_IP"/32 table "$TABLE" 2>/dev/null || true
ip rule add from "$NEW_IP"/32 table "$TABLE" priority "$PRIORITY"

echo "[stage1] done. Now test with the new source IP:"
echo "  ping -I $NEW_IP -c 3 $GW"
echo "  curl --interface $NEW_IP http://<router-or-peer>"
ip rule show | grep "$NEW_IP" || true
ip route show table "$TABLE" || true
