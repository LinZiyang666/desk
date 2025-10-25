#!/usr/bin/env bash
set -euo pipefail

# 中转网关 & 表名/表号 & ipset 名称
GW=155.98.36.110
TABLE=clusterroute
TABLE_ID=200
SET=cluster

# 1) 在 /etc/iproute2/rt_tables 里登记策略路由表（若已存在则跳过）
grep -q -E "^[[:space:]]*${TABLE_ID}[[:space:]]+${TABLE}$" /etc/iproute2/rt_tables \
  || echo "${TABLE_ID} ${TABLE}" | sudo tee -a /etc/iproute2/rt_tables

# 2) 创建并填充 ipset（重复执行安全）
sudo ipset create "${SET}" hash:ip -exist
sudo ipset flush "${SET}"
for ip in \
155.98.36.73 155.98.36.111 155.98.36.154 155.98.36.140 155.98.36.62 \
155.98.36.28 155.98.36.121 155.98.36.35 155.98.36.75 155.98.36.92  \
155.98.36.33  155.98.36.251155.98.36.57 155.98.36.120
do
  sudo ipset add "${SET}" "$ip"
done

# 3) 在自定义表里设置经由中转机的默认路由（存在则覆盖）
sudo ip route replace default via "${GW}" table "${TABLE}"

# 4) 把被标记的流量导到自定义表（若已添加则跳过）
sudo ip rule | grep -q "fwmark 0x1 lookup ${TABLE}" \
  || sudo ip rule add fwmark 0x1 lookup "${TABLE}"

# 5) 用 iptables(mangle) 标记“目的地址 ∈ ipset”的本机发包
#    -C 是检查规则是否已存在；不存在再 -A 添加，避免重复
sudo iptables -t mangle -C OUTPUT -m set --match-set "${SET}" dst -j MARK --set-mark 0x1 2>/dev/null \
  || sudo iptables -t mangle -A OUTPUT -m set --match-set "${SET}" dst -j MARK --set-mark 0x1

# 可选：把连接标记保存/恢复（长连接/多包一致性更好）
sudo iptables -t mangle -C OUTPUT -m set --match-set "${SET}" dst -j CONNMARK --save-mark 2>/dev/null \
  || sudo iptables -t mangle -A OUTPUT -m set --match-set "${SET}" dst -j CONNMARK --save-mark
sudo iptables -t mangle -C OUTPUT -j CONNMARK --restore-mark 2>/dev/null \
  || sudo iptables -t mangle -A OUTPUT -j CONNMARK --restore-mark

# 6) 验证（可选）
echo "==== ipset ====" && sudo ipset list "${SET}" | head -n 20
echo "==== ip rule ====" && ip rule show | sed -n '1,200p'
echo "==== route table ====" && ip route show table "${TABLE}"
