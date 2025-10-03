#!/usr/bin/env bash
set -euo pipefail
# Router LAN interface
IFACE="${IFACE:-enp6s0f0}"
# Management IP (10.0.0.1/24)
MGMT_IP="${MGMT_IP:-10.0.0.1/24}"
# 14 subnets for nodes: 10.10.0.0/24 .. 10.23.0.0/24 (gateway .1 on router)
SUBNET_PREFIXES=(10.10 10.11 10.12 10.13 10.14 10.15 10.16 10.17 10.18 10.19 10.20 10.21 10.22 10.23)
# Shared bandwidth cap for all nodes (downstream Router->Nodes)
SHARED_BW="${SHARED_BW:-100mbit}"

echo "[router] iface=$IFACE mgmt=$MGMT_IP shared_bw=$SHARED_BW"

# Bring interface up
ip link set "$IFACE" up

# Clean old qdisc (safe if absent)
tc qdisc del dev "$IFACE" root 2>/dev/null || true

# Assign management IP (10.0.0.1/24)
if ! ip -4 addr show dev "$IFACE" | grep -q " 10\.0\.0\.1/"; then
  ip addr add "$MGMT_IP" dev "$IFACE"
fi

# Assign 14 gateway IPs: 10.X.0.1/24 on the same interface
for p in "${SUBNET_PREFIXES[@]}"; do
  if ! ip -4 addr show dev "$IFACE" | grep -q " ${p}\.0\.1/"; then
    ip addr add "${p}.0.1/24" dev "$IFACE"
  fi
done

# Enable IPv4 forwarding (LAN-only)
sysctl -w net.ipv4.ip_forward=1 >/dev/null
if grep -q '^net.ipv4.ip_forward' /etc/sysctl.conf; then
  sed -i 's/^net.ipv4.ip_forward.*/net.ipv4.ip_forward=1/' /etc/sysctl.conf
else
  echo 'net.ipv4.ip_forward=1' >> /etc/sysctl.conf
fi

# Set shared bandwidth cap on IFACE (egress Router->Nodes)
tc qdisc add dev "$IFACE" root handle 1: htb default 10
tc class add dev "$IFACE" parent 1: classid 1:10 htb rate "$SHARED_BW" ceil "$SHARED_BW"
tc qdisc add dev "$IFACE" parent 1:10 handle 10: fq

echo "[router] configured."
ip -4 addr show dev "$IFACE"
tc -s qdisc show dev "$IFACE"
tc -s class show dev "$IFACE"
