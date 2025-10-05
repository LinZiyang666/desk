#!/usr/bin/env bash
# 限制“转发(Transit)”流量的最大带宽（Mbps），不影响本机OUTPUT流量
# 依赖：iproute2, iptables, tc，内核需支持 ifb、sch_htb、sch_fq_codel、act_mirred、cls_fw 等

set -euo pipefail

#==== 参数 ====
WAN_IF="${WAN_IF:-eno1}"         # 外网口（你的机器：eno1）
MAX_Mbps="${1:-100}"             # 最大带宽（Mbps），可用参数传入：./set_transit_bw.sh 300
MARK_HEX="0x66"                  # fwmark，用于区分转发流量
IFB_DEV="${IFB_DEV:-ifb0}"       # IFB 名称
ROOT_HTB_ID="1:"
ROOT_INGRESS_ID="ffff:"
CLS_TRANSIT="1:66"
DEFAULT_CLASS="1:10"

#==== 检查 ====
if [[ $EUID -ne 0 ]]; then
  echo "请用 root 运行" >&2
  exit 1
fi

command -v tc >/dev/null || { echo "缺少 tc (iproute2)"; exit 1; }
command -v iptables >/dev/null || { echo "缺少 iptables"; exit 1; }

#==== 开启路由转发 ====
sysctl -w net.ipv4.ip_forward=1 >/dev/null

#==== 给“转发流量”打标（FORWARD链），只标转发，不标本机流量 ====
# 幂等：先检查，不存在再添加
if ! iptables -t mangle -C FORWARD -j MARK --set-mark ${MARK_HEX} >/dev/null 2>&1; then
  iptables -t mangle -A FORWARD -j MARK --set-mark ${MARK_HEX}
fi
# 保存/恢复连接标记，确保长连接后续包一致
if ! iptables -t mangle -C FORWARD -j CONNMARK --save-mark >/dev/null 2>&1; then
  iptables -t mangle -A FORWARD -j CONNMARK --save-mark
fi
if ! iptables -t mangle -C FORWARD -j CONNMARK --restore-mark >/dev/null 2>&1; then
  iptables -t mangle -A FORWARD -j CONNMARK --restore-mark
fi

#==== egress 整形（上行：从路由器发往外网）====
# 清理旧的 qdisc（幂等）
tc qdisc del dev "${WAN_IF}" root 2>/dev/null || true

# 根 HTB
tc qdisc add dev "${WAN_IF}" root handle ${ROOT_HTB_ID} htb default ${DEFAULT_CLASS#1:}

# 默认类（未打标或非转发等走这里，不限速；如果你想对未分类也限速，可把 rate 也设置为 MAX）
tc class add dev "${WAN_IF}" parent ${ROOT_HTB_ID} classid ${DEFAULT_CLASS} htb rate 10000mbit ceil 10000mbit

# 转发类（受限速）
tc class add dev "${WAN_IF}" parent ${ROOT_HTB_ID} classid ${CLS_TRANSIT} htb rate ${MAX_Mbps}mbit ceil ${MAX_Mbps}mbit

# 队列算法
tc qdisc add dev "${WAN_IF}" parent ${CLS_TRANSIT} fq_codel
tc qdisc add dev "${WAN_IF}" parent ${DEFAULT_CLASS} fq_codel

# fwmark 过滤到转发类
# 注意：cls_fw 匹配 fwmark
tc filter add dev "${WAN_IF}" parent ${ROOT_HTB_ID} protocol ip handle ${MARK_HEX} fw flowid ${CLS_TRANSIT}
tc filter add dev "${WAN_IF}" parent ${ROOT_HTB_ID} protocol ipv6 handle ${MARK_HEX} fw flowid ${CLS_TRANSIT} 2>/dev/null || true

#==== ingress 整形（下行：外网进来发往内网）====
# 需要 IFB 设备
modprobe ifb numifbs=1 2>/dev/null || true
ip link add "${IFB_DEV}" type ifb 2>/dev/null || true
ip link set dev "${IFB_DEV}" up

# 清理旧的 ingress/ifb 配置
tc qdisc del dev "${WAN_IF}" ingress 2>/dev/null || true
tc qdisc del dev "${IFB_DEV}" root 2>/dev/null || true

# 在 WAN_IF 挂 ingress qdisc，并把进入的包重定向到 ifb
tc qdisc add dev "${WAN_IF}" handle ${ROOT_INGRESS_ID} ingress
# 把 ingress 包（我们也顺带设置个 mark）重定向到 ifb
tc filter add dev "${WAN_IF}" parent ${ROOT_INGRESS_ID} protocol all u32 match u32 0 0 \
  action skbedit mark ${MARK_HEX} \
  action mirred egress redirect dev "${IFB_DEV}"

# 在 IFB 上做和 egress 同样的 HTB 限速
tc qdisc add dev "${IFB_DEV}" root handle 2: htb default 10
tc class add dev "${IFB_DEV}" parent 2: classid 2:10 htb rate 10000mbit ceil 10000mbit
tc class add dev "${IFB_DEV}" parent 2: classid 2:66 htb rate ${MAX_Mbps}mbit ceil ${MAX_Mbps}mbit
tc qdisc add dev "${IFB_DEV}" parent 2:10 fq_codel
tc qdisc add dev "${IFB_DEV}" parent 2:66 fq_codel
tc filter add dev "${IFB_DEV}" parent 2: protocol ip handle ${MARK_HEX} fw flowid 2:66
tc filter add dev "${IFB_DEV}" parent 2: protocol ipv6 handle ${MARK_HEX} fw flowid 2:66 2>/dev/null || true

echo "[OK] Transit 限速已生效：${MAX_Mbps} Mbps （双向，接口=${WAN_IF}, ifb=${IFB_DEV})"
echo "== 检查命令 =="
echo "tc -s class show dev ${WAN_IF}"
echo "tc -s class show dev ${IFB_DEV}"
echo "iptables -t mangle -S FORWARD | sed -n '1,200p'"
