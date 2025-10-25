#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate multiple machines' layer timing JSON and compute per-device "capacity"
relative to a chosen reference machine.

Usage:
  python aggregate_capacities.py \
      --inputs cpu1.json cpu2.json ... cpu8.json \
      --ref_host cpu1 \
      --out capacities.json

Capacity definition:
  capacity[host] = median_over_layers( tf_ref / tf_host )
  If backward_time_s is available on both, it will average forward/backward ratios.

All measurements must use the SAME batch size and seq_len.
"""

import argparse, json, statistics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    #we just let the first one becomes reference host...
    #ap.add_argument("--ref_host", required=True)
    ap.add_argument("--out", default="capacities.json")
    args = ap.parse_args()

    datas = [json.load(open(p,"r")) for p in args.inputs]
    by_host = {}
    for i in range(len(datas)):
        name = str(datas[i]["host"]) + f'_{i}'
        by_host[name] = datas[i]
    #{d["host"]: d for d in datas}
    #assert args.ref_host in by_host, "ref_host not in inputs"


    ref = by_host[name]
    ref_host = name
    B = ref["batch"]; T = ref["seq_len"]

    caps = {}
    for host, d in by_host.items():
        assert d["batch"] == B and d["seq_len"] == T, "All inputs must share batch and seq_len"
        r = []
        for lr, lh in zip(ref["layers"], d["layers"]):
            if lr["forward_time_s"] > 0 and lh["forward_time_s"] > 0:
                r.append(lr["forward_time_s"] / max(lh["forward_time_s"], 1e-9))
        caps[host] = float(statistics.median(r)) if r else 1.0

    json.dump({"batch": B, "seq_len": T, "ref_host": ref_host, "capacity": caps}, open(args.out,"w"), indent=2)

if __name__ == "__main__":
    main()
