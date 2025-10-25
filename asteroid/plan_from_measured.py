#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run your HPP planner using ONLY measured data.

Inputs:
  - layers_ref.json         (from measure_layers.py on the reference machine)
  - capacities.json         (from aggregate_capacities.py)
  - bandwidth_matrix.json   (from parse_bandwidth_csv.py)
  - memory_model.json       (from calibrate_memory_model.py)
  - devices.json            (your 8 devices' memory budgets, e.g. {"cpu1":24,...} in GB)
  - tied_embed: true/false  (whether lm_head shares weights with embed)

Usage:
  python plan_from_measured.py \
      --layers layers_cpu1.json \
      --capacities capacities.json \
      --bandwidth bandwidth_matrix.json \
      --memory_model memory_model.json \
      --devices devices.json \
      --tied_embed true \
      --max_stages  None \
      --out_dir out_measured

This file calls your Algorithm 2 (which calls Algorithm 1).
"""

import argparse, json
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd

from asteroid_Hpp_plan import Device, plan_hpp_dynamic
from asteroid_Batch_plan import allocate_microbatch_samples

def load_bandwidth_matrix(path: str) -> Dict[Tuple[str,str], float]:
    raw = json.load(open(path,"r"))
    d = {}
    for k,v in raw["bytes_per_sec"].items():
        src, dst = k.split("|")
        d[(src, dst)] = float(v)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--capacities", required=True)
    ap.add_argument("--bandwidth", required=True)
    ap.add_argument("--memory_model", required=True)
    ap.add_argument("--devices", required=True, help='JSON like {"cpu1":24, ...} in GB')
    ap.add_argument("--tied_embed", type=str, default="true", choices=["true","false"])
    ap.add_argument("--max_stages", default=None)
    ap.add_argument("--out_dir", default="out_measured")
    ap.add_argument("--nbatch", default=5)
    args = ap.parse_args()

    layers = json.load(open(args.layers,"r"))
    caps = json.load(open(args.capacities,"r"))
    bm = load_bandwidth_matrix(args.bandwidth)
    mm = json.load(open(args.memory_model,"r"))
    mem_gb = json.load(open(args.devices,"r"))

    T = layers["seq_len"]; B = layers["batch"]
    tied = (args.tied_embed == "true")

    hosts = list(mem_gb.keys())
    devices = [Device(name=h, memory_budget=int(float(mem_gb[h])*(1024**3)), capacity=float(caps["capacity"].get(h,1.0))) for h in hosts]
    dev_caps = {d.name: d.capacity for d in devices}

    # Build per-layer bytes from measurement
    ws = [lr["param_bytes"] for lr in layers["layers"]]
    as_ = [lr["activation_bytes_per_sample"] for lr in layers["layers"]]

    # Fold head/tail params into first/last
    ws[0]  += layers.get("embed_param_bytes", 0)
    tail = layers.get("tail_param_bytes", 0)
    if tied:
        # if tied, avoid double-counting; keep only norm part from tail (heuristic)
        tail = 0
    ws[-1] += tail
    # No outgoing boundary on last
    as_[-1] = 0

    L = len(ws)

    # Latency function built from measured tf/tb on the REF host scaled by per-device capacity
    tf_ref = [max(1e-9, lr["forward_time_s"]) for lr in layers["layers"]]
    tb_ref = [max(1e-9, lr["backward_time_s"] if lr["backward_time_s"]>0 else lr["forward_time_s"]*2.0) for lr in layers["layers"]]  # fallback if bwd missing

    def latency_of_layer(dev_name: str, layer_idx: int, batch: int):
        #assert batch == B, "Measured times are for batch=%d; please measure again for batch=%d" % (B, batch)
        scale = batch/B
        cap = max(dev_caps.get(dev_name,1.0), 1e-6)
        return tf_ref[layer_idx] / cap *scale, tb_ref[layer_idx] / cap*scale

    # Memory model fitted from measurements
    base = float(mm["base"]); alpha = float(mm["alpha"]); Kp = float(mm["Kp"])
    def memory_of_stage(dev_name: str, span: Tuple[int,int], batch: int) -> int:
        s,e = span
        sum_w = sum(ws[i]  for i in range(s,e))
        sum_a = sum(as_[i] for i in range(s,e))
        return int(base + alpha*sum_w + Kp*sum_a*batch)

    # Run planner
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_hpp_dynamic(
        num_layers=L,
        weights_size=ws,
        activation_size=as_,
        devices=devices,
        bandwidth_matrix=bm,
        micro_batch_size=B,
        num_micro_batches=layers.get("num_micro_batches", int(args.nbatch)),
        latency_of_layer=latency_of_layer,
        memory_of_stage=memory_of_stage,
        allocate_fn=allocate_microbatch_samples,
        block_size=1,
        max_stages=None if args.max_stages in (None,"None","none") else int(args.max_stages),
    )

    # Write outputs
    all_dev_names = [d.name for d in devices]
    stage_rows = []
    for i, st in enumerate(plan.stages):
        row = {
            "stage_id": i,
            "layers": f"[{st.layer_start}, {st.layer_end})",
            "devices": ",".join(st.device_names),
            "Ef(s)": round(st.Ef, 6),
            "Eb(s)": round(st.Eb, 6),
            "Ta(s)": round(st.Ta, 6),
        }
        for name in all_dev_names:
            row[f"y[{name}]"] = st.y_allocation.get(name, 0)
        stage_rows.append(row)
    pd.DataFrame(stage_rows).to_csv(out_dir / "stages.csv", index=False)

    Ef, Eb, Ta = plan.meta["Ef"], plan.meta["Eb"], plan.meta["Ta"]
    Tw, Te, totals = plan.meta["Tw"], plan.meta["Te"], plan.meta["totals"]
    dm = plan.meta["dominant_step_index"]

    steps = []
    for s in range(len(Ef)):
        steps.append({
            "step_id": s,
            "Ef": round(Ef[s], 6),
            "Eb": round(Eb[s], 6),
            "Ta": round(Ta[s], 6),
            "Tw": round(Tw[s], 6),
            "Te": round(Te[s], 6),
            "Tw+Te+Ta": round(totals[s], 6),
            "dominant": bool(s == dm),
        })
    pd.DataFrame(steps).to_csv(out_dir / "steps.csv", index=False)

    meta = {
        "round_latency_s": plan.round_latency,
        "dominant_step_index": plan.meta["dominant_step_index"],
        "Ef": Ef, "Eb": Eb, "Ta": Ta,
        "Tw": Tw, "Te": Te, "totals": totals,
    }
    json.dump(meta, open(out_dir / "plan_meta.json","w"), indent=2)

    lines = []
    lines.append(f"Chosen number of stages: {len(plan.stages)}")
    lines.append(f"Pipeline round latency (objective): {plan.round_latency:.6f} seconds")
    lines.append("Stages:")
    for i, st in enumerate(plan.stages):
        lines.append(f"  - Stage {i}: layers [{st.layer_start}, {st.layer_end}), Ef={st.Ef:.6f}, Eb={st.Eb:.6f}, Ta={st.Ta:.6f}")
        for name in st.device_names:
            lines.append(f"      y[{name}] = {st.y_allocation.get(name, 0)}")
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
