#!/usr/bin/env python3
import argparse, json, os, sys
from pathlib import Path
from typing import List
import torch, torch.distributed as dist

def compute_total_time_ms_from_timeline(timeline_path: Path) -> float | None:
    try:
        with open(timeline_path, "r") as f:
            events = json.load(f)
    except Exception:
        return None
    starts = [e.get("start_ns") for e in events if e.get("action") == "FORWARD" and e.get("mb_idx") == 0 and isinstance(e.get("start_ns"), int)]
    ends = [e.get("end_ns") for e in events if e.get("action") == "FULL_BACKWARD" and e.get("mb_idx") == 0 and isinstance(e.get("end_ns"), int)]
    if not starts or not ends:
        return None
    start_ns = min(starts)
    end_ns = max(ends)
    return (end_ns - start_ns) / 1e6

def compute_avg_cpu_ratio_from_jsonl(cpu_jsonl_path: Path) -> float | None:
    vals = []
    try:
        with open(cpu_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = obj.get("cpu_utilization")
                if isinstance(v, (int, float)) and v <= 50:
                    vals.append(float(v))
    except FileNotFoundError:
        return None
    if not vals:
        return None
    avg = sum(vals) / len(vals)
    return avg / 50.0

def infer_keys(root: Path, leaf: Path) -> dict | None:
    """Infer (model_name, cap, layers, mb_size) from the directory layout.
       Layout: model_name/{cap}/{layers}/{mb_size}
       where {cap} looks like 'cap30.0' (保留一位小数)
    """
    try:
        rel_parts = leaf.relative_to(root).parts
    except ValueError:
        # leaf is not under root (shouldn't happen)
        return None

    # We expect rel_parts to end with [cap, layers, mb_size]
    # If rel_parts has 4+ parts, assume last 4 are [model_name, cap, layers, mb_size]
    # If rel_parts has exactly 3, assume root.name is model_name
    if len(rel_parts) >= 4:
        model_name, cap_dir, layers, mb_size = rel_parts[-4:]
    elif len(rel_parts) == 3:
        model_name = root.name
        cap_dir, layers, mb_size = rel_parts[-3:]
    else:
        return None

    # 解析 cap 目录名，如 "cap30.0"
    cap_str = cap_dir
    cap_val = None
    if cap_dir.startswith("cap"):
        try:
            cap_val = float(cap_dir[3:])
        except ValueError:
            cap_val = None

    return {
        "model_name": model_name,
        "cap_dir": cap_str,
        "cap": cap_val,
        "layers": layers,
        "mb_size": mb_size,
    }

# ----------------- 分布式工具 -----------------
def _dist_available() -> bool:
    return dist is not None

def _maybe_init_dist():
    """在 torchrun 环境下用 env:// 初始化；若已初始化则跳过；若无 torch 则跳过。"""
    if not _dist_available():
        return False
    if dist.is_available() and not dist.is_initialized():
        # 优先用 gloo（CPU 环境更通用）；如果你确保 NCCL 可用可改为 nccl
        try:
            dist.init_process_group(backend="gloo", init_method="env://")
        except Exception:
            # 可能并非 torchrun 启动，忽略
            return False
    return dist.is_initialized()

def _rank() -> int:
    if not _dist_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def _world_size() -> int:
    if not _dist_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


ap = argparse.ArgumentParser(description="Traverse profiles and extract metrics into JSON (distributed-friendly).")
ap.add_argument(
    "--root",
    type=str,
    required=True,
    help="Root directory containing model_name/{cap}/{layers}/{mb_size} structure, or a single model_name directory.",
)
ap.add_argument("--output", type=str, default="profiles_summary.json", help="Where to write the combined JSON output.")
ap.add_argument("--timeline-name", type=str, default="timeline_rank0.json", help="Timeline filename to parse in each leaf dir.")
ap.add_argument("--cpu-name", type=str, default="cpu_mem_rank0.jsonl", help="CPU jsonl filename to parse in each leaf dir.")
ap.add_argument(
    "--only-rank0-write",
    action="store_true",
    help="If set (default), only rank0 writes the final merged JSON (recommended under torchrun).",
    default=True,
)
args = ap.parse_args()
def main():
    
    _maybe_init_dist()
    rank = _rank()
    world = _world_size()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root path not found: {root}", file=sys.stderr)
        sys.exit(1)

    local_results: List[dict] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        # Only consider directories that contain both files
        if args.timeline_name in filenames and args.cpu_name in filenames:
            keys = infer_keys(root, dp)
            if not keys:
                continue
            timeline_path = dp / args.timeline_name
            cpu_path = dp / args.cpu_name

            total_ms = compute_total_time_ms_from_timeline(timeline_path)
            cpu_ratio = compute_avg_cpu_ratio_from_jsonl(cpu_path)

            local_results.append({
                **keys,
                "path": str(dp),
                "total_time_ms": total_ms,
                "avg_cpu_ratio": cpu_ratio
            })

    # 分布式聚合
    if _dist_available() and dist.is_initialized():
        gathered = [None for _ in range(world)]
        # all_gather_object 会把每个 rank 的列表收集到 gathered
        dist.all_gather_object(gathered, local_results)
        if rank == 0:
            merged = []
            for part in gathered:
                if isinstance(part, list):
                    merged.extend(part)
        else:
            merged = None
    else:
        merged = local_results

    # 仅 rank0 写文件（默认）
    if (not args.only_rank0_write) or (rank == 0):
        with open(args.output, "w") as f:
            json.dump(merged or [], f, indent=2)
        print(f"[rank {rank}/{world}] Wrote {len(merged or [])} records to {args.output}")
        if not merged:
            print("Hint: Ensure your tree looks like model_name/{cap}/{layers}/{mb_size}/ and files are named exactly as arguments.", file=sys.stderr)
    else:
        print(f"[rank {rank}/{world}] Local results: {len(local_results)} (not writing, rank0 will).")



if __name__ == "__main__":
    main()
