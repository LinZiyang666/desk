#!/usr/bin/env python3
import argparse, json, os, sys
from pathlib import Path

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
    """Infer (model_name, frequency, layers, mb_size) from the directory layout.
       Works whether root is the global root or already a model_name dir.
    """
    try:
        rel_parts = leaf.relative_to(root).parts
    except ValueError:
        # leaf is not under root (shouldn't happen)
        return None

    # We expect rel_parts to end with [frequency, layers, mb_size]
    # If rel_parts has 4+ parts, assume last 4 are [model_name, frequency, layers, mb_size]
    # If rel_parts has exactly 3, assume root.name is model_name
    if len(rel_parts) >= 4:
        model_name, frequency, layers, mb_size = rel_parts[-4:]
    elif len(rel_parts) == 3:
        model_name = root.name
        frequency, layers, mb_size = rel_parts[-3:]
    else:
        return None

    return {
        "model_name": model_name,
        "frequency": frequency,
        "layers": layers,
        "mb_size": mb_size
    }

def main():
    ap = argparse.ArgumentParser(description="Traverse profiles and extract metrics into JSON.")
    ap.add_argument("--root", type=str, required=True, help="Root directory containing model_name/{frequency}/{layers}/{mb_size} structure, or a single model_name directory.")
    ap.add_argument("--output", type=str, default="profiles_summary.json", help="Where to write the combined JSON output.")
    ap.add_argument("--timeline-name", type=str, default="timeline_rank0.json", help="Timeline filename to parse in each leaf dir.")
    ap.add_argument("--cpu-name", type=str, default="cpu_mem_rank0.jsonl", help="CPU jsonl filename to parse in each leaf dir.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root path not found: {root}", file=sys.stderr)
        sys.exit(1)

    results = []
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

            results.append({
                **keys,
                "path": str(dp),
                "total_time_ms": total_ms,
                "avg_cpu_ratio": cpu_ratio
            })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} records to {args.output}")
    if len(results) == 0:
        print("Hint: Ensure your tree looks like model_name/{frequency}/{layers}/{mb_size}/ and files are named exactly as arguments.", file=sys.stderr)

if __name__ == "__main__":
    main()
