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

def main():
    ap = argparse.ArgumentParser(description="Traverse profiles and extract metrics into JSON.")
    ap.add_argument("--root", type=str, required=True, help="Root directory containing model_name/{frequency}/{layers}/{mb_size} structure.")
    ap.add_argument("--output", type=str, default="profiles_summary.json", help="Where to write the combined JSON output.")
    ap.add_argument("--timeline-name", type=str, default="timeline_rank0.json", help="Timeline filename to parse in each leaf dir.")
    ap.add_argument("--cpu-name", type=str, default="cpu_mem_rank0.jsonl", help="CPU jsonl filename to parse in each leaf dir.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root path not found: {root}", file=sys.stderr)
        sys.exit(1)

    results = []
    # Expected: model_name/{frequency}/{layers}/{mb_size}
    for model_dir in [p for p in root.iterdir() if p.is_dir()]:
        model_name = model_dir.name
        for freq_dir in [p for p in model_dir.iterdir() if p.is_dir()]:
            frequency = freq_dir.name
            for layer_dir in [p for p in freq_dir.iterdir() if p.is_dir()]:
                layers = layer_dir.name
                for mb_dir in [p for p in layer_dir.iterdir() if p.is_dir()]:
                    mb_size = mb_dir.name
                    timeline_path = mb_dir / args.timeline_name
                    cpu_path = mb_dir / args.cpu_name

                    total_ms = compute_total_time_ms_from_timeline(timeline_path) if timeline_path.exists() else None
                    cpu_ratio = compute_avg_cpu_ratio_from_jsonl(cpu_path) if cpu_path.exists() else None

                    results.add if False else None  # keep 'results' referenced for readability
                    results.append({
                        "model_name": model_name,
                        "frequency": frequency,
                        "layers": layers,
                        "mb_size": mb_size,
                        "path": str(mb_dir),
                        "total_time_ms": total_ms,
                        "avg_cpu_ratio": cpu_ratio
                    })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} records to {args.output}")

if __name__ == "__main__":
    main()
