#!/usr/bin/env python3
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="List of names, e.g., CPU100 CPU100 CPU100")
    parser.add_argument("--out", default="devices.json",
                        help="Output JSON file")
    parser.add_argument("--value", type=int, default=55,
                        help="Value to assign to each entry (default: 60)")
    args = parser.parse_args()

    data = {}
    # enumerate all inputs, add suffix
    for i, name in enumerate(args.inputs):
        base = name.split("_")[0]   # take only the first part
        key = f"{base}_{i}"
        data[key] = args.value

    # write JSON
    with open(args.out, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()