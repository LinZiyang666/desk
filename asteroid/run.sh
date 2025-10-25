#!/bin/bash

# Usage:
#   ./run_aggregate.sh CPU100_bs5 CPU70_bs4 ...

if [ $# -eq 0 ]; then
  echo "Usage: $0 <#mbatch> <name1> <name2> ..."
  exit 1
fi


NBATCH=$1
shift 

inputs=""
pureinputs=""
for f in "$@"; do
  pureinputs="$pureinputs ${f}"
  inputs="$inputs ../Profile/${f}.json"
done

LAST_INPUT=${!#}   # the last argument, e.g., CPU100_bs6
lastinput="../Profile/${LAST_INPUT}.json"

echo "inputs = $inputs"
echo "last input = $lastinput"

echo "Running:..."
python3 aggregate_capacities.py --inputs $inputs
python3 parse_bandwidth_csv.py --inputs $inputs
python3 aggregate_mem_profile.py --inputs $pureinputs

python3 plan_from_measured.py \
  --layers $lastinput \
  --capacities capacities.json \
  --bandwidth bandwidth_matrix.json \
  --memory_model memory_model.json \
  --devices devices.json \
  --tied_embed true \
  --max_stages None \
  --out_dir out_measured \
  --nbatch $NBATCH