#!/bin/sh
# Runs after the main/curriculum queue: scaling (Exp 3) then growth (Exp 4).
cd "$(dirname "$0")"
while pgrep -f "^xargs -P" > /dev/null || pgrep -f "^python3 run.py (main|curriculum)" > /dev/null; do sleep 20; done
python3 run.py scale > results/logs/scale.log 2>&1
for s in 0 1 2; do
  python3 run.py grow --seed $s > results/logs/grow_hier_$s.log 2>&1 &
  python3 run.py grow --flat --seed $s > results/logs/grow_flat_$s.log 2>&1 &
done
wait
