#!/bin/sh
# Exp 5 queue: waits for after.sh (scale + grow), then runs open-ended acquisition.
cd "$(dirname "$0")"
while pgrep -f "after.sh" > /dev/null; do sleep 30; done
for s in 0 1 2; do for m in smart dup scratch; do echo "acquire --mode $m --seed $s"; done; done |
  xargs -P 4 -I{} sh -c 'python3 run.py {} > "results/logs/$(echo {} | tr " " _ | tr -d -).log" 2>&1'
