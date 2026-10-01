#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "grow_spell.py" > /dev/null; do sleep 30; done
printf "1\n2\n4\n8\n16\n32\n" | xargs -P 4 -I{} sh -c 'python3 monolith.py {} > results/logs/monolith_{}.log 2>&1'
