#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
for s in main charlr budget threshold; do
  python3 -u -m research.suite --suite $s >> results/research/queue1.log 2>&1
done
echo QUEUE1_DONE >> results/research/queue1.log
