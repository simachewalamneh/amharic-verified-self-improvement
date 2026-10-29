#!/usr/bin/env bash
# Resumable: suites skip runs already in the jsonl files. Safe to relaunch.
cd "$(dirname "$0")/../.."
L=results/research/queue_all.log
for s in main charlr budget threshold; do
  python3 -u -m paper.research.suite --suite $s >> $L 2>&1
done
python3 -u -m paper.research.diagnostics >> $L 2>&1
python3 -u -m paper.research.latency_e2e >> $L 2>&1
python3 -u -m paper.research.distill_ext >> $L 2>&1
[ -f results/task3_seeds3to9/all_runs.csv ] || python3 -u -m task3_merged.run --seeds 3 4 5 6 7 8 9 --out results/task3_seeds3to9 >> $L 2>&1
python3 -u -m paper.research.stats >> $L 2>&1
echo QUEUE_ALL_DONE >> $L
