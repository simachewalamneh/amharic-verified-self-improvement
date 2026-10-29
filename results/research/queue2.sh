#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
L=results/research/queue2.log
python3 -u -m paper.research.distill_ext --set "task2.temperatures=[2,4]" "task2.alphas=[0.3,0.7]" >> $L 2>&1
[ -f results/task3_seeds3to9/all_runs.csv ] || python3 -u -m task3_merged.run --seeds 3 4 5 6 7 8 9 --out results/task3_seeds3to9 >> $L 2>&1
python3 -u -m paper.research.stats >> $L 2>&1
echo QUEUE2_DONE >> $L
