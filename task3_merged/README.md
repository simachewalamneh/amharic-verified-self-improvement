# Task 3: Merge
Run the Task 1 loop for a final teacher, then distill round-0 and final teachers into the Task 2 student architecture using (T, alpha) from Task 2.
Hard labels exist only for the seed set during distillation; soft targets cover every training input.
Also reports a seed-only hard student, a hard-pseudo-label student, and Task 2-style full-label references. `results/task3/gain_survival.csv` gives per-seed (final - round0) gains for teachers and students.
Run: `python -m task3_merged.run`.
