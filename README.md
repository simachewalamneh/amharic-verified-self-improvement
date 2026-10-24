# amharic-verified-self-improvement

Verified recursive self-training and knowledge distillation for Amharic news topic classification (MasakhaNEWS), with a merged pipeline.
iCog Labs AI Internship, *Emerging AI Paradigms* assignment. Three separate tasks in one repo.

## Data
Amharic MasakhaNEWS, official splits: 1,311 train / 188 dev / 376 test, 4 classes (business, health, politics, sports).
`python data/download.py` fetches the TSVs into `data/raw/`. **License not verified**: check the dataset card/paper before redistributing.

Protocol: seed set = 20 labelled articles per class (80); rest of train = unlabeled pool (gold labels used only to *measure* pseudo-label precision);
dev decides accept/rollback and (T, alpha) selection; test is used only for final reporting.

## Tasks (run from the repo root)
| Task | Command | Output |
|---|---|---|
| 1 Recursive learning | `python -m task1_recursive.run` | `results/task1/` |
| 2 Knowledge distillation | `python -m task2_distillation.run` | `results/task2/` |
| 3 Merge (loop, then distill) | `python -m task3_merged.run` (reads Task 2's best T, alpha) | `results/task3/` |

Run Task 2 before Task 3. Add `--seeds 0` for a quick run, `--set data.source=synthetic` for a smoke test (never report synthetic results).
Report: `report/report.tex` + `report/report.pdf` (regenerate with `python report/make_report.py`, then `pdflatex`).

## Results (TF-IDF features, 3 seeds, test macro-F1)
- Task 1: verified loop 0.841 vs round-0 teacher 0.846 (no gain); naive loop without rollback 0.713 (degradation despite ~0.97-1.0 pseudo-label precision).
- Task 2: teacher 0.908 (2.56M params), hard student 0.901, distilled student 0.899 (8,004 params, ~3.4x faster). Distillation did not beat hard labels on F1; it lowered ECE.
- Task 3: the loop's gain did not appear on test, so there is nothing to survive distillation (null result).
- A character n-gram logistic regression is a strong baseline (0.925 with all labels, 0.874 with the seed set only).

## Caveats
- AfroXLMR (`features.backend: afroxlmr`) and Hugging Face (`data.source: hf`) paths are implemented but untested.
- Notebooks (`task*/notebook.ipynb`, regenerate with `python make_notebooks.py`) call the same code as the runners.
- `research/` is reserved for the research extension (not implemented yet).
