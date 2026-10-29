# amharic-verified-self-improvement

Verified recursive self-training and knowledge distillation for Amharic news topic classification (MasakhaNEWS), with a merged pipeline.
iCog Labs AI Internship, *Emerging AI Paradigms* assignment. Three separate tasks in one repo.

## Data
Amharic MasakhaNEWS, official splits: 1,311 train / 188 dev / 376 test, 4 classes (business, health, politics, sports).
`python data/download.py` fetches the TSVs into `data/raw/`. 

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

## Research extension (added after peer review)
The three tasks above are unchanged. The `research/` package adds the experiments suggested by reviewers; `research.loop` reproduces Task 1's verified variant exactly.

| Module | Purpose |
|---|---|
| `research/loop.py` | generalised loop: selection rules (`threshold`, `cap`, `balanced`), char-LR teacher, temperature scaling, rollback margin, verifier-disagreement logging |
| `research/suite.py` | resumable suites: `main` (10 seeds), `charlr` (10 seeds), `budget` (5/10/40 labels per class), `threshold` (0.5/0.7) |
| `research/diagnostics.py` | round-0 per-class confidence, reliability bins, confusion, teacher-verifier error correlation |
| `research/distill_ext.py` | linear and MLP students, full-label and 80-label conditions, paired tests |
| `research/latency_e2e.py` | end-to-end (vectorise + forward) latency |
| `research/stats.py` | paired bootstrap CI, Wilcoxon, TOST, per-example bootstrap |

```bash
python -m research.suite --suite all        # about 1 hour on one CPU core
python -m research.diagnostics
python -m research.distill_ext --set "task2.temperatures=[2,4]" "task2.alphas=[0.3,0.7]"
python -m research.latency_e2e
python -m task3_merged.run --seeds 3 4 5 6 7 8 9 --out results/task3_seeds3to9
python -m research.stats
python paper/make_tables2.py && (cd paper && pdflatex paper_v2 && bibtex paper_v2 && pdflatex paper_v2 && pdflatex paper_v2)
```

Architecture diagrams for the three tasks: `docs/ARCHITECTURE.md` (PlantUML sources in `docs/*.puml`, PNGs in `docs/img/`).
Revised paper: `paper/paper_v2.tex` / `paper_v2.pdf` (the three-seed version stays in `paper/paper.tex`).

## Headline findings (10 paired seeds, 20 labels per class, MLP teacher, test macro-F1)
- Seed-only 0.843; verified loop equivalent within +-0.01; naive loop 0.697 (precision 0.984, 78% of sports vs 1% of business pool accepted).
- Balanced per-class selection: +0.016 (10/10 seeds). A calibrated char-LR teacher improves with every variant; the second-view verifier adds nothing.
- Distillation: no accuracy change with a full-label teacher, a loss with the 80-label teacher.
