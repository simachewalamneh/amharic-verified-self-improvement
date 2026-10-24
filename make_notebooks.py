"""Generates the three task notebooks. Run once: python make_notebooks.py"""
import nbformat as nbf
from pathlib import Path

HEAD = '''import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / "configs").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.utils import load_config
from src.data import load_base, make_split
from src.features import attach_views

DATA_SOURCE = "local"   # "local" (data/raw TSVs), "hf", or "synthetic" (smoke test only)
cfg = load_config(overrides={"data.source": DATA_SOURCE})
base = attach_views(load_base(cfg), cfg)
cfg["_n_classes"] = base.n_classes
print(base.label_names, {k: len(v) for k, v in base.texts.items()})'''


def nb(cells, path):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(n, path)


nb([
    ("md", "# Task 1: Verified recursive self-training (Amharic news topic classification)\n"
           "Loop: generate pseudo-labels -> verify (confidence + independent char-n-gram model) -> retrain a copy -> "
           "accept/rollback on **dev** -> repeat. Gold labels of the pool are used only to *measure* pseudo-label precision. "
           "The test split is used only in the final cell."),
    ("code", HEAD),
    ("code", "from task1_recursive.selftrain import run_selftrain\nSEED = 0\nsplit = make_split(base, SEED, cfg['data']['seed_per_class'])\n"
             "print('seed set:', len(split.seed_idx), 'pool:', len(split.pool_idx))\n"
             "res = run_selftrain(base, split, cfg, SEED)\nhist = pd.DataFrame(res.history)\nhist"),
    ("code", "fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))\nax[0].plot(hist['round'], hist['dev_macro_f1'], marker='o'); ax[0].set_title('Dev macro-F1 per round'); ax[0].set_xlabel('round')\n"
             "ax[1].plot(hist['round'], hist['pl_precision'], marker='o', color='tab:red'); ax[1].set_title('Pseudo-label precision (vs hidden gold)'); ax[1].set_xlabel('round')\nplt.tight_layout(); plt.show()"),
    ("code", "from src.eval import scores\nXt = base.X['teacher']; y = base.y['test']\n"
             "for name, m in [('round-0 teacher', res.teacher_round0), ('final teacher', res.teacher_final)]:\n    print(name, {k: round(v, 4) for k, v in scores(m.proba(Xt['test']), y).items()})"),
    ("md", "Full multi-seed comparison (verified vs confidence-only vs naive): `python -m task1_recursive.run`."),
], "task1_recursive/notebook.ipynb")

nb([
    ("md", "# Task 2: Knowledge distillation\nTeacher (MLP) vs hard-label student vs distilled student, loss "
           "`alpha*CE + (1-alpha)*T^2*KL(teacher||student)`. Sweep T and alpha; select on **dev**; report test, size, latency."),
    ("code", HEAD),
    ("code", "from task2_distillation.distill import run_distillation, select_best\ncfg['task2']['seeds'] = [0, 1, 2]\n"
             "df = run_distillation(base, cfg, cfg['task2']['seeds'])\nbest = select_best(df); print('best on dev:', best)"),
    ("code", "d = df[df.model == 'student_distilled'].groupby(['T', 'alpha'])[['dev_macro_f1', 'test_macro_f1']].mean().round(4); d"),
    ("code", "chosen = df[(df.model == 'student_distilled') & (df['T'] == best['T']) & (df.alpha == best['alpha'])].assign(model='student_distilled(best_on_dev)')\n"
             "comp = pd.concat([df[df.model.isin(['teacher', 'student_hard', 'baseline_charLR'])], chosen])\n"
             "table = comp.groupby('model')[['test_macro_f1', 'test_ece', 'params', 'size_kb', 'latency_ms']].agg(['mean', 'std']).round(4); table"),
    ("code", "m = comp.groupby('model')['test_macro_f1'].mean()\nplt.figure(figsize=(6, 3.5)); m.plot(kind='bar'); plt.ylabel('test macro-F1'); plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.show()"),
    ("md", "Full run that also writes `results/task2/`: `python -m task2_distillation.run`."),
], "task2_distillation/notebook.ipynb")

nb([
    ("md", "# Task 3: Merge - recursive learning, then distillation\nRun the Task 1 loop to obtain a final teacher, distill it into the student (hard labels only on the seed set), "
           "and ask whether the loop's gain survives distillation."),
    ("code", HEAD),
    ("code", "import json\nfrom task3_merged import run as t3\nprint('see task3_merged/run.py for the full pipeline')\n"
             "df = t3.main(['--seeds', '0', '--out', 'results/task3_notebook', '--set', f'data.source={DATA_SOURCE}'])"),
    ("code", "gain = pd.read_csv(ROOT / 'results/task3_notebook/gain_survival.csv'); gain"),
    ("code", "df.groupby('model')[['test_macro_f1', 'test_ece', 'params', 'latency_ms']].mean().round(4)"),
    ("md", "Multi-seed results: `python -m task3_merged.run` (writes `results/task3/`). A null result (the gain does not survive distillation) is a valid finding."),
], "task3_merged/notebook.ipynb")
print("notebooks written")
