# Does local learning preserve plasticity?

**Forward-Forward versus backpropagation in continual learning** — code, notebooks and complete results.

Dedi Irawan, Sudarmaji, Imam Samsudin · Faculty of Computer Science, Muhammadiyah University of Metro, Indonesia
Manuscript submitted to *Neural Networks* (Elsevier).

---

## In one paragraph

Neural networks trained by backpropagation slowly lose the ability to learn new tasks when training never stops (*loss of plasticity*). We ask whether this is caused by end-to-end gradient propagation, by comparing backpropagation with the Forward-Forward algorithm, a local learning rule. A ladder of controlled conditions changes one factor at a time:

| Condition | What it is | Isolates |
|---|---|---|
| **BP** | MLP + softmax, trained end to end | — |
| **BP-N** | BP + the normalization between layers used by FF | normalization |
| **BP-G** | FF objective (layer-wise goodness, label in the input), **global** gradients | objective |
| **FF** | Forward-Forward: same as BP-G, but gradients stay inside each layer | locality |
| **BP + reset**, **FF + reset** | + utility-based unit reinitialization | remedy |
| **DFA**, **DFA-N** | direct feedback alignment (± normalization) | another local rule |
| **LC-N** vs **GC-N** | layer-wise local classifiers vs the same heads with global gradients | locality under cross-entropy |

**Main findings.** In the memorization regime, BP collapsed to chance level with Adam and with SGD, in an MLP and in a CNN. FF lost plasticity at its tuned learning rate, more gradually than BP; at matched learning rates it never retained more than BP-G. Two further local rules (normalized DFA and layer-wise local classifiers) also lost more plasticity than their global counterparts. Normalization between layers protected plasticity in the memorization regime but not on a continual CIFAR-100 benchmark, where plasticity loss was milder. Retained plasticity tracked the effective rank of hidden representations.

---

## Quick start: rebuild every table and figure (no GPU, < 1 minute)

```bash
git clone <this repository URL>
cd ff-plasticity
pip install -r requirements.txt
python src/make_paper_outputs.py
```

The script reads the logs in `results/` and writes all tables to `tables/` and all figures to `figures/`. The files already in those folders were produced this way.

---

## What is in this repository

```
ff-plasticity/
├── README.md                  this file
├── LICENSE                    MIT (code) + CC BY 4.0 (results, tables, figures)
├── CITATION.cff               how to cite
├── requirements.txt           Python packages
│
├── src/
│   ├── ffplast.py             experiment library: task streams, the six conditions (MLP and CNN),
│   │                          metrics, resumable training, statistics
│   ├── paper_figures.py       figure style (colour-blind-safe palette, journal widths)
│   └── make_paper_outputs.py  rebuilds all tables and figures from results/
│
├── notebooks/                 Google Colab notebooks that produced the results (free T4 GPU)
│   ├── 01_control_regime_permuted_streams.ipynb   control regime (abundant data per task)
│   ├── 02_main_random_label_bpn_width.ipynb       main experiment, BP-N control, width ablation
│   ├── 03_ff_learning_rate_sensitivity.ipynb      FF at a smaller learning rate
│   ├── 04_robustness_sgd_cnn.ipynb                SGD with momentum and a CNN
│   └── 05_other_local_rules_cifar100.ipynb        DFA, local classifiers, Continual CIFAR-100
│
├── results/                   raw per-task logs of every run (one file per experiment)
│   ├── main_random_label/      runs.jsonl + config.json, pilot.json, plan.json, lr_choice.json
│   ├── control_abundant_data/  runs.jsonl + config.json, plan.json, lr_choice.json
│   ├── lr_sensitivity/         runs.jsonl
│   ├── robustness_sgd_cnn/     mlp_sgdm/runs.jsonl, cnn_adam/runs.jsonl, lr_choice.json
│   ├── other_local_rules/      runs.jsonl + lr_choice.json (DFA, DFA-N, LC-N, GC-N)
│   └── continual_cifar100/     runs.jsonl + lr_choice.json (two new CIFAR-100 classes per task)
│
├── tables/                    CSV/TXT tables of the paper
└── figures/                   Figures 1–5, A.1, A.2 (PDF + PNG, 300 dpi)
```

### Where each paper item comes from

| Paper item | File |
|---|---|
| Table 4 (plasticity, main regime) | `tables/table4_plasticity.csv` |
| Table 5 (pairwise tests) | `tables/table5_tests.csv` |
| Spearman correlations (Section 5.6) | `tables/spearman.txt` |
| Table A.1 (learning rates) | `tables/tableA1_learning_rates.csv` |
| Table A.2 (FF learning-rate sensitivity) | `tables/tableA2_lr_sensitivity.csv`, `tables/tableA2_lr_sensitivity_tests.csv` |
| Table A.3 and Section 5.9 (SGD, CNN) | `tables/tableA3_robustness.csv`, `tables/tableA3_robustness_tests.csv`, `tables/spearman_robustness.txt` |
| Table A.4 and Section 5.10 (other local rules) | `tables/tableA4_other_local_rules.csv`, `tables/tableA4_other_local_rules_tests.csv` |
| Table A.5 and Section 5.11 (Continual CIFAR-100) | `tables/tableA5_continual_cifar100.csv`, `tables/tableA5_continual_cifar100_tests.csv` |
| Table A.6 (control regime) | `tables/tableA6_control_regime.csv` |
| Fig. 1–5, Fig. A.1–A.4 | `figures/Figure_1` … `Figure_5`, `figures/Figure_A1` … `figures/Figure_A4` |

---

## Data format

Each `runs.jsonl` file holds one JSON object per line, and each line is **one task of one run**:

| Field | Meaning |
|---|---|
| `run` | run identifier, e.g. `rlmnist_ff_w500_s3` = stream, condition, width, seed |
| `bench`, `method`, `width`, `seed`, `lr` | stream (`rlmnist`, `rlcifar`, `c100pairs`, …), condition (`bp`, `bp_cbp`, `bpn`, `bpg`, `ff`, `ff_cbp`, `dfa`, `dfan`, `lcn`, `gcn`), hidden width, seed, learning rate |
| `arch`, `optimizer` | `mlp`/`cnn`, `adam`/`sgdm` (robustness runs only) |
| `task` | task index (0 = first task) |
| `acc_end`, `acc_q1` | accuracy at the end of the task and after one quarter of its training |
| `dormant`, `erank`, `wmag` | per hidden layer: fraction of dormant units, effective rank, mean absolute weight |
| `dormant_neutral`, `erank_neutral` | the same, probed with a neutral label code (goodness-based networks only) |
| `goodness_sep` | mean goodness of positive minus negative data (goodness-based networks) |
| `n_reset` | cumulative number of reinitialized units (reset conditions) |

Runs whose identifier starts with `tune_`, `cal_` or `pilot_` belong to learning-rate selection, timing calibration or the pilot run and are excluded from the paper's statistics. The robustness tables use seeds 0–2.

Load the logs in Python:

```python
import sys; sys.path.insert(0, "src")
import ffplast as fp
df = fp.load_results("results/main_random_label")       # one row per task
per_run, _ = fp.plasticity_table(df[df.width == 500])   # peak, last, retention R per run
```

---

## Run the experiments from scratch

1. Open a notebook from `notebooks/` in Google Colab and choose **Runtime → Change runtime type → T4 GPU**.
2. Click **Runtime → Run all** and allow access to Google Drive.
3. Results are written to `MyDrive/ff_plasticity/` after every task, with a checkpoint every 10 tasks. Each session stops itself after about two hours; run **Run all** again to continue where it stopped. At the end, a ZIP with all results is downloaded automatically.

Recommended order: notebook 01 → 02 → 03 → 04 → 05. MNIST and CIFAR-10 are downloaded automatically through `torchvision`. The notebooks write one log file per run; `fp.load_results` reads a folder of such files or a merged `runs.jsonl` in the same way.

Software used for the reported runs: Python 3.13, PyTorch 2.11 (CUDA 13.0), NVIDIA T4 GPU.

### Main settings

| Setting | Value |
|---|---|
| Network | MLP, 3 hidden layers × 500 ReLU units (robustness: LeNet-style CNN) |
| Optimizer | Adam, batch size 64 (robustness: SGD with momentum 0.9) |
| Task stream | 83 tasks; each task = 1,000 images with fresh random labels, 20 epochs |
| Seeds | 5 (width ablation, robustness checks, other local rules, CIFAR-100: 3) |
| Learning rate | chosen per condition on the first 5 tasks with a separate seed (`lr_choice.json`) |

---

## License

Code: MIT. Results, tables and figures: CC BY 4.0. See `LICENSE`.

## Citation

See `CITATION.cff`. Once the article is published, please cite the article and the archived version of this repository (Zenodo DOI).
