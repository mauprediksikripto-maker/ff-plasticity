"""Rebuild every table and figure of the paper from the per-task logs in results/.

Usage (from the repository root):
    python src/make_paper_outputs.py

Outputs:
    figures/Figure_1..5.{pdf,png}, figures/Figure_A1.{pdf,png}
    tables/table4_plasticity.csv, tables/table5_tests.csv, tables/tableA1_learning_rates.csv,
    tables/tableA6_control_regime.csv, tables/tableA2_lr_sensitivity.csv, tables/spearman.txt
No GPU and no training are needed; runs in under a minute on a laptop.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
from scipy import stats                 # noqa: E402
import ffplast as fp                    # noqa: E402
import paper_figures as pf              # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
TAB = os.path.join(ROOT, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


def load(folder):
    df = fp.load_results(os.path.join(RES, folder))
    return df[~df.run.str.startswith(("tune_", "cal_", "pilot_"))]


def main():
    main_df = load("main_random_label")
    per_run, _ = fp.plasticity_table(main_df, window=10, last=10)
    w500 = per_run[per_run.width == 500]

    # Table 4
    t4 = w500.groupby(["bench", "method"]).agg(
        n=("seed", "count"), peak=("acc_peak", "mean"), peak_sd=("acc_peak", "std"),
        last=("acc_last", "mean"), last_sd=("acc_last", "std"),
        R=("retention", "mean"), R_sd=("retention", "std"),
        dormant=("dormant_last", "mean"), erank=("erank_last", "mean"), erank_sd=("erank_last", "std"),
        dormant_neutral=("dormant_neutral_last", "mean"), erank_neutral=("erank_neutral_last", "mean"))
    t4.round(4).to_csv(os.path.join(TAB, "table4_plasticity.csv"))

    # Table 5 (+ width comparisons)
    tests = pd.concat([fp.pairwise_tests(per_run, "retention"), fp.pairwise_tests(per_run, "acc_last")])
    tests.round(6).to_csv(os.path.join(TAB, "table5_tests.csv"), index=False)

    # Table A.1
    lr = json.load(open(os.path.join(RES, "main_random_label", "lr_choice.json")))
    pd.DataFrame(lr).T.to_csv(os.path.join(TAB, "tableA1_learning_rates.csv"))

    # Table A.6 control regime
    ctrl = load("control_abundant_data")
    pc, _ = fp.plasticity_table(ctrl, window=10, last=10)
    pc.groupby(["bench", "method"]).agg(n=("seed", "count"), peak=("acc_peak", "mean"),
                                        R=("retention", "mean"), R_sd=("retention", "std"),
                                        dormant=("dormant_last", "mean")).round(4) \
      .to_csv(os.path.join(TAB, "tableA6_control_regime.csv"))

    # Spearman (Section 5.6)
    agg = pf.summary_points(per_run)
    m5 = agg[agg.width == 500]
    dorm = per_run.groupby(["bench", "method", "width"]).dormant_last.mean().reset_index()
    lines = []
    for name, d in (("width 500", m5), ("all widths", agg)):
        dd = d.merge(dorm, on=["bench", "method", "width"])
        r1 = stats.spearmanr(dd.R, dd.erank)
        r2 = stats.spearmanr(dd.R, dd.dormant_last)
        lines.append(f"{name}: n={len(dd)} rho(R, erank)={r1[0]:.3f} p={r1[1]:.2e}; "
                     f"rho(R, dormant)={r2[0]:.3f} p={r2[1]:.2e}")
    open(os.path.join(TAB, "spearman.txt"), "w").write("\n".join(lines) + "\n")

    # Figures 1-5
    pf.fig1_conditions(os.path.join(FIG, "Figure_1"))
    pf.fig2_accuracy(main_df, os.path.join(FIG, "Figure_2"))
    pf.fig3_mechanisms(main_df, os.path.join(FIG, "Figure_3"))
    rho = (f"Spearman ρ = {stats.spearmanr(m5.R, m5.erank)[0]:.2f} (width 500, n = {len(m5)})\n"
           f"ρ = {stats.spearmanr(agg.R, agg.erank)[0]:.2f} (all widths, n = {len(agg)})")
    pts = [dict(bench=r.bench, method=r.method, width=r.width, R=r.R, erank=r.erank) for r in agg.itertuples()]
    pf.fig4_rank_vs_retention(pts, os.path.join(FIG, "Figure_4"), rho=rho)
    wr = agg[(agg.bench == "rlmnist") & agg.method.isin(["bp", "bpg", "ff"])]
    pf.fig5_width([dict(method=r.method, width=r.width, R_mean=r.R, R_sd=r.R_sd) for r in wr.itertuples()],
                  os.path.join(FIG, "Figure_5"))

    # Table A.2 + Figure A1 (learning-rate sensitivity of FF)
    sens = load("lr_sensitivity")
    ps, _ = fp.plasticity_table(sens, window=10, last=10)
    ps["method"] = "ff@1e-3"
    ref = w500[w500.method.isin(["ff", "bpg"])]
    allr = pd.concat([ref, ps])
    ta3 = allr.groupby(["bench", "method"]).agg(n=("seed", "count"), peak=("acc_peak", "mean"),
                                                peak_sd=("acc_peak", "std"), last=("acc_last", "mean"),
                                                last_sd=("acc_last", "std"), R=("retention", "mean"),
                                                R_sd=("retention", "std"), dormant=("dormant_last", "mean"),
                                                erank=("erank_last", "mean"), erank_sd=("erank_last", "std"))
    rows = []
    for bk in ("rlmnist", "rlcifar"):
        a = allr[(allr.bench == bk) & (allr.method == "ff@1e-3")].retention
        for b in ("ff", "bpg"):
            x = allr[(allr.bench == bk) & (allr.method == b)].retention
            rows.append(dict(bench=bk, comparison=f"ff@1e-3 vs {b}",
                             welch_p=stats.ttest_ind(a, x, equal_var=False).pvalue,
                             mwu_p=stats.mannwhitneyu(a, x, alternative="two-sided").pvalue))
    ta3.round(4).to_csv(os.path.join(TAB, "tableA2_lr_sensitivity.csv"))
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "tableA2_lr_sensitivity_tests.csv"), index=False)
    _figure_a1(main_df, sens)
    robustness()
    other_local_rules(main_df)
    continual_cifar100()
    print("Done. Figures in", FIG, "| tables in", TAB)
    print("\n".join(lines))


def _figure_a1(main_df, sens):
    import matplotlib.pyplot as plt
    series = [(main_df[(main_df.method == "ff") & (main_df.width == 500)], "FF, LR 3·10⁻³ (main)", "#D55E00", "-"),
              (sens[sens.method == "ff"], "FF, LR 10⁻³", "#D55E00", "--"),
              (main_df[(main_df.method == "bpg") & (main_df.width == 500)], "BP-G (main)", "#7B5EA7", "-")]
    fig, axes = plt.subplots(1, 2, figsize=(pf.FULL, 62 * pf.MM), sharey=True)
    for ax, (bk, title) in zip(axes, pf.STREAM.items()):
        for d, lab, col, ls in series:
            g = d[d.bench == bk]
            piv = g.pivot_table(index="task", columns="seed", values="acc_end").rolling(3, min_periods=1).mean()
            mu, sd = piv.mean(axis=1), piv.std(axis=1).fillna(0)
            ax.plot(mu.index, mu, color=col, ls=ls, label=lab)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        ax.axhline(0.1, color="0.5", lw=0.6, ls=":")
        ax.set_title(title); ax.set_xlabel("Task number"); ax.set_ylim(0, 1.04); ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("Accuracy at end of task")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.2)
    pf._save(fig, os.path.join(FIG, "Figure_A1"))


ROBUST = (("mlp_sgdm", "MLP, SGD (momentum 0.9)"), ("cnn_adam", "CNN, Adam"))


def robustness(seeds=(0, 1, 2)):
    """Table A.3, robustness tests and Figure A.2: SGD with momentum (MLP) and a LeNet-style CNN."""
    import matplotlib.pyplot as plt
    lr = json.load(open(os.path.join(RES, "robustness_sgd_cnn", "lr_choice.json")))
    key = {"mlp_sgdm": "A_sgd", "cnn_adam": "B_cnn"}
    rows, tests, dfs = [], [], {}
    for part, _ in ROBUST:
        df = fp.load_results(os.path.join(RES, "robustness_sgd_cnn", part))
        df = df[df.seed.isin(seeds)]
        dfs[part] = df
        pr, _ = fp.plasticity_table(df, window=10, last=10)
        pr["setting"] = part
        pr["lr"] = [lr[f"{key[part]}/{b}"][m] for b, m in zip(pr.bench, pr.method)]
        rows.append(pr)
        t = fp.pairwise_tests(pr, "retention", pairs=(("ff", "bpg"), ("ff", "bp"), ("bpg", "bp"),
                                                       ("bpn", "bp"), ("bpg", "bpn")))
        t["setting"] = part
        tests.append(t)
    pr = pd.concat(rows)
    tab = pr.groupby(["setting", "bench", "method", "lr"]).agg(
        n=("seed", "count"), peak=("acc_peak", "mean"), peak_sd=("acc_peak", "std"),
        last=("acc_last", "mean"), last_sd=("acc_last", "std"), R=("retention", "mean"), R_sd=("retention", "std"),
        dormant=("dormant_last", "mean"), erank=("erank_last", "mean"), erank_sd=("erank_last", "std"))
    tab.round(4).to_csv(os.path.join(TAB, "tableA3_robustness.csv"))
    pd.concat(tests).round(6).to_csv(os.path.join(TAB, "tableA3_robustness_tests.csv"), index=False)
    agg = pr.groupby(["setting", "bench", "method"]).agg(R=("retention", "mean"), erank=("erank_last", "mean"),
                                                         dormant=("dormant_last", "mean")).reset_index()
    r1 = stats.spearmanr(agg.R, agg.erank); r2 = stats.spearmanr(agg.R, agg.dormant)
    with open(os.path.join(TAB, "spearman_robustness.txt"), "w") as f:
        f.write(f"n={len(agg)} rho(R, erank)={r1[0]:.3f} p={r1[1]:.2e}; rho(R, dormant)={r2[0]:.3f} p={r2[1]:.2e}\n")
    # Figure A.2: 2 x 2 panels
    col = {"bp": "#0072B2", "bpn": "#009E73", "bpg": "#7B5EA7", "ff": "#D55E00"}
    lab = {"bp": "BP", "bpn": "BP-N", "bpg": "BP-G", "ff": "FF"}
    fig, axes = plt.subplots(2, 2, figsize=(pf.FULL, 118 * pf.MM), sharex=True, sharey=True)
    for i, (part, ptitle) in enumerate(ROBUST):
        for j, (bk, stitle) in enumerate(pf.STREAM.items()):
            ax = axes[i, j]; d = dfs[part]
            for m in ("bp", "bpn", "bpg", "ff"):
                g = d[(d.bench == bk) & (d.method == m)]
                piv = g.pivot_table(index="task", columns="seed", values="acc_end").rolling(3, min_periods=1).mean()
                mu, sd = piv.mean(axis=1), piv.std(axis=1).fillna(0)
                ax.plot(mu.index, mu, color=col[m], label=lab[m])
                ax.fill_between(mu.index, (mu - sd).clip(0, 1), (mu + sd).clip(0, 1), color=col[m], alpha=0.15, lw=0)
            ax.axhline(0.1, color="0.5", lw=0.6, ls=":")
            ax.set_title(f"({'abcd'[2 * i + j]}) {ptitle}: {stitle}")
            ax.set_ylim(0, 1.04); ax.grid(alpha=0.25, lw=0.4)
            if i == 1: ax.set_xlabel("Task number")
            if j == 0: ax.set_ylabel("Accuracy at end of task")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    pf._save(fig, os.path.join(FIG, "Figure_A2"))
    print(tab.round(3).to_string()); print(open(os.path.join(TAB, "spearman_robustness.txt")).read())


COLX = {"bp": "#0072B2", "bpn": "#009E73", "bpg": "#7B5EA7", "ff": "#D55E00",
        "dfa": "#8C564B", "dfan": "#8C564B", "lcn": "#CC79A7", "gcn": "#56B4E9"}
LSX = {"dfan": "--"}
LABX = {"bp": "BP", "bpn": "BP-N", "bpg": "BP-G", "ff": "FF", "dfa": "DFA", "dfan": "DFA-N", "lcn": "LC-N", "gcn": "GC-N"}


def _summ(pr, keys):
    return pr.groupby(keys).agg(
        n=("seed", "count"), peak=("acc_peak", "mean"), peak_sd=("acc_peak", "std"),
        last=("acc_last", "mean"), last_sd=("acc_last", "std"), R=("retention", "mean"), R_sd=("retention", "std"),
        dormant=("dormant_last", "mean"), erank=("erank_last", "mean"), erank_sd=("erank_last", "std"))


def other_local_rules(main_df):
    """Table A.4 and Figure A.3: DFA, DFA-N, LC-N and GC-N (MLP, Adam, random-label streams),
    compared with BP and BP-N from the main experiment."""
    import matplotlib.pyplot as plt
    new = fp.load_results(os.path.join(RES, "other_local_rules"))
    ref = main_df[(main_df.width == 500) & main_df.method.isin(["bp", "bpn"])]
    allr = pd.concat([ref, new])
    pr, _ = fp.plasticity_table(allr, window=10, last=10)
    lr = json.load(open(os.path.join(RES, "other_local_rules", "lr_choice.json")))
    _summ(pr, ["bench", "method"]).round(4).to_csv(os.path.join(TAB, "tableA4_other_local_rules.csv"))
    fp.pairwise_tests(pr, "retention", pairs=(("lcn", "gcn"), ("dfan", "bpn"), ("dfa", "bp"), ("dfan", "dfa"))) \
      .round(6).to_csv(os.path.join(TAB, "tableA4_other_local_rules_tests.csv"), index=False)
    json.dump(lr, open(os.path.join(TAB, "tableA4_learning_rates.json"), "w"), indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(pf.FULL, 62 * pf.MM), sharey=True)
    for ax, (bk, title) in zip(axes, pf.STREAM.items()):
        for m in ("bp", "bpn", "dfa", "dfan", "lcn", "gcn"):
            g = allr[(allr.bench == bk) & (allr.method == m)]
            piv = g.pivot_table(index="task", columns="seed", values="acc_end").rolling(3, min_periods=1).mean()
            mu, sd = piv.mean(axis=1), piv.std(axis=1).fillna(0)
            ref_line = m in ("bp", "bpn")
            ax.plot(mu.index, mu, color=COLX[m], ls=LSX.get(m, "-"), lw=0.8 if ref_line else 1.3,
                    alpha=0.55 if ref_line else 1.0, label=LABX[m] + (" (main)" if ref_line else ""))
            if not ref_line:
                ax.fill_between(mu.index, (mu - sd).clip(0, 1), (mu + sd).clip(0, 1), color=COLX[m], alpha=0.12, lw=0)
        ax.axhline(0.1, color="0.5", lw=0.6, ls=":")
        ax.set_title(title); ax.set_xlabel("Task number"); ax.set_ylim(0, 1.04); ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("Accuracy at end of task")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.2)
    pf._save(fig, os.path.join(FIG, "Figure_A3"))
    print(_summ(pr, ["bench", "method"]).round(3).to_string())


def continual_cifar100():
    """Table A.5 and Figure A.4: Continual CIFAR-100 (two new classes per task, held-out test accuracy)."""
    import matplotlib.pyplot as plt
    d = fp.load_results(os.path.join(RES, "continual_cifar100"))
    pr, _ = fp.plasticity_table(d, window=10, last=10)
    first = d.sort_values("task").groupby(["method", "seed"]).acc_end.apply(lambda a: a.head(10).mean()).rename("first10")
    pr = pr.merge(first.reset_index(), on=["method", "seed"])
    tab = _summ(pr, ["bench", "method"])
    tab["first10"] = pr.groupby(["bench", "method"]).first10.mean()
    tab.round(4).to_csv(os.path.join(TAB, "tableA5_continual_cifar100.csv"))
    fp.pairwise_tests(pr, "acc_last", pairs=(("ff", "bpg"), ("lcn", "gcn"), ("bpn", "bp"), ("ff", "bp"), ("bpg", "bp"))) \
      .round(6).to_csv(os.path.join(TAB, "tableA5_continual_cifar100_tests.csv"), index=False)
    fig, ax = plt.subplots(figsize=(pf.FULL, 70 * pf.MM))
    for m in ("bp", "bpn", "bpg", "ff", "lcn", "gcn"):
        g = d[d.method == m]
        piv = g.pivot_table(index="task", columns="seed", values="acc_end").rolling(10, min_periods=1).mean()
        mu, sd = piv.mean(axis=1), piv.std(axis=1).fillna(0)
        ax.plot(mu.index, mu, color=COLX[m], label=LABX[m])
        ax.fill_between(mu.index, (mu - sd).clip(0.45, 1), (mu + sd).clip(0.45, 1), color=COLX[m], alpha=0.12, lw=0)
    ax.axhline(0.5, color="0.5", lw=0.6, ls=":")
    ax.set_xlabel("Task number"); ax.set_ylabel("Test accuracy at end of task"); ax.set_ylim(0.45, 1.0)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    pf._save(fig, os.path.join(FIG, "Figure_A4"))
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
