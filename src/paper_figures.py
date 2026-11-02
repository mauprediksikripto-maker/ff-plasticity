# paper_figures.py — publication figures for the FF vs BP plasticity manuscript.
# Elsevier: vector PDF + 300-dpi PNG/TIFF, one file per figure, colour-blind-safe palette,
# single column = 90 mm, full width = 190 mm, text >= 7 pt.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

MM = 1 / 25.4
SINGLE, FULL = 90 * MM, 190 * MM
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.3,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300,
})
# validated (CVD-safe) categorical palette: BP, BP-N, BP-G, FF ; reset = same hue, dashed
COL = {"bp": "#0072B2", "bp_cbp": "#0072B2", "bpn": "#009E73", "bpg": "#7B5EA7",
       "ff": "#D55E00", "ff_cbp": "#D55E00"}
LS = {"bp": "-", "bp_cbp": "--", "bpn": "-", "bpg": "-", "ff": "-", "ff_cbp": "--"}
MK = {"bp": "o", "bp_cbp": "P", "bpn": "s", "bpg": "D", "ff": "^", "ff_cbp": "v"}
NAME = {"bp": "BP", "bp_cbp": "BP + reset", "bpn": "BP-N", "bpg": "BP-G", "ff": "FF", "ff_cbp": "FF + reset"}
ORDER = ["bp", "bp_cbp", "bpn", "bpg", "ff", "ff_cbp"]
STREAM = {"rlmnist": "Random-label MNIST", "rlcifar": "Random-label CIFAR-10"}


def _save(fig, path_noext):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_noext}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path_noext


def _curve(ax, df, method, metric, smooth=3):
    g = df[df.method == method]
    if g.empty:
        return False
    piv = g.pivot_table(index="task", columns="seed", values=metric).rolling(smooth, min_periods=1).mean()
    mu, sd = piv.mean(1), piv.std(1).fillna(0)
    ax.plot(mu.index, mu.values, color=COL[method], ls=LS[method], label=NAME[method])
    ax.fill_between(mu.index, mu - sd, mu + sd, color=COL[method], alpha=0.15, lw=0)
    return True


# ---------------------------------------------------------------- Fig. 1
def fig1_conditions(path):
    """Schematic of the conditions: normalisation, objective and gradient scope."""
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 62 * MM))
    specs = [("BP", False, "softmax", "global"), ("BP-N", True, "softmax", "global"),
             ("BP-G", True, "goodness", "global"), ("FF", True, "goodness", "local")]
    grad = "#C0392B"
    for ax, (title, norm, obj, scope) in zip(axes, specs):
        ax.set_xlim(-1, 11); ax.set_ylim(0, 13); ax.axis("off")
        ax.set_title(title, fontweight="bold", pad=2)
        ys = [2.2, 5.0, 7.8]
        ax.add_patch(FancyBboxPatch((1.7, 0.2), 5.6, 1.1, boxstyle="round,pad=0.05,rounding_size=0.3",
                                    fc="white", ec="0.35", lw=0.7))
        ax.text(4.5, 0.75, "image + label" if obj == "goodness" else "image", ha="center", va="center", fontsize=6.3)
        prev_top = 1.3
        for i, y in enumerate(ys):
            if norm:
                ax.text(4.75, y - 0.62, "RMS norm", ha="left", va="center", fontsize=5.6, color="0.3")
            ax.annotate("", xy=(4.5, y - 0.05), xytext=(4.5, prev_top),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.35", mutation_scale=6))
            ax.add_patch(FancyBboxPatch((1.7, y), 5.6, 1.3, boxstyle="round,pad=0.05,rounding_size=0.3",
                                        fc="#EEF3F8", ec="0.35", lw=0.7))
            ax.text(4.5, y + 0.65, f"layer {i + 1}", ha="center", va="center", fontsize=6.5)
            if obj == "goodness":
                ax.text(7.6, y + 0.65, rf"$\mathcal{{L}}_{i + 1}$", ha="left", va="center", fontsize=7, color=grad)
            prev_top = y + 1.3
        if obj == "softmax":
            ax.annotate("", xy=(4.5, 10.35), xytext=(4.5, prev_top),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.35", mutation_scale=6))
            ax.add_patch(FancyBboxPatch((1.7, 10.4), 5.6, 1.2, boxstyle="round,pad=0.05,rounding_size=0.3",
                                        fc="white", ec="0.35", lw=0.7))
            ax.text(4.5, 11.0, "softmax · CE", ha="center", va="center", fontsize=6.3)
            ax.text(7.6, 11.0, r"$\mathcal{L}$", ha="left", va="center", fontsize=7, color=grad)
        else:
            ax.text(4.5, 10.7, r"loss $=\sum_l \mathcal{L}_l$", ha="center", va="center", fontsize=6.5)
        if scope == "global":
            top = 11.0 if obj == "softmax" else 8.45
            ax.add_patch(FancyArrowPatch((0.9, top), (0.9, 2.6), arrowstyle="-|>", mutation_scale=7,
                                         color=grad, lw=1.1, connectionstyle="arc3,rad=0"))
            ax.text(0.3, (top + 2.6) / 2, "global gradient", rotation=90, ha="center", va="center",
                    fontsize=6, color=grad)
        else:
            for y in ys:
                ax.add_patch(FancyArrowPatch((0.9, y + 1.2), (0.9, y + 0.1), arrowstyle="-|>", mutation_scale=6,
                                             color=grad, lw=1.1))
            ax.text(-0.2, 5.6, "local gradient per layer", rotation=90, ha="center",
                    va="center", fontsize=6, color=grad)
            ax.text(4.5, 10.0, "stop-gradient between layers", ha="center", va="center", fontsize=5.6, color=grad)
    fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.9, bottom=0.02)
    return _save(fig, path)


# ---------------------------------------------------------------- Fig. 2
def fig2_accuracy(df, path, width=500):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 62 * MM), sharey=True)
    for ax, (bk, title) in zip(axes, STREAM.items()):
        sub = df[(df.bench == bk) & (df.width == width)]
        for m in ORDER:
            _curve(ax, sub, m, "acc_end")
        ax.axhline(0.1, color="0.5", lw=0.6, ls=":")
        ax.text(sub.task.max() if len(sub) else 80, 0.115, "chance", ha="right", va="bottom", fontsize=6, color="0.4")
        ax.set_title(title); ax.set_xlabel("Task number"); ax.set_ylim(0, 1.04)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("Accuracy at end of task")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.2)
    return _save(fig, path)


# ---------------------------------------------------------------- Fig. 3
def fig3_mechanisms(df, path, width=500):
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 108 * MM), sharex=True)
    rows = [("dormant_mean", "Dormant units (fraction)"), ("erank_mean", "Effective rank")]
    for r, (metric, lab) in enumerate(rows):
        for c, (bk, title) in enumerate(STREAM.items()):
            ax = axes[r, c]
            sub = df[(df.bench == bk) & (df.width == width)]
            for m in ORDER:
                _curve(ax, sub, m, metric)
            if r == 0:
                ax.set_title(title); ax.set_ylim(0, 1)
            else:
                ax.set_xlabel("Task number"); ax.set_ylim(bottom=0)
            if c == 0:
                ax.set_ylabel(lab)
            ax.grid(alpha=0.25, lw=0.4)
            ax.text(-0.13 if c == 0 else -0.06, 1.04, "abcd"[2 * r + c], transform=ax.transAxes,
                    fontweight="bold", fontsize=8.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=0.8, w_pad=1.2)
    return _save(fig, path)


# ---------------------------------------------------------------- Fig. 4
def fig4_rank_vs_retention(points, path, rho=None):
    """points: list of dicts {method, bench, width, R, erank}."""
    fig, ax = plt.subplots(figsize=(SINGLE, 80 * MM))
    seen = set()
    for p in points:
        main = p["width"] == 500
        filled = p["bench"] == "rlmnist"
        lab = NAME[p["method"]] if (main and p["method"] not in seen) else None
        if main:
            seen.add(p["method"])
        ax.scatter(p["erank"], p["R"], s=26 if main else 14, marker=MK[p["method"]],
                   facecolors=COL[p["method"]] if filled else "white", edgecolors=COL[p["method"]],
                   linewidths=0.9, alpha=1 if main else 0.6, label=lab, zorder=3)
    ax.set_xscale("log"); ax.set_xlabel("Effective rank (last 10 tasks, log scale)")
    ax.set_ylabel("Retention R"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.25, lw=0.4, which="major")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=3, handletextpad=0.3, columnspacing=1.0)
    for hnd in leg.legend_handles:
        hnd.set_facecolor(hnd.get_edgecolor())
    if rho is not None:
        ax.text(0.03, 0.96, rho, transform=ax.transAxes, va="top", fontsize=7)
    ax.text(0.03, 0.80, "filled: MNIST · open: CIFAR-10\nsmall: width 128 / 1,000",
            transform=ax.transAxes, va="top", fontsize=6, color="0.35")
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------- Fig. 5
def fig5_width(rows, path):
    """rows: list of dicts {method, width, R_mean, R_sd} (random-label MNIST)."""
    fig, ax = plt.subplots(figsize=(SINGLE, 62 * MM))
    for m in ("bp", "bpg", "ff"):
        rr = sorted([r for r in rows if r["method"] == m], key=lambda r: r["width"])
        x = [r["width"] for r in rr]; y = [r["R_mean"] for r in rr]; e = [r["R_sd"] for r in rr]
        ax.errorbar(x, y, yerr=e, color=COL[m], marker=MK[m], ms=4.5, capsize=2, lw=1.2, label=NAME[m])
        ax.text(x[-1] * 1.08, y[-1], NAME[m], color="0.15", fontsize=6.5, va="center")
    ax.set_xscale("log"); ax.set_xticks([128, 500, 1000]); ax.set_xticklabels(["128", "500", "1,000"])
    ax.set_xlim(100, 1600); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Hidden units per layer"); ax.set_ylabel("Retention R")
    ax.grid(alpha=0.25, lw=0.4); ax.legend(loc="center left", frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def summary_points(per_run):
    """Aggregate plasticity_table per_run output into points for fig4/fig5."""
    g = per_run.groupby(["bench", "method", "width"]).agg(R=("retention", "mean"), R_sd=("retention", "std"),
                                                           erank=("erank_last", "mean")).reset_index()
    return g
