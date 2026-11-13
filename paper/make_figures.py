"""Regenerate Figures 2 and 4 of the PJMS manuscript from the per-fold rank configurations (Table 3).
The configurations reproduce every reported per-fold and averaged metric to four decimals."""
import math, numpy as np, matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

# fold -> (m = fold size, ranks of the positive vertices)
FOLDS = [(9, [6, 7, 8, 9]), (9, [5, 6, 7, 8, 9]), (9, [1, 2, 3, 4, 5]), (9, [1, 2, 3, 4, 5]),
         (8, [1, 2, 3, 4]), (8, [1, 2, 3, 4]), (8, [1, 2, 3, 4]), (8, [1, 2, 3, 4]), (8, [1, 2, 3, 4]),
         (8, [1, 2, 3, 8])]
K = np.arange(1, 21)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, BAND = "#2a78d6", "#eb6834", "#c9c8c3"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "axes.unicode_minus": False})

def dcg(ranks): return sum(1 / math.log2(r + 1) for r in ranks)

def at_k(m, ranks, k):
    n = len(ranks); c = sum(r <= k for r in ranks)
    ndcg = dcg([r for r in ranks if r <= k]) / dcg(range(1, min(n, k) + 1))
    return c / k, c / n, ndcg

def bound_k(m, n, k):
    c = min(n, k); return c / k, c / n, 1.0

TW = 360 / 72.27   # text width of the amsart 11pt layout in inches; figures are placed at 100%
rng = np.random.default_rng(0)
def random_k(m, n, k, trials=20000):
    acc = np.zeros(3)
    for _ in range(trials):
        acc += at_k(m, sorted(rng.choice(np.arange(1, m + 1), n, replace=False)), k)
    return acc / trials

obs = np.array([[at_k(m, r, k) for k in K] for m, r in FOLDS]).mean(0)
upp = np.array([[bound_k(m, len(r), k) for k in K] for m, r in FOLDS]).mean(0)
rnd = np.array([[random_k(m, len(r), k) for k in K] for m, r in FOLDS]).mean(0)

# ---------------- Figure 2 ----------------
fig, axes = plt.subplots(1, 3, figsize=(TW, 2.35), sharey=True)
names = ["Precision@$k$", "Recall@$k$", "NDCG@$k$"]
for j, ax in enumerate(axes):
    ax.axvspan(9, 20.6, color="#efeeeb", lw=0, zorder=0)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.plot(K, upp[:, j], ls="--", color=MUTED, lw=1.2, zorder=2)
    ax.plot(K, rnd[:, j], ls=":", color=ORANGE, lw=1.6, zorder=2)
    ax.plot(K, obs[:, j], color=BLUE, lw=2, zorder=3)
    sel = np.isin(K, [1, 3, 5, 10, 20])
    ax.plot(K[sel], obs[sel, j], "o", ms=4.2, color=BLUE, mec="white", mew=0.8, zorder=4)
    ax.set_title(names[j], color=INK, pad=4)
    ax.set_xticks([1, 3, 5, 10, 15, 20]); ax.set_xlim(0.5, 20.6); ax.set_ylim(0, 1.05)
    ax.set_xlabel("$k$")
axes[0].set_ylabel("mean over 10 folds")
for ax in axes:
    ax.text(14.8, 0.95 if ax is axes[0] else 0.06, "$k\\geq m$:\nmodel-\nindependent",
            ha="center", va="top" if ax is axes[0] else "bottom", fontsize=7.5, color=MUTED)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=BLUE, lw=2, marker="o", ms=4, mec="white"),
            Line2D([], [], color=MUTED, lw=1.2, ls="--"), Line2D([], [], color=ORANGE, lw=1.6, ls=":")],
           ["KG-GCN (observed)", "upper bound (Prop. 4.2)", "random (expected)"],
           loc="lower center", ncol=3, frameon=False, fontsize=8, handlelength=2.2, columnspacing=1.2,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=0.6)
fig.savefig("fig2_metrics_vs_k.pdf"); plt.close(fig)

# ---------------- Figure 4 ----------------
def fold_metrics(m, r):
    n = len(r); Hn = sum(1 / j for j in range(1, n + 1))
    ap = np.mean([(i + 1) / x for i, x in enumerate(r)])
    nmrr = np.mean([1 / x for x in r]) * n / Hn
    nd = dcg(r) / dcg(range(1, n + 1)); mr = np.mean(r)
    low = list(range(m - n + 1, m + 1))
    lo = (np.mean([(i + 1) / x for i, x in enumerate(low)]), np.mean([1 / x for x in low]) * n / Hn,
          dcg(low) / dcg(range(1, n + 1)), (n + 1) / 2)
    hi = (1.0, 1.0, 1.0, m - (n - 1) / 2)
    return (ap, nmrr, nd, mr), lo, hi

vals = [fold_metrics(m, r) for m, r in FOLDS]
fig, axes = plt.subplots(2, 2, figsize=(TW, 3.6)); axes = axes.ravel()
titles = ["AP", "normalised MRR$^{+}$", "NDCG", "mean rank"]
x = np.arange(1, 11)
for j, ax in enumerate(axes):
    o = np.array([v[0][j] for v in vals]); lo = np.array([v[1][j] for v in vals]); hi = np.array([v[2][j] for v in vals])
    ax.vlines(x, lo, hi, color=BAND, lw=5, zorder=1, capstyle="butt")
    good = hi if j < 3 else lo           # the optimal end of each range
    ax.plot(x, o, "o", ms=4.6, color=BLUE, mec="white", mew=0.8, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in x])
    ax.set_xlabel("fold" if j >= 2 else ""); ax.set_title(titles[j], color=INK, pad=4)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    if j < 3: ax.set_ylim(0, 1.06)
    else: ax.set_ylim(9.5, 1); ax.set_yticks([2, 4, 6, 8])
fig.legend([Line2D([], [], color=BAND, lw=5), Line2D([], [], color=BLUE, ls="", marker="o", ms=4.6, mec="white")],
           ["attainable range (Prop. 4.3)", "KG-GCN (observed)"],
           loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.0, h_pad=0.8)
fig.savefig("fig3_per_fold.pdf"); plt.close(fig)

print("P@k", np.round(obs[[0, 2, 4, 9, 19], 0], 4), "R@k", np.round(obs[[0, 2, 4, 9, 19], 1], 4),
      "NDCG@k", np.round(obs[[0, 2, 4, 9, 19], 2], 4))
print("random P@1..", np.round(rnd[[0, 2, 4], 0], 3), "random NDCG@m", np.round(rnd[19, 2], 3))
