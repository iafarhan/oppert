"""Figure 3 -- Cross-method comparison. DA, Pearson, cosine across Norman/K562/RPE1."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, pearsonr, spearmanr

H = Path(__file__).parent
NPZ = H / "data" / "h2h_npz"
OUT = H / "out" / "fig_comp2_results"
OUT.parent.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.6,
    "axes.labelsize": 7.0,
    "axes.titlesize": 8.0,
    "axes.labelpad": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "lines.solid_capstyle": "round",
    "legend.fontsize": 6.4,
    "legend.frameon": True,
    "legend.handlelength": 1.0,
    "legend.handletextpad": 0.4,
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.20,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

INDIGO = "#3B5BDB"
INDIGO_DARK = "#1F3A93"
RUST = "#A0522D"
INK = "#1A1A1A"
PALE = "#6B6B6B"

# DATA
base = pd.read_csv(H / "data" / "published_baselines.csv")
base = base.rename(columns={"rho_DEG": "rho20", "DA_DEG": "DA20"})
pm = pd.read_csv(H / "data" / "csvs_real_norman" / "pair_metrics.csv")
pw = pd.read_csv(H / "data" / "csvs_real_norman" / "pathway_enrichment.csv")

# OpPert Norman s3 per-pert pred + actual for the volcano
sc = np.load(NPZ / "scfate_norman_s3.npz", allow_pickle=True)
sc_names = list(sc["names"])

DSETS = ["Norman", "K562", "RPE1", "SciPlex3"]

# FIGURE
fig = plt.figure(figsize=(15.0, 7.6))
outer = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.42,
                         width_ratios=[1.30, 1.05, 1.05],
                         height_ratios=[1, 1])
ax_a = fig.add_subplot(outer[0, 0])
ax_b = fig.add_subplot(outer[0, 1])
ax_c = fig.add_subplot(outer[0, 2])
ax_d = fig.add_subplot(outer[1, 0])
ax_e = fig.add_subplot(outer[1, 1:])      # spans last 2/3 of bottom row

# (a) cross-dataset receipts: 3 rows × 11 methods, ρ20 annotated
ax = ax_a
methods_all = (base[base.dataset == "Norman"].sort_values("rho20")
               .method.tolist())   # ordering by Norman ρ20 ascending
n_methods = len(methods_all)

DS_OFFSETS = {"Norman": -0.33, "K562": -0.11, "RPE1": 0.11, "SciPlex3": 0.33}
DS_COL = {"Norman": INDIGO_DARK, "K562": "#7C3FAE",
          "RPE1": "#3DA34D", "SciPlex3": "#D9822B"}

y_pos = {m: i for i, m in enumerate(methods_all)}
for ds in DSETS:
    sub = base[base.dataset == ds].set_index("method")
    for m in methods_all:
        if m not in sub.index:
            continue
        v = float(sub.loc[m, "rho20"])
        is_ours = m == "OpPert"
        color = DS_COL[ds]
        size = 50 if is_ours else 18
        a = 1.0 if is_ours else 0.85
        y = y_pos[m] + DS_OFFSETS[ds]
        ax.hlines(y, 0, v, color=color, lw=0.4, alpha=0.30, zorder=2)
        ax.scatter([v], [y], s=size,
                   color=color if is_ours else "white",
                   edgecolor=color, lw=0.7, alpha=a, zorder=4)
        ax.text(v + 1.0, y, f"{v:.1f}",
                fontsize=5.4, va="center", ha="left",
                color=color, fontweight="semibold" if is_ours else "normal")

ax.set_yticks(list(y_pos.values()))
ax.set_yticklabels([m for m in methods_all], fontsize=6.2)
for tl, m in zip(ax.get_yticklabels(), methods_all):
    if m == "OpPert":
        tl.set_color(INK); tl.set_fontweight("bold")

ax.set_xlim(0, 100)
ax.set_xlabel(r"$\rho_{20}$  (\%)")
ax.tick_params(axis="y", length=0, pad=2)
ax.xaxis.grid(True, lw=0.4, color="0.92", zorder=0)
ax.set_axisbelow(True)
ax.text(-0.30, 1.04, "a", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top", color=INK)
ax.set_title("Cross-method × cross-dataset receipts",
             fontsize=8.2, pad=8, color=INK, loc="left",
             fontweight="semibold")
ax.text(0.0, 1.005,
        rf"three datasets per method:  Norman / K562 / RPE1",
        transform=ax.transAxes, fontsize=6.4,
        color=PALE, ha="left", va="bottom")

# Inline cluster-color legend
handles_a = [Line2D([], [], marker="o", lw=0, mec=DS_COL[ds],
                    mfc=DS_COL[ds], ms=4.5, label=ds) for ds in DSETS]
leg_a = ax.legend(handles=handles_a, loc="lower right",
                  facecolor="white", edgecolor="0.80",
                  framealpha=0.95)
leg_a.get_frame().set_linewidth(0.6)

# (b) volcano: OpPert pred vs measured for representative OOD pert
ax = ax_b
# Pick the most-effect OOD combo with strong signal
# Use the index of CEBPA+JUN if present, else max-effect pert
def canon(s): return "+".join(sorted(s.split("+")))
sc_canon = [canon(n) for n in sc_names]
target = canon("CEBPA+JUN")
if target in sc_canon:
    pert_idx = sc_canon.index(target)
else:
    eff = np.linalg.norm(sc["actual"], axis=1)
    pert_idx = int(np.argmax(eff))
pert_name = sc_names[pert_idx]

actual = sc["actual"][pert_idx]
pred = sc["pred"][pert_idx]
n_genes = len(actual)

# treat actual as measured log-FC, pred as predicted log-FC
# colour points by abs error; size by combined effect
err = np.abs(pred - actual)
combined_eff = np.maximum(np.abs(actual), np.abs(pred))

# Only show genes with reasonable effect to declutter
mask_show = combined_eff > np.quantile(combined_eff, 0.85)
ax.scatter(actual[~mask_show], pred[~mask_show],
           s=4, color="0.85", alpha=0.5, lw=0, zorder=2)
sc_show = ax.scatter(actual[mask_show], pred[mask_show],
                     s=18, c=err[mask_show], cmap="magma_r",
                     alpha=0.85, edgecolor="white", lw=0.3, zorder=4)

# y = x diagonal
xy_lo = float(min(actual.min(), pred.min())) * 1.05
xy_hi = float(max(actual.max(), pred.max())) * 1.05
ax.plot([xy_lo, xy_hi], [xy_lo, xy_hi],
        color="0.55", lw=0.7, ls=(0, (3, 2)), zorder=3)

# Annotate top-K agreement genes (large measured AND large predicted, same sign)
agree = np.sign(actual) == np.sign(pred)
topK = np.argsort(-(combined_eff * agree.astype(float)))[:6]
for idx in topK:
    ax.annotate(f"g{idx}",
                (actual[idx], pred[idx]),
                xytext=(3, 2), textcoords="offset points",
                fontsize=5.4, color=INK)

# Pearson on top-100 DE genes
top100 = np.argsort(-np.abs(actual))[:100]
r_b, _ = pearsonr(actual[top100], pred[top100])

ax.text(0.04, 0.96,
        rf"pert: {pert_name}",
        transform=ax.transAxes, fontsize=6.6,
        color=INK, fontweight="semibold", va="top")
ax.text(0.04, 0.89,
        rf"top-100 DE  $\rho$ = {r_b:+.2f}",
        transform=ax.transAxes, fontsize=6.4,
        color=INK, va="top")

ax.set_xlabel(r"measured  log-FC")
ax.set_ylabel(r"OpPert-predicted  log-FC")
ax.text(-0.18, 1.04, "b", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top", color=INK)
ax.set_title("Gene-level prediction (representative OOD combo)",
             fontsize=8.2, pad=8, color=INK, loc="left",
             fontweight="semibold")
ax.set_xlim(xy_lo, xy_hi); ax.set_ylim(xy_lo, xy_hi)
ax.set_aspect("equal", adjustable="box")
ax.xaxis.grid(True, lw=0.4, color="0.92", zorder=0)
ax.yaxis.grid(True, lw=0.4, color="0.92", zorder=0)
ax.set_axisbelow(True)

# (c) per-pert ρ20 KDE per method on Norman OOD pairs
ax = ax_c
method_order_c = ["OpPert", "GEARS", "AttentionPert", "CPA",
                  "linear", "additive", "mean"]
method_color_c = {
    "OpPert":        INDIGO_DARK,
    "GEARS":         "#A0522D",
    "AttentionPert": "#3DA34D",
    "CPA":           "#9C36B5",
    "linear":        "#7C8794",
    "additive":      "#B0BEC5",
    "mean":          "#CFD8DC",
}
xs_kde = np.linspace(-0.2, 1.0, 400)
for m in method_order_c:
    v = pm[pm.method == m]["value"].to_numpy()
    if len(v) < 3:
        continue
    kde = gaussian_kde(v, bw_method=0.20)(xs_kde)
    kde = kde / kde.max()
    is_ours = (m == "OpPert")
    color = method_color_c[m]
    lw = 1.8 if is_ours else 0.9
    a = 1.0 if is_ours else 0.85
    z = 5 if is_ours else 3
    if is_ours:
        ax.fill_between(xs_kde, 0, kde, color=color, alpha=0.20, lw=0,
                        zorder=z - 1)
    ax.plot(xs_kde, kde, color=color, lw=lw, alpha=a, zorder=z, label=m)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.10)
ax.set_xlabel(r"per-pair top-20 DE Pearson, $\rho_{20}$")
ax.set_ylabel("share of pairs")
ax.set_yticks([])
ax.yaxis.grid(True, lw=0.4, color="0.92", zorder=0)
ax.set_axisbelow(True)
ax.text(-0.16, 1.04, "c", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top", color=INK)
ax.set_title("Per-pair accuracy distribution per method (Norman)",
             fontsize=8.2, pad=8, color=INK, loc="left",
             fontweight="semibold")
leg_c = ax.legend(loc="upper left", ncol=2, columnspacing=0.6,
                  fontsize=5.8, facecolor="white",
                  edgecolor="0.80", framealpha=0.95)
leg_c.get_frame().set_linewidth(0.6)

# (d) difficulty-stratified curves (multi-method)
ax = ax_d
SPLIT_ORDER = [
    "both genes observed, pair held out",
    "exactly one gene observed",
    "neither gene observed",
]
SPLIT_LABEL = ["seen2", "seen1", "seen0"]
xs_d = np.arange(len(SPLIT_ORDER))
rng = np.random.default_rng(0)
B = 800
for m in method_order_c:
    sub = pm[pm.method == m]
    means, los, his = [], [], []
    for sp in SPLIT_ORDER:
        v = sub[sub.split_category == sp]["value"].to_numpy()
        boots = np.array([np.median(rng.choice(v, len(v), replace=True))
                          for _ in range(B)])
        means.append(float(np.median(v)))
        los.append(float(np.percentile(boots, 2.5)))
        his.append(float(np.percentile(boots, 97.5)))
    means = np.array(means); los = np.array(los); his = np.array(his)
    is_ours = (m == "OpPert")
    color = method_color_c[m]
    lw = 2.0 if is_ours else 0.9
    ms = 5.5 if is_ours else 3.2
    ax.fill_between(xs_d, los, his, color=color,
                    alpha=0.18 if is_ours else 0.10, lw=0,
                    zorder=2 if is_ours else 1)
    ax.plot(xs_d, means, color=color, lw=lw,
            marker="o", ms=ms, mec="white", mew=0.5,
            zorder=5 if is_ours else 3, label=m)
ax.set_xticks(xs_d)
ax.set_xticklabels(SPLIT_LABEL, fontsize=6.6)
ax.set_xlim(-0.4, len(SPLIT_ORDER) - 0.6)
ax.set_ylabel(r"$\rho_{20}$ (median)")
ax.tick_params(axis="x", length=0, pad=2)
ax.yaxis.grid(True, lw=0.4, color="0.92", zorder=0)
ax.set_axisbelow(True)
ax.text(-0.16, 1.04, "d", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top", color=INK)
ax.set_title("Combo-difficulty stratification (Norman)",
             fontsize=8.2, pad=8, color=INK, loc="left",
             fontweight="semibold")

# (e) DEG recall@K curves per dataset (OpPert)
ax = ax_e
def recall_at_k(pred, actual, ks):
    n = pred.shape[0]
    out = np.zeros((n, len(ks)))
    for i in range(n):
        a_top = np.argsort(-np.abs(actual[i]))
        p_top = np.argsort(-np.abs(pred[i]))
        for j, k in enumerate(ks):
            out[i, j] = len(set(a_top[:k]) & set(p_top[:k])) / k
    return out

KS = np.array([1, 5, 10, 20, 50, 100, 200, 500])
DSET_PATHS = [("Norman", "scfate_norman_s3.npz", INDIGO_DARK),
              ("K562",   "scfate_k562_reflow.npz", "#7C3FAE"),
              ("RPE1",   "scfate_rpe1_s1.npz",     "#3DA34D")]
for ds, fn, color in DSET_PATHS:
    d = np.load(NPZ / fn, allow_pickle=True)
    n_g = d["pred"].shape[1]
    valid = KS[KS <= n_g]
    rec = recall_at_k(d["pred"], d["actual"], valid)
    means = rec.mean(axis=0)
    ses = rec.std(axis=0) / np.sqrt(rec.shape[0])
    ax.fill_between(valid, means - ses, means + ses,
                    color=color, alpha=0.18, lw=0, zorder=2)
    ax.plot(valid, means, color=color, lw=1.6,
            marker="o", ms=3.4, mec="white", mew=0.5,
            zorder=4, label=ds)
ax.set_xscale("log")
ax.set_xlabel("top-K rank threshold")
ax.set_ylabel("DEG recall@K")
ax.set_xlim(0.85, 600)
ax.set_ylim(0, 1.02)
ax.yaxis.grid(True, lw=0.4, color="0.92", zorder=0)
ax.set_axisbelow(True)
ax.text(-0.18, 1.04, "e", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top", color=INK)
ax.set_title("Top-K DEG retrieval (OpPert)",
             fontsize=8.2, pad=8, color=INK, loc="left",
             fontweight="semibold")
leg_e = ax.legend(loc="lower right", facecolor="white",
                  edgecolor="0.80", framealpha=0.95)
leg_e.get_frame().set_linewidth(0.6)

# Panel (f) dropped — was redundant with (c). Keep code below for reference.
__SKIP__ = """
ax = ax_f
methods_f = ["OpPert", "GEARS", "AttentionPert", "CPA",
             "linear", "additive", "mean"]
data_per_method = {
    m: pm[pm["method"] == m]["value"].to_numpy() for m in methods_f
}
order_f = sorted(methods_f,
                 key=lambda m: -float(np.median(data_per_method[m])))
n_rows = len(order_f)

rng_jit = np.random.default_rng(0)
H_VIOL = 0.42
STRIP_OFFSET = 0.18
xs_v = np.linspace(-0.05, 1.05, 300)

for k, m in enumerate(order_f):
    sub = data_per_method[m]
    n = len(sub)
    if n == 0:
        continue
    median = float(np.median(sub))
    lo = float(np.percentile(sub, 5))
    hi = float(np.percentile(sub, 95))
    is_ours = (m == "OpPert")
    color = INDIGO_DARK if is_ours else "0.40"
    fill_color = INDIGO if is_ours else "0.65"

    # half-violin (KDE) above; tighter bandwidth → sharper shapes
    if sub.std() > 1e-6:
        kde = gaussian_kde(sub, bw_method=0.18)(xs_v)
        kde = kde / kde.max() * H_VIOL
        ax.fill_between(xs_v, k, k + kde,
                        color=fill_color,
                        alpha=0.32 if is_ours else 0.18,
                        lw=0, zorder=2)
        ax.plot(xs_v, k + kde,
                color=color,
                lw=0.7 if is_ours else 0.5, zorder=3)

    # raw strip below
    yj = k - STRIP_OFFSET + rng_jit.uniform(-0.04, 0.04, n)
    ax.scatter(sub, yj, s=3, color=color,
               alpha=0.40 if is_ours else 0.28, lw=0, zorder=3)

    # 5/95 line + median dot at row baseline
    ax.hlines(k, lo, hi, color=color,
              lw=1.0 if is_ours else 0.7, zorder=4)
    ax.scatter([median], [k], s=46 if is_ours else 30,
               color=color, edgecolor="white", lw=0.6, zorder=6)

    # right-margin median annotation
    ax.text(1.04, k, f"{median:.2f}",
            fontsize=5.8,
            color=color,
            fontweight="semibold" if is_ours else "normal",
            va="center", ha="left")

ax.set_yticks(np.arange(n_rows))
ax.set_yticklabels(order_f, fontsize=6.4)
for tl, m in zip(ax.get_yticklabels(), order_f):
    if m == "OpPert":
        tl.set_color(INDIGO_DARK); tl.set_fontweight("bold")
ax.set_xlim(0, 1.18)
ax.set_xlabel(r"per-pair top-20 DE Pearson, $\rho_{20}$")
ax.xaxis.grid(True, lw=0.4, color="0.92", zorder=0)
ax.set_axisbelow(True)
ax.text(-0.18, 1.04, "f", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top", color=INK)
ax.set_title(r"Per-method per-pair $\rho_{20}$ distribution (Norman)",
             fontsize=8.2, pad=8, color=INK, loc="left",
             fontweight="semibold")
"""
# end skip block

# save
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=400)
fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)
print("wrote:", OUT.with_suffix(".pdf"))
