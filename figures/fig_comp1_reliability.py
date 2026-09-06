"""Figure 4 -- Reliability analysis. PCA map, bracket recovery, and operating curve."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap
from scipy.stats import pearsonr

H = Path(__file__).parent
NPZ = H / "data" / "h2h_npz"
OUT = H / "out" / "fig_comp1_reliability"
OUT.parent.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.labelpad": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.0,
    "lines.solid_capstyle": "round",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.4,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

INDIGO = "#3B5BDB"
INDIGO_DARK = "#1F3A93"
RUST = "#D9822B"
INK = "#1A1A1A"

# Norman 2019 functional categories
CAT = {
    # Granulocyte / apoptosis
    "CEBPA": "Granulocyte", "CEBPE": "Granulocyte", "CEBPB": "Granulocyte",
    "SPI1":  "Granulocyte", "HES7":  "Granulocyte",
    # Erythroid markers
    "KLF1": "Erythroid", "BPGM": "Erythroid", "GATA1": "Erythroid",
    "ZBTB1": "Erythroid", "UBASH3B": "Erythroid", "CNN1": "Erythroid",
    "IGDCC3": "Erythroid", "ZBTB25": "Erythroid", "SAMD1": "Erythroid",
    "ZBTB10": "Erythroid", "CDKN1A": "Erythroid", "CDKN1B": "Erythroid",
    "CDKN1C": "Erythroid",
    # Megakaryocyte markers
    "ETS2": "Megakaryocyte", "FOSB": "Megakaryocyte",
    "COL2A1": "Megakaryocyte", "ZNF318": "Megakaryocyte",
    "FOXF1": "Megakaryocyte", "TMSB4X": "Megakaryocyte",
    "RUNX1T1": "Megakaryocyte", "BCL2L11": "Megakaryocyte",
    "MAP2K3": "Megakaryocyte", "SLC38A2": "Megakaryocyte",
    "SLC4A1": "Megakaryocyte", "TSC22D1": "Megakaryocyte",
    "ZC3HAV1": "Megakaryocyte",
    # Pioneer factors
    "FOXA1": "Pioneer", "FOXA3": "Pioneer", "FOXM1": "Pioneer",
    "FOXL2": "Pioneer", "HOXB9": "Pioneer", "HOXB13": "Pioneer",
    "MEIS1": "Pioneer", "PLK4": "Pioneer", "STIL": "Pioneer",
    "POU3F2": "Pioneer", "ISL2": "Pioneer", "DUSP9": "Pioneer",
    "DLX2": "Pioneer", "SNAI1": "Pioneer",
    # Progrowth
    "MAPK1": "Progrowth", "MAP4K5": "Progrowth",
    "ARRDC3": "Progrowth", "ELMSAN1": "Progrowth",
    "TGFBR2": "Progrowth", "CBL": "Progrowth", "KIF2C": "Progrowth",
    "MAP4K3": "Progrowth", "SLC6A9": "Progrowth", "SET": "Progrowth",
    "TBX3": "Progrowth", "C19ORF26": "Progrowth", "C3ORF72": "Progrowth",
    "NCL": "Progrowth", "AHR": "Progrowth",
}

CAT_COLORS = {
    "Granulocyte":      "#EC4D9D",
    "Erythroid":        "#8B3FBF",
    "Megakaryocyte":    "#E14C4C",
    "Pioneer":          "#F18C26",
    "Progrowth":        "#3DAA52",
    "High erythroid":   "#2F7BD9",
    "Mixed/uncertain":  "#F4C430",
    "Other":            "#9AA0A6",
}

def categorize(name: str) -> str:
    if "+" in name:
        parts = name.split("+")
        cats = [CAT.get(p, "Other") for p in parts]
        non_other = [c for c in cats if c != "Other"]
        if non_other:
            return non_other[0]
        return "Other"
    return CAT.get(name, "Other")

# (a) BCH composition landscape — GEARS Fig 4c idiom.
# All 75 training singles + all C(75,2)=2775 BCH-composed pairs of singles,
# each rendered as the latent z_treated = R · z_ctrl_mean. We then t-SNE
# this 2850×128 cloud, k-means cluster, and colour by Norman category.
# Training singles are marked with × (seen).
LANDSCAPE = H / "data" / "bch_landscape.npz"
L = np.load(LANDSCAPE, allow_pickle=True)
z_all       = L["z_all"]                       # (N, 128)
names       = list(L["names_all"])             # (N,)
is_combo    = L["is_combo"].astype(bool)       # (N,)
n_perts     = z_all.shape[0]
M_singles   = int(L["M_singles"])
combo_mask  = is_combo

# direction-normalise for shape, not magnitude
z_n = z_all / np.linalg.norm(z_all, axis=1, keepdims=True).clip(1e-9)

# k-means on full landscape
K_CLUSTERS = 5
km = KMeans(n_clusters=K_CLUSTERS, random_state=0, n_init=20).fit(z_n)
cluster_id = km.labels_

# embed
emb = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.65,
                spread=1.4, metric="cosine",
                random_state=0).fit_transform(z_n)

# Re-cluster on the UMAP 2D coordinates so colours match visual position.
# Bumped to 8 for more granularity; we backfill names with extra categories
# that aren't in the Norman base set.
K_CLUSTERS = 7
km = KMeans(n_clusters=K_CLUSTERS, random_state=0, n_init=20).fit(emb)
cluster_id = km.labels_

# Cluster naming from dominant gene-family in members
from collections import Counter
remaining = ["Granulocyte", "Erythroid", "Megakaryocyte", "Pioneer",
             "Progrowth"]
extra_pool = ["High erythroid", "Mixed/uncertain"]
order_clusters = sorted(range(K_CLUSTERS),
                        key=lambda ci: -np.sum(cluster_id == ci))
cluster_names = {}
extra_iter = iter(extra_pool)
for ci in order_clusters:
    members = [names[i] for i in range(n_perts) if cluster_id[i] == ci]
    flavs = Counter()
    for m in members:
        for g in m.split("+"):
            f = CAT.get(g, "Other")
            if f != "Other":
                flavs[f] += 1
    pick = next((f for f, _ in flavs.most_common() if f in remaining), None)
    if pick is None:
        pick = next(extra_iter, "Other")
    else:
        remaining.remove(pick)
    cluster_names[ci] = pick

# (b) bracket × measured epistasis — independent OOD-combo data
def canon(s: str) -> str:
    return "+".join(sorted(s.split("+")))

geom = np.load(H / "data" / "fig2_geometry_data.npz", allow_pickle=True)
gs   = np.load(H / "data" / "fig2_genespace_data.npz", allow_pickle=True)

add_gene  = gs["combo_additive_gene"]
meas_gene = gs["combo_measured_gene"]
top20_g = np.argsort(-np.abs(meas_gene), axis=1)[:, :20]
ep_proper = np.array([
    np.sqrt(np.mean((meas_gene[i, top20_g[i]] - add_gene[i, top20_g[i]]) ** 2))
    for i in range(len(gs["combo_names"]))
])
gs_lookup = dict(zip([canon(n) for n in gs["combo_names"]], ep_proper))

br_lookup = dict(zip(
    [canon(n) for n in geom["combo_names"]],
    geom["combo_comm_norm"].astype(float),
))

DROP = {"CEBPA+CEBPE"}
br_arr, ep_arr = [], []
for cn, br_v in br_lookup.items():
    if cn in DROP or cn not in gs_lookup:
        continue
    br_arr.append(br_v)
    ep_arr.append(gs_lookup[cn])
br_arr = np.array(br_arr); ep_arr = np.array(ep_arr)
r_be, p_be = pearsonr(br_arr, ep_arr)
from scipy.stats import spearmanr as _sp
sr_be, sp_be = _sp(br_arr, ep_arr)

# (c) bracket-epistasis scatter (data loaded inline below)

# figure
fig = plt.figure(figsize=(14.0, 4.4))
outer = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.25, 1.05],
                         wspace=0.22,
                         left=0.04, right=0.97, bottom=0.12, top=0.88)
ax_a = fig.add_subplot(outer[0, 0])   # OOD-only BCH landscape
ax_b = fig.add_subplot(outer[0, 1])   # bracket lift
ax_c = fig.add_subplot(outer[0, 2])   # reliability

ax = ax_a
# colors per cluster, using the Norman palette by inferred label
cluster_color = {ci: CAT_COLORS.get(cluster_names[ci], "#9AA0A6")
                 for ci in range(K_CLUSTERS)}

# Panel (a): OOD-only — BCH-composed combos (no seen training perts shown).
sel_combo = np.where(combo_mask)[0]
emb_combo = emb[sel_combo]

# Soft KDE hulls per cluster for the modern "biological-region" feel
from scipy.stats import gaussian_kde as _gk
xpad = (emb_combo[:, 0].max() - emb_combo[:, 0].min()) * 0.08
ypad = (emb_combo[:, 1].max() - emb_combo[:, 1].min()) * 0.08
xs = np.linspace(emb_combo[:, 0].min() - xpad,
                 emb_combo[:, 0].max() + xpad, 320)
ys = np.linspace(emb_combo[:, 1].min() - ypad,
                 emb_combo[:, 1].max() + ypad, 320)
XG, YG = np.meshgrid(xs, ys)
positions = np.vstack([XG.ravel(), YG.ravel()])
for ci in range(K_CLUSTERS):
    sel = np.where((cluster_id == ci) & combo_mask)[0]
    if len(sel) < 8:
        continue
    pts = emb[sel].T
    kde = _gk(pts, bw_method=0.20)
    Z = kde(positions).reshape(XG.shape); Z = Z / Z.max()
    ax.contourf(XG, YG, Z, levels=[0.18, 1.001],
                colors=[cluster_color[ci]],
                alpha=0.13, zorder=1)
    ax.contour(XG, YG, Z, levels=[0.18],
               colors=[cluster_color[ci]],
               linewidths=0.45, alpha=0.55, zorder=2)

# Dots
combo_colors = [cluster_color[cluster_id[i]] for i in sel_combo]
ax.scatter(emb_combo[:, 0], emb_combo[:, 1],
           s=11, c=combo_colors, alpha=0.78, lw=0, zorder=4)

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(ys[0], ys[-1])

# corner legend with cluster names by colour swatch
from matplotlib.lines import Line2D as _L2D
order_legend = sorted(range(K_CLUSTERS),
                      key=lambda ci: -np.sum(cluster_id == ci))
handles = [_L2D([], [], marker="o", lw=0, ms=5.2,
                mfc=cluster_color[ci], mec="white", mew=0.5,
                label=cluster_names[ci])
           for ci in order_legend]
leg_a = ax_a.legend(handles=handles, loc="upper left",
                    fontsize=7.5, frameon=True,
                    handlelength=0.6, handletextpad=0.35,
                    borderpad=0.35, labelspacing=0.18,
                    facecolor="white", edgecolor="0.80",
                    framealpha=0.95)
leg_a.get_frame().set_linewidth(0.6)

ax.set_xlabel("UMAP 1", fontsize=7.5)
ax.set_ylabel("UMAP 2", fontsize=7.5)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(True)
    s.set_linewidth(0.6)
    s.set_color("0.50")
ax.text(-0.06, 1.08, "d", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="top", color=INK)
ax.set_title("Predicted atlas of unseen pairwise perturbations",
             fontsize=9, pad=8, color=INK, loc="left",
             fontweight="semibold")
# subtitle removed

# NOTE — accuracy-vs-difficulty plot moved to comparison figure F3.
# This panel introspects what OpPert LEARNED: the rotation eigenangles.
__SKIPPED__ = """
# Norman OOD combos split by how many of the two component genes were
# observed during training: seen2 (easy) > seen1 > seen0 (hardest).
# Plot median top-20 DE Pearson at each difficulty for every method,
# bootstrap-CI ribbon, OpPert highlighted.
pm_b = pd.read_csv(H / "data" / "csvs_real_norman" / "pair_metrics.csv")
SPLIT_ORDER = [
    "both genes observed, pair held out",      # seen2
    "exactly one gene observed",               # seen1
    "neither gene observed",                   # seen0
]
SPLIT_LABEL = {
    "both genes observed, pair held out": "seen2\nboth genes\nobserved",
    "exactly one gene observed":           "seen1\none gene\nobserved",
    "neither gene observed":               "seen0\nneither gene\nobserved",
}

METHOD_DISPLAY = {
    "OpPert":        "OpPert",
    "GEARS":         "GEARS",
    "AttentionPert": "AttentionPert",
    "CPA":           "CPA",
    "linear":        "Linear",
    "additive":      "Additive",
    "mean":          "Mean",
}

# bootstrap median per (method, split)
rng = np.random.default_rng(0)
B = 1000
xs_b = np.arange(len(SPLIT_ORDER))
method_order = ["OpPert", "GEARS", "AttentionPert", "CPA",
                "linear", "additive", "mean"]
method_color = {
    "OpPert":        INDIGO_DARK,
    "GEARS":         "#A0522D",
    "AttentionPert": "#3DA34D",
    "CPA":           "#9C36B5",
    "linear":        "#7C8794",
    "additive":      "#B0BEC5",
    "mean":          "#CFD8DC",
}

for m in method_order:
    sub = pm_b[pm_b.method == m]
    means, los, his = [], [], []
    for sp in SPLIT_ORDER:
        v = sub[sub.split_category == sp]["value"].to_numpy()
        n = len(v)
        boots = np.array([np.median(rng.choice(v, n, replace=True))
                          for _ in range(B)])
        means.append(float(np.median(v)))
        los.append(float(np.percentile(boots, 2.5)))
        his.append(float(np.percentile(boots, 97.5)))
    means = np.array(means); los = np.array(los); his = np.array(his)
    is_ours = (m == "OpPert")
    color = method_color[m]
    lw = 2.0 if is_ours else 0.9
    ms = 6.0 if is_ours else 3.5
    a = 1.0 if is_ours else 0.85
    ax_b.fill_between(xs_b, los, his,
                      color=color, alpha=0.18 if is_ours else 0.10,
                      lw=0, zorder=2 if is_ours else 1)
    ax_b.plot(xs_b, means, color=color, lw=lw, alpha=a,
              marker="o", ms=ms, mec="white", mew=0.6,
              zorder=5 if is_ours else 3,
              label=METHOD_DISPLAY[m])

ax_b.set_xticks(xs_b)
ax_b.set_xticklabels([SPLIT_LABEL[sp] for sp in SPLIT_ORDER],
                     fontsize=7.5)
ax_b.set_ylabel(r"per-pair top-20 DE Pearson, $\rho_{20}$",
                fontsize=8)
ax_b.set_xlim(-0.4, len(SPLIT_ORDER) - 0.6)
ax_b.tick_params(axis="x", length=0, pad=2)
ax_b.set_ylim(0.20, max(0.72,
                        ax_b.get_ylim()[1]))
for s in ("top", "right"):
    ax_b.spines[s].set_visible(False)
ax_b.yaxis.grid(True, lw=0.4, color="0.92", zorder=0)
ax_b.set_axisbelow(True)

ax_b.legend(loc="upper right", fontsize=7.5, frameon=False,
            handlelength=1.2, handletextpad=0.4,
            labelspacing=0.30, borderpad=0.2,
            ncol=2, columnspacing=0.6)
"""
geom_b = np.load(H / "data" / "fig2_geometry_data.npz", allow_pickle=True)
ea = geom_b["eigenangles"].astype(float)
n_ea = len(ea)

PI8 = np.pi / 8
PI4 = np.pi / 4
PI2 = np.pi / 2

from scipy.stats import gaussian_kde as _gk_b
xs_ea = np.linspace(0, max(PI2 + 0.10, ea.max() + 0.05), 700)
kde_ea = _gk_b(ea, bw_method=0.16)(xs_ea)
kde_ea = kde_ea / kde_ea.max()

# Soft graduated "BCH-comfort" wash (0 → π/4)
RUST = "#A0522D"
n_steps = 60
for i in range(n_steps):
    a0 = i / n_steps
    a1 = (i + 1) / n_steps
    x0 = a0 * PI4
    x1 = a1 * PI4
    ax_b.axvspan(x0, x1, color=RUST, alpha=0.10 * (1 - a0 * 0.7),
                 lw=0, zorder=1)

# Filled density — gradient look via stacked alphas
for k_alpha, frac in [(0.10, 1.00), (0.18, 0.65), (0.30, 0.30)]:
    cut = np.where(kde_ea >= 1 - frac, kde_ea, np.nan)
    ax_b.fill_between(xs_ea, 0, cut, color=INDIGO,
                      alpha=k_alpha, lw=0, zorder=3)
ax_b.plot(xs_ea, kde_ea, color=INDIGO_DARK, lw=1.7, zorder=5)

# Histogram strip at the very bottom for raw count read
hist, edges = np.histogram(ea, bins=80, range=(0, PI2 + 0.05))
hist = hist / hist.max() * 0.06
for i in range(len(hist)):
    ax_b.add_patch(plt.Rectangle((edges[i], -0.10),
                                  edges[i + 1] - edges[i], hist[i],
                                  facecolor=INDIGO_DARK, alpha=0.55,
                                  edgecolor="none", zorder=2))

# reference vertical lines (BCH-validity rings at π/8, π/4, π/2)
for x, lbl, sub_lbl, col in [
    (PI8, r"$\pi/8$", "tight",                "#A0522D"),
    (PI4, r"$\pi/4$", "BCH$_2$ comfort",      "#7C2D12"),
    (PI2, r"$\pi/2$", "BCH$_2$ validity",     "#3E1A0F"),
]:
    ax_b.axvline(x, color=col, lw=1.0, ls=(0, (3, 2)),
                 zorder=6, alpha=0.85)
    ax_b.text(x + 0.010, 1.05, lbl, color=col, fontsize=8,
              fontweight="semibold", va="top", ha="left")
    ax_b.text(x + 0.010, 0.99, sub_lbl, color=col, fontsize=7,
              va="top", ha="left")

median_ea = float(np.median(ea))
frac_pi8 = float((ea < PI8).mean())
frac_pi4 = float((ea < PI4).mean())

# Stats in a light-bordered box (matches legend style on A and C)
from matplotlib.patches import Rectangle as _Rect
box_x, box_y, box_w, box_h = 0.49, 0.92, 0.25, 0.16
ax_b.add_patch(_Rect((box_x, box_y - box_h), box_w, box_h,
                     transform=ax_b.transAxes,
                     facecolor="white", edgecolor="0.80",
                     lw=0.6, alpha=0.95, zorder=10))
ax_b.text(box_x + 0.012, box_y - 0.025,
          rf"$\widetilde{{\theta}}$ = {median_ea:.3f} rad",
          transform=ax_b.transAxes,
          fontsize=6.6, fontweight="semibold",
          color=INDIGO_DARK, va="top", ha="left", zorder=11)
ax_b.text(box_x + 0.012, box_y - 0.075,
          rf"{100*frac_pi8:.0f}% inside $\pi/8$",
          transform=ax_b.transAxes,
          fontsize=7.5, color=INK, va="top", ha="left", zorder=11)
ax_b.text(box_x + 0.012, box_y - 0.118,
          rf"{100*frac_pi4:.1f}% inside $\pi/4$",
          transform=ax_b.transAxes,
          fontsize=7.5, color=INK, va="top", ha="left", zorder=11)

# median tick on x-axis
ax_b.axvline(median_ea, color=INDIGO_DARK, lw=0.7, ls=":",
             alpha=0.7, zorder=4)

ax_b.set_xlim(-0.02, PI2 + 0.25)
ax_b.set_ylim(-0.13, 1.10)
ax_b.set_xlabel(r"learned rotation eigenangle $\theta$  (rad)",
                fontsize=8)
ax_b.set_ylabel(r"share of learned rotations", fontsize=8)
ax_b.set_yticks([])
ax_b.set_xticks([0, PI8, PI4, PI2])
ax_b.set_xticklabels([
    "0", r"$\pi/8$", r"$\pi/4$", r"$\pi/2$"], fontsize=8)
for s in ("top", "right", "left"):
    ax_b.spines[s].set_visible(False)
ax_b.spines["bottom"].set_color("0.65")
ax_b.tick_params(axis="x", color="0.65")

ax_b.text(-0.06, 1.08, "e", transform=ax_b.transAxes,
          fontsize=12, fontweight="bold", va="top", color=INK)
ax_b.set_title(r"Learned rotations stay within the local BCH regime",
               fontsize=9, pad=8, color=INK, loc="left",
               fontweight="semibold")
# subtitle removed

ax = ax_c

with open(H / "data" / "panel_d_cam_e115.json") as _fd:
    _Dc = json.load(_fd)

_genes_c = _Dc["genes"]
_Mc = np.array(_Dc["matrix"])
_gc = len(_genes_c)
_vmax_c = float(_Mc.max())

# Blue sequential to match figure's indigo palette
from matplotlib.colors import LinearSegmentedColormap as _LSC
_blue_cmap = _LSC.from_list("indigo_seq",
    ["#F7F9FC", "#C5D5EA", "#7BA3CC", "#3B5BDB", "#1F3A93", "#0C1D4A"])
im = ax.imshow(_Mc, cmap=_blue_cmap, vmin=0, vmax=_vmax_c,
               aspect="equal", interpolation="nearest", origin="upper")

ax.set_xticks(np.arange(_gc))
ax.set_yticks(np.arange(_gc))
ax.set_xticklabels(_genes_c, rotation=45, ha="right", fontsize=7.5,
                   rotation_mode="anchor")
ax.set_yticklabels(_genes_c, fontsize=7.5)
ax.tick_params(length=0)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
cbar.set_label(r"$\|\frac{1}{2}[\theta_p,\theta_q]\|_F$" "\n(Frobenius norm)",
               fontsize=7.5, labelpad=4)
cbar.ax.tick_params(labelsize=5.8, length=2)
cbar.outline.set_linewidth(0.4)

for s in ax.spines.values():
    s.set_visible(False)

ax.text(-0.06, 1.08, "f", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="top", color=INK)
ax.set_title(r"Commutator norm  $\|[\theta_p,\theta_q]\|_F$",
             fontsize=9, pad=8, color=INK, loc="left",
             fontweight="semibold")
# subtitle removed

n_pairs_c = _gc

# save
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=400)
fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)

# companion
companion = {
    "panel_a": {
        "metric": "PCA-2D of post-perturbation gene-delta predictions",
        "n_perts": int(n_perts),
        "n_combos": int(combo_mask.sum()),
        "embedding": "t-SNE on cosine of L2-normalised θ_p",
        "categories": list(CAT_COLORS.keys()),
    },
    "panel_b": {
        "metric": "bracket norm vs gene-space top-20 DE residual ||measured - additive||",
        "n_combos_with_data": int(len(br_arr)),
        "pearson_r": float(r_be),
        "spearman_r": float(sr_be),
        "p_value": float(p_be),
        "outlier_removed": "CEBPA+CEBPE",
    },
    "panel_c": {
        "metric": "rotation generator cosine similarity, sorted by Norman category",
        "n_singles": n_pairs_c,
    },
}
with open(OUT.with_suffix("._data.json"), "w") as f:
    json.dump(companion, f, indent=2)
print(json.dumps(companion, indent=2))
print("\nwrote:", OUT.with_suffix(".pdf"))
