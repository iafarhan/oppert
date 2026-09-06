#!/usr/bin/env python3
"""Figure 1 -- Headline. Non-additivity, bracket-epistasis, uncertainty split."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde, spearmanr
from pathlib import Path

from figstyle import (
    apply_style, METHOD_COLORS, DATASET_COLORS, INK,
    BAR_ALPHA, BAR_EDGE_LW,
    BRACKET_LW, BRACKET_COLOR,
    PANEL_LABEL_SIZE, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE,
    darker,
)

apply_style()
import matplotlib as mpl
mpl.rcParams.update({
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

DATA = Path(__file__).parent / "data" / "csvs_real_norman"
FIG = Path(__file__).parent / "out"
FIG.mkdir(exist_ok=True)

db = pd.read_csv(DATA / "commutator_epistasis.csv")

# Load flow uncertainty data
with open(DATA.parent / "uncertainty_per_pert.json") as f:
    unc_raw = json.load(f)

OPPERT = METHOD_COLORS["OpPert"]
CORAL = "#C0564E"
CORAL_LT = "#EACED0"
MUTED = "#999999"
FAINT = "#CCCCCC"
BRACKET_BLUE = "#3366AA"
BRACKET_BLUE_LT = "#C5D5EA"


def draw_violin(ax, data, pos, color, color_lt, width=0.32):
    if len(data) < 4:
        return
    kde = gaussian_kde(data, bw_method=0.35)
    y = np.linspace(data.min() - 0.05 * np.ptp(data),
                    data.max() + 0.05 * np.ptp(data), 200)
    d = kde(y)
    d = d / d.max() * width
    ax.fill_betweenx(y, pos - d, pos + d, color=color_lt, alpha=0.50, lw=0)
    ax.plot(pos - d, y, color=color, lw=0.55, alpha=0.65)
    ax.plot(pos + d, y, color=color, lw=0.55, alpha=0.65)


fig = plt.figure(figsize=(7.2, 2.55), facecolor='white')
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                       width_ratios=[0.9, 1.1, 0.9],
                       left=0.055, right=0.98, bottom=0.19, top=0.83)

_T = 9    # title
_L = 8    # axis labels
_K = 7    # ticks
_P = 11   # panel labels
lbl = dict(fontsize=_P, fontweight='bold', color=INK, va='top', ha='left')

ax = fig.add_subplot(gs[0])
ax.text(-0.15, 1.13, 'a', transform=ax.transAxes, **lbl)

ratio = db['measured_epistasis'] / db['additive_delta_norm']

ax.hist(ratio, bins=14, density=True, color=CORAL_LT, edgecolor=darker(CORAL),
        linewidth=0.4, alpha=0.65, zorder=1)
kde_r = gaussian_kde(ratio, bw_method=0.35)
xg = np.linspace(ratio.min() - 0.05, ratio.max() + 0.05, 300)
dens = kde_r(xg)
ax.plot(xg, dens, color=CORAL, lw=1.0, zorder=2)

for v in ratio:
    ax.plot([v, v], [-0.08 * dens.max(), -0.025 * dens.max()],
            color=CORAL, lw=0.25, alpha=0.4, solid_capstyle='round')

med = np.median(ratio)
ax.axvline(med, color=CORAL, lw=0.6, ls='--', alpha=0.45, ymin=0.05, ymax=0.85,
           zorder=3)
ax.text(med + 0.02, dens.max() * 0.92,
        f'median = {med:.2f}', fontsize=_K, color=CORAL, va='top',
        fontstyle='italic')

ax.set_xlabel(r'$\|\delta_{pq} - \delta_p - \delta_q\|\;/\;\|\delta_p + \delta_q\|$',
              labelpad=3, fontsize=_L)
ax.set_ylabel('Density', labelpad=2, fontsize=_L)
ax.set_title('Held-out pairs are non-additive', color=INK, pad=6,
             fontsize=_T, fontweight='bold')
ax.set_xlim(ratio.min() - 0.08, ratio.max() + 0.06)
ax.set_ylim(-0.14 * dens.max(), dens.max() * 1.12)
ax.text(0.97, 0.95, f'n = {len(ratio)} pairs', transform=ax.transAxes,
        fontsize=_K, va='top', ha='right', color=MUTED)

ax = fig.add_subplot(gs[1])
ax.text(-0.12, 1.13, 'b', transform=ax.transAxes, **lbl)

eps = db['epsilon'].values
epi = db['measured_epistasis'].values

# Spearman correlation
rho, pval = spearmanr(eps, epi)

# Background shading for low vs high bracket
ax.axvspan(eps.min() - 0.01, np.median(eps), color=BRACKET_BLUE_LT, alpha=0.12, zorder=0)
ax.axvspan(np.median(eps), eps.max() + 0.01, color=CORAL_LT, alpha=0.18, zorder=0)

# Scatter with size proportional to additive norm (larger = bigger effect)
add_norm = db['additive_delta_norm'].values
sizes = 12 + 28 * (add_norm - add_norm.min()) / np.ptp(add_norm)
ax.scatter(eps, epi, s=sizes, color=BRACKET_BLUE, alpha=0.55,
           edgecolors=darker(BRACKET_BLUE, 0.6), linewidths=0.35, zorder=3)

# LOWESS-like trend: robust polynomial
z = np.polyfit(eps, epi, 1)
xfit = np.linspace(eps.min() - 0.005, eps.max() + 0.005, 100)
ax.plot(xfit, np.polyval(z, xfit), color=darker(BRACKET_BLUE, 0.5), lw=1.4,
        ls='--', alpha=0.6, zorder=2)

# Annotation — top left for rho
ax.text(0.03, 0.97,
        f'$\\rho_s$ = {rho:.2f}\n$p$ = {pval:.1e}',
        transform=ax.transAxes, fontsize=_K + 1,
        va='top', ha='left', color=INK,
        fontweight='bold', linespacing=1.4,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=FAINT, alpha=0.85))

# "no pair supervision" callout
ax.text(0.97, 0.03,
        'no pair supervision\nrequired',
        transform=ax.transAxes, fontsize=_K,
        va='bottom', ha='right', color=MUTED,
        fontstyle='italic', linespacing=1.3)

ax.set_xlabel(r'Bracket score $\varepsilon(p,q)$',
              labelpad=3, fontsize=_L)
ax.set_ylabel(r'Measured epistasis $\|\delta_{pq} - \delta_p - \delta_q\|$',
              labelpad=2, fontsize=_L)
ax.set_title('Lie bracket predicts\nnon-additivity from singles', color=INK, pad=6,
             fontsize=_T, fontweight='bold')

ax = fig.add_subplot(gs[2])
ax.text(-0.16, 1.13, 'c', transform=ax.transAxes, **lbl)

k562_pp = unc_raw["k562"]["per_pert"]
unc_k562 = np.array([p.get("unc_topk_norm", p.get("unc_std_topk", 0)) for p in k562_pp])
da_k562 = np.array([p["da"] for p in k562_pp])
n_k562 = len(k562_pp)

order = np.argsort(unc_k562)
q25 = max(1, n_k562 // 4)
conf = da_k562[order[:q25]] * 100
unc_bot = da_k562[order[-q25:]] * 100

CONF_COLOR = DATASET_COLORS["K562"]
UNC_COLOR = CORAL

draw_violin(ax, conf, 0, CONF_COLOR, "#E5DBFF", width=0.36)
draw_violin(ax, unc_bot, 1, UNC_COLOR, CORAL_LT, width=0.36)

for pos, data, col in [(0, conf, CONF_COLOR), (1, unc_bot, UNC_COLOR)]:
    q1, med_v, q3 = np.percentile(data, [25, 50, 75])
    ax.plot([pos, pos], [q1, q3], color=col, lw=1.8, alpha=0.45,
            solid_capstyle='round', zorder=3)
    ax.plot([pos - 0.09, pos + 0.09], [med_v, med_v],
            color=col, lw=1.4, solid_capstyle='round', zorder=4)

m_c, m_u = conf.mean(), unc_bot.mean()
for pos, m, col, ha, xoff in [(0, m_c, CONF_COLOR, 'right', -8),
                                (1, m_u, UNC_COLOR, 'left', 8)]:
    ax.plot(pos, m, 'o', color=col, ms=3.5, zorder=5,
            markeredgecolor='white', markeredgewidth=0.5)
    ax.annotate(f'{m:.1f}%', (pos, m), xytext=(xoff, 0),
                textcoords='offset points', fontsize=_K,
                fontweight='bold', color=col, ha=ha, va='center')

gap = m_c - m_u
mid_y = (m_c + m_u) / 2
ax.plot([0.42, 0.42], [m_u, m_c], '-', color=INK, lw=BRACKET_LW * 0.5, zorder=2)
ax.plot([0.39, 0.45], [m_c, m_c], '-', color=INK, lw=0.4)
ax.plot([0.39, 0.45], [m_u, m_u], '-', color=INK, lw=0.4)
ax.text(0.48, mid_y, f'+{gap:.0f}pp', fontsize=_K,
        color=INK, fontweight='bold', va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Top 25%\n(confident)', 'Bottom 25%\n(uncertain)'],
                    fontsize=_K, linespacing=1.3)
ax.set_ylabel(r'DA$^{\rm DEG}$ (%)', labelpad=2, fontsize=_L)
ax.set_title('Flow variance flags\nunreliable predictions (K562)', color=INK, pad=6,
             fontsize=_T, fontweight='bold')
ax.set_xlim(-0.55, 1.55)
ax.set_ylim(-5, 110)

ax.text(0, -4, f'n={len(conf)}', fontsize=_K - 1, color=MUTED, ha='center')
ax.text(1, -4, f'n={len(unc_bot)}', fontsize=_K - 1, color=MUTED, ha='center')

for ext in ("png", "pdf", "svg"):
    fig.savefig(FIG / f"fig_headline_v5_bracket.{ext}", dpi=300,
                facecolor='white', bbox_inches='tight')
plt.close()
print("Saved fig_headline_v5_bracket.{png,pdf,svg}")
