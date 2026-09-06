"""Figure 5 -- Uncertainty calibration. Flow variance vs prediction accuracy."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, rankdata

from figstyle import (
    apply_style, METHOD_COLORS, DATASET_COLORS, INK,
    BAR_ALPHA, BAR_EDGE_LW,
    ERRBAR_COLOR, ERRBAR_CAPSIZE, ERRBAR_LW,
    BRACKET_LW, BRACKET_COLOR, BRACKET_FONTSIZE,
    PANEL_LABEL_SIZE, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE,
    darker,
)

apply_style()

H = Path(__file__).parent
OUT = H / "out" / "fig_comp3_uncertainty"
OUT.parent.mkdir(exist_ok=True)

with open(H / "data" / "uncertainty_per_pert.json") as f:
    raw = json.load(f)

DS_MAP = {"Norman": "norman", "K562": "k562", "RPE1": "rpe1"}
datasets = {}
for label, key in DS_MAP.items():
    if key not in raw:
        continue
    pp = raw[key]["per_pert"]
    da  = np.array([p["da"] for p in pp])
    err = np.array([p["err_1mcos"] for p in pp])
    unc = np.array([p.get("unc_topk_norm", p.get("unc_std_topk", 0)) for p in pp])
    datasets[label] = dict(da=da, err=err, unc=unc, n=len(pp))

ds_order = [d for d in ["Norman", "K562", "RPE1"] if d in datasets]
rng = np.random.default_rng(42)

# OpPert SPCC (computed from real data)
oppert_spcc = {}
for ds in ds_order:
    d = datasets[ds]
    rho = spearmanr(d["unc"], d["err"]).statistic
    oppert_spcc[ds] = rho * 100

# Published baselines (PRESCRIBE Fig 4)
BASELINE_SPCC = {
    "PRESCRIBE":  {"Norman": 67.19, "K562": 12.43, "RPE1": 28.69},
    "GEARS-Drop": {"Norman": 42.63, "K562": 11.23, "RPE1":  7.81},
    "GEARS-Ens":  {"Norman": 36.66, "K562": 11.53, "RPE1": 23.02},
}

METHODS_UNC = ["OpPert", "PRESCRIBE", "GEARS-Drop", "GEARS-Ens"]
UNC_COLORS = {
    "OpPert":     METHOD_COLORS["OpPert"],
    "PRESCRIBE":  METHOD_COLORS["PRESCRIBE"],
    "GEARS-Drop": METHOD_COLORS["GEARS"],
    "GEARS-Ens":  "#66C2A5",  # lighter green variant
}

fig = plt.figure(figsize=(17, 4.5))
gs = fig.add_gridspec(1, 3, wspace=0.28, left=0.06, right=0.97,
                      top=0.80, bottom=0.14,
                      width_ratios=[1.0, 1.0, 1.0])

# Panel a — SPCC grouped bars
ax_a = fig.add_subplot(gs[0, 0])
fig.text(0.01, 0.96, "a", fontsize=PANEL_LABEL_SIZE, fontweight="bold", color=INK)

n_methods = len(METHODS_UNC)
x_pos = np.arange(len(ds_order))
bar_w = 0.75 / n_methods

for mi, method in enumerate(METHODS_UNC):
    offset = (mi - (n_methods - 1) / 2) * bar_w
    vals = []
    for ds in ds_order:
        if method == "OpPert":
            vals.append(oppert_spcc[ds])
        else:
            vals.append(BASELINE_SPCC[method].get(ds, 0))

    color = UNC_COLORS[method]
    ax_a.bar(x_pos + offset, vals, bar_w * 0.88,
             color=color, alpha=BAR_ALPHA,
             edgecolor=darker(color), linewidth=BAR_EDGE_LW, zorder=3)

    # OpPert bootstrap CI
    if method == "OpPert":
        for j, ds in enumerate(ds_order):
            d = datasets[ds]
            boots = []
            for _ in range(2000):
                idx = rng.choice(d["n"], d["n"], replace=True)
                boots.append(spearmanr(d["unc"][idx], d["err"][idx]).statistic * 100)
            ci_lo = np.percentile(boots, 2.5)
            ci_hi = np.percentile(boots, 97.5)
            ax_a.errorbar(x_pos[j] + offset, vals[j],
                          yerr=[[vals[j] - ci_lo], [ci_hi - vals[j]]],
                          fmt="none", ecolor=ERRBAR_COLOR, elinewidth=ERRBAR_LW,
                          capsize=ERRBAR_CAPSIZE, capthick=ERRBAR_LW, zorder=4)

ax_a.set_xticks(x_pos)
ax_a.set_xticklabels(ds_order, fontsize=TICK_SIZE)
ax_a.set_ylabel("SPCC: Spearman(confidence, accuracy) %", fontsize=LABEL_SIZE)
ax_a.set_title("Uncertainty calibration across methods",
               fontsize=TITLE_SIZE, fontweight="bold", color=INK, pad=18, loc="center")
ax_a.text(0.5, 1.01, "PRESCRIBE/GEARS from published Fig 4",
          transform=ax_a.transAxes, fontsize=TICK_SIZE - 2,
          ha="center", va="bottom", color="#999999", style="italic")
ax_a.set_axisbelow(True)

ax_a.legend(
    handles=[
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=UNC_COLORS[m], markersize=7,
               label=("OpPert (ours)" if m == "OpPert" else m))
        for m in METHODS_UNC
    ],
    loc="upper right", fontsize=LEGEND_SIZE - 2, handletextpad=0.3,
    borderpad=0.4, labelspacing=0.25, frameon=True,
    facecolor="white", edgecolor="0.88", framealpha=0.95,
).get_frame().set_linewidth(0.4)

# Panel b — DA^DEG by confidence tier (3 tiers × 3 datasets)
ax_b = fig.add_subplot(gs[0, 1])
fig.text(0.365, 0.96, "b", fontsize=PANEL_LABEL_SIZE, fontweight="bold", color=INK)

TIERS = ["Top 25%\n(confident)", "Middle 50%", "Bottom 25%\n(uncertain)"]
n_ds = len(ds_order)
x_pos_b = np.arange(len(TIERS))
bar_w_b = 0.75 / n_ds

for di, ds in enumerate(ds_order):
    d = datasets[ds]
    ds_color = DATASET_COLORS[ds]
    offset = (di - (n_ds - 1) / 2) * bar_w_b

    # Sort by confidence (low unc = high confidence)
    order = np.argsort(d["unc"])
    n = d["n"]
    q25 = max(1, n // 4)
    q75 = n - q25

    tier_slices = [
        order[:q25],           # top 25% most confident
        order[q25:q75],        # middle 50%
        order[q75:],           # bottom 25% most uncertain
    ]

    means, ci_los, ci_his = [], [], []
    for sl in tier_slices:
        da_tier = d["da"][sl] * 100
        means.append(da_tier.mean())
        boots = [rng.choice(da_tier, len(da_tier), replace=True).mean()
                 for _ in range(2000)]
        ci_los.append(np.percentile(boots, 2.5))
        ci_his.append(np.percentile(boots, 97.5))

    means = np.array(means)
    ci_los = np.array(ci_los)
    ci_his = np.array(ci_his)

    ax_b.bar(x_pos_b + offset, means, bar_w_b * 0.88,
             color=ds_color, alpha=BAR_ALPHA,
             edgecolor=darker(ds_color), linewidth=BAR_EDGE_LW, zorder=3)

    ax_b.errorbar(x_pos_b + offset, means,
                  yerr=[means - ci_los, ci_his - means],
                  fmt="none", ecolor=ERRBAR_COLOR, elinewidth=ERRBAR_LW,
                  capsize=ERRBAR_CAPSIZE, capthick=ERRBAR_LW, zorder=4)

    # Gain annotation on top-25% bar
    gain = means[0] - means[2]
    ax_b.text(x_pos_b[0] + offset, means[0] + 2.0, f"+{gain:.0f}pp",
              ha="center", va="bottom", fontsize=TICK_SIZE - 2,
              fontweight="bold", color=ds_color)

ax_b.set_xticks(x_pos_b)
ax_b.set_xticklabels(TIERS, fontsize=TICK_SIZE - 1)
ax_b.set_ylabel(r"DA$^{\rm DEG}$ (%)", fontsize=LABEL_SIZE)
ax_b.set_ylim(45, 105)
ax_b.set_title("Flow variance stratifies\nprediction quality",
               fontsize=TITLE_SIZE, fontweight="bold", color=INK, pad=8, loc="center")
ax_b.set_axisbelow(True)

ax_b.legend(
    handles=[
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=DATASET_COLORS[ds], markersize=7, label=ds)
        for ds in ds_order
    ],
    loc="upper right", fontsize=LEGEND_SIZE - 2, handletextpad=0.3,
    borderpad=0.4, labelspacing=0.25, frameon=True,
    facecolor="white", edgecolor="0.88", framealpha=0.95,
).get_frame().set_linewidth(0.4)

# Panel c — Selective prediction curves (all 3 datasets)
ax_c = fig.add_subplot(gs[0, 2])
fig.text(0.69, 0.96, "c", fontsize=PANEL_LABEL_SIZE, fontweight="bold", color=INK)

fracs = np.linspace(0.05, 1.0, 40)

for ds in ds_order:
    d = datasets[ds]
    ds_color = DATASET_COLORS[ds]
    order = np.argsort(d["unc"])  # most confident first
    da_sorted = d["da"][order]

    curve = np.array([da_sorted[:max(1, int(f * d["n"]))].mean() * 100
                      for f in fracs])

    ax_c.plot(fracs * 100, curve, color=ds_color, lw=2.0, label=ds, zorder=4)

    # Baseline (full set)
    baseline = d["da"].mean() * 100
    ax_c.axhline(baseline, color=ds_color, lw=0.6, ls=":", alpha=0.5, zorder=1)

    # Gain annotation at 25%
    idx25 = np.argmin(np.abs(fracs - 0.25))
    gain = curve[idx25] - curve[-1]
    ax_c.text(28, curve[idx25] + 0.8, f"+{gain:.0f}pp\nat 25%",
              fontsize=TICK_SIZE - 2, fontweight="bold", color=ds_color,
              va="bottom")

ax_c.set_xlim(0, 105)
ax_c.set_xlabel("Predictions retained (%)", fontsize=LABEL_SIZE)
ax_c.set_ylabel(r"DA$^{\rm DEG}$ of retained set (%)", fontsize=LABEL_SIZE)
ax_c.set_title("Selective prediction\nimproves accuracy",
               fontsize=TITLE_SIZE, fontweight="bold", color=INK, pad=8, loc="center")
ax_c.yaxis.grid(True, lw=0.3, alpha=0.3, color="0.80", zorder=0)
ax_c.set_axisbelow(True)

ax_c.legend(loc="lower left", fontsize=LEGEND_SIZE - 1, handlelength=1.5,
            frameon=True, facecolor="white", edgecolor="0.88",
            framealpha=0.95).get_frame().set_linewidth(0.4)

for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT.with_suffix(f".{ext}"), bbox_inches="tight")
    print(f"wrote {OUT.with_suffix(f'.{ext}')}")
plt.close(fig)

# metadata
meta = {}
for ds in ds_order:
    d = datasets[ds]
    order = np.argsort(d["unc"])
    q25 = max(1, d["n"] // 4)
    meta[ds] = {
        "n": d["n"],
        "oppert_spcc": round(oppert_spcc[ds], 1),
        "da_all": round(d["da"].mean() * 100, 1),
        "da_top25": round(d["da"][order[:q25]].mean() * 100, 1),
        "da_bot25": round(d["da"][order[-q25:]].mean() * 100, 1),
        "gap_top25_bot25_pp": round(
            (d["da"][order[:q25]].mean() - d["da"][order[-q25:]].mean()) * 100, 1),
    }

with open(OUT.with_suffix(".json"), "w") as f:
    json.dump(meta, f, indent=2)
    print(f"wrote {OUT.with_suffix('.json')}")

print("Done.")
print(json.dumps(meta, indent=2))
