"""Modern paper style for scFATE NeurIPS 2026 figures.

RResNet-aligned: clean schematic geometry, restrained palette, sans-serif math.
"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# ── core palette ──────────────────────────────────────────────────────
INK        = "#1a1d29"   # near-black text
INK_SOFT   = "#4a5568"   # secondary text
PAPER      = "#ffffff"
GRID       = "#e2e8f0"

ROT       = "#5b3df2"    # vivid indigo — rotations / scFATE
ROT_DARK  = "#3a25b8"
ROT_PALE  = "#ddd5ff"
ADD       = "#ff8c69"    # peach — additive baselines
ADD_DARK  = "#d05533"
BRACKET   = "#e02d8a"    # magenta — bracket / closure gap
TEAL      = "#14b8a6"    # accent / data manifold
TEAL_PALE = "#cdebe5"
GOLD      = "#f5b83d"    # validity highlight

PALETTE = {
    "scfate":   ROT,
    "scfate_dark": ROT_DARK,
    "scfate_pale": ROT_PALE,
    "additive": ADD,
    "additive_dark": ADD_DARK,
    "real":     INK,
    "accent":   BRACKET,
    "neutral":  "#cbd5e0",
    "manifold": TEAL,
    "manifold_pale": TEAL_PALE,
    "gold":     GOLD,
}

# Surface colormap: pale mint→pale violet, low saturation, perceptual
SURFACE_CMAP = LinearSegmentedColormap.from_list(
    "scfate_surface",
    [(0.92, 0.95, 0.94), (0.78, 0.86, 0.92), (0.78, 0.74, 0.94)],
    N=128,
)

CMAP_SEQ = "viridis"
CMAP_DENSITY = "magma_r"


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "mathtext.fontset": "cm",
        "mathtext.default": "it",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#3a4250",
        "axes.labelcolor": INK,
        "axes.titleweight": "regular",
        "axes.titlepad": 6,
        "xtick.color": "#3a4250",
        "ytick.color": "#3a4250",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "lines.markeredgewidth": 0,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "patch.linewidth": 0.5,
        "patch.edgecolor": "white",
    })


def thin_axes(ax, which=("left", "bottom")):
    for side, sp in ax.spines.items():
        sp.set_visible(side in which)
        if side in which:
            sp.set_linewidth(0.7)
            sp.set_color("#3a4250")


def hide_3d_chrome(ax):
    """Strip everything except the surface from a mplot3d axes."""
    ax.set_axis_off()
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_visible(False)
        axis.pane.fill = False
    ax.set_facecolor(PAPER)


def label_panel(ax, letter, x=-0.04, y=1.04, fontsize=12):
    fig = ax.figure
    bbox = ax.get_position()
    fx = bbox.x0 + x * bbox.width
    fy = bbox.y0 + y * bbox.height
    fig.text(fx, fy, letter, fontsize=fontsize, fontweight="bold",
             color=INK, va="bottom", ha="left")
