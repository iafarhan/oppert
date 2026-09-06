"""Shared figure style for OpPert paper. Colorblind-safe palette."""
import matplotlib as mpl
import numpy as np

METHOD_COLORS = {
    "OpPert":       "#1A1A1A",
    "PRESCRIBE":    "#E69F00",
    "GEARS":        "#009E73",
    "scGPT":        "#0072B2",
    "scFoundation": "#56B4E9",
    "CPA":          "#CC79A7",
    "Linear":       "#C0C0C0",
    "DecoderOnly":  "#888888",
    "SAMS-VAE":     "#A65628",
    "AvgKnown":     "#E0E0E0",
    "CellOracle":   "#BDB76B",
    "GraphVCI":     "#9970AB",
}

DATASET_COLORS = {
    "Norman": "#4C72B0",
    "K562":   "#8172B2",
    "RPE1":   "#4F9D9D",
}

INK = "#1A1A1A"

BAR_HEIGHT_GROUPED = 0.75
BAR_HEIGHT_SINGLE  = 0.85
BAR_EDGE_LW        = 0.6
BAR_ALPHA           = 0.9

SEED_DOT_SIZE       = 35
SEED_DOT_EDGE       = "black"
SEED_DOT_FACE       = "white"
SEED_DOT_EDGE_LW    = 0.8

BRACKET_LW          = 1.2
BRACKET_COLOR       = INK
BRACKET_FONTSIZE    = 10

ERRBAR_COLOR        = "black"
ERRBAR_CAPSIZE      = 4
ERRBAR_LW           = 1.0

TITLE_SIZE          = 11
LABEL_SIZE          = 10
TICK_SIZE           = 9
LEGEND_SIZE         = 9
PANEL_LABEL_SIZE    = 14


def darker(hex_color, factor=0.7):
    """Return a darker version of hex_color (for bar edges)."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


def apply_style():
    """Set global rcParams. Call once at top of each figure script."""
    mpl.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset":     "dejavusans",
        "font.size":            TICK_SIZE,
        "axes.linewidth":       0.6,
        "axes.labelsize":       LABEL_SIZE,
        "axes.titlesize":       TITLE_SIZE,
        "axes.labelpad":        4,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "xtick.labelsize":      TICK_SIZE,
        "ytick.labelsize":      TICK_SIZE,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.size":     3,
        "ytick.major.size":     3,
        "xtick.major.width":    0.6,
        "ytick.major.width":    0.6,
        "lines.solid_capstyle": "round",
        "savefig.dpi":          300,
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
    })


def method_label(m):
    """Display label for legend."""
    return "OpPert (ours)" if m == "OpPert" else m


def seed_dots(ax, x, values, color=SEED_DOT_FACE, jitter_std=0.04, rng=None):
    """Overlay seed dots on a bar. values = array of per-seed values."""
    if rng is None:
        rng = np.random.default_rng(0)
    jitter = rng.normal(0, jitter_std, size=len(values))
    ax.scatter(x + jitter, values, s=SEED_DOT_SIZE,
               facecolors=SEED_DOT_FACE, edgecolors=SEED_DOT_EDGE,
               linewidths=SEED_DOT_EDGE_LW, zorder=6)
