#!/usr/bin/env python3
"""Figure 2 -- Rotation geometry. Real-data rotation orbits in latent space."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from _style_modern import apply_style, hide_3d_chrome, PALETTE, INK, INK_SOFT

apply_style()

HERE = Path(__file__).resolve().parent
DATA = np.load(HERE / "data" / "fig2_geometry_data.npz", allow_pickle=True)

# CLI: argv[1] = hero combo name, argv[2] = output stem (without ext)
HERO_OVERRIDE = sys.argv[1] if len(sys.argv) > 1 else None
OUT_STEM      = sys.argv[2] if len(sys.argv) > 2 else "fig2_geometry"
OUT  = HERE / "out" / OUT_STEM
OUT.parent.mkdir(exist_ok=True)

z_train     = DATA["z_train"]                        # (5000, 128)
z_ctrl_mean = DATA["z_ctrl_mean"]                    # (128,)
ood_names   = [n.item() for n in DATA["ood_names"]]
ood_combo   = DATA["ood_is_combo"].astype(bool)
ood_z_rot   = DATA["ood_z_rot"]                      # (105, 128)
combo_names = [n.item() for n in DATA["combo_names"]]

# Real per-component rotation predictions (one per combo)
combo_z_a  = DATA["combo_z_a"]                       # (79, 128) = R_a · z̄
combo_z_b  = DATA["combo_z_b"]                       # (79, 128) = R_b · z̄

# Pick combo: CLI override OR default to CEBPA+JUN OR fallback
def find_pair():
    if HERO_OVERRIDE:
        if HERO_OVERRIDE in combo_names:
            a, b = HERO_OVERRIDE.split("+")
            return HERO_OVERRIDE, a, b
        rev = "+".join(HERO_OVERRIDE.split("+")[::-1])
        if rev in combo_names:
            a, b = rev.split("+")
            return rev, a, b
    for s in ("CEBPA+JUN", "JUN+CEBPA"):
        if s in combo_names:
            a, b = s.split("+")
            return s, a, b
    name = combo_names[0]
    a, b = name.split("+")
    return name, a, b

hero_combo, P1_NAME, P2_NAME = find_pair()
combo_idx_in_combos = combo_names.index(hero_combo)
combo_idx_in_ood    = ood_names.index(hero_combo)

# Real positions in 128-D — all from OpPert model output
z_bar      = z_ctrl_mean                             # control mean
R_p1_z     = combo_z_a[combo_idx_in_combos]          # R_{p₁} · z̄
combo_pred = ood_z_rot[combo_idx_in_ood]             # BCH(θ_{p₁},θ_{p₂}) · z̄

# Theta magnitudes (proxy: latent displacement norms)
theta_p1_norm   = float(np.linalg.norm(R_p1_z   - z_bar))
theta_p2_norm   = float(np.linalg.norm(combo_pred - R_p1_z))

print(f"[fig1A] hero={hero_combo} ({P1_NAME}, {P2_NAME})  "
      f"‖θ_{P1_NAME}‖₁₂₈={theta_p1_norm:.2f}  "
      f"‖θ_{P2_NAME}‖₁₂₈={theta_p2_norm:.2f}")

fit_data = np.vstack([z_train, z_bar[None], R_p1_z[None], combo_pred[None]])
pca = PCA(n_components=3, random_state=0).fit(fit_data)
def proj(z): return pca.transform(np.atleast_2d(z))

z_train_3 = proj(z_train)
z_bar_3   = proj(z_bar)[0]
R_p1_3    = proj(R_p1_z)[0]
combo_3   = proj(combo_pred)[0]

nn_full = NearestNeighbors(n_neighbors=200).fit(z_train)

def local_tangent(point_128, target_128):
    """Local PCA at point_128 (200-NN train cells), with e1 oriented
    toward target_128 in PCA(3) for clean visual flow."""
    _, idx = nn_full.kneighbors(point_128[None])
    local = z_train[idx[0]] - point_128
    lp = PCA(n_components=2, random_state=0).fit(local)
    # In 128-D the local axes:
    e1_128 = lp.components_[0]
    e2_128 = lp.components_[1]
    # Project local axes into PCA(3)
    e1_3 = pca.components_ @ e1_128
    e2_3 = pca.components_ @ e2_128
    # Re-orthonormalize in PCA(3)
    e1_3 /= np.linalg.norm(e1_3) + 1e-9
    e2_3 -= e1_3 * (e1_3 @ e2_3)
    e2_3 /= np.linalg.norm(e2_3) + 1e-9
    # Surface normal in PCA(3): cross of e1, e2 (in PCA(3))
    n_3 = np.cross(e1_3, e2_3)
    n_3 /= np.linalg.norm(n_3) + 1e-9
    # Re-orient e1 to point toward target (within tangent plane)
    target_3 = pca.transform(target_128[None])[0]
    base_3   = pca.transform(point_128[None])[0]
    d = target_3 - base_3
    d_tan = d - (d @ n_3) * n_3
    if np.linalg.norm(d_tan) > 1e-6:
        e1_new = d_tan / np.linalg.norm(d_tan)
        e2_new = np.cross(n_3, e1_new)
    else:
        e1_new, e2_new = e1_3, e2_3
    return n_3, e1_new, e2_new, base_3

N0, e10, e20, P0_3 = local_tangent(z_bar, R_p1_z)

# Parallel-transport the first plane's basis to R_p1_z (rather than
# trying local PCA there — R_p1_z is OFF the training-cell manifold
# so its 200-NN are biased "back toward the cluster" and give a
# degenerate normal). The transport rotation is the smallest 3-D
# rotation that aligns z̄ → R_p1_z directions.
def parallel_transport(e1, e2, n, p_from_3, p_to_3, center_3):
    v_from = p_from_3 - center_3
    v_to   = p_to_3 - center_3
    nf = np.linalg.norm(v_from); nt = np.linalg.norm(v_to)
    if nf < 1e-6 or nt < 1e-6:
        return e1, e2, n
    v_from = v_from / nf
    v_to   = v_to   / nt
    axis = np.cross(v_from, v_to)
    sin_a = float(np.linalg.norm(axis))
    if sin_a < 1e-6:
        return e1, e2, n
    axis = axis / sin_a
    cos_a = float(v_from @ v_to)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R3 = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
    return R3 @ e1, R3 @ e2, R3 @ n

P1_3 = pca.transform(R_p1_z[None])[0]
center_train = z_train_3.mean(axis=0)
e1_t, e2_t, N1 = parallel_transport(e10, e20, N0, P0_3, P1_3, center_train)

# Re-orient e1 of the transported basis to point toward combo (so the
# arrow direction reads naturally; e2 stays orthogonal in the plane).
combo_3_pre = pca.transform(combo_pred[None])[0]
d = combo_3_pre - P1_3
d_tan = d - (d @ N1) * N1
if np.linalg.norm(d_tan) > 1e-6:
    e11 = d_tan / np.linalg.norm(d_tan)
    e21 = np.cross(N1, e11)
else:
    e11, e21 = e1_t, e2_t

# Plane size scaled to a fraction of the trajectory step length so it
# reads at the right scale relative to the perturbation displacements.
step_len     = 0.5 * (np.linalg.norm(R_p1_3 - P0_3) + np.linalg.norm(combo_3 - R_p1_3))
PLANE_SIZE   = 0.40 * step_len
PLANE_LIFT   = 0.18 * PLANE_SIZE
ARROW_LIFT   = 0.05 * PLANE_SIZE
ARROW_LEN_1  = min(0.88 * PLANE_SIZE, 0.6 * np.linalg.norm(R_p1_3 - P0_3))
ARROW_LEN_2  = min(0.88 * PLANE_SIZE, 0.6 * np.linalg.norm(combo_3 - P1_3))

fig = plt.figure(figsize=(7.0, 4.0))
ax = fig.add_subplot(111, projection="3d")
ax.set_proj_type("ortho")
ax.set_position((0.00, 0.04, 1.00, 0.94))
hide_3d_chrome(ax)
ax.view_init(elev=22, azim=-58)

# View extent — tight on the trajectory + tangent planes
focus = np.vstack([
    P0_3[None], P1_3[None], combo_3[None],
    (P0_3 + 1.2 * PLANE_SIZE * e10)[None],
    (P0_3 - 1.2 * PLANE_SIZE * e10)[None],
])
center = focus.mean(axis=0)
extent = float(np.linalg.norm(focus - center, axis=1).max()) * 1.20
ax.set_xlim(center[0] - extent, center[0] + extent)
ax.set_ylim(center[1] - extent, center[1] + extent)
ax.set_zlim(center[2] - extent * 0.7, center[2] + extent * 0.7)
try:
    ax.set_box_aspect((1.0, 1.0, 0.7))
except Exception:
    pass

# the training-cell cloud in PCA(3)) — gives 3-D shape from data ─────
mean_train = z_train_3.mean(axis=0)
cov_train  = np.cov(z_train_3.T)
ev_w, ev_v = np.linalg.eigh(cov_train)
ord_      = np.argsort(-ev_w)
ev_w      = ev_w[ord_]
ev_v      = ev_v[:, ord_]
ellip_radii = 1.8 * np.sqrt(ev_w)                    # ~1.8-σ envelope
nu, nv = 70, 40
uu = np.linspace(0, 2 * np.pi, nu)
vv = np.linspace(0, np.pi, nv)
xs_e = np.outer(np.cos(uu), np.sin(vv))
ys_e = np.outer(np.sin(uu), np.sin(vv))
zs_e = np.outer(np.ones_like(uu), np.cos(vv))
sphere_local = np.stack([xs_e, ys_e, zs_e], axis=-1)
ellip = mean_train + sphere_local @ (ev_v * ellip_radii).T
EX = ellip[..., 0]; EY = ellip[..., 1]; EZ = ellip[..., 2]

# Soft pastel surface (data-derived shape, not schematic)
from matplotlib.colors import LightSource
ls = LightSource(azdeg=210, altdeg=42)
ENV_CMAP = LinearSegmentedColormap.from_list(
    "env_pastel",
    [(0.62, 0.74, 0.84), (0.78, 0.86, 0.78), (0.95, 0.85, 0.82)], N=64)
shaded = ls.shade(EZ, cmap=ENV_CMAP, vert_exag=1.0, blend_mode="soft")
ax.plot_surface(EX, EY, EZ, facecolors=shaded, rstride=2, cstride=2,
                linewidth=0, antialiased=True, alpha=0.35, zorder=1)
ax.plot_wireframe(EX, EY, EZ, rstride=10, cstride=12,
                  color="#3a4a5e", linewidth=0.35, alpha=0.30, zorder=2)

# Faint cell scatter inside the envelope
mask = np.linalg.norm(z_train_3 - center, axis=1) < extent * 1.10
inside = np.where(mask)[0]
rng = np.random.RandomState(0)
sub = rng.choice(inside, size=min(1800, len(inside)), replace=False)
ax.scatter(z_train_3[sub, 0], z_train_3[sub, 1], z_train_3[sub, 2],
           s=3, c=PALETTE["neutral"], alpha=0.18, depthshade=False,
           linewidths=0, zorder=3)

def draw_tangent_plane(ax, base, e1, e2, normal, size, n=4,
                       face="#dad4ec", edge="#22243a",
                       face_alpha=0.20, edge_alpha=0.95, lw=0.85,
                       lift=PLANE_LIFT):
    base_l = base + lift * normal
    corners = np.array([
        base_l + size * (-e1 - e2),
        base_l + size * ( e1 - e2),
        base_l + size * ( e1 + e2),
        base_l + size * (-e1 + e2),
    ])
    poly = Poly3DCollection([corners], facecolor=face, alpha=face_alpha,
                            edgecolor="none", linewidth=0)
    poly.set_zorder(8)
    ax.add_collection3d(poly)
    grid_segs = []
    for k in range(n + 1):
        t = -size + 2 * size * k / n
        grid_segs.append([base_l + t * e1 - size * e2,
                          base_l + t * e1 + size * e2])
        grid_segs.append([base_l - size * e1 + t * e2,
                          base_l + size * e1 + t * e2])
    grid_lc = Line3DCollection(grid_segs, colors=edge, linewidths=lw,
                               alpha=edge_alpha, zorder=9)
    ax.add_collection3d(grid_lc)
    border = np.vstack([corners, corners[:1]])
    ax.plot(border[:, 0], border[:, 1], border[:, 2],
            color=edge, linewidth=0.45, alpha=edge_alpha, zorder=10)
    ax.plot(*zip(base_l, base), color=edge, linewidth=0.6,
            alpha=0.55, linestyle=(0, (2, 3)), zorder=7)
    return base_l

P0_l = draw_tangent_plane(ax, P0_3, e10, e20, N0, PLANE_SIZE,
                          face="#cdc4eb", edge="#3a2f60", lw=0.22,
                          face_alpha=0.22)
P1_l = draw_tangent_plane(ax, P1_3, e11, e21, N1, PLANE_SIZE,
                          face="#9485c8", edge="#231948", lw=0.25,
                          face_alpha=0.32)

# Arrow positions (tip in tangent plane, in direction of next base)
V1_base = P0_l + ARROW_LIFT * N0
V2_base = P1_l + ARROW_LIFT * N1
V1_tip  = V1_base + ARROW_LEN_1 * e10
V2_tip  = V2_base + ARROW_LEN_2 * e11

def draw_arrow(ax, p0, p1, normal, color="#a31e25", shaft_w=0.06,
               head_len=0.22, head_w=0.18, alpha=1.0, zorder=20):
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return
    d = d / L
    perp = np.cross(normal, d)
    pn = np.linalg.norm(perp)
    if pn < 1e-9:
        perp = np.cross(np.array([0.0, 0.0, 1.0]), d)
        pn = np.linalg.norm(perp)
    perp = perp / pn
    head_base = p0 + (L - head_len) * d
    sw, hw = 0.5 * shaft_w, 0.5 * head_w
    verts = [
        tuple(p0 + sw * perp),
        tuple(head_base + sw * perp),
        tuple(head_base + hw * perp),
        tuple(p1),
        tuple(head_base - hw * perp),
        tuple(head_base - sw * perp),
        tuple(p0 - sw * perp),
    ]
    poly = Poly3DCollection([verts], facecolor=color, edgecolor=color,
                            linewidth=0.5, alpha=alpha, joinstyle="miter")
    poly.set_zorder(zorder)
    ax.add_collection3d(poly)

# Scale arrow geometry to plane size — finer strokes
arrow_shaft_w = 0.025 * PLANE_SIZE
arrow_head_l  = 0.16  * PLANE_SIZE
arrow_head_w  = 0.10  * PLANE_SIZE
draw_arrow(ax, V1_base, V1_tip, N0, color="#b22a2f",
           shaft_w=arrow_shaft_w, head_len=arrow_head_l, head_w=arrow_head_w)
draw_arrow(ax, V2_base, V2_tip, N1, color="#b22a2f",
           shaft_w=arrow_shaft_w, head_len=arrow_head_l, head_w=arrow_head_w)

def smooth_arc(ax, p_start, p_end, normal_end, color="#d77820",
               lw=1.8, n=80, head_len=None, head_w=None,
               bow=0.25, zorder=18):
    """Quadratic Bezier from p_start to p_end with a control point lifted
    midway. No projection onto a fake surface — pure 3-D curve between
    two real PCA(3) points."""
    if head_len is None:
        head_len = 0.18 * PLANE_SIZE
    if head_w is None:
        head_w = 0.12 * PLANE_SIZE
    midpoint = 0.5 * (p_start + p_end)
    chord = p_end - p_start
    chord_len = np.linalg.norm(chord) + 1e-9
    # Bow direction: average of the two end-normals, perpendicular to chord
    bow_dir = normal_end.copy()
    bow_dir -= (bow_dir @ chord) / (chord_len ** 2) * chord
    bow_dir /= np.linalg.norm(bow_dir) + 1e-9
    ctrl = midpoint + bow * chord_len * bow_dir

    ts = np.linspace(0, 1, n)
    pts = np.array([
        (1 - t) ** 2 * p_start + 2 * (1 - t) * t * ctrl + t ** 2 * p_end
        for t in ts
    ])
    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    cutoff = max(cum[-1] - head_len, 1e-3)
    keep = cum <= cutoff
    arc_pts = pts[keep] if keep.sum() >= 2 else pts[:2]
    ax.plot(arc_pts[:, 0], arc_pts[:, 1], arc_pts[:, 2],
            color=color, linewidth=lw, solid_capstyle="round",
            zorder=zorder, alpha=0.96)
    # Arrowhead
    d = pts[-1] - arc_pts[-1]
    nd = np.linalg.norm(d)
    if nd < 1e-9:
        return
    d /= nd
    perp = np.cross(normal_end, d)
    pn = np.linalg.norm(perp)
    if pn < 1e-9:
        perp = np.cross(np.array([0.0, 0.0, 1.0]), d)
        pn = np.linalg.norm(perp)
    perp /= pn
    base = arc_pts[-1]
    head = Poly3DCollection([[
        tuple(pts[-1]),
        tuple(base + 0.5 * head_w * perp),
        tuple(base - 0.5 * head_w * perp),
    ]], facecolor=color, edgecolor=color, linewidth=0.4)
    head.set_zorder(zorder + 2)
    ax.add_collection3d(head)

# Arc endpoints: tangent-vector tips → next plane base (lifted, real)
N1_at_landing = N1
N2_at_landing = local_tangent(combo_pred, R_p1_z)[0]
P2_l = combo_3 + PLANE_LIFT * N2_at_landing
smooth_arc(ax, V1_tip, P1_l, normal_end=N1, color="#d77820", lw=1.0,
           head_len=0.13 * PLANE_SIZE, head_w=0.08 * PLANE_SIZE)
smooth_arc(ax, V2_tip, P2_l, normal_end=N2_at_landing, color="#d77820",
           lw=1.0, head_len=0.13 * PLANE_SIZE, head_w=0.08 * PLANE_SIZE)

for P in (P0_3, P1_3, combo_3):
    ax.scatter(P[0], P[1], P[2], s=18, c=INK,
               depthshade=False, zorder=22, linewidths=0)

def label_with_line(ax, anchor_3d, text, off=(0, 0), align="left",
                    fontsize=10, color=INK, weight="regular",
                    style="italic", line=True):
    x2, y2, _ = proj3d.proj_transform(anchor_3d[0], anchor_3d[1],
                                       anchor_3d[2], ax.get_proj())
    arrow = dict(arrowstyle="-", color="black", linewidth=0.5,
                 linestyle=(0, (3, 3)),
                 shrinkA=0.0, shrinkB=2.0) if line else None
    ax.annotate(text, (float(x2), float(y2)),
                xycoords=ax.transData,
                xytext=off, textcoords="offset points",
                fontsize=fontsize, color=color,
                fontweight=weight, fontstyle=style,
                ha=align, va="center",
                arrowprops=arrow,
                zorder=30, clip_on=False)

label_with_line(ax, P0_3, r"$\bar z$",
                off=(-22, -16), align="right", fontsize=14, weight="bold")

label_with_line(ax, V1_tip, rf"$\theta_{{\rm {P1_NAME}}}$",
                off=(-12, 24), align="right", fontsize=11,
                color="#7a1e22", weight="bold")

label_with_line(ax, P1_3,
                rf"$R_{{\rm {P1_NAME}}}\,\bar z$",
                off=(0, -28), align="center", fontsize=11)

label_with_line(ax, V2_tip, rf"$\theta_{{\rm {P2_NAME}}}$",
                off=(16, 24), align="left", fontsize=11,
                color="#7a1e22", weight="bold")

label_with_line(ax, combo_3,
                rf"$R_{{\rm {P2_NAME}}}R_{{\rm {P1_NAME}}}\,\bar z$",
                off=(34, 0), align="left", fontsize=11)

# Tangent space ID, anchored to LEFT plane
T0_anchor = P0_l + PLANE_SIZE * (-e10 + e20)
label_with_line(ax, T0_anchor,
                r"$T_{\bar z}\mathcal{M}\!\cong\!\mathfrak{so}(4)^{32}$",
                off=(-30, 28), align="right", fontsize=10,
                color=PALETTE["scfate_dark"], weight="medium")

# Manifold ID
M_anchor = z_train_3[sub[np.argmax(z_train_3[sub, 0])]]
label_with_line(ax, M_anchor,
                r"$\mathcal{M}\subset\mathbb{R}^{128}$",
                off=(20, 18), align="left", fontsize=12,
                weight="medium", line=False)

# Panel letter
fig.text(0.02, 0.95, "a", fontsize=18, fontweight="bold", color=INK,
         ha="left", va="top")

# Real-data caption (small, italic)
fig.text(0.02, 0.05,
         f"PCA(3) of training-cell latents · {hero_combo} = real OOD Norman combo · "
         r"$\bar z, R_p\bar z, R_{p'}R_p\bar z$ from OpPert; tangent planes from local PCA on 200-NN cells.",
         fontsize=8.0, color=INK_SOFT, fontstyle="italic",
         ha="left", va="bottom", linespacing=1.4)

fig.savefig(str(OUT) + ".svg", bbox_inches="tight")
fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
fig.savefig(str(OUT) + ".png", dpi=400, bbox_inches="tight")

(HERE / "out" / "fig2_geometry_data.json").write_text(json.dumps({
    "panel_a": {
        "framework": "matplotlib",
        "design": "real-data PCA(3): training-cell cloud + real positions + local-PCA tangent planes",
        "hero_combo": hero_combo,
        "p1": P1_NAME, "p2": P2_NAME,
        "theta_p1_norm_128d": theta_p1_norm,
        "theta_p2_norm_128d": theta_p2_norm,
        "convention": (
            "Manifold = PCA(3) of training-cell latents (real). "
            "Base points z̄, R_{p1} z̄, R_{p2}R_{p1} z̄ from OpPert model output. "
            "Tangent planes = top-2 local PCs of 200-NN of each base point in 128-D. "
            "Arrow length scaled to plane size for legibility."
        ),
    },
}, indent=2))
print(f"[fig1A] saved {OUT}.pdf, {OUT}.png  hero combo: {hero_combo}")
