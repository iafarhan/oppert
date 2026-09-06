#!/usr/bin/env python3
"""Fair comparison: rotation KRR vs direct delta KRR on identical OOD sets."""
import json, sys, os, logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

os.chdir(str(Path(__file__).resolve().parent.parent))

# FLOW_EVAL_PATCH: parse optional --flow_ckpt and --multiview from argv
_argv = list(sys.argv[1:])
FLOW_CKPT = None
FLOW_CKPTS_EXTRA = []  # additional flow ckpts for multi-seed ensemble (paths beyond the primary)
MV_PATH = None
FLOW_NSAMPLES_OVERRIDE = None
FLOW_NSTEPS_OVERRIDE = None
FLOW_SIGMA_INFERENCE_OVERRIDE = None  # if set, overrides per-member sigma used as initial noise scale at inference
FLOW_TAG_OVERRIDE = None
BRACKET_SCALE_OVERRIDE = None
FLOW_BLEND_ALPHA = None  # blend alpha*KRR_prior + (1-alpha)*flow_mean in theta space
FLOW_INTEGRATOR = "euler"  # ODE integrator: "euler" (default) or "heun"
FLOW_CFG_SCALE = 1.0  # classifier-free guidance scale w (1.0 = off, >1.0 strengthens cond)
FLOW_TIME_SCHEDULE = "uniform"  # 'uniform' | 'cosine_end' | 'cosine_start' | 'poly2_end' | 'poly2_start'
FLOW_ANTITHETIC = False  # pair ensemble samples as (noise, -noise) for MC variance reduction
OOD_JSON_PATH = None  # optional JSON override for OOD pert set (keys: train_drugs, ood_drugs)
_cleaned = []
i = 0
while i < len(_argv):
    a = _argv[i]
    if a == "--flow_ckpt" and i + 1 < len(_argv):
        FLOW_CKPT = _argv[i + 1]; i += 2
    elif a == "--flow_ckpts" and i + 1 < len(_argv):
        _paths = [p for p in _argv[i + 1].split(",") if p]
        if _paths:
            FLOW_CKPT = _paths[0]
            FLOW_CKPTS_EXTRA = _paths[1:]
        i += 2
    elif a == "--multiview" and i + 1 < len(_argv):
        MV_PATH = _argv[i + 1]; i += 2
    elif a == "--flow_nsamples" and i + 1 < len(_argv):
        FLOW_NSAMPLES_OVERRIDE = int(_argv[i + 1]); i += 2
    elif a == "--flow_nsteps" and i + 1 < len(_argv):
        FLOW_NSTEPS_OVERRIDE = int(_argv[i + 1]); i += 2
    elif a == "--flow_sigma_inference" and i + 1 < len(_argv):
        FLOW_SIGMA_INFERENCE_OVERRIDE = float(_argv[i + 1]); i += 2
    elif a == "--flow_tag_suffix" and i + 1 < len(_argv):
        FLOW_TAG_OVERRIDE = _argv[i + 1]; i += 2
    elif a == "--bracket_scale" and i + 1 < len(_argv):
        BRACKET_SCALE_OVERRIDE = float(_argv[i + 1]); i += 2
    elif a == "--flow_blend_alpha" and i + 1 < len(_argv):
        FLOW_BLEND_ALPHA = float(_argv[i + 1]); i += 2
    elif a == "--integrator" and i + 1 < len(_argv):
        FLOW_INTEGRATOR = _argv[i + 1]; i += 2
    elif a == "--cfg_scale" and i + 1 < len(_argv):
        FLOW_CFG_SCALE = float(_argv[i + 1]); i += 2
    elif a == "--flow_time_schedule" and i + 1 < len(_argv):
        FLOW_TIME_SCHEDULE = _argv[i + 1]; i += 2
    elif a == "--flow_antithetic" and i + 1 < len(_argv):
        FLOW_ANTITHETIC = _argv[i + 1].lower() in ("1", "true", "yes", "y", "t"); i += 2
    elif a == "--ood_json" and i + 1 < len(_argv):
        OOD_JSON_PATH = _argv[i + 1]; i += 2
    else:
        _cleaned.append(a); i += 1
_argv = _cleaned

DATASET = _argv[0] if len(_argv) > 0 else "replogle_k562"
CKPT_PATH = Path(_argv[1]) if len(_argv) > 1 else None
EMB_PATH = Path(_argv[2]) if len(_argv) > 2 else Path("data/gene_embeddings/genept_bge_large.pt")
print(f"[flow-fair] FLOW_CKPT={FLOW_CKPT}  MV_PATH={MV_PATH}")

# Default checkpoints
if CKPT_PATH is None:
    if "norman" in DATASET:
        CKPT_PATH = Path("runs/exp_block_norman_500_s1/checkpoints/OpPert_epoch015_best.pt")
    else:
        CKPT_PATH = Path("runs/blk_rep_v2_s1/checkpoints/OpPert_epoch050_best.pt")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
else:
    DEVICE = torch.device("cpu")
SEED = 42
TOPK_DE = 50

torch.manual_seed(SEED)
np.random.seed(SEED)

from OpPert.evaluate_full import _load_config, _build_data, _load_model
from omegaconf import OmegaConf

ds_cfg, dl_cfg = _load_config(DATASET)
ds_cfg = OmegaConf.merge(ds_cfg, {"pert_subsample": None})
dm = _build_data(ds_cfg, dl_cfg)
dataset = dm.dataset
obs = dataset.adata.obs
n_genes = dataset.genes.shape[1]
pert_key = getattr(dataset.cfg, "perturbation_key", "perturbation")

model, meta = _load_model(str(CKPT_PATH), "tuned", DEVICE)
model.eval()

train_indices = dataset.indices.get("train", [])
ood_indices = dataset.indices.get("ood", dataset.indices.get("test", []))

# --ood_json override: redefine train/ood pert sets from a JSON split file
# (needed when the dataset's built-in ood split != the flow's OOD split,
# e.g. SciPlex3 where backbone saw all 187 drugs but flow uses a 25-drug holdout).
if OOD_JSON_PATH is not None:
    import json as _json
    with open(OOD_JSON_PATH) as _f:
        _spec = _json.load(_f)
    _ood_set = set(_spec.get("ood_drugs", []))
    _train_set = set(_spec.get("train_drugs", []))
    logger.info(f"--ood_json override: train_drugs={len(_train_set)}, ood_drugs={len(_ood_set)}")
    # OOD eval cells: every cell whose pert name is in ood_drugs (and is not control)
    _all_idx = np.arange(len(obs))
    _not_control = obs['control'].astype(int).values != 1
    _pert_vals = obs[pert_key].values
    train_perts = set([p for p in _train_set if p in set(_pert_vals)])
    ood_perts_raw = set([p for p in _ood_set if p in set(_pert_vals)])
else:
    # Get OOD perturbation names (same filtering as perturbench)
    train_perts = set(obs.iloc[train_indices].loc[
        obs.iloc[train_indices]['control'].astype(int) != 1, pert_key].unique())
    ood_perts_raw = set(obs.iloc[ood_indices].loc[
        obs.iloc[ood_indices]['control'].astype(int) != 1, pert_key].unique())

# Filter: need ≥5 cells, no combos for Replogle, skip controls
ood_perts_filtered = []
for pname in sorted(ood_perts_raw):
    if "control" in pname.lower() or "dmso" in pname.lower():
        continue
    if OOD_JSON_PATH is not None:
        # Pull cells from the FULL dataset, ignoring the built-in split bucketing.
        ood_mask = (obs[pert_key].values == pname) & \
                   (obs['control'].astype(int).values != 1)
        ood_idx = np.nonzero(ood_mask)[0]
    else:
        ood_mask = (obs.iloc[ood_indices][pert_key].values == pname) & \
                   (obs.iloc[ood_indices]['control'].astype(int).values != 1)
        ood_idx = np.array(ood_indices)[ood_mask]
    if ood_idx.size < 5:
        continue
    ood_perts_filtered.append((pname, ood_idx))

logger.info(f"Dataset: {DATASET}")
logger.info(f"Train perts: {len(train_perts)}, OOD perts: {len(ood_perts_filtered)}")

raw_emb = torch.load(EMB_PATH, map_location="cpu")
emb_data = {k.upper(): v for k, v in raw_emb.items()}

if OOD_JSON_PATH is not None:
    # When using a JSON split, pull controls from the whole dataset (not just the
    # built-in train bucket) since the built-in split may segregate drugs differently.
    ctrl_mask_full = obs['control'].astype(int).values == 1
    ctrl_idx = np.nonzero(ctrl_mask_full)[0]
else:
    ctrl_mask = obs.iloc[train_indices]['control'].astype(int).values == 1
    ctrl_idx = np.array(train_indices)[ctrl_mask]
rng = np.random.RandomState(SEED)
sample_ctrl = rng.choice(ctrl_idx, min(500, len(ctrl_idx)), replace=False)
ctrl_raw_mean = dataset.genes[sample_ctrl].mean(0)

ctrl_genes_dev = dataset.genes[sample_ctrl].to(DEVICE)
with torch.no_grad():
    z_ctrl = model.get_latent(ctrl_genes_dev)
z_basal_mean = z_ctrl.mean(0)

rotation = model.rotation
all_gen_params = rotation.generator_params.detach()
gen_dim = all_gen_params.shape[1]
block_size = rotation.block_size
num_blocks = rotation.num_blocks
basis = getattr(rotation, "basis", None)  # BlockRotation only; Cayley/Householder have no fixed basis
pert_name_to_idx = {v: k for k, v in dataset.pert_dict.items()}

logger.info(f"Generator dim: {gen_dim}, block_size: {block_size}")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

# Method 1: Rotation KRR (BGE → generator params)
rot_train_embs, rot_train_targets, rot_train_names = [], [], []
# Method 2: Direct Delta KRR (BGE → expression deltas)
dd_train_embs, dd_train_deltas, dd_train_names = [], [], []

for pname in sorted(train_perts):
    if "+" in pname:
        continue  # Skip combos for training (consistent with both methods)
    pu = pname.upper()
    if pu not in emb_data:
        continue

    # For rotation: need generator params
    if pname in pert_name_to_idx:
        pidx = pert_name_to_idx[pname]
        rot_train_embs.append(emb_data[pu].numpy())
        rot_train_targets.append(all_gen_params[pidx].cpu().numpy())
        rot_train_names.append(pname)

    # For direct delta: need expression delta
    train_treated_mask = (obs.iloc[train_indices][pert_key].values == pname)
    train_treated_idx = np.array(train_indices)[train_treated_mask]
    if train_treated_idx.size < 3:
        continue
    treated_mean = dataset.genes[train_treated_idx].mean(0)
    delta = (treated_mean - ctrl_raw_mean).cpu().numpy()
    dd_train_embs.append(emb_data[pu].numpy())
    dd_train_deltas.append(delta)
    dd_train_names.append(pname)

rot_X = np.stack(rot_train_embs)
rot_Y = np.stack(rot_train_targets)
dd_X = np.stack(dd_train_embs)
dd_Y = np.stack(dd_train_deltas)

logger.info(f"Rotation KRR training: {rot_X.shape[0]} perts → {rot_Y.shape[1]}D generators")
logger.info(f"Direct Delta KRR training: {dd_X.shape[0]} perts → {dd_Y.shape[1]}D deltas")

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    "gamma": [0.001, 0.01, 0.1, None],
}

# Rotation KRR
scaler_rot = StandardScaler()
rot_X_sc = scaler_rot.fit_transform(rot_X)
krr_rot = GridSearchCV(KernelRidge(kernel="rbf"), param_grid,
                       cv=min(5, len(rot_X)), scoring="neg_mean_squared_error", n_jobs=-1)
logger.info("Fitting Rotation CV-KRR...")
krr_rot.fit(rot_X_sc, rot_Y)
logger.info(f"  Rotation KRR best: {krr_rot.best_params_}")
# FLOW_EVAL_PATCH: load flow net + multiview if provided, define flow_predict
_flow_net = None; _mv_embed = None; _mv_genes = None; _flow_sigma = 1.0; _flow_nsamples = 32; _flow_nsteps = 40
if FLOW_NSAMPLES_OVERRIDE is not None:
    _flow_nsamples = FLOW_NSAMPLES_OVERRIDE
if FLOW_NSTEPS_OVERRIDE is not None:
    _flow_nsteps = FLOW_NSTEPS_OVERRIDE
_flow_prior_kind = "none"; _flow_krr = None
_flow_members = []  # multi-seed ensemble: list of dicts with net, sigma, prior_kind, krr

def _build_flow_member(ckpt_path):
    """Load a flow ckpt into a member dict for the multi-seed ensemble."""
    from oppert.flow import BlockTransformerVelocityNet, ConditionalVelocityNet
    fd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = fd["v_net_state_dict"]
    velocity_arch = fd.get("velocity_arch")
    if velocity_arch is None:
        velocity_arch = "transformer" if "pos_embed" in sd else "mlp"
    if velocity_arch == "transformer":
        net = BlockTransformerVelocityNet(
            theta_dim=fd["theta_dim"], d_embed=fd["d_embed"],
            d_token=fd.get("tx_d_token", 128),
            d_cond=fd.get("d_hidden", 512),
            n_layers=fd.get("tx_n_layers", 4),
            n_heads=fd.get("tx_n_heads", 4),
            mlp_mult=fd.get("tx_mlp_mult", 4.0),
        ).to(DEVICE)
        logger.info(f"[ensemble] {ckpt_path}: transformer")
    else:
        head_kind = fd.get("head_kind")
        if head_kind is None:
            head_kind = "skew_block" if any(k.startswith("out_proj.split") or k.startswith("out_proj.to_coef") for k in sd.keys()) else "flat"
        d_block_head = fd.get("d_block_head", 16)
        if head_kind == "skew_block" and "out_proj.split.weight" in sd:
            d_block_head = sd["out_proj.split.weight"].shape[0] // 32
        net = ConditionalVelocityNet(
            theta_dim=fd["theta_dim"], d_embed=fd["d_embed"],
            d_hidden=fd.get("d_hidden", 512), n_blocks=fd.get("n_blocks", 4),
            head_kind=head_kind, d_block_head=d_block_head,
        ).to(DEVICE)
        logger.info(f"[ensemble] {ckpt_path}: {head_kind}")
    net.load_state_dict(sd)
    net.eval()
    prior_kind = fd.get("prior", "none") or "none"
    krr = None
    if prior_kind == "krr":
        import pickle as _pickle
        from sklearn.kernel_ridge import KernelRidge as _KR
        with open(os.path.join(os.path.dirname(ckpt_path), "krr_prior.pkl"), "rb") as _pf:
            kd = _pickle.load(_pf)
        krr = _KR(alpha=kd["alpha"], kernel="rbf", gamma=kd["gamma"])
        krr.fit(kd["X_train"], kd["Y_train"])
    return {"net": net, "sigma": fd.get("sigma", 1.0), "prior_kind": prior_kind,
            "krr": krr, "theta_dim": fd["theta_dim"], "d_embed": fd["d_embed"]}

if FLOW_CKPT and MV_PATH:
    from oppert.flow import sample_theta_ensemble as _sample_ens
    _primary = _build_flow_member(FLOW_CKPT)
    _flow_members = [_primary] + [_build_flow_member(p) for p in FLOW_CKPTS_EXTRA]
    # Primary-member exports for back-compat with downstream logic.
    _flow_net = _primary["net"]; _flow_sigma = _primary["sigma"]
    _flow_prior_kind = _primary["prior_kind"]; _flow_krr = _primary["krr"]
    _mv = torch.load(MV_PATH, map_location=DEVICE, weights_only=False)
    _mv_genes = {g.upper(): i for i, g in enumerate(_mv["gene_names"])}
    _mv_embed = _mv["embed"].float().to(DEVICE)
    logger.info(f"Flow loaded: {len(_flow_members)} member(s)  theta_dim={_primary['theta_dim']}  d_embed={_primary['d_embed']}  mv_genes={len(_mv_genes)}")
    if FLOW_BLEND_ALPHA is not None:
        logger.info(f"Flow-KRR blend alpha: {FLOW_BLEND_ALPHA:.2f} (alpha*KRR_prior + (1-alpha)*flow_mean)")

def _predict_rot(pu):
    """Return generator prediction for pert-name pu. Uses flow if enabled, else KRR.

    Multi-seed ensemble: averages samples across all loaded flow members before
    returning the flow mean. Single-member case is a strict subset.
    """
    if _flow_net is not None and pu in _mv_genes:
        e = _mv_embed[_mv_genes[pu]].unsqueeze(0)
        # For blend/skip logic use the primary member's prior.
        _prior_t = None
        if _flow_prior_kind == "krr" and _flow_krr is not None:
            _prior_np = _flow_krr.predict(e.cpu().numpy())
            _prior_t = torch.from_numpy(_prior_np).float().to(DEVICE)
        if FLOW_BLEND_ALPHA is not None and _prior_t is not None and FLOW_BLEND_ALPHA >= 1.0:
            return _prior_t.squeeze(0).cpu().numpy()
        # Collect samples from every member (each with its own prior).
        all_samples = []
        for _m in _flow_members:
            m_prior_t = None
            if _m["prior_kind"] == "krr" and _m["krr"] is not None:
                m_prior_np = _m["krr"].predict(e.cpu().numpy())
                m_prior_t = torch.from_numpy(m_prior_np).float().to(DEVICE)
            _sigma_eff = _m["sigma"] if FLOW_SIGMA_INFERENCE_OVERRIDE is None else FLOW_SIGMA_INFERENCE_OVERRIDE
            _s = _sample_ens(_m["net"], e, n_samples=_flow_nsamples, n_steps=_flow_nsteps,
                             sigma_noise=_sigma_eff, prior=m_prior_t,
                             integrator=FLOW_INTEGRATOR,
                             cfg_scale=FLOW_CFG_SCALE,
                             time_schedule=FLOW_TIME_SCHEDULE,
                             antithetic=FLOW_ANTITHETIC)
            all_samples.append(_s)
        samples = torch.cat(all_samples, dim=0) if len(all_samples) > 1 else all_samples[0]
        flow_mean = samples.mean(dim=0).squeeze(0)
        if FLOW_BLEND_ALPHA is not None and _prior_t is not None and FLOW_BLEND_ALPHA > 0.0:
            blended = FLOW_BLEND_ALPHA * _prior_t.squeeze(0) + (1.0 - FLOW_BLEND_ALPHA) * flow_mean
            return blended.cpu().numpy()
        return flow_mean.cpu().numpy()
    emb = scaler_rot.transform(emb_data[pu].numpy().reshape(1, -1))
    return krr_rot.predict(emb)[0]


# Direct Delta KRR
scaler_dd = StandardScaler()
dd_X_sc = scaler_dd.fit_transform(dd_X)
krr_dd = GridSearchCV(KernelRidge(kernel="rbf"), param_grid,
                      cv=min(5, len(dd_X)), scoring="neg_mean_squared_error", n_jobs=-1)
logger.info("Fitting Direct Delta CV-KRR...")
krr_dd.fit(dd_X_sc, dd_Y)
logger.info(f"  Direct Delta KRR best: {krr_dd.best_params_}")

def gen_to_blocks(g):
    """Convert flat generator params to block rotation matrices."""
    coeffs = g.reshape(num_blocks, -1)
    upper_dim = block_size * (block_size - 1) // 2
    blocks = []
    for b in range(num_blocks):
        A = torch.zeros(block_size, block_size, device=g.device)
        idx = 0
        for i in range(block_size):
            for j in range(i+1, block_size):
                A[i, j] = coeffs[b, idx]
                A[j, i] = -coeffs[b, idx]
                idx += 1
        A_exp = torch.matrix_exp(A)
        blocks.append(A_exp)
    return torch.block_diag(*blocks)

def apply_rotation_and_decode(gen_params):
    """Apply rotation to basal and decode to expression space."""
    g = torch.tensor(gen_params, dtype=torch.float32, device=DEVICE)
    R = gen_to_blocks(g)
    z_rotated = (R @ z_basal_mean.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        decoded, _ = model._decode(z_rotated.unsqueeze(0))
        if decoded.dim() == 3:
            decoded = decoded[:, :, 0]
        if decoded.shape[-1] > n_genes:
            decoded = decoded[:, :n_genes]
    return decoded.squeeze(0).cpu()

# BCH composition for combos
bracket_scale = getattr(model.rotation, 'bracket_scale', None)
if bracket_scale is not None:
    bracket_scale = float(bracket_scale.item()) if hasattr(bracket_scale, 'item') else float(bracket_scale)
else:
    bracket_scale = 1.0
if BRACKET_SCALE_OVERRIDE is not None:
    logger.info(f"BCH bracket_scale override: {bracket_scale:.4f} -> {BRACKET_SCALE_OVERRIDE:.4f}")
    bracket_scale = BRACKET_SCALE_OVERRIDE
logger.info(f"BCH bracket_scale: {bracket_scale:.4f}")

def bch_compose(g1, g2, scale=bracket_scale):
    """BCH composition: g1 + g2 + scale/2 * [g1, g2] in block-diagonal form."""
    g1t = torch.tensor(g1, dtype=torch.float32)
    g2t = torch.tensor(g2, dtype=torch.float32)
    c1 = g1t.reshape(num_blocks, -1)
    c2 = g2t.reshape(num_blocks, -1)
    upper_dim = block_size * (block_size - 1) // 2
    result = []
    for b in range(num_blocks):
        A1 = torch.zeros(block_size, block_size)
        A2 = torch.zeros(block_size, block_size)
        idx = 0
        for i in range(block_size):
            for j in range(i+1, block_size):
                A1[i, j] = c1[b, idx]; A1[j, i] = -c1[b, idx]
                A2[i, j] = c2[b, idx]; A2[j, i] = -c2[b, idx]
                idx += 1
        bracket = A1 @ A2 - A2 @ A1
        A_combo = A1 + A2 + (scale / 2) * bracket
        coeffs = []
        for i in range(block_size):
            for j in range(i+1, block_size):
                coeffs.append(A_combo[i, j].item())
        result.extend(coeffs)
    return np.array(result)

rot_da, rot_cos, rot_pde, rot_mse = [], [], [], []
dd_da, dd_cos, dd_pde, dd_mse = [], [], [], []
rot_da_s, rot_da_c, dd_da_s, dd_da_c = [], [], [], []
n_eval = 0
n_skip = 0

with torch.no_grad():
    g_ctrl_out, _ = model._decode(z_basal_mean.unsqueeze(0))
    if g_ctrl_out.dim() == 3:
        g_ctrl_out = g_ctrl_out[:, :, 0]
    if g_ctrl_out.shape[-1] > n_genes:
        g_ctrl_out = g_ctrl_out[:, :n_genes]
    g_ctrl_out = g_ctrl_out.squeeze(0).cpu()

# Track pert name + combo_seen category per eval for per-category breakdown
rot_names, dd_names = [], []
rot_cat, dd_cat = [], []  # 0=single, 1=combo_seen0, 2=combo_seen1, 3=combo_seen2

for pname, treat_idx in ood_perts_filtered:
    is_combo = "+" in pname
    # combo_seen category: for combos, count how many components are in training
    if is_combo:
        _parts_name = [p.strip().upper() for p in pname.split("+")]
        _train_upper = {t.upper() for t in train_perts}
        _seen_count = sum(1 for p in _parts_name if p in _train_upper)
        # 1=combo_seen0, 2=combo_seen1, 3=combo_seen2
        _cat_code = 1 + _seen_count
    else:
        _cat_code = 0  # single OOD

    # Get actual expression and delta (SAME for both methods)
    actual_genes = dataset.genes[treat_idx].mean(0)
    actual_delta = actual_genes - ctrl_raw_mean  # expression space delta

    # Top-K DE genes
    actual_delta = actual_delta.cpu(); actual_abs = actual_delta.abs()
    topk_idx = actual_abs.topk(TOPK_DE).indices

    # ── Method 1: Rotation KRR ──
    if is_combo:
        parts = [p.strip() for p in pname.split("+")]
        skip_rot = False
        gen_parts = []
        for p in parts:
            pu = p.upper()
            if pu not in emb_data:
                skip_rot = True
                break
            gen_parts.append(_predict_rot(pu))
        if not skip_rot:
            combo_gen = bch_compose(gen_parts[0], gen_parts[1])
            rot_pred_genes = apply_rotation_and_decode(combo_gen)
            rot_pred_delta = rot_pred_genes - g_ctrl_out
        else:
            skip_rot = True
    else:
        pu = pname.upper()
        if pu not in emb_data:
            n_skip += 1
            continue
        gen_pred = _predict_rot(pu)
        rot_pred_genes = apply_rotation_and_decode(gen_pred)
        rot_pred_delta = rot_pred_genes - g_ctrl_out
        skip_rot = False

    # ── Method 2: Direct Delta KRR ──
    if is_combo:
        parts = [p.strip() for p in pname.split("+")]
        skip_dd = False
        dd_parts = []
        for p in parts:
            pu = p.upper()
            if pu not in emb_data:
                skip_dd = True
                break
            emb = scaler_dd.transform(emb_data[pu].numpy().reshape(1, -1))
            dd_parts.append(krr_dd.predict(emb)[0])
        if not skip_dd:
            dd_pred_delta = torch.tensor(sum(dd_parts), dtype=torch.float32)
        else:
            skip_dd = True
    else:
        pu = pname.upper()
        if pu not in emb_data:
            n_skip += 1
            continue
        emb = scaler_dd.transform(emb_data[pu].numpy().reshape(1, -1))
        dd_pred_delta = torch.tensor(krr_dd.predict(emb)[0], dtype=torch.float32)
        skip_dd = False

    # Only evaluate if BOTH methods can predict
    if (is_combo and (skip_rot or skip_dd)):
        n_skip += 1
        continue

    # device-align before .numpy() / indexing
    rot_pred_delta = rot_pred_delta.cpu() if isinstance(rot_pred_delta, torch.Tensor) else rot_pred_delta
    if not skip_dd and isinstance(dd_pred_delta, torch.Tensor):
        dd_pred_delta = dd_pred_delta.cpu()
    actual_delta_c = actual_delta.cpu() if isinstance(actual_delta, torch.Tensor) else actual_delta
    # ── Compute metrics (IDENTICAL for both) ──
    # Rotation metrics
    if not skip_rot:
        r_da = ((actual_delta[topk_idx] > 0).float() == (rot_pred_delta[topk_idx] > 0).float()).float().mean().item()
        rot_da.append(r_da)
        r_cos = F.cosine_similarity(rot_pred_delta[topk_idx].unsqueeze(0), actual_delta[topk_idx].unsqueeze(0)).item()
        rot_cos.append(r_cos)
        ad = actual_delta[topk_idx].cpu().numpy()
        pd_np = rot_pred_delta[topk_idx].cpu().numpy()
        if np.std(ad) > 0 and np.std(pd_np) > 0:
            r_val = np.corrcoef(ad, pd_np)[0, 1]
            if not np.isnan(r_val):
                rot_pde.append(r_val)
        top20 = actual_abs.topk(20).indices
        rot_mse.append(F.mse_loss(rot_pred_delta[top20], actual_delta[top20]).item())
        if is_combo:
            rot_da_c.append(r_da)
        else:
            rot_da_s.append(r_da)
        rot_names.append(pname)
        rot_cat.append(_cat_code)

    # Direct delta metrics
    if not skip_dd:
        d_da = ((actual_delta[topk_idx] > 0).float() == (dd_pred_delta[topk_idx] > 0).float()).float().mean().item()
        dd_da.append(d_da)
        d_cos = F.cosine_similarity(dd_pred_delta[topk_idx].unsqueeze(0), actual_delta[topk_idx].unsqueeze(0)).item()
        dd_cos.append(d_cos)
        ad = actual_delta[topk_idx].cpu().numpy()
        pd_np = dd_pred_delta[topk_idx].cpu().numpy()
        if np.std(ad) > 0 and np.std(pd_np) > 0:
            r_val = np.corrcoef(ad, pd_np)[0, 1]
            if not np.isnan(r_val):
                dd_pde.append(r_val)
        top20 = actual_abs.topk(20).indices
        dd_mse.append(F.mse_loss(dd_pred_delta[top20], actual_delta[top20]).item())
        if is_combo:
            dd_da_c.append(d_da)
        else:
            dd_da_s.append(d_da)
        dd_names.append(pname)
        dd_cat.append(_cat_code)

    n_eval += 1

logger.info(f"\n{'='*70}")
logger.info(f"FAIR COMPARISON: Rotation KRR vs Direct Delta KRR")
logger.info(f"Dataset: {DATASET}")
logger.info(f"Checkpoint: {CKPT_PATH}")
logger.info(f"Embedding: {EMB_PATH}")
logger.info(f"Eval perts: {n_eval} (skip={n_skip})")
logger.info(f"BCH bracket_scale: {bracket_scale:.4f}")
logger.info(f"{'='*70}")

logger.info(f"\n{'Method':<20} {'DA':>7} {'cos':>7} {'PDE':>7} {'MSE20':>8} | {'DA_s':>6} {'DA_c':>6}")
logger.info("-" * 70)

if rot_da:
    r_s = f"{np.mean(rot_da_s)*100:.1f}" if rot_da_s else "—"
    r_c = f"{np.mean(rot_da_c)*100:.1f}" if rot_da_c else "—"
    logger.info(f"{'Rotation KRR':<20} {np.mean(rot_da)*100:6.1f}% {np.mean(rot_cos):7.3f} {np.mean(rot_pde):7.3f} {np.mean(rot_mse):8.4f} | {r_s:>6} {r_c:>6}")

if dd_da:
    d_s = f"{np.mean(dd_da_s)*100:.1f}" if dd_da_s else "—"
    d_c = f"{np.mean(dd_da_c)*100:.1f}" if dd_da_c else "—"
    logger.info(f"{'Direct Delta KRR':<20} {np.mean(dd_da)*100:6.1f}% {np.mean(dd_cos):7.3f} {np.mean(dd_pde):7.3f} {np.mean(dd_mse):8.4f} | {d_s:>6} {d_c:>6}")

if rot_da and dd_da:
    diff_da = (np.mean(dd_da) - np.mean(rot_da)) * 100
    diff_cos = np.mean(dd_cos) - np.mean(rot_cos)
    logger.info(f"\nΔ (Direct - Rotation): DA={diff_da:+.1f}pp  cos={diff_cos:+.3f}")

def _cat_slice(arr, cats, code):
    return [x for x, c in zip(arr, cats) if c == code]

_cat_names = {0: "single",
              1: "combo_seen0",
              2: "combo_seen1",
              3: "combo_seen2"}

rot_per_cat = {}
dd_per_cat = {}
for _code, _cname in _cat_names.items():
    _rda = _cat_slice(rot_da, rot_cat, _code)
    _rco = _cat_slice(rot_cos, rot_cat, _code)
    _rpd = _cat_slice(rot_pde, rot_cat, _code)
    _rms = _cat_slice(rot_mse, rot_cat, _code)
    _dda = _cat_slice(dd_da, dd_cat, _code)
    _dco = _cat_slice(dd_cos, dd_cat, _code)
    _dpd = _cat_slice(dd_pde, dd_cat, _code)
    _dms = _cat_slice(dd_mse, dd_cat, _code)
    if _rda:
        rot_per_cat[_cname] = {
            "n": len(_rda),
            "da": float(np.mean(_rda)),
            "cos": float(np.mean(_rco)) if _rco else 0,
            "pde": float(np.mean(_rpd)) if _rpd else 0,
            "mse20": float(np.mean(_rms)) if _rms else 0,
        }
    if _dda:
        dd_per_cat[_cname] = {
            "n": len(_dda),
            "da": float(np.mean(_dda)),
            "cos": float(np.mean(_dco)) if _dco else 0,
            "pde": float(np.mean(_dpd)) if _dpd else 0,
            "mse20": float(np.mean(_dms)) if _dms else 0,
        }

logger.info(f"\n{'Category':<15} {'n':>4} {'Rot DA':>8} {'DD DA':>8} {'Rot cos':>8} {'DD cos':>8}")
logger.info("-" * 65)
for _cname in ["single", "combo_seen0", "combo_seen1", "combo_seen2"]:
    _r = rot_per_cat.get(_cname)
    _d = dd_per_cat.get(_cname)
    if _r is None and _d is None:
        continue
    _n = _r.get("n", 0) if _r else (_d.get("n", 0) if _d else 0)
    _rda_s = f"{_r['da']*100:5.1f}%" if _r else "    —"
    _dda_s = f"{_d['da']*100:5.1f}%" if _d else "    —"
    _rco_s = f"{_r['cos']:+.3f}" if _r else "    —"
    _dco_s = f"{_d['cos']:+.3f}" if _d else "    —"
    logger.info(f"{_cname:<15} {_n:>4} {_rda_s:>8} {_dda_s:>8} {_rco_s:>8} {_dco_s:>8}")

# Save
out = {
    "dataset": DATASET,
    "checkpoint": str(CKPT_PATH),
    "embedding": str(EMB_PATH),
    "n_eval": n_eval,
    "bracket_scale": bracket_scale,
    "flow_blend_alpha": FLOW_BLEND_ALPHA,
    # NB: when FLOW_CKPT is set, `rotation_krr.*` holds the FLOW-via-rotation
    # prediction (see _predict_rot() → flow_mean branch at line ~297), NOT a
    # pure-KRR baseline. `direct_delta_krr.*` is always the pure DD-KRR
    # baseline. The legacy field name survives for JSON-schema compatibility
    # with iter-0-era pure-KRR runs.
    "flow_loaded": bool(FLOW_CKPT),
    "flow_n_members": len(_flow_members) if FLOW_CKPT else 0,
    "rotation_krr": {
        "da": float(np.mean(rot_da)) if rot_da else 0,
        "da_singles": float(np.mean(rot_da_s)) if rot_da_s else 0,
        "da_combos": float(np.mean(rot_da_c)) if rot_da_c else 0,
        "cos": float(np.mean(rot_cos)) if rot_cos else 0,
        "pde": float(np.mean(rot_pde)) if rot_pde else 0,
        "mse20": float(np.mean(rot_mse)) if rot_mse else 0,
        "n_singles": len(rot_da_s),
        "n_combos": len(rot_da_c),
        "krr_params": krr_rot.best_params_,
        # Per-perturbation arrays for bootstrap CIs
        "per_pert_da": [float(x) for x in rot_da],
        "per_pert_cos": [float(x) for x in rot_cos],
        "per_pert_pde": [float(x) for x in rot_pde],
        "per_pert_mse20": [float(x) for x in rot_mse],
        "per_pert_name": rot_names,
        "per_pert_combo_seen": rot_cat,
        "per_category": rot_per_cat,
    },
    "direct_delta_krr": {
        "da": float(np.mean(dd_da)) if dd_da else 0,
        "da_singles": float(np.mean(dd_da_s)) if dd_da_s else 0,
        "da_combos": float(np.mean(dd_da_c)) if dd_da_c else 0,
        "cos": float(np.mean(dd_cos)) if dd_cos else 0,
        "pde": float(np.mean(dd_pde)) if dd_pde else 0,
        "mse20": float(np.mean(dd_mse)) if dd_mse else 0,
        "n_singles": len(dd_da_s),
        "n_combos": len(dd_da_c),
        "krr_params": krr_dd.best_params_ if not skip_dd else {},
        # Per-perturbation arrays for bootstrap CIs
        "per_pert_da": [float(x) for x in dd_da],
        "per_pert_cos": [float(x) for x in dd_cos],
        "per_pert_pde": [float(x) for x in dd_pde],
        "per_pert_mse20": [float(x) for x in dd_mse],
        "per_pert_name": dd_names,
        "per_pert_combo_seen": dd_cat,
        "per_category": dd_per_cat,
    },
}
out_dir = Path("experiments/results/fair_comparison")
out_dir.mkdir(parents=True, exist_ok=True)
if "norman" in DATASET:
    ds_short = "norman"
elif "sciplex" in DATASET:
    ds_short = "sciplex3"
elif "rpe1" in DATASET:
    ds_short = "rpe1"
else:
    ds_short = "replogle"
# Avoid overwriting canonical pure-KRR result when running flow eval
if FLOW_CKPT:
    _flow_tag = Path(FLOW_CKPT).parent.name  # e.g. b200_norman_flow_e115_krrinit_v1
    if FLOW_TAG_OVERRIDE:
        _flow_tag = f"{_flow_tag}_{FLOW_TAG_OVERRIDE}"
    _out_name = f"{ds_short}_rotation_vs_direct__flow__{_flow_tag}.json"
else:
    _out_name = f"{ds_short}_rotation_vs_direct.json"
with open(out_dir / _out_name, "w") as f:
    json.dump(out, f, indent=2, default=str)
logger.info(f"\nSaved to {out_dir}/{_out_name}")
