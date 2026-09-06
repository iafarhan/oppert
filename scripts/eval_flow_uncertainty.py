#!/usr/bin/env python3
"""Uncertainty calibration for flow-sampled rotations."""
import argparse, json, os, pickle, logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

os.chdir(str(Path(__file__).resolve().parent.parent))

from sklearn.kernel_ridge import KernelRidge  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from OpPert.evaluate_full import _load_config, _build_data, _load_model  # noqa: E402
from oppert.flow import ConditionalVelocityNet, sample_theta_ensemble  # noqa: E402

SEED = 42
TOPK_DE = 50


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="crispra_norman_gears")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--flow_ckpt", required=True)
    p.add_argument("--multiview", required=True)
    p.add_argument("--embedding", default="data/gene_embeddings/genept_bge_large.pt")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n_steps", type=int, default=40)
    p.add_argument("--out", required=True)
    return p.parse_args()


def build_block_skew(g, num_blocks, block_size, device):
    coeffs = g.reshape(num_blocks, -1)
    upper_dim = block_size * (block_size - 1) // 2  # noqa: F841
    blocks = []
    for b in range(num_blocks):
        A = torch.zeros(block_size, block_size, device=device)
        idx = 0
        for i in range(block_size):
            for j in range(i + 1, block_size):
                A[i, j] = coeffs[b, idx]
                A[j, i] = -coeffs[b, idx]
                idx += 1
        blocks.append(torch.matrix_exp(A))
    return torch.block_diag(*blocks)


def bch_compose(g1, g2, num_blocks, block_size, bracket_scale):
    c1 = g1.reshape(num_blocks, -1)
    c2 = g2.reshape(num_blocks, -1)
    out = []
    for b in range(num_blocks):
        A1 = torch.zeros(block_size, block_size)
        A2 = torch.zeros(block_size, block_size)
        idx = 0
        for i in range(block_size):
            for j in range(i + 1, block_size):
                A1[i, j] = c1[b, idx]; A1[j, i] = -c1[b, idx]
                A2[i, j] = c2[b, idx]; A2[j, i] = -c2[b, idx]
                idx += 1
        bracket = A1 @ A2 - A2 @ A1
        A_combo = A1 + A2 + (bracket_scale / 2) * bracket
        for i in range(block_size):
            for j in range(i + 1, block_size):
                out.append(A_combo[i, j].item())
    return torch.tensor(out, dtype=torch.float32)


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))
    torch.manual_seed(SEED); np.random.seed(SEED)

    # ---- Data ----
    ds_cfg, dl_cfg = _load_config(args.dataset)
    ds_cfg = OmegaConf.merge(ds_cfg, {"pert_subsample": None})
    dm = _build_data(ds_cfg, dl_cfg)
    dataset = dm.dataset
    obs = dataset.adata.obs
    n_genes = dataset.genes.shape[1]
    pert_key = getattr(dataset.cfg, "perturbation_key", "perturbation")

    # ---- Model ----
    model, _ = _load_model(args.ckpt, "tuned", device)
    model.eval()
    rotation = model.rotation
    num_blocks = rotation.num_blocks
    block_size = rotation.block_size
    bracket_scale = getattr(rotation, "bracket_scale", None)
    if bracket_scale is not None and hasattr(bracket_scale, "item"):
        bracket_scale = float(bracket_scale.item())
    else:
        bracket_scale = 1.0 if bracket_scale is None else float(bracket_scale)
    logger.info(f"num_blocks={num_blocks} block_size={block_size} bracket_scale={bracket_scale:.3f}")

    # ---- Flow ----
    fd = torch.load(args.flow_ckpt, map_location=device, weights_only=False)
    v_net = ConditionalVelocityNet(
        theta_dim=fd["theta_dim"], d_embed=fd["d_embed"],
        d_hidden=fd.get("d_hidden", 512), n_blocks=fd.get("n_blocks", 4),
    ).to(device)
    v_net.load_state_dict(fd["v_net_state_dict"]); v_net.eval()
    flow_sigma = fd.get("sigma", 0.02)
    prior_kind = fd.get("prior", "none") or "none"
    logger.info(f"flow: theta_dim={fd['theta_dim']} sigma={flow_sigma} prior={prior_kind}")

    mv = torch.load(args.multiview, map_location=device, weights_only=False)
    mv_genes = {g.upper(): i for i, g in enumerate(mv["gene_names"])}
    mv_embed = mv["embed"].float().to(device)

    flow_krr = None
    if prior_kind == "krr":
        prior_pkl = os.path.join(os.path.dirname(args.flow_ckpt), "krr_prior.pkl")
        with open(prior_pkl, "rb") as f:
            kd = pickle.load(f)
        flow_krr = KernelRidge(alpha=kd["alpha"], kernel="rbf", gamma=kd["gamma"])
        flow_krr.fit(kd["X_train"], kd["Y_train"])
        logger.info(f"KRR prior restored: n_train={kd['X_train'].shape[0]}")

    # ---- OOD perts ----
    train_indices = dataset.indices.get("train", [])
    ood_indices = dataset.indices.get("ood", dataset.indices.get("test", []))
    ood_perts_raw = set(obs.iloc[ood_indices].loc[
        obs.iloc[ood_indices]["control"].astype(int) != 1, pert_key].unique())
    ood_perts = []
    for pname in sorted(ood_perts_raw):
        if "control" in pname.lower() or "dmso" in pname.lower():
            continue
        mask = (obs.iloc[ood_indices][pert_key].values == pname) & \
               (obs.iloc[ood_indices]["control"].astype(int).values != 1)
        idx = np.array(ood_indices)[mask]
        if idx.size < 5:
            continue
        ood_perts.append((pname, idx))
    logger.info(f"OOD perts: {len(ood_perts)}")

    # ---- Control basal ----
    ctrl_mask = obs.iloc[train_indices]["control"].astype(int).values == 1
    ctrl_idx = np.array(train_indices)[ctrl_mask]
    rng = np.random.RandomState(SEED)
    sample_ctrl = rng.choice(ctrl_idx, min(500, len(ctrl_idx)), replace=False)
    ctrl_raw_mean = dataset.genes[sample_ctrl].mean(0)
    with torch.no_grad():
        z_ctrl = model.get_latent(dataset.genes[sample_ctrl].to(device))
    z_basal_mean = z_ctrl.mean(0)
    with torch.no_grad():
        g_ctrl, _ = model._decode(z_basal_mean.unsqueeze(0))
        if g_ctrl.dim() == 3:
            g_ctrl = g_ctrl[:, :, 0]
        if g_ctrl.shape[-1] > n_genes:
            g_ctrl = g_ctrl[:, :n_genes]
        g_ctrl = g_ctrl.squeeze(0).cpu()

    def flow_samples(pu):
        """Return [K, theta_dim] flow samples for pert name pu."""
        if pu not in mv_genes:
            return None
        e = mv_embed[mv_genes[pu]].unsqueeze(0)
        prior_t = None
        if flow_krr is not None:
            prior_np = flow_krr.predict(e.cpu().numpy())
            prior_t = torch.from_numpy(prior_np).float().to(device)
        s = sample_theta_ensemble(v_net, e, n_samples=args.k, n_steps=args.n_steps,
                                  sigma_noise=flow_sigma, prior=prior_t)
        return s.squeeze(1)  # [K, theta_dim]

    def decode_theta(theta):
        g = theta.to(device)
        R = build_block_skew(g, num_blocks, block_size, device)
        z_rot = (R @ z_basal_mean.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            decoded, _ = model._decode(z_rot.unsqueeze(0))
            if decoded.dim() == 3:
                decoded = decoded[:, :, 0]
            if decoded.shape[-1] > n_genes:
                decoded = decoded[:, :n_genes]
        return decoded.squeeze(0).cpu()

    # ---- Eval ----
    per_pert = []
    n_skip = 0
    for pname, treat_idx in ood_perts:
        is_combo = "+" in pname
        actual_genes = dataset.genes[treat_idx].mean(0)
        actual_delta = actual_genes - ctrl_raw_mean
        actual_delta = actual_delta.cpu()
        topk_idx = actual_delta.abs().topk(TOPK_DE).indices

        if is_combo:
            parts = [p.strip().upper() for p in pname.split("+")]
            ssets = [flow_samples(p) for p in parts]
            if any(s is None for s in ssets):
                n_skip += 1
                continue
            K = min(s.shape[0] for s in ssets)
            deltas = []
            for k in range(K):
                theta_k = bch_compose(ssets[0][k].cpu(), ssets[1][k].cpu(),
                                      num_blocks, block_size, bracket_scale)
                pred_genes = decode_theta(theta_k)
                deltas.append(pred_genes - g_ctrl)
        else:
            pu = pname.upper()
            s = flow_samples(pu)
            if s is None:
                n_skip += 1
                continue
            deltas = [decode_theta(s[k].cpu()) - g_ctrl for k in range(s.shape[0])]

        deltas = torch.stack(deltas, dim=0)  # [K, n_genes]
        delta_mean = deltas.mean(dim=0)
        delta_std = deltas.std(dim=0)

        # metrics
        cos_mean = F.cosine_similarity(delta_mean[topk_idx].unsqueeze(0),
                                       actual_delta[topk_idx].unsqueeze(0)).item()
        da_mean = ((actual_delta[topk_idx] > 0).float() ==
                   (delta_mean[topk_idx] > 0).float()).float().mean().item()
        # uncertainty summaries
        unc_topk = float(delta_std[topk_idx].mean().item())
        unc_global = float(delta_std.mean().item())
        # also sample-space: ensemble std norm in theta (correlates with exploration)
        # (only for singles — combos mix two distributions)

        per_pert.append({
            "pert": pname,
            "is_combo": is_combo,
            "cos": cos_mean,
            "da": da_mean,
            "err_1mcos": 1.0 - cos_mean,
            "unc_std_topk": unc_topk,
            "unc_std_global": unc_global,
            "k": deltas.shape[0],
        })

    def grouped(xs, key):
        return [p[key] for p in xs]

    def summary(pp):
        if not pp:
            return {}
        err = grouped(pp, "err_1mcos")
        unc = grouped(pp, "unc_std_topk")
        unc_g = grouped(pp, "unc_std_global")
        return {
            "n": len(pp),
            "da_mean": float(np.mean(grouped(pp, "da"))),
            "cos_mean": float(np.mean(grouped(pp, "cos"))),
            "err_mean": float(np.mean(err)),
            "unc_topk_mean": float(np.mean(unc)),
            "unc_global_mean": float(np.mean(unc_g)),
            "pearson_err_vs_unc_topk": pearson(err, unc),
            "pearson_err_vs_unc_global": pearson(err, unc_g),
            "spearman_err_vs_unc_topk": pearson(
                list(np.argsort(np.argsort(err))), list(np.argsort(np.argsort(unc)))
            ),
        }

    all_pp = per_pert
    single_pp = [p for p in per_pert if not p["is_combo"]]
    combo_pp = [p for p in per_pert if p["is_combo"]]

    out = {
        "dataset": args.dataset,
        "ckpt": args.ckpt,
        "flow_ckpt": args.flow_ckpt,
        "K": args.k,
        "n_steps": args.n_steps,
        "sigma": float(flow_sigma),
        "prior": prior_kind,
        "n_eval": len(per_pert),
        "n_skip": n_skip,
        "overall": summary(all_pp),
        "singles": summary(single_pp),
        "combos": summary(combo_pp),
        "per_pert": per_pert,
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"n_eval={len(per_pert)}  n_skip={n_skip}  K={args.k}")
    for label, s in [("overall", out["overall"]), ("singles", out["singles"]), ("combos", out["combos"])]:
        if not s:
            continue
        logger.info(f"[{label:>7}] n={s['n']:3d}  DA={s['da_mean']*100:5.1f}%  "
                    f"cos={s['cos_mean']:.3f}  err={s['err_mean']:.3f}  "
                    f"unc_topk={s['unc_topk_mean']:.3f}  "
                    f"r(err,unc_topk)={s['pearson_err_vs_unc_topk']:.3f}  "
                    f"r(err,unc_global)={s['pearson_err_vs_unc_global']:.3f}")
    logger.info(f"Saved {outp}")


if __name__ == "__main__":
    main()
