"""SciPlex3 delta flow — conditional flow matching on HVG deltas."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oppert.flow import (
    ConditionalVelocityNet,
    flow_matching_loss,
    sample_theta_ensemble,
)


def compute_per_drug_delta_hvg(adata, hvg_idx, pert_col="perturbation",
                                ctrl_name="control"):
    """Returns {drug_name: Δ_vec_hvg}. Uses ctrl-pooled mean across all control cells."""
    import scipy.sparse as sp
    X = adata.X
    sparse_X = sp.issparse(X)
    perts = adata.obs[pert_col].astype(str).values

    def _mean_rows(mask):
        if sparse_X:
            v = np.asarray(X[mask].mean(axis=0)).ravel()
        else:
            v = np.asarray(X[mask].mean(axis=0)).ravel()
        return v[hvg_idx].astype(np.float32)

    ctrl_mask = perts == ctrl_name
    ctrl_mean = _mean_rows(ctrl_mask)

    names = sorted({p for p in perts if p != ctrl_name and "+" not in p})
    out = {}
    for name in names:
        mask = perts == name
        if mask.sum() < 5:
            continue
        dm = _mean_rows(mask)
        out[name] = (dm - ctrl_mean).astype(np.float32)
    return out, ctrl_mean


def compute_per_drug_dose_delta_hvg(adata, hvg_idx,
                                    pert_col="perturbation",
                                    dose_col="dose_character",
                                    ctrl_name="control",
                                    min_cells=5):
    """Iter 167 dose-aware Δgene. Returns {(drug, dose_str): Δ_vec_hvg}.
    Control pooled across all control cells (dose-independent).
    Keeps only non-zero doses that have >= min_cells in this drug."""
    import scipy.sparse as sp
    X = adata.X
    sparse_X = sp.issparse(X)
    perts = adata.obs[pert_col].astype(str).values
    doses = adata.obs[dose_col].astype(str).values

    def _mean_rows(mask):
        if sparse_X:
            v = np.asarray(X[mask].mean(axis=0)).ravel()
        else:
            v = np.asarray(X[mask].mean(axis=0)).ravel()
        return v[hvg_idx].astype(np.float32)

    ctrl_mask = perts == ctrl_name
    ctrl_mean = _mean_rows(ctrl_mask)

    drug_names = sorted({p for p in perts if p != ctrl_name and "+" not in p})
    # non-zero dose labels sorted numerically
    uniq_doses = sorted({d for d in doses if d not in ("0", "0.0", "nan")},
                        key=lambda x: float(x))
    out = {}
    for name in drug_names:
        drug_mask = perts == name
        for d in uniq_doses:
            m = drug_mask & (doses == d)
            if m.sum() < min_cells:
                continue
            dm = _mean_rows(m)
            out[(name, d)] = (dm - ctrl_mean).astype(np.float32)
    return out, ctrl_mean, uniq_doses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_h5ad", required=True)
    ap.add_argument("--multiview", required=True)
    ap.add_argument("--ood_json", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--sigma_inf", type=float, default=0.10)
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument("--K_eval", type=int, default=128)
    ap.add_argument("--d_hidden", type=int, default=512)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--topk_de", type=int, default=50)
    ap.add_argument("--zscore_target", action="store_true",
                    help="z-score Δgene per-feature across training drugs "
                         "before flow-matching (stabilizes loss for 2000-dim).")
    ap.add_argument("--prior", type=str, default="none", choices=["none", "krr"],
                    help="Prior for A_0: 'none' (Gaussian) or 'krr' (RBF KRR on multiview).")
    ap.add_argument("--krr_gamma", type=float, default=None,
                    help="RBF gamma; if None with --prior krr, CV-picked.")
    ap.add_argument("--krr_alpha", type=float, default=None,
                    help="Ridge alpha; if None with --prior krr, CV-picked.")
    ap.add_argument("--dose_aware", action="store_true",
                    help="Iter 167: use per-(drug,dose) Δgene instead of drug-averaged. "
                         "Descriptor gets a normalized log-dose scalar appended. "
                         "OOD eval averages the 4 dose predictions per drug.")
    ap.add_argument("--bch_aug", action="store_true",
                    help="Iter 168: linear-BCH synthetic drug-pair augmentation. "
                         "Δ_combo := Δ_i + Δ_j, descriptor := (e_i+e_j)/2. "
                         "Virtual pairs appended to singleton training set; OOD is single drugs only.")
    ap.add_argument("--bch_n_pairs", type=int, default=0,
                    help="Cap synthetic pairs (0 = use all C(n,2) ≈ 11781).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    # 1. Load adata + compute per-drug Δ over HVG genes
    import scanpy as sc
    t0 = time.time()
    adata = sc.read_h5ad(args.dataset_h5ad)
    print(f"adata: {adata.shape}  t={time.time()-t0:.1f}s")

    hvg_mask = adata.var["highly_variable"].values.astype(bool) \
        if "highly_variable" in adata.var.columns else np.ones(adata.shape[1], bool)
    if hvg_mask.sum() == 0:
        hvg_mask = np.ones(adata.shape[1], bool)
    hvg_idx = np.where(hvg_mask)[0]
    print(f"HVG: {hvg_idx.size}/{adata.shape[1]}")

    # Always compute drug-averaged Δgene — used for OOD ground truth even in dose-aware mode.
    delta_by_drug, ctrl_mean = compute_per_drug_delta_hvg(adata, hvg_idx)
    print(f"Computed per-drug Δgene for {len(delta_by_drug)} drugs  t={time.time()-t0:.1f}s")
    if args.dose_aware:
        delta_by_dd, _, uniq_doses = compute_per_drug_dose_delta_hvg(
            adata, hvg_idx, dose_col="dose_character")
        print(f"Computed per-(drug,dose) Δgene for {len(delta_by_dd)} pairs "
              f"(doses={uniq_doses})  t={time.time()-t0:.1f}s")
        # dose scalar: log10(dose_nM) normalized to [0,1] over uniq_doses
        log_doses = np.array([np.log10(float(d)) for d in uniq_doses])
        log_min, log_max = log_doses.min(), log_doses.max()
        def _dose_scalar(d):
            return float((np.log10(float(d)) - log_min) / (log_max - log_min + 1e-8))
        drug_pair_names = sorted({n for (n, _d) in delta_by_dd.keys()})

    # 2. Split + multiview
    with open(args.ood_json) as f:
        split = json.load(f)
    train_set = set(split["train_drugs"])
    ood_set = set(split["ood_drugs"])

    mv = torch.load(args.multiview, map_location="cpu", weights_only=False)
    mv_names = list(mv["gene_names"])
    mv_embed = mv["embed"].float()
    mv_g2i = {g: i for i, g in enumerate(mv_names)}

    if args.dose_aware:
        train_names = [n for n in drug_pair_names if n in train_set and n in mv_g2i]
        ood_names = [n for n in drug_pair_names if n in ood_set and n in mv_g2i]
        print(f"Train drugs (dose-aware): {len(train_names)}  OOD drugs: {len(ood_names)}  "
              f"(ood_json has train={len(train_set)}/ood={len(ood_set)})")
        # Expand to (drug, dose) train pairs: append dose scalar to descriptor
        train_pair_list, X_rows, Y_rows = [], [], []
        for n in train_names:
            desc = mv_embed[mv_g2i[n]]
            for d in uniq_doses:
                key = (n, d)
                if key in delta_by_dd:
                    dose_s = _dose_scalar(d)
                    X_rows.append(torch.cat([desc, torch.tensor([dose_s])]))
                    Y_rows.append(torch.tensor(delta_by_dd[key]))
                    train_pair_list.append(key)
        X_train = torch.stack(X_rows)
        Y_train = torch.stack(Y_rows).float()
        keep = Y_train.norm(dim=1) > 1e-6
        n_drop = int((~keep).sum().item())
        Y_train, X_train = Y_train[keep], X_train[keep]
        train_pair_list = [train_pair_list[i] for i in np.where(keep.numpy())[0]]
        print(f"After zero-norm mask: {len(train_pair_list)} (drug,dose) train pairs "
              f"(dropped {n_drop}; avg {len(train_pair_list)/max(1,len(train_names)):.2f} doses/drug)")
    else:
        train_names = [n for n in delta_by_drug if n in train_set and n in mv_g2i]
        ood_names = [n for n in delta_by_drug if n in ood_set and n in mv_g2i]
        print(f"Train drugs: {len(train_names)}  OOD drugs: {len(ood_names)}  "
              f"(ood_json has train={len(train_set)}/ood={len(ood_set)})")

        Y_train = torch.tensor(np.stack([delta_by_drug[n] for n in train_names]),
                               dtype=torch.float32)
        X_train = mv_embed[[mv_g2i[n] for n in train_names]]
        keep = Y_train.norm(dim=1) > 1e-6
        n_drop = int((~keep).sum().item())
        Y_train, X_train = Y_train[keep], X_train[keep]
        train_names = [train_names[i] for i in np.where(keep.numpy())[0]]
        print(f"After zero-norm mask: {len(train_names)} train pairs (dropped {n_drop})")

        if args.bch_aug:
            n_sing = Y_train.shape[0]
            rng = np.random.default_rng(args.seed + 99991)
            tri_i, tri_j = np.triu_indices(n_sing, k=1)
            n_pairs = tri_i.size
            if 0 < args.bch_n_pairs < n_pairs:
                pick = rng.choice(n_pairs, size=args.bch_n_pairs, replace=False)
                tri_i, tri_j = tri_i[pick], tri_j[pick]
            Y_aug = Y_train[tri_i] + Y_train[tri_j]
            X_aug = 0.5 * (X_train[tri_i] + X_train[tri_j])
            Y_train = torch.cat([Y_train, Y_aug], dim=0)
            X_train = torch.cat([X_train, X_aug], dim=0)
            aug_names = [f"__bch__{train_names[i]}+{train_names[j]}" for i, j in zip(tri_i, tri_j)]
            train_names = train_names + aug_names
            print(f"BCH-aug: {n_sing} singletons + {len(aug_names)} synthetic pairs "
                  f"(Δ_ij = Δ_i+Δ_j, e_ij = (e_i+e_j)/2) → {Y_train.shape[0]} total train rows")

    # 3. Target z-score (optional)
    if args.zscore_target:
        y_mean = Y_train.mean(dim=0, keepdim=True)
        y_std = Y_train.std(dim=0, keepdim=True).clamp(min=1e-4)
        Y_train_fit = (Y_train - y_mean) / y_std
    else:
        y_mean = torch.zeros(1, Y_train.shape[1])
        y_std = torch.ones(1, Y_train.shape[1])
        Y_train_fit = Y_train

    theta_dim = Y_train.shape[1]
    d_embed = X_train.shape[1]
    print(f"theta_dim={theta_dim} d_embed={d_embed}  "
          f"Y std(before scaling)={Y_train.std().item():.4f}  "
          f"scaled_std={Y_train_fit.std().item():.4f}")

    # 4. Optional KRR-init prior (fit on z-scored target)
    prior_train_d = None
    prior_ood_d = None
    krr_info = None
    if args.prior == "krr":
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import KFold
        X_train_np = X_train.detach().cpu().numpy()
        Y_train_np = Y_train_fit.detach().cpu().numpy()
        if args.krr_gamma is None or args.krr_alpha is None:
            print("KRR CV: picking gamma, alpha...")
            best_err = float("inf"); best_g = None; best_a = None
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            for g_ in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                for a_ in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                    errs = []
                    for tr, te in kf.split(X_train_np):
                        m = KernelRidge(alpha=a_, kernel="rbf", gamma=g_)
                        m.fit(X_train_np[tr], Y_train_np[tr])
                        errs.append(float(np.mean(
                            (m.predict(X_train_np[te]) - Y_train_np[te]) ** 2)))
                    e = float(np.mean(errs))
                    if e < best_err:
                        best_err = e; best_g = g_; best_a = a_
            args.krr_gamma = best_g; args.krr_alpha = best_a
            print(f"KRR best gamma={best_g}, alpha={best_a}, CV MSE={best_err:.4f}")
        krr = KernelRidge(alpha=args.krr_alpha, kernel="rbf", gamma=args.krr_gamma)
        krr.fit(X_train_np, Y_train_np)
        Y_pred_train_np = krr.predict(X_train_np)
        krr_train_cos = float(np.mean([
            float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            for a, b in zip(Y_pred_train_np, Y_train_np)
        ]))
        print(f"KRR train recon cos (zscored scale): {krr_train_cos:.4f}")
        prior_train_d = torch.from_numpy(Y_pred_train_np).float().to(device)
        krr_info = {
            "gamma": float(args.krr_gamma),
            "alpha": float(args.krr_alpha),
            "train_cos": krr_train_cos,
        }
        # Stash the fitted KRR for OOD inference prior
        _fitted_krr = krr
    else:
        _fitted_krr = None

    # 5. Model + flow matching training
    v_net = ConditionalVelocityNet(
        theta_dim=theta_dim, d_embed=d_embed, d_hidden=args.d_hidden,
        n_blocks=args.n_blocks, dropout=args.dropout, head_kind="flat",
    ).to(device)
    n_params = sum(p.numel() for p in v_net.parameters())
    print(f"v_net params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(v_net.parameters(), lr=args.lr, weight_decay=args.wd)

    Y_fit_d = Y_train_fit.to(device)
    X_fit_d = X_train.to(device)
    N = Y_fit_d.shape[0]
    bsz = min(args.batch, N)

    best_loss = float("inf")
    best_sd = None
    t0 = time.time()
    for ep in range(args.epochs + 1):
        v_net.train()
        batch_idx = torch.randint(0, N, (bsz,), device=device)
        loss = flow_matching_loss(
            v_net, theta_train=Y_fit_d, embed_train=X_fit_d,
            batch_idx=batch_idx, sigma_noise=args.sigma,
            prior_train=prior_train_d, interpolant="ot", t_schedule="uniform",
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
        opt.step()

        l = float(loss.item())
        if l < best_loss:
            best_loss = l
            best_sd = {k: v.detach().cpu().clone() for k, v in v_net.state_dict().items()}
        if ep % args.log_every == 0:
            print(f"E{ep:05d}  loss={l:.5f}  best={best_loss:.5f}  "
                  f"t={time.time()-t0:.1f}s")

    print(f"\nTraining done. best_loss={best_loss:.5f}  total={time.time()-t0:.1f}s")

    # 5. Save
    ckpt_path = Path(args.output) / "flow_best.pt"
    save_train_names = (
        [f"{n}|{d}" for (n, d) in train_pair_list] if args.dose_aware else train_names
    )
    torch.save({
        "state_dict": best_sd,
        "theta_dim": theta_dim, "d_embed": d_embed,
        "y_mean": y_mean, "y_std": y_std,
        "train_names": save_train_names,
        "hparams": vars(args),
    }, ckpt_path)

    # 6. OOD eval — K-sample ensemble
    v_net.load_state_dict(best_sd)
    v_net.eval()
    Y_ood = torch.tensor(np.stack([delta_by_drug[n] for n in ood_names]),
                         dtype=torch.float32).to(device)
    y_mean_d, y_std_d = y_mean.to(device), y_std.to(device)

    if args.dose_aware:
        # Per OOD drug, evaluate at each of the uniq_doses and average in scaled space.
        desc_ood = mv_embed[[mv_g2i[n] for n in ood_names]]  # [n_ood, d_desc]
        pred_scaled_doses = []
        krr_ood_cos_doses = []
        for d in uniq_doses:
            dose_s = _dose_scalar(d)
            X_ood_d = torch.cat(
                [desc_ood, torch.full((desc_ood.shape[0], 1), dose_s)], dim=1
            ).to(device)
            prior_ood_dd = None
            if _fitted_krr is not None:
                X_ood_np = X_ood_d.detach().cpu().numpy()
                prior_ood_np = _fitted_krr.predict(X_ood_np)
                prior_ood_dd = torch.from_numpy(prior_ood_np).float().to(device)
                # diagnostic: KRR prior vs TRUE per-(drug,dose) Δ (where available)
                y_dd_list, mask = [], []
                for nm in ood_names:
                    if (nm, d) in delta_by_dd:
                        y_dd_list.append(delta_by_dd[(nm, d)])
                        mask.append(True)
                    else:
                        y_dd_list.append(np.zeros(theta_dim, np.float32))
                        mask.append(False)
                y_dd = np.stack(y_dd_list)
                y_dd_scaled = (y_dd - y_mean.numpy()) / y_std.numpy()
                m = np.array(mask)
                if m.any():
                    cos = float(np.mean([
                        float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                        for a, b, kept in zip(prior_ood_np, y_dd_scaled, mask) if kept
                    ]))
                    krr_ood_cos_doses.append((d, cos, int(m.sum())))
            with torch.no_grad():
                samp = sample_theta_ensemble(
                    v_net, e=X_ood_d, n_samples=args.K_eval,
                    n_steps=args.n_steps, sigma_noise=args.sigma_inf,
                    prior=prior_ood_dd, antithetic=True,
                )
            pred_scaled_doses.append(samp.mean(dim=0))  # [n_ood, theta_dim]
        pred_mean_scaled = torch.stack(pred_scaled_doses, dim=0).mean(dim=0)
        for d, cos, n_ok in krr_ood_cos_doses:
            print(f"KRR OOD recon cos (zscored) dose={d}: {cos:.4f}  (n={n_ok})")
        pred_delta = pred_mean_scaled * y_std_d + y_mean_d
    else:
        X_ood = mv_embed[[mv_g2i[n] for n in ood_names]].to(device)
        prior_ood_d = None
        if _fitted_krr is not None:
            X_ood_np = X_ood.detach().cpu().numpy()
            prior_ood_np = _fitted_krr.predict(X_ood_np)
            prior_ood_d = torch.from_numpy(prior_ood_np).float().to(device)
            # Diagnostic: how well does the KRR prior alone predict OOD Δ on zscored scale?
            y_ood_np_scaled = ((Y_ood - y_mean.to(device)) / y_std.to(device)).detach().cpu().numpy()
            krr_ood_cos = float(np.mean([
                float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                for a, b in zip(prior_ood_np, y_ood_np_scaled)
            ]))
            print(f"KRR OOD recon cos (zscored): {krr_ood_cos:.4f}")

        with torch.no_grad():
            samples = sample_theta_ensemble(
                v_net, e=X_ood, n_samples=args.K_eval,
                n_steps=args.n_steps, sigma_noise=args.sigma_inf,
                prior=prior_ood_d, antithetic=True,
            )  # [K, n_ood, theta_dim]
        pred_mean_scaled = samples.mean(dim=0)  # [n_ood, theta_dim]
        pred_delta = pred_mean_scaled * y_std_d + y_mean_d

    results = []
    for i, name in enumerate(ood_names):
        actual = Y_ood[i]
        pred = pred_delta[i]
        topk = actual.abs().topk(args.topk_de).indices
        da = ((actual[topk] > 0).float() == (pred[topk] > 0).float()).float().mean().item()
        cos_topk = F.cosine_similarity(
            pred[topk].unsqueeze(0), actual[topk].unsqueeze(0)).item()
        cos_full = F.cosine_similarity(
            pred.unsqueeze(0), actual.unsqueeze(0)).item()
        top20 = actual.abs().topk(20).indices
        mse20 = F.mse_loss(pred[top20], actual[top20]).item()
        results.append({
            "drug": name, "da": da,
            "cos_topk": cos_topk, "cos_full": cos_full, "mse_top20": mse20,
        })

    da_mean = float(np.mean([r["da"] for r in results]))
    cos_topk_mean = float(np.mean([r["cos_topk"] for r in results]))
    cos_full_mean = float(np.mean([r["cos_full"] for r in results]))

    if args.dose_aware:
        _iter_tag, _model_tag = 167, "sciplex3_delta_flow_multiview_dose_aware"
    elif args.bch_aug:
        _iter_tag, _model_tag = 168, "sciplex3_delta_flow_multiview_bchaug"
    else:
        _iter_tag, _model_tag = 162, "sciplex3_delta_flow_multiview_v2"
    out = {
        "iter": _iter_tag,
        "model": _model_tag,
        "n_train": (len(train_pair_list) if args.dose_aware else len(train_names)),
        "n_train_drugs": len(train_names),
        "dose_aware": args.dose_aware,
        "bch_aug": args.bch_aug,
        "bch_n_pairs": args.bch_n_pairs if args.bch_aug else 0,
        "uniq_doses": (uniq_doses if args.dose_aware else None),
        "n_ood": len(ood_names),
        "theta_dim": theta_dim,
        "d_embed": d_embed,
        "K_eval": args.K_eval,
        "sigma": args.sigma,
        "sigma_inf": args.sigma_inf,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": bsz,
        "seed": args.seed,
        "zscore_target": args.zscore_target,
        "prior": args.prior,
        "krr_info": krr_info,
        "best_loss": best_loss,
        "da_mean": da_mean,
        "cos_topk_mean": cos_topk_mean,
        "cos_full_mean": cos_full_mean,
        "per_drug": results,
    }
    res_path = Path(args.output) / "eval.json"
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== OOD RESULTS (n={len(ood_names)}) ===")
    print(f"  DA={da_mean*100:.2f}%  cos_topk={cos_topk_mean:+.4f}  cos_full={cos_full_mean:+.4f}")
    for r in sorted(results, key=lambda x: -x["cos_topk"])[:5]:
        print(f"  best  {r['drug']:24s}  da={r['da']:.2f}  cos={r['cos_topk']:+.3f}")
    for r in sorted(results, key=lambda x: x["cos_topk"])[:5]:
        print(f"  worst {r['drug']:24s}  da={r['da']:.2f}  cos={r['cos_topk']:+.3f}")
    print(f"\nWrote {res_path}")


if __name__ == "__main__":
    main()
