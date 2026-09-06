"""Train conditional velocity network on rotation generators."""
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
    BlockTransformerVelocityNet,
    ConditionalVelocityNet,
    _build_so4_basis,
    flow_matching_loss,
    sample_theta,
    sample_theta_ensemble,
)


def load_generators(ckpt_path: str, device: torch.device):
    """Extract rotation.generator_params [num_perts, 192] from a tuple-format OpPert ckpt."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ck, tuple):
        sd = ck[0]
    elif isinstance(ck, dict):
        sd = ck.get("state_dict", ck)
    else:
        sd = ck
    # Find rotation.generator_params
    gp_key = None
    for k in sd:
        if k.endswith("rotation.generator_params"):
            gp_key = k
            break
    if gp_key is None:
        raise ValueError(f"Could not find rotation.generator_params in {ckpt_path}")
    gp = sd[gp_key].float().to(device)
    return gp  # [num_perts, 192]


def load_multiview(mv_path: str, device: torch.device):
    """Load multi-view descriptor. Returns dict with 'gene_names' and 'embed'."""
    d = torch.load(mv_path, map_location=device, weights_only=False)
    return d


def map_perts_to_embed(pert_names: list, mv_gene_names: list):
    """Return (pert_idx, mv_idx) pairs for training singles."""
    g2idx = {g: i for i, g in enumerate(mv_gene_names)}
    pairs = []
    for p_idx, p_name in enumerate(pert_names):
        if "+" in p_name or p_name == "control":
            continue
        if p_name in g2idx:
            pairs.append((p_idx, g2idx[p_name]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset_h5ad", required=True)
    ap.add_argument("--multiview", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--d_hidden", type=int, default=512)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--prior", type=str, default="none", choices=["none", "krr"],
                    help="Prior for A_0: 'none' (Gaussian) or 'krr' (RBF KRR on multiview).")
    ap.add_argument("--krr_gamma", type=float, default=None,
                    help="RBF gamma; if None with --prior krr, CV-picked.")
    ap.add_argument("--krr_alpha", type=float, default=None,
                    help="Ridge alpha; if None with --prior krr, CV-picked.")
    ap.add_argument("--mask_zero_theta", action="store_true",
                    help="Drop training pairs with ||theta_i|| < 1e-6 before KRR fit + flow training.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed torch+numpy+python RNGs for reproducibility. Unset = nondeterministic.")
    ap.add_argument("--head", type=str, default="flat", choices=["flat", "skew_block"],
                    help="Velocity-net output head. 'flat' = Linear(d_hidden, theta_dim); "
                         "'skew_block' = per-so(4)-block shared-weight head (lever #1, "
                         "Lie-algebra-aware parameterization).")
    ap.add_argument("--d_block_head", type=int, default=16,
                    help="Intermediate dim for skew_block head (ignored if head=flat).")
    ap.add_argument("--ema_decay", type=float, default=0.0,
                    help="If >0, maintain EMA of v_net weights with this decay; "
                         "validation + checkpoint use EMA weights. 0.999 is typical.")
    ap.add_argument("--t_schedule", type=str, default="uniform",
                    choices=["uniform", "logit_normal"],
                    help="Time-sampling schedule (lever #5). 'uniform' = t~U[0,1] (default). "
                         "'logit_normal' = t=sigmoid(u), u~N(m,s^2); biases mass toward t=0.5 "
                         "(SD3-style weighting on harder-to-predict midpoints).")
    ap.add_argument("--t_logit_m", type=float, default=0.0,
                    help="Mean of logit-normal t sampler (negative biases toward t=0; positive toward t=1)")
    ap.add_argument("--t_logit_s", type=float, default=1.0,
                    help="Std of logit-normal t sampler (smaller s = tighter mass near sigmoid(m))")
    ap.add_argument("--interpolant", type=str, default="ot", choices=["ot", "cosine"],
                    help="Stochastic interpolant (lever #4). 'ot' = (1-t)A_0 + t A_1 "
                         "(straight paths, default); 'cosine' = cos(pi t/2) A_0 + sin(pi t/2) A_1.")
    ap.add_argument("--bch_aug_rate", type=float, default=0.0,
                    help="Fraction of each batch slot to replace with BCH-composed pseudo-combos "
                         "(lever #1 from directives). 0.0 = disabled. Typical: 0.25-0.75.")
    ap.add_argument("--bch_embed_mode", type=str, default="mean", choices=["mean", "sum"],
                    help="How to combine multi-view embeddings for the pseudo-combo: 'mean' "
                         "(e_a+e_b)/2 or 'sum' e_a+e_b.")
    ap.add_argument("--velocity_arch", type=str, default="mlp", choices=["mlp", "transformer"],
                    help="Velocity-net architecture (lever #2). 'mlp' = flat AdaLN-MLP "
                         "(default, back-compat); 'transformer' = DiT-style self-attention "
                         "over 32 so(4)-block tokens.")
    ap.add_argument("--tx_d_token", type=int, default=128,
                    help="Per-block token dim for velocity_arch=transformer.")
    ap.add_argument("--tx_n_layers", type=int, default=4,
                    help="Transformer depth (n layers) for velocity_arch=transformer.")
    ap.add_argument("--tx_n_heads", type=int, default=4,
                    help="Attention heads for velocity_arch=transformer.")
    ap.add_argument("--tx_mlp_mult", type=float, default=4.0,
                    help="Feed-forward expansion for velocity_arch=transformer.")
    ap.add_argument("--cfg_drop", type=float, default=0.0,
                    help="Classifier-free guidance training dropout rate (lever #6). "
                         "Per-sample probability of replacing the conditioning embedding "
                         "with zeros during training. 0.0 = off. Typical: 0.10.")
    ap.add_argument("--ood_json", type=str, default=None,
                    help="Optional split JSON with keys 'train_drugs' and 'ood_drugs' "
                         "(list[str]). When set, training pairs are restricted to perts "
                         "in train_drugs and perts in ood_drugs are excluded. Used for "
                         "SciPlex3 where the backbone sees all drugs and the flow's OOD "
                         "split must be enforced at the flow-training boundary.")
    args = ap.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"[seed] set torch/np/random to {args.seed}")

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    # 1. Load dataset to get pert_names
    import scanpy as sc
    adata = sc.read_h5ad(args.dataset_h5ad)
    # Replicate dataset processing: split pert names, collect uniques (singles only for flow)
    perts_names = adata.obs["perturbation"].astype(str).values
    unique_singles = sorted(set(g for p in perts_names for g in p.split("+") if g != "control"))
    # But we care about the order matching the model's pert indices. Look at perturbation_key uniques:
    # OpPert uses self.perts_names_unique (sorted singles). See OpPert/data/norman.py:203
    # It gets: set of all parts of all perts (splitting by +), excluding control.
    all_parts = set()
    for pp in perts_names:
        for part in pp.split("+"):
            all_parts.add(part)  # keep "control" so count matches model
    pert_names_ordered = np.array(sorted(all_parts))  # matches model's generator_params row order

    # 2. Load trained generators
    gp = load_generators(args.ckpt, device)  # [num_perts, 192]
    num_perts, theta_dim = gp.shape
    assert num_perts == len(pert_names_ordered), (
        f"pert count mismatch: ckpt has {num_perts}, dataset has {len(pert_names_ordered)}"
    )
    print(f"Loaded {num_perts} generators of dim {theta_dim} from {args.ckpt}")

    # 3. Load multi-view
    mv = load_multiview(args.multiview, device)
    mv_genes = mv["gene_names"]
    mv_embed = mv["embed"].float().to(device)  # [M, d_embed]
    d_embed = mv_embed.shape[1]
    print(f"Multi-view: {len(mv_genes)} genes, d_embed={d_embed}")

    # 4. Map pert_names → multiview rows; build training tensors
    pairs = map_perts_to_embed(pert_names_ordered.tolist(), mv_genes)
    if len(pairs) < 10:
        raise RuntimeError(f"Only {len(pairs)} training pairs — check name matching")

    # Optional: filter by JSON split (SciPlex3 v2 — backbone saw all drugs, so the
    # OOD split must be enforced here at the flow-training boundary).
    if args.ood_json is not None:
        with open(args.ood_json) as f:
            split = json.load(f)
        train_set = set(split.get("train_drugs", []))
        ood_set = set(split.get("ood_drugs", []))
        pert_names_list = pert_names_ordered.tolist()
        before = len(pairs)
        filtered = []
        dropped_ood = 0
        dropped_notrain = 0
        for p_idx, m_idx in pairs:
            name = pert_names_list[p_idx]
            if name in ood_set:
                dropped_ood += 1
                continue
            if train_set and name not in train_set:
                dropped_notrain += 1
                continue
            filtered.append((p_idx, m_idx))
        pairs = filtered
        print(f"[ood_json] {args.ood_json}: kept {len(pairs)}/{before} pairs "
              f"(dropped {dropped_ood} OOD, {dropped_notrain} not-in-train_drugs)")

    pert_idx = torch.tensor([p for p, _ in pairs], dtype=torch.long, device=device)
    mv_idx = torch.tensor([m for _, m in pairs], dtype=torch.long, device=device)
    theta_train = gp[pert_idx].detach()  # [N_train, 192]
    embed_train = mv_embed[mv_idx].detach()  # [N_train, d_embed]
    N = pert_idx.shape[0]
    print(f"Training pairs: {N} singles")
    if args.mask_zero_theta:
        tnorm_all = theta_train.norm(dim=1)
        keep = tnorm_all > 1e-6
        n_drop = int((~keep).sum().item())
        if n_drop > 0:
            pert_idx = pert_idx[keep]
            mv_idx = mv_idx[keep]
            theta_train = theta_train[keep].contiguous()
            embed_train = embed_train[keep].contiguous()
            N = pert_idx.shape[0]
            print(f"[mask_zero_theta] dropped {n_drop} zero-norm pairs; {N} remain")
    print(f"theta_train: shape={theta_train.shape} mean={theta_train.float().mean():.4f} std={theta_train.float().std():.4f}")

    # 4b. Optional KRR prior: predict A_0 base from multiview descriptor.
    prior_train = None
    krr_info = None
    if args.prior == "krr":
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import KFold
        X_all = mv_embed.detach().cpu().numpy()
        X_train_np = X_all[mv_idx.cpu().numpy()]
        Y_train_np = theta_train.detach().cpu().numpy()
        if args.krr_gamma is None or args.krr_alpha is None:
            print("KRR CV: picking gamma, alpha...")
            best_err = float("inf"); best_g = None; best_a = None
            kf = KFold(n_splits=min(5, N), shuffle=True, random_state=0)
            for g_ in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                for a_ in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                    errs = []
                    for tr, te in kf.split(X_train_np):
                        m = KernelRidge(alpha=a_, kernel="rbf", gamma=g_)
                        m.fit(X_train_np[tr], Y_train_np[tr])
                        errs.append(float(np.mean((m.predict(X_train_np[te]) - Y_train_np[te]) ** 2)))
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
        print(f"KRR train recon cos (on training perts): {krr_train_cos:.4f}")
        prior_train = torch.from_numpy(Y_pred_train_np).float().to(device)
        # Save full-KRR params so we can reconstruct predictions for OOD perts later.
        krr_info = {
            "gamma": float(args.krr_gamma),
            "alpha": float(args.krr_alpha),
            "train_cos": krr_train_cos,
            "mv_gene_names": mv_genes,
            "train_mv_idx": mv_idx.detach().cpu().numpy().tolist(),
            "train_pert_idx": pert_idx.detach().cpu().numpy().tolist(),
            "X_train": X_train_np,
            "Y_train": Y_train_np,
        }
        import pickle
        with open(os.path.join(args.output, "krr_prior.pkl"), "wb") as f:
            pickle.dump({"gamma": args.krr_gamma, "alpha": args.krr_alpha,
                         "X_train": X_train_np, "Y_train": Y_train_np,
                         "mv_gene_names": mv_genes}, f)
        print(f"Saved KRR prior artifacts to {os.path.join(args.output, 'krr_prior.pkl')}")

    # 5. Init velocity net
    if args.velocity_arch == "mlp":
        v_net = ConditionalVelocityNet(
            theta_dim=theta_dim,
            d_embed=d_embed,
            d_hidden=args.d_hidden,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            head_kind=args.head,
            d_block_head=args.d_block_head,
        ).to(device)
    elif args.velocity_arch == "transformer":
        v_net = BlockTransformerVelocityNet(
            theta_dim=theta_dim,
            d_embed=d_embed,
            d_token=args.tx_d_token,
            d_cond=args.d_hidden,
            n_layers=args.tx_n_layers,
            n_heads=args.tx_n_heads,
            mlp_mult=args.tx_mlp_mult,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"unknown velocity_arch={args.velocity_arch!r}")
    n_params = sum(p.numel() for p in v_net.parameters())
    print(f"VelocityNet[{args.velocity_arch}] params: {n_params:,}")

    opt = torch.optim.AdamW(v_net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Pre-build so(4) basis for BCH augmentation (constant per-device).
    bch_basis = _build_so4_basis(device) if args.bch_aug_rate > 0.0 else None
    if args.bch_aug_rate > 0.0:
        print(f"[bch_aug] rate={args.bch_aug_rate}  embed_mode={args.bch_embed_mode}")

    # Optional EMA shadow net (separate weight tensors, synced after every opt.step()).
    v_net_ema = None
    if args.ema_decay > 0.0:
        import copy
        v_net_ema = copy.deepcopy(v_net).to(device)
        for p in v_net_ema.parameters():
            p.requires_grad_(False)
        print(f"[ema] enabled with decay={args.ema_decay}")

    # 6. Training loop
    metrics_path = os.path.join(args.output, "flow_metrics.jsonl")
    best_loss = float("inf")
    t_start = time.time()
    for epoch in range(args.epochs):
        v_net.train()
        perm = torch.randperm(N, device=device)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, N, args.batch):
            b_idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = flow_matching_loss(
                v_net, theta_train, embed_train, b_idx, sigma_noise=args.sigma,
                prior_train=prior_train, interpolant=args.interpolant,
                t_schedule=args.t_schedule,
                t_logit_m=args.t_logit_m, t_logit_s=args.t_logit_s,
                bch_aug_rate=args.bch_aug_rate,
                bch_basis=bch_basis,
                bch_embed_mode=args.bch_embed_mode,
                cfg_drop_prob=args.cfg_drop,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
            opt.step()
            if v_net_ema is not None:
                d = args.ema_decay
                with torch.no_grad():
                    for p_ema, p in zip(v_net_ema.parameters(), v_net.parameters()):
                        p_ema.mul_(d).add_(p.detach(), alpha=1.0 - d)
                    for b_ema, b in zip(v_net_ema.buffers(), v_net.buffers()):
                        b_ema.copy_(b)
            ep_loss += float(loss)
            n_batches += 1
        sched.step()
        avg_loss = ep_loss / max(1, n_batches)
        lr_cur = sched.get_last_lr()[0]

        # Validate: sample theta for training perts, measure reconstruction of lookup
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            v_eval = v_net_ema if v_net_ema is not None else v_net
            v_eval.eval()
            sampled = sample_theta(
                v_eval, embed_train, n_steps=args.n_steps, sigma_noise=args.sigma,
                prior=prior_train,
            )
            # Mask out zero-norm training targets from cos (some ckpt gen-param slots
            # are identically zero: control / combo-only / untrained rows). F.cosine_similarity
            # returns 0 for zero vectors, dragging the mean down to an artifact.
            tnorm = theta_train.norm(dim=1)
            nz_mask = tnorm > 1e-6
            cos_per = F.cosine_similarity(sampled, theta_train, dim=1)
            cos = cos_per[nz_mask].mean().item() if nz_mask.any() else float('nan')
            cos_all = cos_per.mean().item()  # legacy metric, for comparison
            mse = F.mse_loss(sampled, theta_train).item()
            elapsed = time.time() - t_start
            n_nz = int(nz_mask.sum().item())
            print(f"E{epoch:04d}  loss={avg_loss:.4f}  recon_cos={cos:.3f}  (nz={n_nz}/{cos_per.numel()}; legacy_cos={cos_all:.3f})  mse={mse:.3f}  lr={lr_cur:.2e}  ({elapsed:.1f}s)")
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "loss": avg_loss, "recon_cos": cos,
                    "recon_cos_legacy": cos_all, "n_nonzero": n_nz,
                    "recon_mse": mse, "lr": lr_cur, "elapsed": elapsed,
                }) + "\n")
            if avg_loss < best_loss:
                best_loss = avg_loss
                state_to_save = v_net_ema.state_dict() if v_net_ema is not None else v_net.state_dict()
                torch.save({
                    "v_net_state_dict": state_to_save,
                    "theta_dim": theta_dim,
                    "d_embed": d_embed,
                    "d_hidden": args.d_hidden,
                    "n_blocks": args.n_blocks,
                    "sigma": args.sigma,
                    "n_steps": args.n_steps,
                    "epoch": epoch,
                    "loss": avg_loss,
                    "pert_names": pert_names_ordered.tolist(),
                    "mv_gene_names": mv_genes,
                    "prior": args.prior,
                    "krr_gamma": args.krr_gamma if args.prior == "krr" else None,
                    "krr_alpha": args.krr_alpha if args.prior == "krr" else None,
                    "head_kind": args.head,
                    "d_block_head": args.d_block_head,
                    "velocity_arch": args.velocity_arch,
                    "tx_d_token": args.tx_d_token,
                    "tx_n_layers": args.tx_n_layers,
                    "tx_n_heads": args.tx_n_heads,
                    "tx_mlp_mult": args.tx_mlp_mult,
                    "cfg_drop": args.cfg_drop,
                }, os.path.join(args.output, "flow_best.pt"))

    print(f"Training done. Best loss={best_loss:.4f}")
    print(f"Checkpoint: {os.path.join(args.output, 'flow_best.pt')}")


if __name__ == "__main__":
    main()
