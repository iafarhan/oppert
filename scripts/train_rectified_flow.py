"""Rectified flow distillation from a trained teacher flow."""
from __future__ import annotations

import argparse
import json
import os
import pickle
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
    bch_compose,
    block_bracket_norm_sq,
    sample_theta,
)


def load_generators(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ck, tuple):
        sd = ck[0]
    elif isinstance(ck, dict):
        sd = ck.get("state_dict", ck)
    else:
        sd = ck
    gp_key = next((k for k in sd if k.endswith("rotation.generator_params")), None)
    if gp_key is None:
        raise ValueError(f"Could not find rotation.generator_params in {ckpt_path}")
    return sd[gp_key].float().to(device)


def build_velocity_net_from_ckpt(fd: dict, device: torch.device):
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
        ).to(device)
    else:
        head_kind = fd.get("head_kind")
        if head_kind is None:
            head_kind = "skew_block" if any(
                k.startswith("out_proj.split") or k.startswith("out_proj.to_coef")
                for k in sd.keys()
            ) else "flat"
        d_block_head = fd.get("d_block_head", 16)
        if head_kind == "skew_block" and "out_proj.split.weight" in sd:
            d_block_head = sd["out_proj.split.weight"].shape[0] // 32
        net = ConditionalVelocityNet(
            theta_dim=fd["theta_dim"], d_embed=fd["d_embed"],
            d_hidden=fd.get("d_hidden", 512), n_blocks=fd.get("n_blocks", 4),
            head_kind=head_kind, d_block_head=d_block_head,
        ).to(device)
    net.load_state_dict(sd)
    return net, velocity_arch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="Teacher flow_best.pt to distill from.")
    ap.add_argument("--ckpt", required=True, help="Base OpPert ckpt (for theta_train).")
    ap.add_argument("--dataset_h5ad", required=True)
    ap.add_argument("--multiview", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--K", type=int, default=4, help="Teacher samples per training pert.")
    ap.add_argument("--teacher_n_steps", type=int, default=20,
                    help="ODE steps for teacher rollouts (matches teacher's training n_steps).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--mask_zero_theta", action="store_true",
                    help="Drop pairs with ||theta_i|| < 1e-6 from the KRR prior fit + training set.")
    ap.add_argument("--init_from_teacher", action="store_true",
                    help="Warm-start the student v_net from the teacher's weights instead of random init. "
                         "Turns distillation into a fine-tune toward straightened trajectories.")
    ap.add_argument("--target_noise_sigma", type=float, default=0.0,
                    help="If >0, perturb A1 targets with per-batch Gaussian noise of this std. "
                         "Tests whether explicit target-noise replicates the K-scaling implicit "
                         "regularization benefit observed in iter53/54 (K=4 cold-start > K=8). "
                         "Typical: 0.05-0.20.")
    ap.add_argument("--t_sampling", type=str, default="uniform",
                    choices=["uniform", "logit_normal", "beta_left", "beta_right"],
                    help="Distribution for sampling interpolation time t during training. "
                         "'uniform' = U(0,1) (baseline). 'logit_normal' = sigmoid(N(0,1)), SD3-style, "
                         "concentrated near t=0.5. 'beta_left' = Beta(2,5) biased to t=0 (early/noisy). "
                         "'beta_right' = Beta(5,2) biased to t=1 (late/clean).")
    ap.add_argument("--bch_aug_rate", type=float, default=0.0,
                    help="If >0, append bch_aug_rate*N BCH-pseudo-combo pairs per K-rollout to "
                         "the distilled corpus. theta_combo = BCH(A1_a, A1_b) on teacher rollouts, "
                         "A0_combo = BCH(A0_a, A0_b), e_combo = mean(e_a, e_b). Doubles or more the "
                         "effective training set without requiring real combo perts. Directive lever #1.")
    ap.add_argument("--bch_embed_mode", type=str, default="mean", choices=["mean", "sum"],
                    help="How to combine the two single-pert embeddings into a pseudo-combo embed.")
    ap.add_argument("--student_arch", type=str, default="", choices=["", "mlp", "transformer"],
                    help="Student velocity-net architecture. '' = match teacher (default, back-compat). "
                         "'transformer' = DiT-style self-attention over 32 so(4)-block tokens (directive "
                         "lever #2), distilled from a (likely MLP) teacher's K-rollouts. Arch mismatch "
                         "between teacher and student is fine — teacher only produces (A0, A1) pairs.")
    ap.add_argument("--student_tx_d_token", type=int, default=128)
    ap.add_argument("--student_tx_n_layers", type=int, default=4)
    ap.add_argument("--student_tx_n_heads", type=int, default=4)
    ap.add_argument("--student_tx_mlp_mult", type=float, default=4.0)
    ap.add_argument("--cfg_drop", type=float, default=0.0,
                    help="Classifier-free guidance training drop rate. If >0, zero the "
                         "embedding with probability cfg_drop per-sample so the student "
                         "learns both p(v|e) and p(v|∅). Directive lever #6. At inference "
                         "pass --cfg_scale w>1 to eval_fair_comparison.py; typical w in [1.5, 3].")
    ap.add_argument("--bracket_reg", type=float, default=0.0,
                    help="Weight on block-wise ||[A_t, v(A_t)]||_F^2 penalty (directive lever #4). "
                         "Encourages v(A_t) to commute with A_t block-by-block — i.e., the flow "
                         "trajectory to be a geodesic in SO(n). 0.0 = off. Typical: 1e-3..1e-1.")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    # 1. Load teacher net + ckpt metadata
    print(f"Loading teacher: {args.teacher}")
    fd_t = torch.load(args.teacher, map_location=device, weights_only=False)
    teacher_net, teacher_arch = build_velocity_net_from_ckpt(fd_t, device)
    teacher_net.eval()
    teacher_sigma = float(fd_t.get("sigma", 1.0))
    teacher_prior_kind = fd_t.get("prior", "none") or "none"
    teacher_krr_alpha = fd_t.get("krr_alpha", None)
    teacher_krr_gamma = fd_t.get("krr_gamma", None)
    theta_dim = fd_t["theta_dim"]
    d_embed = fd_t["d_embed"]
    print(f"Teacher: arch={teacher_arch} sigma={teacher_sigma} prior={teacher_prior_kind} "
          f"theta_dim={theta_dim} d_embed={d_embed}")

    # 2. Load dataset → pert_names_ordered (matches base ckpt's generator_params row order)
    import scanpy as sc
    adata = sc.read_h5ad(args.dataset_h5ad)
    perts_names = adata.obs["perturbation"].astype(str).values
    all_parts = set()
    for pp in perts_names:
        for part in pp.split("+"):
            all_parts.add(part)
    pert_names_ordered = np.array(sorted(all_parts))

    # 3. Load base ckpt generators + multiview, build training pairs
    gp = load_generators(args.ckpt, device)
    assert gp.shape[0] == len(pert_names_ordered), \
        f"pert count mismatch: ckpt {gp.shape[0]} vs dataset {len(pert_names_ordered)}"
    mv = torch.load(args.multiview, map_location=device, weights_only=False)
    mv_genes = mv["gene_names"]
    mv_embed = mv["embed"].float().to(device)
    g2idx = {g: i for i, g in enumerate(mv_genes)}
    pairs = []
    for p_idx, p_name in enumerate(pert_names_ordered.tolist()):
        if "+" in p_name or p_name == "control":
            continue
        if p_name in g2idx:
            pairs.append((p_idx, g2idx[p_name]))
    pert_idx = torch.tensor([p for p, _ in pairs], dtype=torch.long, device=device)
    mv_idx = torch.tensor([m for _, m in pairs], dtype=torch.long, device=device)
    theta_train = gp[pert_idx].detach()
    embed_train = mv_embed[mv_idx].detach()

    if args.mask_zero_theta:
        keep = theta_train.norm(dim=1) > 1e-6
        n_drop = int((~keep).sum().item())
        if n_drop > 0:
            pert_idx = pert_idx[keep]; mv_idx = mv_idx[keep]
            theta_train = theta_train[keep].contiguous()
            embed_train = embed_train[keep].contiguous()
            print(f"[mask_zero_theta] dropped {n_drop} zero-norm pairs")
    N = pert_idx.shape[0]
    print(f"Training perts: N={N}")

    # 4. Build KRR prior — load from teacher's krr_prior.pkl if present, else CV-fit.
    prior_train = None
    if teacher_prior_kind == "krr":
        from sklearn.kernel_ridge import KernelRidge
        krr_pkl = os.path.join(os.path.dirname(args.teacher), "krr_prior.pkl")
        if os.path.exists(krr_pkl):
            with open(krr_pkl, "rb") as f:
                kp = pickle.load(f)
            X_train_np = kp["X_train"]; Y_train_np = kp["Y_train"]
            gamma = kp["gamma"]; alpha = kp["alpha"]
            print(f"Loaded teacher KRR prior: gamma={gamma} alpha={alpha} (from {krr_pkl})")
        else:
            from sklearn.model_selection import KFold
            X_all = mv_embed.detach().cpu().numpy()
            X_train_np = X_all[mv_idx.cpu().numpy()]
            Y_train_np = theta_train.detach().cpu().numpy()
            gamma = teacher_krr_gamma; alpha = teacher_krr_alpha
            if gamma is None or alpha is None:
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
                gamma = best_g; alpha = best_a
                print(f"KRR best gamma={gamma} alpha={alpha} CV MSE={best_err:.4f}")
        krr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
        krr.fit(X_train_np, Y_train_np)
        # Predict on the current training embeddings (size N) — when --mask_zero_theta
        # is off, N can exceed the teacher's 815 support set.
        X_flow_np = mv_embed[mv_idx].detach().cpu().numpy()
        Y_pred_train_np = krr.predict(X_flow_np)
        prior_train = torch.from_numpy(Y_pred_train_np).float().to(device)
        Y_train_ref = theta_train.detach().cpu().numpy()
        krr_train_cos = float(np.mean([
            float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            for a, b in zip(Y_pred_train_np, Y_train_ref)
        ]))
        print(f"KRR train recon cos: {krr_train_cos:.4f}")
        # Save the same krr_prior.pkl in output dir so eval can find it.
        with open(os.path.join(args.output, "krr_prior.pkl"), "wb") as f:
            pickle.dump({"gamma": gamma, "alpha": alpha,
                         "X_train": X_train_np, "Y_train": Y_train_np,
                         "mv_gene_names": mv_genes}, f)
        teacher_krr_gamma, teacher_krr_alpha = gamma, alpha

    # 5. Generate teacher rollouts: K samples per pert, in chunks to fit GPU.
    print(f"Distilling: K={args.K} samples per pert, teacher_n_steps={args.teacher_n_steps}")
    bch_basis = None
    n_bch = 0
    if args.bch_aug_rate > 0.0:
        bch_basis = _build_so4_basis(device)
        n_bch = int(round(args.bch_aug_rate * N))
        print(f"[bch_aug] rate={args.bch_aug_rate} -> n_bch={n_bch} pseudo-combo pairs per K "
              f"(embed_mode={args.bch_embed_mode})")
    A0_list = []; A1_list = []; e_list = []
    A0_bch_list = []; A1_bch_list = []; e_bch_list = []
    t_distill = time.time()
    for k in range(args.K):
        # Per-k seed for reproducibility + diversity across K
        torch.manual_seed(args.seed * 10000 + k)
        torch.cuda.manual_seed_all(args.seed * 10000 + k)
        noise = torch.randn(N, theta_dim, device=device)
        if prior_train is not None:
            A0_k = prior_train + teacher_sigma * noise
        else:
            A0_k = teacher_sigma * noise
        # Roll out teacher: A1 = A0 + integral of v over [0,1].
        # Re-implement euler so we control A0 (sample_theta re-samples noise internally).
        A_t = A0_k.clone()
        dt = 1.0 / args.teacher_n_steps
        with torch.no_grad():
            for step in range(args.teacher_n_steps):
                t = torch.full((N,), step * dt, device=device)
                v = teacher_net(A_t, t, embed_train)
                A_t = A_t + dt * v
        A1_k = A_t  # teacher's A_1 prediction
        A0_list.append(A0_k); A1_list.append(A1_k); e_list.append(embed_train)

        if n_bch > 0:
            # Pseudo-combo pairs from teacher singles via BCH.
            a_idx = torch.randint(0, N, (n_bch,), device=device)
            b_idx = torch.randint(0, N, (n_bch,), device=device)
            eq = a_idx == b_idx
            if eq.any():
                b_idx = torch.where(eq, (b_idx + 1) % N, b_idx)
            A0_combo = bch_compose(A0_k[a_idx], A0_k[b_idx], basis=bch_basis)
            A1_combo = bch_compose(A1_k[a_idx], A1_k[b_idx], basis=bch_basis)
            if args.bch_embed_mode == "mean":
                e_combo = 0.5 * (embed_train[a_idx] + embed_train[b_idx])
            else:  # "sum"
                e_combo = embed_train[a_idx] + embed_train[b_idx]
            A0_bch_list.append(A0_combo); A1_bch_list.append(A1_combo); e_bch_list.append(e_combo)
    A0_flat = torch.cat(A0_list + A0_bch_list, dim=0).contiguous()  # [N*K + n_bch*K, theta_dim]
    A1_flat = torch.cat(A1_list + A1_bch_list, dim=0).contiguous()
    e_flat = torch.cat(e_list + e_bch_list, dim=0).contiguous()
    M = A0_flat.shape[0]
    M_singles = N * args.K
    M_combos = M - M_singles
    print(f"Distilled corpus: M={M} pairs ({M_singles} singles + {M_combos} BCH-combos) "
          f"in {time.time()-t_distill:.1f}s")
    # Diagnostic: how close does the teacher get to theta_train? (singles portion only)
    with torch.no_grad():
        A1_singles = A1_flat[:M_singles]
        A0_singles = A0_flat[:M_singles]
        A1_K = A1_singles.view(args.K, N, theta_dim).mean(dim=0)  # K-sample mean
        cos_to_theta = F.cosine_similarity(A1_K, theta_train, dim=1).mean().item()
        # Average straight-line displacement norm (whole corpus incl. BCH)
        disp_norm = (A1_flat - A0_flat).norm(dim=1).mean().item()
        print(f"Teacher vs theta_train: K-mean cos={cos_to_theta:.4f}  "
              f"avg ||A1-A0||={disp_norm:.4f} (over M={M} pairs)")

    teacher_state = None
    if args.init_from_teacher:
        teacher_state = {k: v.detach().clone() for k, v in teacher_net.state_dict().items()}

    # Free teacher memory
    del teacher_net
    torch.cuda.empty_cache()

    # 6. Init student v_net. Default: match teacher arch. Override via --student_arch.
    student_arch = args.student_arch or teacher_arch
    if student_arch == "transformer":
        if teacher_arch == "transformer":
            stu_d_token = fd_t.get("tx_d_token", 128)
            stu_n_layers = fd_t.get("tx_n_layers", 4)
            stu_n_heads = fd_t.get("tx_n_heads", 4)
            stu_mlp_mult = fd_t.get("tx_mlp_mult", 4.0)
        else:
            stu_d_token = args.student_tx_d_token
            stu_n_layers = args.student_tx_n_layers
            stu_n_heads = args.student_tx_n_heads
            stu_mlp_mult = args.student_tx_mlp_mult
        student = BlockTransformerVelocityNet(
            theta_dim=theta_dim, d_embed=d_embed,
            d_token=stu_d_token,
            d_cond=fd_t.get("d_hidden", 512),
            n_layers=stu_n_layers,
            n_heads=stu_n_heads,
            mlp_mult=stu_mlp_mult,
        ).to(device)
    else:
        head_kind = fd_t.get("head_kind", "flat") or "flat"
        student = ConditionalVelocityNet(
            theta_dim=theta_dim, d_embed=d_embed,
            d_hidden=fd_t.get("d_hidden", 512),
            n_blocks=fd_t.get("n_blocks", 4),
            head_kind=head_kind,
            d_block_head=fd_t.get("d_block_head", 16),
        ).to(device)
    print(f"Student[{student_arch}] (teacher={teacher_arch}) params: {sum(p.numel() for p in student.parameters()):,}")

    if teacher_state is not None:
        if student_arch != teacher_arch:
            print(f"[init_from_teacher] skipped — arch mismatch "
                  f"(teacher={teacher_arch}, student={student_arch})")
        else:
            missing, unexpected = student.load_state_dict(teacher_state, strict=False)
            print(f"[init_from_teacher] loaded teacher state into student "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    bracket_basis = _build_so4_basis(device, torch.float32) if args.bracket_reg > 0 else None

    # 7. Training loop on coupled pairs.
    metrics_path = os.path.join(args.output, "flow_metrics.jsonl")
    best_loss = float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        student.train()
        perm = torch.randperm(M, device=device)
        ep_loss = 0.0; n_batches = 0
        for i in range(0, M, args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            A0_b = A0_flat[b]; A1_b = A1_flat[b]; e_b = e_flat[b]
            if args.target_noise_sigma > 0.0:
                A1_b = A1_b + args.target_noise_sigma * torch.randn_like(A1_b)
            nb = b.shape[0]
            if args.t_sampling == "uniform":
                t = torch.rand(nb, device=device)
            elif args.t_sampling == "logit_normal":
                t = torch.sigmoid(torch.randn(nb, device=device))
            elif args.t_sampling == "beta_left":
                t = torch.distributions.Beta(2.0, 5.0).sample((nb,)).to(device)
            elif args.t_sampling == "beta_right":
                t = torch.distributions.Beta(5.0, 2.0).sample((nb,)).to(device)
            else:
                raise ValueError(f"Unknown t_sampling: {args.t_sampling}")
            tt = t[:, None]
            A_t = (1.0 - tt) * A0_b + tt * A1_b
            target = A1_b - A0_b  # straight-line OT velocity
            if args.cfg_drop > 0.0:
                drop_mask = (torch.rand(nb, device=device) < args.cfg_drop).float().unsqueeze(1)
                e_in = e_b * (1.0 - drop_mask)
            else:
                e_in = e_b
            v_pred = student(A_t, t, e_in)
            loss = F.mse_loss(v_pred, target)
            if args.bracket_reg > 0.0:
                bp = block_bracket_norm_sq(A_t, v_pred, basis=bracket_basis).mean()
                loss = loss + args.bracket_reg * bp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss); n_batches += 1
        sched.step()
        avg_loss = ep_loss / max(1, n_batches)
        lr_cur = sched.get_last_lr()[0]

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            student.eval()
            with torch.no_grad():
                # Recon: sample student on train embeds with deterministic noise (seed=0)
                gen = torch.Generator(device=device).manual_seed(0)
                z = torch.randn(N, theta_dim, device=device, generator=gen)
                A_s = (prior_train if prior_train is not None else 0.0) + teacher_sigma * z
                dt = 1.0 / args.teacher_n_steps
                for step in range(args.teacher_n_steps):
                    t_s = torch.full((N,), step * dt, device=device)
                    A_s = A_s + dt * student(A_s, t_s, embed_train)
                cos_per = F.cosine_similarity(A_s, theta_train, dim=1)
                tnorm = theta_train.norm(dim=1)
                nz_mask = tnorm > 1e-6
                cos = cos_per[nz_mask].mean().item() if nz_mask.any() else float('nan')
                mse = F.mse_loss(A_s, theta_train).item()
            elapsed = time.time() - t0
            n_nz = int(nz_mask.sum().item())
            print(f"E{epoch:04d}  loss={avg_loss:.4f}  recon_cos={cos:.3f}  "
                  f"(nz={n_nz}/{cos_per.numel()})  mse={mse:.3f}  lr={lr_cur:.2e}  "
                  f"({elapsed:.1f}s)")
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "loss": avg_loss, "recon_cos": cos,
                    "recon_mse": mse, "lr": lr_cur, "elapsed": elapsed,
                }) + "\n")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "v_net_state_dict": student.state_dict(),
                    "theta_dim": theta_dim,
                    "d_embed": d_embed,
                    "d_hidden": fd_t.get("d_hidden", 512),
                    "n_blocks": fd_t.get("n_blocks", 4),
                    "sigma": teacher_sigma,
                    "n_steps": args.teacher_n_steps,
                    "epoch": epoch,
                    "loss": avg_loss,
                    "pert_names": pert_names_ordered.tolist(),
                    "mv_gene_names": mv_genes,
                    "prior": teacher_prior_kind,
                    "krr_gamma": teacher_krr_gamma,
                    "krr_alpha": teacher_krr_alpha,
                    "head_kind": fd_t.get("head_kind", "flat") or "flat",
                    "d_block_head": fd_t.get("d_block_head", 16),
                    "velocity_arch": student_arch,
                    "teacher_arch": teacher_arch,
                    "tx_d_token": (fd_t.get("tx_d_token", 128) if teacher_arch == "transformer"
                                   else args.student_tx_d_token),
                    "tx_n_layers": (fd_t.get("tx_n_layers", 4) if teacher_arch == "transformer"
                                    else args.student_tx_n_layers),
                    "tx_n_heads": (fd_t.get("tx_n_heads", 4) if teacher_arch == "transformer"
                                   else args.student_tx_n_heads),
                    "tx_mlp_mult": (fd_t.get("tx_mlp_mult", 4.0) if teacher_arch == "transformer"
                                    else args.student_tx_mlp_mult),
                    "cfg_drop": args.cfg_drop,
                    # rectified-flow distillation provenance
                    "rectified_from": args.teacher,
                    "rectified_K": args.K,
                    "rectified_teacher_n_steps": args.teacher_n_steps,
                    "target_noise_sigma": args.target_noise_sigma,
                    "t_sampling": args.t_sampling,
                    "bch_aug_rate": args.bch_aug_rate,
                    "bch_embed_mode": args.bch_embed_mode,
                    "bracket_reg": args.bracket_reg,
                }, os.path.join(args.output, "flow_best.pt"))

    print(f"Done. best_loss={best_loss:.4f}  ckpt={os.path.join(args.output, 'flow_best.pt')}")


if __name__ == "__main__":
    main()
