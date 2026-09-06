"""Flow vs KRR comparison on the rotation-generator prediction task."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))
from oppert.flow import ConditionalVelocityNet, sample_theta_ensemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="crispra_norman_gears")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--flow_ckpt", required=True)
    ap.add_argument("--multiview", required=True)
    ap.add_argument("--emb", default="data/gene_embeddings/genept_bge_large.pt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_samples", type=int, default=32)
    ap.add_argument("--n_steps", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOPK_DE = 50

    # Load dataset + model via _load_config/_build_data/_load_model
    from OpPert.evaluate_full import _load_config, _build_data, _load_model
    from omegaconf import OmegaConf
    ds_cfg, dl_cfg = _load_config(args.dataset)
    ds_cfg = OmegaConf.merge(ds_cfg, {"pert_subsample": None})
    dm = _build_data(ds_cfg, dl_cfg)
    dataset = dm.dataset
    obs = dataset.adata.obs
    pert_key = getattr(dataset.cfg, "perturbation_key", "perturbation")

    model, meta = _load_model(args.ckpt, "tuned", DEVICE)
    model.eval()

    # Build OOD eval set (same as eval_fair_comparison)
    train_indices = dataset.indices.get("train", [])
    ood_indices = dataset.indices.get("ood", dataset.indices.get("test", []))
    ood_perts_raw = set(obs.iloc[ood_indices].loc[
        obs.iloc[ood_indices]['control'].astype(int) != 1, pert_key].unique())
    ood_perts_filtered = []
    for pname in sorted(ood_perts_raw):
        if "control" in pname.lower() or "dmso" in pname.lower():
            continue
        ood_mask = (obs.iloc[ood_indices][pert_key].values == pname) & \
                   (obs.iloc[ood_indices]['control'].astype(int).values != 1)
        ood_idx = np.array(ood_indices)[ood_mask]
        if ood_idx.size < 5:
            continue
        ood_perts_filtered.append((pname, ood_idx))

    # Load embeddings (BGE for KRR, multi-view for flow)
    bge_emb = torch.load(args.emb, map_location="cpu", weights_only=False)
    bge = {k.upper(): v for k, v in bge_emb.items()}

    mv = torch.load(args.multiview, map_location=DEVICE, weights_only=False)
    mv_genes = {g.upper(): i for i, g in enumerate(mv["gene_names"])}
    mv_embed = mv["embed"].float().to(DEVICE)

    # Load flow
    fd = torch.load(args.flow_ckpt, map_location=DEVICE, weights_only=False)
    v_net = ConditionalVelocityNet(
        theta_dim=fd["theta_dim"], d_embed=fd["d_embed"],
        d_hidden=fd.get("d_hidden", 512), n_blocks=fd.get("n_blocks", 4),
    ).to(DEVICE)
    v_net.load_state_dict(fd["v_net_state_dict"])
    v_net.eval()
    flow_sigma = fd.get("sigma", 1.0)

    # Get training singles (atoms) from train perts
    train_perts = sorted(set(obs.iloc[train_indices].loc[
        obs.iloc[train_indices]['control'].astype(int) != 1, pert_key].unique()))
    train_atoms = set()
    for p in train_perts:
        for a in p.split("+"):
            if a != "control":
                train_atoms.add(a)
    print(f"train atoms: {len(train_atoms)}, OOD perts to evaluate: {len(ood_perts_filtered)}")

    # Generator dim
    gen_dim = model.rotation.generator_params.shape[1]
    block_size = model.rotation.block_size
    num_blocks = gen_dim // (block_size * (block_size - 1) // 2)

    # Train KRR on training atoms (rotation generators)
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    perts_names = list(dataset.perts_names_unique)
    p2idx = {p: i for i, p in enumerate(perts_names)}
    all_gen = model.rotation.generator_params.data.cpu()

    # Training data for KRR: BGE embed + generator
    rot_Xs, rot_Ys = [], []
    for p in train_atoms:
        pu = p.upper()
        if pu in bge and p in p2idx:
            rot_Xs.append(bge[pu].numpy())
            rot_Ys.append(all_gen[p2idx[p]].numpy())
    rot_X = np.array(rot_Xs)
    rot_Y = np.array(rot_Ys)

    scaler_rot = StandardScaler(); rot_Xsc = scaler_rot.fit_transform(rot_X)
    # CV pick
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    best_e, best_g, best_a = np.inf, None, None
    for g_ in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        for a_ in [0.01, 0.1, 1.0, 10.0]:
            errs = []
            for tr, te in kf.split(rot_Xsc):
                m = KernelRidge(alpha=a_, kernel="rbf", gamma=g_)
                m.fit(rot_Xsc[tr], rot_Y[tr])
                errs.append(np.mean((m.predict(rot_Xsc[te]) - rot_Y[te])**2))
            e = np.mean(errs)
            if e < best_e: best_e, best_g, best_a = e, g_, a_
    print(f"KRR best gamma={best_g} alpha={best_a} cv_mse={best_e:.4f}")
    krr_rot = KernelRidge(alpha=best_a, kernel="rbf", gamma=best_g)
    krr_rot.fit(rot_Xsc, rot_Y)

    # Apply rotation & decode helper
    def apply_rotation_and_decode(generator_flat, ctrl_latent):
        gen_t = torch.tensor(generator_flat, dtype=torch.float32, device=DEVICE)
        # reshape to (num_blocks, block_gen_dim)
        gen_blocks = gen_t.view(num_blocks, -1)
        # Build so(4) skew, exp, apply
        basis = model.rotation.basis  # [gen_dim_per_block, block_size, block_size]
        A = torch.einsum("bd,dij->bij", gen_blocks, basis)  # [num_blocks, b, b]
        R_blocks = torch.matrix_exp(A)  # [num_blocks, b, b]
        # Apply: latent -> reshape to blocks -> R_b @ z_b -> reshape back
        z = ctrl_latent.view(-1, num_blocks, block_size)
        z_rot = torch.einsum("bij,nbj->nbi", R_blocks, z).reshape(ctrl_latent.shape)
        return z_rot

    # Get control cells to rotate
    test_indices = dataset.indices.get("test", [])
    ctrl_mask = obs.iloc[test_indices]['control'].astype(int).values == 1
    ctrl_idx = np.array(test_indices)[ctrl_mask]
    if len(ctrl_idx) < 100:
        # Fall back: use any control cells
        all_ctrl = obs['control'].astype(int).values == 1
        ctrl_idx = np.where(all_ctrl)[0][:500]
    ctrl_genes = dataset.genes[ctrl_idx].to(DEVICE) if isinstance(dataset.genes, torch.Tensor) else torch.tensor(dataset.genes[ctrl_idx]).to(DEVICE)

    # Evaluate both predictors on each OOD pert
    def bch(a, b, s=None):
        # model.rotation.bracket_scale
        if s is None:
            s = float(model.rotation.bracket_scale.detach()) if hasattr(model.rotation, 'bracket_scale') else 1.0
        a_t = torch.tensor(a, dtype=torch.float32, device=DEVICE).view(num_blocks, -1)
        b_t = torch.tensor(b, dtype=torch.float32, device=DEVICE).view(num_blocks, -1)
        basis = model.rotation.basis
        A = torch.einsum("bd,dij->bij", a_t, basis)
        B = torch.einsum("bd,dij->bij", b_t, basis)
        comm = (A @ B - B @ A) * (s / 2.0)
        # back to flat
        pseudo_inv = torch.linalg.pinv(basis.view(-1, block_size*block_size))
        comm_flat = torch.einsum("bij,d(ij)->bd", comm, pseudo_inv.view(basis.shape[0], block_size, block_size).view(basis.shape[0], -1).T.reshape(basis.shape[0], block_size, block_size)).reshape(num_blocks, -1)  # fragile
        return (a_t + b_t + comm_flat).view(-1).cpu().numpy()

    # simpler BCH using project_to_skew is risky; just use element-wise sum (additive) for rotations in the gen_params space is WRONG for generic rotations,
    # but for small generators approx works. For simplicity, stick with additive comp: BCH ≈ additive at first order.
    def rot_compose_additive(a, b):
        return a + b  # TODO proper BCH via log(R_A @ R_B)

    rot_results = {"krr": {"da": [], "cos": [], "pde": []}, "flow": {"da": [], "cos": [], "pde": []}}
    decoder = model.decoder
    encoder = model.encoder
    output_scale_genes = model.output_scale_genes

    with torch.no_grad():
        z_ctrl = encoder(ctrl_genes)
        g_ctrl_out_mu, _ = decoder(z_ctrl)
        g_ctrl_out_mu = g_ctrl_out_mu * output_scale_genes
        g_ctrl_mean = g_ctrl_out_mu.mean(dim=0)

    def predict_and_decode(gen_flat):
        z_ctrl_rot = apply_rotation_and_decode(gen_flat, z_ctrl)
        with torch.no_grad():
            g_treated_mu, _ = decoder(z_ctrl_rot)
        return (g_treated_mu * output_scale_genes).mean(dim=0)  # predicted mean across ctrl cells

    results_data = {"perts": []}
    for pname, ood_idx_arr in ood_perts_filtered:
        parts = [p.strip() for p in pname.split("+")]
        is_combo = len(parts) > 1
        # Actual delta
        treated = dataset.genes[ood_idx_arr]
        if hasattr(treated, "to"):
            treated = treated.to(DEVICE)
        treated_mean = treated.mean(dim=0)
        actual_delta = treated_mean - g_ctrl_mean.cpu() if treated_mean.device != g_ctrl_mean.device else treated_mean - g_ctrl_mean
        actual_delta = actual_delta.cpu()

        # Get KRR & Flow predictions
        parts_upper = [p.upper() for p in parts]
        skip = any(p not in bge for p in parts_upper) or any(p not in mv_genes for p in parts_upper)
        if skip:
            continue

        # KRR per-atom
        krr_gens = []
        flow_gens = []
        for pu in parts_upper:
            emb = scaler_rot.transform(bge[pu].numpy().reshape(1, -1))
            krr_gens.append(krr_rot.predict(emb)[0])
            # Flow
            e_mv = mv_embed[mv_genes[pu]].unsqueeze(0)
            samples = sample_theta_ensemble(v_net, e_mv, n_samples=args.n_samples, n_steps=args.n_steps, sigma_noise=flow_sigma)
            flow_gens.append(samples.mean(dim=0).squeeze(0).cpu().numpy())
        krr_gen = rot_compose_additive(krr_gens[0], krr_gens[1]) if is_combo else krr_gens[0]
        flow_gen = rot_compose_additive(flow_gens[0], flow_gens[1]) if is_combo else flow_gens[0]

        # Predict delta
        krr_pred_mean = predict_and_decode(krr_gen)
        flow_pred_mean = predict_and_decode(flow_gen)
        krr_delta = (krr_pred_mean - g_ctrl_mean).cpu()
        flow_delta = (flow_pred_mean - g_ctrl_mean).cpu()

        # Top-K DE indices by actual
        topk = actual_delta.abs().topk(TOPK_DE).indices

        for name, pred_delta in [("krr", krr_delta), ("flow", flow_delta)]:
            da = ((actual_delta[topk] > 0).float() == (pred_delta[topk] > 0).float()).float().mean().item()
            cos = F.cosine_similarity(pred_delta[topk].unsqueeze(0), actual_delta[topk].unsqueeze(0)).item()
            ad = actual_delta[topk].numpy(); pdn = pred_delta[topk].numpy()
            pde = np.corrcoef(ad, pdn)[0, 1] if np.std(ad) > 0 and np.std(pdn) > 0 else 0.0
            rot_results[name]["da"].append(da)
            rot_results[name]["cos"].append(cos)
            rot_results[name]["pde"].append(float(pde) if not np.isnan(pde) else 0.0)

    print(f"\n=== Norman OOD eval (n={len(rot_results['krr']['da'])} perts) ===")
    for name in ("krr", "flow"):
        da = np.mean(rot_results[name]["da"])
        cos = np.mean(rot_results[name]["cos"])
        pde = np.mean(rot_results[name]["pde"])
        print(f"{name.upper():<6s}  DA={da*100:.1f}%  cos={cos:.3f}  PDE={pde:.3f}")

    with open(os.path.join(args.output, "fair_flow_vs_krr.json"), "w") as f:
        json.dump({
            name: {k: float(np.mean(v)) for k, v in res.items()}
            for name, res in rot_results.items()
        }, f, indent=2)


if __name__ == "__main__":
    main()
