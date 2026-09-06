#!/usr/bin/env python3
"""Reproduce Table 1 — OpPert main results.

Usage:
    python scripts/reproduce_table1.py --dataset norman --data_dir data/
    python scripts/reproduce_table1.py --dataset k562 --data_dir data/
    python scripts/reproduce_table1.py --dataset rpe1 --data_dir data/

Datasets: see README for download instructions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIGS = {
    "norman": {
        "h5ad": "norman2019_gears_split.h5ad",
        "backbone": "checkpoints/norman/backbone.pt",
        "split_key": "gears_split",
        "pert_key": "perturbation",
        "expected_da20": 87.33,
    },
    "k562": {
        "h5ad": "replogle_k562.h5ad",
        "backbone": "checkpoints/k562/backbone.pt",
        "split_key": "split",
        "pert_key": "perturbation",
        "expected_da20": 81.57,
    },
    "rpe1": {
        "h5ad": "replogle_rpe1.h5ad",
        "backbone": "checkpoints/rpe1/backbone.pt",
        "split_key": "split",
        "pert_key": "perturbation",
        "expected_da20": 87.77,
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_backbone(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict, _, _, init_args, _, _ = ckpt
    hp = init_args["hparams"]

    num_genes = init_args["num_genes"]
    num_perts = init_args["num_perts"]
    latent_dim = hp["dim"]
    block_size = hp.get("rotation_block_size", 4)
    enc_w = hp.get("autoencoder_width", 256)
    enc_d = hp.get("autoencoder_depth", 4)
    num_covariates = init_args.get("num_covariates", [0])
    if isinstance(num_covariates, int):
        num_covariates = [num_covariates]
    recon_loss = hp.get("recon_loss_genes", hp.get("recon_loss", "nll"))

    from oppert.layers import ResidualMLP, GatedResidualDecoder, MLP
    from oppert.rotation import BlockRotation

    encoder = ResidualMLP([num_genes] + [enc_w] * enc_d + [latent_dim])
    rotation = BlockRotation(num_perts, latent_dim, block_size=block_size)

    g_out_dim = num_genes if recon_loss in ("mse", "gnmse", "wmse") else num_genes * 2
    decoder_type = hp.get("decoder_type", "gated_residual")
    if decoder_type == "gated_residual":
        gh = hp.get("gated_residual_gate_hidden", hp.get("gate_hidden", 128))
        gib = hp.get("gated_residual_init_gate_bias", hp.get("gate_init_bias", -2.0))
        decoder = GatedResidualDecoder(latent_dim, g_out_dim, gate_hidden=gh, init_gate_bias=gib)
    else:
        decoder = MLP([latent_dim, enc_w, enc_w, g_out_dim])

    cov_embeddings = torch.nn.ModuleList()
    for nc in num_covariates:
        if nc > 0:
            cov_embeddings.append(torch.nn.Embedding(nc, latent_dim))

    enc_sd = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
    dec_sd = {k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder.")}
    rot_sd = {k.replace("rotation.", ""): v for k, v in state_dict.items() if k.startswith("rotation.")}
    encoder.load_state_dict(enc_sd)
    decoder.load_state_dict(dec_sd)
    rotation.load_state_dict(rot_sd)

    for k, emb in enumerate(cov_embeddings):
        prefix = f"covariates_embeddings.{k}."
        emb_sd = {key.replace(prefix, ""): v for key, v in state_dict.items() if key.startswith(prefix)}
        if emb_sd:
            emb.load_state_dict(emb_sd)

    output_scale = state_dict.get("output_scale_genes", torch.ones(1))
    is_nll = recon_loss not in ("mse", "gnmse", "wmse")

    encoder.to(DEVICE).eval()
    decoder.to(DEVICE).eval()
    rotation.to(DEVICE).eval()
    cov_embeddings.to(DEVICE).eval()
    output_scale = output_scale.to(DEVICE)

    return encoder, decoder, rotation, cov_embeddings, output_scale, num_genes, num_perts, is_nll


def decode_mean(decoder, z, output_scale, num_genes, is_nll):
    out = decoder(z)
    if is_nll:
        return output_scale * out[:, :num_genes]
    return output_scale * out


def compute_da(pred_delta, true_delta, topk=20):
    idx = true_delta.abs().argsort(descending=True)[:topk]
    return ((pred_delta[idx] > 0) == (true_delta[idx] > 0)).float().mean().item()


def compute_pearson(pred_delta, true_delta, topk=20):
    idx = true_delta.abs().argsort(descending=True)[:topk]
    p, t = pred_delta[idx], true_delta[idx]
    p, t = p - p.mean(), t - t.mean()
    return ((p * t).sum() / (p.norm() * t.norm()).clamp(min=1e-8)).item()


def compute_cosine(pred_delta, true_delta, topk=200):
    idx = true_delta.abs().argsort(descending=True)[:topk]
    return F.cosine_similarity(pred_delta[idx].unsqueeze(0), true_delta[idx].unsqueeze(0)).item()


def main():
    ap = argparse.ArgumentParser(description="Reproduce OpPert Table 1")
    ap.add_argument("--dataset", required=True, choices=list(CONFIGS.keys()))
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_ctrl", type=int, default=500)
    args = ap.parse_args()

    cfg = CONFIGS[args.dataset]
    h5ad_path = Path(args.data_dir) / cfg["h5ad"]
    if not h5ad_path.exists():
        print(f"ERROR: {h5ad_path} not found")
        sys.exit(1)

    print(f"Loading backbone from {cfg['backbone']}...")
    encoder, decoder, rotation, cov_embeddings, output_scale, num_genes, num_perts, is_nll = \
        load_backbone(ROOT / cfg["backbone"])

    print(f"Loading dataset from {h5ad_path}...")
    import scanpy as sc
    adata = sc.read_h5ad(str(h5ad_path))

    pert_key = cfg["pert_key"]
    split_key = cfg["split_key"]

    if "control" in adata.obs.columns:
        ctrl_mask = adata.obs["control"].astype(bool).values
    else:
        ctrl_mask = (adata.obs[pert_key] == "control").values

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = torch.tensor(X, dtype=torch.float32)

    atom_set = set()
    for pname in adata.obs[pert_key].unique():
        if "control" not in pname.lower() and pname.lower() != "dmso":
            for a in pname.split("+"):
                a = a.strip()
                if a:
                    atom_set.add(a)
    atoms = sorted(atom_set)
    atom_to_idx = {a: i for i, a in enumerate(atoms[:num_perts])}

    ctrl_idx = np.where(ctrl_mask)[0]
    if len(ctrl_idx) > args.max_ctrl:
        rng = np.random.RandomState(42)
        ctrl_idx = rng.choice(ctrl_idx, args.max_ctrl, replace=False)

    ctrl_genes = X[ctrl_idx].to(DEVICE)
    raw_ctrl_mean = ctrl_genes.mean(0)

    with torch.no_grad():
        h_ctrl = encoder(ctrl_genes)
        z_c = torch.zeros_like(h_ctrl)
        if len(cov_embeddings) > 0:
            z_c = cov_embeddings[0](torch.zeros(len(ctrl_idx), dtype=torch.long, device=DEVICE))
        z_basal = h_ctrl + z_c

    ood_mask = adata.obs[split_key] == "ood"
    ood_perts = sorted(adata.obs.loc[ood_mask, pert_key].unique())

    da20_list, da200_list, rho20_list, cos200_list = [], [], [], []
    n_skipped = 0

    print(f"Evaluating {len(ood_perts)} OOD perturbations...")
    for pname in ood_perts:
        if "control" in pname.lower() or pname.lower() == "dmso":
            continue

        parts = [p.strip() for p in pname.split("+")]
        if any(p not in atom_to_idx for p in parts):
            n_skipped += 1
            continue

        pert_mask = (adata.obs[pert_key] == pname) & ood_mask
        pert_idx = np.where(pert_mask.values if hasattr(pert_mask, "values") else pert_mask)[0]
        if len(pert_idx) < 5:
            n_skipped += 1
            continue

        ohe = torch.zeros(num_perts, device=DEVICE)
        for p in parts:
            ohe[atom_to_idx[p]] = 1.0
        ohe_batch = ohe.unsqueeze(0).expand(len(ctrl_idx), -1)

        with torch.no_grad():
            z_rot = rotation(z_basal, ohe_batch)
            pred_cells = decode_mean(decoder, z_rot, output_scale, num_genes, is_nll)
        pred_mean = pred_cells.mean(0)
        true_mean = X[pert_idx].to(DEVICE).mean(0)

        true_delta = true_mean - raw_ctrl_mean
        pred_delta = pred_mean - raw_ctrl_mean

        da20_list.append(compute_da(pred_delta, true_delta, 20))
        da200_list.append(compute_da(pred_delta, true_delta, 200))
        rho20_list.append(compute_pearson(pred_delta, true_delta, 20))
        cos200_list.append(compute_cosine(pred_delta, true_delta, 200))

    n = len(da20_list)
    da20 = np.mean(da20_list) * 100
    da200 = np.mean(da200_list) * 100
    rho20 = np.mean(rho20_list) * 100
    cos200 = np.mean(cos200_list) * 100

    print(f"\n{'=' * 55}")
    print(f"  {args.dataset.upper()} — {n} OOD perturbations ({n_skipped} skipped)")
    print(f"{'=' * 55}")
    print(f"  DA^DEG  (top-20):   {da20:6.2f}%   (expected: {cfg['expected_da20']:.2f}%)")
    print(f"  DA      (top-200):  {da200:6.2f}%")
    print(f"  rho^DEG (top-20):   {rho20:6.2f}%")
    print(f"  Cos     (top-200):  {cos200:6.2f}%")
    print(f"{'=' * 55}")

    out = ROOT / f"results_{args.dataset}.json"
    with open(out, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "n_perts": n, "n_skipped": n_skipped,
            "DA_DEG": round(da20, 2),
            "DA_200": round(da200, 2),
            "rho_DEG": round(rho20, 2),
            "Cos_200": round(cos200, 2),
        }, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
