#!/usr/bin/env python3
"""Train gene-to-rotation network from GenePT embeddings."""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Parse args
run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/ar_full_loo")
checkpoint = sys.argv[2] if len(sys.argv) > 2 else "OpPert_epoch070_best.pt"
ckpt_path = run_dir / "checkpoints" / checkpoint

print(f"Loading model from {ckpt_path}")
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

# Load model and data
from OpPert.evaluate_full import _load_config, _build_data, _load_model

ds_cfg, dl_cfg = _load_config("crispri_norman")
params_path = run_dir / "params.json"
if params_path.exists():
    _params = json.load(open(params_path))
    if _params.get("num_perts", 0) > 500:
        from omegaconf import OmegaConf
        ds_cfg = OmegaConf.merge(ds_cfg, {"pert_subsample": None})

dm = _build_data(ds_cfg, dl_cfg)
model, meta = _load_model(str(ckpt_path), "tuned", device)
model.eval()

# Get perturbation info
dataset = dm.dataset
pert_names = list(dataset.pert_dict.values())  # idx -> name
print(f"Total perturbations: {len(pert_names)}")

# Load gene embeddings (use full GenePT set, not just pert subset)
embedding_dir = Path("data/gene_embeddings")
emb_path = embedding_dir / "genept_ada.pt"
if not emb_path.exists():
    emb_path = embedding_dir / "pert_gene_embeddings_ada.pt"
raw_emb = torch.load(emb_path, map_location="cpu")
# Normalize keys to uppercase for case-insensitive matching
emb_data = {k.upper(): v for k, v in raw_emb.items()}
emb_dim = next(iter(emb_data.values())).shape[0]
print(f"Embedding dim: {emb_dim}, genes with embeddings: {len(emb_data)} (from {emb_path.name})")

# Get rotation parameters for each perturbation
rotation = model.rotation
if not hasattr(rotation, 'generator_params'):
    print("ERROR: Model doesn't use Cayley rotation. Exiting.")
    sys.exit(1)

all_params = rotation.generator_params.detach().cpu()  # (num_perts, latent_dim * rank * 2)
param_dim = all_params.shape[1]
print(f"Rotation params shape: {all_params.shape}")

# Match perturbations to embeddings
# Need: pert_name -> pert_idx, pert_name -> embedding
train_indices = dataset.indices.get("train", [])
ood_indices = dataset.indices.get("ood", dataset.indices.get("test", []))

# Get train and OOD pert names
obs = dataset.adata.obs
train_perts = set(obs.iloc[train_indices].loc[obs.iloc[train_indices]['control'].astype(int) != 1, 'perturbation'].unique())
ood_perts = set(obs.iloc[ood_indices].loc[obs.iloc[ood_indices]['control'].astype(int) != 1, 'perturbation'].unique())

# Map pert names to indices in the model
pert_name_to_idx = {v: k for k, v in dataset.pert_dict.items()}

# Build train set: (embedding, rotation_params) pairs
train_embs = []
train_targets = []
train_names = []
for pname in sorted(train_perts):
    pname_upper = pname.upper()
    if pname_upper in emb_data and pname in pert_name_to_idx:
        idx = pert_name_to_idx[pname]
        train_embs.append(emb_data[pname_upper])
        train_targets.append(all_params[idx])
        train_names.append(pname)

train_embs = torch.stack(train_embs)     # (N_train, emb_dim)
train_targets = torch.stack(train_targets)  # (N_train, param_dim)
print(f"Train pairs: {len(train_embs)} (perts with embeddings AND rotation params)")

# Build OOD set
ood_embs = []
ood_targets = []
ood_names = []
for pname in sorted(ood_perts):
    pname_upper = pname.upper()
    if pname_upper in emb_data and pname in pert_name_to_idx:
        idx = pert_name_to_idx[pname]
        ood_embs.append(emb_data[pname_upper])
        ood_targets.append(all_params[idx])
        ood_names.append(pname)

if ood_embs:
    ood_embs = torch.stack(ood_embs)
    ood_targets = torch.stack(ood_targets)
    print(f"OOD pairs: {len(ood_embs)} (for evaluation)")
else:
    print("WARNING: No OOD pairs found")
    ood_embs = torch.zeros(0, emb_dim)
    ood_targets = torch.zeros(0, param_dim)

# Normalize targets (important for training stability)
target_mean = train_targets.mean(dim=0, keepdim=True)
target_std = train_targets.std(dim=0, keepdim=True).clamp(min=1e-6)
train_targets_norm = (train_targets - target_mean) / target_std
ood_targets_norm = (ood_targets - target_mean) / target_std if len(ood_targets) > 0 else ood_targets

# Define gene-to-rotation network
class Gene2Rot(nn.Module):
    """Maps gene embeddings to rotation parameters."""
    def __init__(self, emb_dim, out_dim, hidden=512, depth=3, dropout=0.1):
        super().__init__()
        layers = []
        in_d = emb_dim
        for i in range(depth):
            layers.append(nn.Linear(in_d, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            in_d = hidden
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Train
net = Gene2Rot(emb_dim, param_dim, hidden=512, depth=3, dropout=0.1).to(device)
train_embs_d = train_embs.to(device)
train_targets_d = train_targets_norm.to(device)

optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)

best_ood_loss = float('inf')
best_epoch = 0

print(f"\nTraining Gene2Rot: {sum(p.numel() for p in net.parameters())} params")
print(f"Input: {emb_dim}, Output: {param_dim}, Hidden: 512x3")

for epoch in range(2000):
    net.train()
    pred = net(train_embs_d)
    loss = nn.functional.mse_loss(pred, train_targets_d)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0 or epoch == 1999:
        net.eval()
        with torch.no_grad():
            # Train loss
            train_pred = net(train_embs_d)
            train_loss = nn.functional.mse_loss(train_pred, train_targets_d).item()

            # OOD loss
            if len(ood_embs) > 0:
                ood_pred = net(ood_embs.to(device))
                ood_loss = nn.functional.mse_loss(ood_pred, ood_targets_norm.to(device)).item()
            else:
                ood_loss = float('inf')

            # Cosine similarity of predicted vs actual rotation effects
            # Denormalize predictions
            pred_denorm = train_pred * target_std.to(device) + target_mean.to(device)
            cos_sim = nn.functional.cosine_similarity(
                pred_denorm, train_targets.to(device), dim=1
            ).mean().item()

            if len(ood_embs) > 0:
                ood_pred_denorm = ood_pred * target_std.to(device) + target_mean.to(device)
                ood_cos = nn.functional.cosine_similarity(
                    ood_pred_denorm, ood_targets.to(device), dim=1
                ).mean().item()
            else:
                ood_cos = 0.0

            print(f"E{epoch:4d} | train_loss={train_loss:.4f} ood_loss={ood_loss:.4f} | "
                  f"train_cos={cos_sim:.3f} ood_cos={ood_cos:.3f} | lr={scheduler.get_last_lr()[0]:.1e}")

            if ood_loss < best_ood_loss:
                best_ood_loss = ood_loss
                best_epoch = epoch
                torch.save({
                    'net_state': net.state_dict(),
                    'target_mean': target_mean,
                    'target_std': target_std,
                    'epoch': epoch,
                    'ood_loss': ood_loss,
                    'ood_cos': ood_cos,
                }, run_dir / f"gene2rot_best.pt")

print(f"\nBest OOD loss: {best_ood_loss:.4f} at epoch {best_epoch}")

# Now evaluate: use predicted rotations for geodesic-free OOD prediction
print("\n" + "="*60)
print("Evaluating Gene2Rot predictions on OOD perts")
print("="*60)

net.eval()
with torch.no_grad():
    # For each OOD pert, predict rotation params from embedding
    if len(ood_embs) > 0:
        ood_pred = net(ood_embs.to(device))
        ood_pred_denorm = ood_pred * target_std.to(device) + target_mean.to(device)

        # Now use these predicted rotation params for directional accuracy
        n = rotation.latent_dim
        r = rotation.rank

        from OpPert.engine import evaluator

        # Get control cells for z_basal
        ctrl_indices = dataset.indices.get("test", dataset.indices.get("train", []))
        ctrl_mask = dataset.adata.obs.iloc[ctrl_indices]['control'].astype(int).values == 1
        ctrl_idx = np.array(ctrl_indices)[ctrl_mask]

        # Sample control cells
        rng = np.random.RandomState(42)
        sample_ctrl = rng.choice(ctrl_idx, min(500, len(ctrl_idx)), replace=False)

        ctrl_genes = dataset.genes[sample_ctrl].to(device)
        with torch.no_grad():
            z_ctrl = model.encode(ctrl_genes)
        z_basal = z_ctrl.mean(dim=0, keepdim=True)  # (1, latent_dim)

        # For each OOD pert: apply predicted rotation, decode, compare to actual
        ood_idx = np.array(ood_indices)
        correct = 0
        total = 0

        for i, pname in enumerate(ood_names):
            # Get actual treated cells
            pert_mask = obs.iloc[ood_idx]['perturbation'].values == pname
            if pert_mask.sum() < 5:
                continue

            pert_cell_idx = ood_idx[pert_mask]
            actual_genes = dataset.genes[pert_cell_idx].mean(dim=0).to(device)
            ctrl_mean_genes = dataset.genes[sample_ctrl].mean(dim=0).to(device)
            actual_delta = actual_genes - ctrl_mean_genes  # (n_genes,)

            # Predicted rotation
            params = ood_pred_denorm[i]  # (param_dim,)
            factors = params.view(2, n, r)
            U, V = factors[0], factors[1]
            A = U @ V.T - V @ U.T
            I_mat = torch.eye(n, device=device)
            R = torch.linalg.solve(I_mat - A / 2, I_mat + A / 2)

            # Apply rotation
            z_rotated = R @ z_basal.squeeze()  # (latent_dim,)

            # Decode
            with torch.no_grad():
                pred_genes = model.decode(z_rotated.unsqueeze(0))
                if isinstance(pred_genes, tuple):
                    pred_genes = pred_genes[0]
                if pred_genes.dim() == 3:
                    pred_genes = pred_genes[:, :, 0]  # mean only
                ctrl_pred = model.decode(z_basal)
                if isinstance(ctrl_pred, tuple):
                    ctrl_pred = ctrl_pred[0]
                if ctrl_pred.dim() == 3:
                    ctrl_pred = ctrl_pred[:, :, 0]
            pred_delta = pred_genes.squeeze() - ctrl_pred.squeeze()

            # Directional accuracy: fraction of genes where sign matches
            actual_sign = (actual_delta > 0).float()
            pred_sign = (pred_delta > 0).float()
            da = (actual_sign == pred_sign).float().mean().item()

            correct += da
            total += 1

        if total > 0:
            avg_da = correct / total
            print(f"\nGene2Rot DA: {avg_da*100:.1f}% ({total} OOD perts)")
        else:
            print("No OOD perts evaluated")

print(f"\nDone!")
