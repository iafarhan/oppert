"""Base perturbation module. Handles encoding, rotation composition, and decoding."""

from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from oppert.layers import MLP, GeneralizedSigmoid
from oppert.rotation import BlockRotation


# Helpers (module-level)

def _device_default():
    if torch.xpu.is_available():
        return "xpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class _MultiOptimizer:
    """Wraps multiple optimizers (e.g. Muon for 2D + AdamW for 1D) as one."""

    def __init__(self, optimizers):
        self._optimizers = optimizers
        self._schedulers = []

    def zero_grad(self, set_to_none=True):
        for o in self._optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for o in self._optimizers:
            o.step(closure)

    @property
    def param_groups(self):
        groups = []
        for o in self._optimizers:
            groups.extend(o.param_groups)
        return groups

    def state_dict(self):
        return [o.state_dict() for o in self._optimizers]

    def load_state_dict(self, state_dicts):
        for o, sd in zip(self._optimizers, state_dicts):
            o.load_state_dict(sd)


class _MultiScheduler:
    """Wraps multiple LR schedulers as one (mirrors _MultiOptimizer)."""

    def __init__(self, schedulers):
        self._schedulers = schedulers

    def step(self):
        for s in self._schedulers:
            s.step()

    def get_last_lr(self):
        lrs = []
        for s in self._schedulers:
            lrs.extend(s.get_last_lr())
        return lrs


# PerturbationModule ABC

class PerturbationModule(ABC, nn.Module):
    """Abstract base for gene-only, flux-only, and multimodal perturbation models."""

    # Subclasses must set these as class attributes
    modalities: Tuple[str, ...] = ()
    _loss_weights: Dict[str, float] = {}

    num_perts: int
    use_perts_idx: bool

    def __init__(
        self,
        num_genes: int,
        num_fluxes: int,
        num_perts: int,
        num_covariates: Sequence[int],
        device: Optional[str] = None,
        seed: int = 0,
        patience: int = 5,
        doser_type: Optional[str] = "logsigm",
        decoder_activation: str = "linear",
        hparams: Union[str, dict] = "",
        pert_embeddings: Optional[nn.Embedding] = None,
        use_perts_idx: bool = False,
        append_layer_width: Optional[int] = None,
        multi_task: bool = False,
        enable_cpa_mode: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._crispr_base_latents = None
        self._crispr_base_latent = None
        self.device = device or _device_default()

        self.num_genes = num_genes
        self.num_fluxes = num_fluxes
        self.num_perts = num_perts
        self.num_covariates = list(num_covariates) if isinstance(num_covariates, (list, tuple)) else [num_covariates]
        self.seed = seed
        self.patience = patience
        self.best_score = -1e9
        self.patience_trials = 0
        self.use_perts_idx = use_perts_idx
        self.multi_task = multi_task
        self.enable_cpa_mode = enable_cpa_mode
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.hparams = self._set_hparams(seed, hparams)

        # Per-modality loss types (must be set before _build_layers so decoders know output size)
        # Backward compat: "recon_loss" sets both; new keys override individually
        default_loss = self.hparams.get("recon_loss", "nll")
        self.recon_loss_type_genes = self.hparams.get("recon_loss_genes") or default_loss
        self.recon_loss_type_fluxes = self.hparams.get("recon_loss_fluxes") or default_loss

        from oppert.losses import StableGaussianNLL
        self.loss_genes = None if self.recon_loss_type_genes == "mse" else StableGaussianNLL(min_var=1e-3, add_const=True)
        self.loss_fluxes = None if self.recon_loss_type_fluxes == "mse" else StableGaussianNLL(min_var=1e-3, add_const=True)

        # Let the subclass build encoders, decoders, fusion
        self._build_layers(
            append_layer_width=append_layer_width,
            decoder_activation=decoder_activation,
        )

        # Perturbation embeddings, adversaries, covariate embeddings
        E = self.hparams["dim"]
        effective_doser = self.hparams.get("doser_type", doser_type)
        if effective_doser == "none":
            effective_doser = None
        self._build_pert_and_covariate_layers(E, pert_embeddings, effective_doser)

        # Multi-task DEG predictor (optional)
        self.degs_predictor = None
        if self.multi_task:
            from oppert.layers import FocalLoss
            self.degs_predictor = MLP(
                [2 * E] + [2 * E] + [num_genes],
                batch_norm=True,
            )
            self.loss_degs = FocalLoss()

        # Optimizers
        embedding_requires_grad = pert_embeddings is None
        self._build_optimizers(E, embedding_requires_grad)

        self.iteration = 0
        self.history = {"epoch": [], "stats_epoch": []}
        self.init_args = {
            "num_genes": num_genes,
            "num_fluxes": num_fluxes,
            "num_perts": num_perts,
            "num_covariates": num_covariates,
            "seed": seed,
            "patience": patience,
            "doser_type": doser_type,
            "decoder_activation": decoder_activation,
            "hparams": hparams,
            "use_perts_idx": use_perts_idx,
        }

        self._crispr_cache: Dict[Tuple[int, str, float, bool], torch.Tensor] = {}
        self._crispr_base_latent: Optional[torch.Tensor] = None
        self._avg_in_vocab_effect_norm: Optional[float] = None

        self.to(self.device)

    # Abstract methods — subclasses MUST implement

    @abstractmethod
    def _build_layers(self, *, append_layer_width: Optional[int], decoder_activation: str):
        """Create encoder(s), decoder(s), fusion, output scales. Called from __init__."""

    @abstractmethod
    def _encode(self, genes: torch.Tensor, fluxes: torch.Tensor) -> torch.Tensor:
        """Return fused basal latent h of shape (B, E)."""

    @abstractmethod
    def _decode(self, latent: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Decode latent into (g_pred, f_pred).
        Each is [mu, var] concatenated along dim=1, or None if that modality is absent.
        g_pred shape: (B, 2*G), f_pred shape: (B, 2*F).
        """

    @abstractmethod
    def _autoencoder_params(self) -> List[nn.Parameter]:
        """Return list of parameters for the autoencoder optimizer (encoders, decoders, fusion, scales)."""

    @property
    @abstractmethod
    def _perturbation_decoder(self) -> nn.Module:
        """The decoder used for CRISPR gradient direction computation."""

    @property
    @abstractmethod
    def _perturbation_output_scale(self) -> nn.Parameter:
        """The output scale parameter for the perturbation decoder."""

    @property
    @abstractmethod
    def _perturbation_output_dim(self) -> int:
        """Number of output features for the perturbation decoder (G or F)."""

    # Shared perturbation + covariate layer construction

    def _build_pert_and_covariate_layers(self, E: int, pert_embeddings, doser_type):
        self.pert_composition = self.hparams.get("pert_composition", "additive")

        if self.num_perts > 0:
            self.adversary_perts = MLP(
                [E] + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"] + [self.num_perts]
            )

            self.loss_adversary_perts = nn.CrossEntropyLoss()

            if self.pert_composition == "rotation":
                block_size = self.hparams.get("rotation_block_size", 4)
                self.rotation = BlockRotation(self.num_perts, E, block_size=block_size)
                self.doser_type = doser_type
                if doser_type is not None:
                    assert doser_type in ("sigm", "logsigm")
                    self.dosers = GeneralizedSigmoid(self.num_perts, self.device, nonlin=doser_type)
                else:
                    self.dosers = None  # raw dose passthrough (dose=1 → scale=1)
                self.pert_embeddings = None
                self.pert_encoder = None
                # Gene-indexed rotation for synthetic training (all genes, not just perts)
                if self.hparams.get("synthetic_rotation", False):
                    self.gene_rotation = BlockRotation(self.num_genes, E, block_size=block_size)
                else:
                    self.gene_rotation = None
            else:
                self.rotation = None
                self.gene_rotation = None
                if pert_embeddings is None:
                    self.pert_embeddings = nn.Embedding(self.num_perts, E)
                else:
                    self.pert_embeddings = pert_embeddings

                if self.enable_cpa_mode:
                    self.pert_encoder = None
                else:
                    self.pert_encoder = MLP(
                        [self.pert_embeddings.embedding_dim]
                        + [self.hparams["embedding_encoder_width"]] * self.hparams["embedding_encoder_depth"]
                        + [E],
                        last_layer_act="linear",
                    )

                assert doser_type in ("mlp", "sigm", "logsigm", "amortized", None)
                self.doser_type = doser_type
                if doser_type == "mlp":
                    self.dosers = nn.ModuleList([
                        MLP([1] + [self.hparams["dosers_width"]] * self.hparams["dosers_depth"] + [1], batch_norm=False)
                        for _ in range(self.num_perts)
                    ])
                elif doser_type == "amortized":
                    assert self.use_perts_idx
                    self.dosers = MLP(
                        [self.pert_embeddings.embedding_dim + 1]
                        + [self.hparams["dosers_width"]] * self.hparams["dosers_depth"]
                        + [1]
                    )
                else:
                    self.dosers = GeneralizedSigmoid(self.num_perts, self.device, nonlin=doser_type)
        else:
            self.pert_composition = "additive"
            self.rotation = None
            self.gene_rotation = None
            self.doser_type = doser_type

        if self.num_covariates and self.num_covariates[0] > 0:
            self.adversary_covariates = nn.ModuleList()
            self.loss_adversary_covariates = nn.ModuleList()
            self.covariates_embeddings = nn.ModuleList()
            for num_cov in self.num_covariates:
                self.covariates_embeddings.append(nn.Embedding(num_cov, E))
                if num_cov <= 1:
                    # Single-class covariate: adversary is degenerate, skip it
                    continue
                self.adversary_covariates.append(
                    MLP([E] + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"] + [num_cov])
                )
                self.loss_adversary_covariates.append(nn.CrossEntropyLoss())
        else:
            self.adversary_covariates = nn.ModuleList()
            self.loss_adversary_covariates = nn.ModuleList()
            self.covariates_embeddings = nn.ModuleList()

    # Shared optimizer construction

    def _make_optimizer(self, params: List[nn.Parameter], lr: float, wd: float) -> torch.optim.Optimizer:
        opt_type = self.hparams.get("optimizer", "adam")
        if opt_type == "muon":
            from torch.optim import Muon
            params_2d = [p for p in params if p.dim() == 2]
            params_other = [p for p in params if p.dim() != 2]
            opts = []
            if params_2d:
                opts.append(Muon(params_2d, lr=lr, weight_decay=wd, momentum=0.95, nesterov=True))
            if params_other:
                adamw_lr = self.hparams.get("muon_adamw_lr") or lr * 0.05
                opts.append(torch.optim.AdamW(params_other, lr=adamw_lr, weight_decay=wd))
            return _MultiOptimizer(opts)
        elif opt_type == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        else:
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)

    def _build_optimizers(self, E: int, embedding_requires_grad: bool):
        params_main = list(self._autoencoder_params())

        if self.num_perts > 0 and self.pert_composition == "rotation":
            params_main += list(self.rotation.parameters())
            if getattr(self, "gene_rotation", None) is not None:
                params_main += list(self.gene_rotation.parameters())
        else:
            if not self.enable_cpa_mode and self.num_perts > 0 and self.pert_encoder is not None:
                params_main += list(self.pert_encoder.parameters())
            if self.num_perts > 0 and embedding_requires_grad and self.pert_embeddings is not None:
                params_main += list(self.pert_embeddings.parameters())
        if self.multi_task and self.degs_predictor is not None:
            params_main += list(self.degs_predictor.parameters())
        for emb in self.covariates_embeddings:
            params_main += list(emb.parameters())

        self.optimizer_autoencoder = self._make_optimizer(
            params_main, lr=self.hparams["autoencoder_lr"], wd=self.hparams["autoencoder_wd"],
        )

        params_adv: List[nn.Parameter] = []
        if self.num_perts > 0:
            params_adv += list(self.adversary_perts.parameters())
        for adv in self.adversary_covariates:
            params_adv += list(adv.parameters())

        # Adversaries always use Adam (Muon not beneficial for small classifiers)
        self.optimizer_adversaries = torch.optim.Adam(
            params_adv,
            lr=self.hparams["adversary_lr"],
            weight_decay=self.hparams["adversary_wd"],
        )

        if self.num_perts > 0 and self.dosers is not None:
            self.optimizer_dosers = torch.optim.Adam(
                self.dosers.parameters(),
                lr=self.hparams["dosers_lr"],
                weight_decay=self.hparams["dosers_wd"],
            )
        else:
            self.optimizer_dosers = None

        sched = self.hparams.get("scheduler", "step")
        self.scheduler_autoencoder = self._make_scheduler(self.optimizer_autoencoder, sched)
        self.scheduler_adversary = self._make_scheduler(self.optimizer_adversaries, sched)
        if self.num_perts > 0 and self.optimizer_dosers is not None:
            self.scheduler_dosers = self._make_scheduler(self.optimizer_dosers, sched)
        else:
            self.scheduler_dosers = None

    def _make_scheduler(self, optimizer, sched_type: str):
        """Create LR scheduler(s). Handles _MultiOptimizer transparently."""
        if isinstance(optimizer, _MultiOptimizer):
            scheds = [self._make_scheduler(o, sched_type) for o in optimizer._optimizers]
            return _MultiScheduler(scheds)
        if sched_type == "cosine":
            T_max = self.hparams.get("cosine_T_max", 200)
            eta_min = self.hparams.get("cosine_eta_min", 1e-6)
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        else:
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.hparams["step_size_lr"], gamma=0.5
            )

    # Hparams

    @staticmethod
    def _set_hparams(seed: int, hparams: Union[str, dict]) -> dict:
        default = seed == 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        hp = {
            "dim": 256 if default else int(np.random.choice([128, 256, 512])),
            "dosers_width": 64 if default else int(np.random.choice([32, 64, 128])),
            "dosers_depth": 2 if default else int(np.random.choice([1, 2, 3])),
            "dosers_lr": 1e-3 if default else float(10 ** np.random.uniform(-4, -2)),
            "dosers_wd": 1e-7 if default else float(10 ** np.random.uniform(-8, -5)),
            "autoencoder_width": 512 if default else int(np.random.choice([256, 512, 1024])),
            "autoencoder_depth": 4 if default else int(np.random.choice([3, 4, 5])),
            "adversary_width": 128 if default else int(np.random.choice([64, 128, 256])),
            "adversary_depth": 3 if default else int(np.random.choice([2, 3, 4])),
            "reg_adversary": 5.0 if default else float(10 ** np.random.uniform(-2, 2)),
            "reg_adversary_cov": 5.0 if default else float(10 ** np.random.uniform(-2, 2)),
            "penalty_adversary": 3.0 if default else float(10 ** np.random.uniform(-2, 1)),
            "autoencoder_lr": 1e-3 if default else float(10 ** np.random.uniform(-4, -2)),
            "adversary_lr": 3e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6 if default else float(10 ** np.random.uniform(-8, -4)),
            "adversary_wd": 1e-4 if default else float(10 ** np.random.uniform(-6, -3)),
            "adversary_steps": 3 if default else int(np.random.choice([1, 2, 3, 4, 5])),
            "batch_size": 128 if default else int(np.random.choice([64, 128, 256, 512])),
            "step_size_lr": 45 if default else int(np.random.choice([15, 25, 45])),
            "embedding_encoder_width": 512,
            "embedding_encoder_depth": 0,
            "reg_multi_task": 1.0,
            "pert_composition": "additive",
            "rotation_block_size": 4,
            "bch_order": 2,
            "recon_loss": "nll",
            "recon_loss_genes": None,   # None → falls back to recon_loss
            "recon_loss_fluxes": None,  # None → falls back to recon_loss
            "optimizer": "adam",        # "adam", "adamw", or "muon"
            "muon_adamw_lr": None,      # None → 5% of autoencoder_lr
            "reg_cov_latent": 0.0,      # latent covariance reg (off-diag penalty)
            "reg_jacobian_rank": 0.0,   # decoder Jacobian effective rank reg
            "jacobian_rank_every": 50,  # compute Jacobian rank reg every N steps (1=every step)
            "reg_curvature": 0.0,       # Riemannian sectional curvature of decoder manifold
            "reg_contrastive_pert": 0.0, # contrastive pert loss: decoded diff ≈ actual diff
            "phase2_lr": None,          # None → use autoencoder_lr
            "phase2_wd": None,          # None → use autoencoder_wd
            "phase2_reg_rotation_recon": 1.0,  # rotation recon weight in phase 2
            "phase3_lr": None,          # None → 10% of autoencoder_lr
            "phase3_wd": None,          # None → use autoencoder_wd
        }
        if hparams != "":
            if isinstance(hparams, str):
                hp.update(json.loads(hparams))
            else:
                hp.update(hparams)
        return hp

    # Keep backward-compat alias used by FateModule.defaults()
    def set_hparams_(self, seed, hparams):
        return self._set_hparams(seed, hparams)

    @classmethod
    def defaults(cls):
        return cls._set_hparams(0, "")

    # predict() — shared control flow

    def predict(
        self,
        genes,
        fluxes,
        perts: Optional[torch.Tensor] = None,
        perts_idx: Optional[torch.Tensor] = None,
        dosages: Optional[torch.Tensor] = None,
        covariates: Optional[List[torch.Tensor]] = None,
        crispr_targets: Optional[torch.Tensor] = None,
        crispr_mode: Optional[str] = None,
        use_cached_crispr_base: bool = False,
        calibrate_crispr_to_known: bool = True,
        crispr_strength: float = 1.0,
        crispr_per_cell_dir: bool = False,
        crispr_cone_cos_min: float = 0.5,
        return_latents: bool = False,
    ):
        # Caller (trainer / evaluator) is responsible for moving data to self.device
        h = self._encode(genes=genes, fluxes=fluxes)

        if len(self.covariates_embeddings) > 0 and covariates is not None:
            cov_vecs = []
            for k, emb in enumerate(self.covariates_embeddings):
                cov_idx = covariates[k].argmax(1)
                cov_vecs.append(emb(cov_idx))
            z_c = torch.stack(cov_vecs, 1).sum(1)
        else:
            z_c = torch.zeros_like(h)

        # CRISPR perturbation branch
        if crispr_targets is not None:
            return self._predict_crispr(
                h=h, z_c=z_c,
                crispr_targets=crispr_targets,
                crispr_mode=crispr_mode,
                use_cached_crispr_base=use_cached_crispr_base,
                calibrate_crispr_to_known=calibrate_crispr_to_known,
                crispr_strength=crispr_strength,
                crispr_per_cell_dir=crispr_per_cell_dir,
                crispr_cone_cos_min=crispr_cone_cos_min,
                return_latents=return_latents,
                genes=genes, fluxes=fluxes,
            )

        # Perturbation composition branch
        if self.num_perts > 0 and self.pert_composition == "rotation":
            z_rotated = self.rotation(h + z_c, perts, dosers=self.dosers)
            latent_treated = z_rotated
            z_d = latent_treated - (h + z_c)  # implicit delta for logging/adversary
        elif self.num_perts > 0:
            z_d = self.compute_pert_embeddings_(perts=perts, perts_idx=perts_idx, dosages=dosages)
            latent_treated = h + z_d + z_c
        else:
            z_d = torch.zeros_like(h)
            latent_treated = h + z_d + z_c

        g_pred, f_pred = self._decode(latent_treated)
        cell_pert_embedding = torch.cat([z_c, z_d], 1)

        if return_latents:
            # For compatibility, fill z_g/z_f with zeros for single-modality models
            z_g = getattr(self, '_last_z_g', torch.zeros_like(h))
            z_f = getattr(self, '_last_z_f', torch.zeros_like(h))
            return g_pred, f_pred, cell_pert_embedding, (z_g, z_f, h, z_d, z_c, latent_treated)
        return g_pred, f_pred, cell_pert_embedding

    def _predict_crispr(
        self, *, h, z_c, crispr_targets, crispr_mode, use_cached_crispr_base,
        calibrate_crispr_to_known, crispr_strength, crispr_per_cell_dir,
        crispr_cone_cos_min, return_latents, genes, fluxes,
    ):
        t = crispr_targets.view(-1)
        gene_index = int(t[0].item())

        base = (
            self._crispr_base_latents.to(self.device)
            if (use_cached_crispr_base and self._crispr_base_latents is not None)
            else (h + z_c)
        )

        if crispr_per_cell_dir:
            v_raw = self._oov_grad_direction_batch(base, gene_index, mode=(crispr_mode or "i"))
            base_for_dir = (
                self._crispr_base_latent.to(self.device)
                if (use_cached_crispr_base and self._crispr_base_latent is not None)
                else h.mean(0)
            )
            v_pop = self._oov_grad_direction(base_for_dir, gene_index, mode=(crispr_mode or "i"))
            v = self._project_to_cone(v_raw, v_pop, cos_min=crispr_cone_cos_min)
            if calibrate_crispr_to_known:
                alpha = self._calibrate_alpha_quantile(
                    base, gene_index, v, mode=(crispr_mode or "i"),
                    target_abs_lfc=float(abs(crispr_strength)), q=0.7,
                )
            else:
                alpha = float(crispr_strength)
        else:
            base_for_dir = (
                self._crispr_base_latent.to(self.device)
                if (use_cached_crispr_base and self._crispr_base_latent is not None)
                else h.mean(0)
            )
            v_pop = self._oov_grad_direction(base_for_dir, gene_index, mode=(crispr_mode or "i"))
            if calibrate_crispr_to_known:
                alpha = self._calibrate_alpha_to_lfc(
                    h, z_c, gene_index, v_pop,
                    target_abs_lfc=float(abs(crispr_strength)),
                    mode=(crispr_mode or "i"),
                )
            else:
                alpha = float(crispr_strength)
            v = v_pop.unsqueeze(0).expand_as(base)

        latent_treated = base + alpha * v
        g_pred, f_pred = self._decode(latent_treated)
        cell_pert_embedding = torch.cat([z_c, torch.zeros_like(z_c)], 1)

        if return_latents:
            z_g = getattr(self, '_last_z_g', torch.zeros_like(h))
            z_f = getattr(self, '_last_z_f', torch.zeros_like(h))
            return g_pred, f_pred, cell_pert_embedding, (z_g, z_f, h, torch.zeros_like(h), z_c, latent_treated)
        return g_pred, f_pred, cell_pert_embedding

    # update() — shared training step

    def update(
        self,
        genes: torch.Tensor,
        fluxes: torch.Tensor,
        perts: Optional[torch.Tensor] = None,
        perts_idx: Optional[torch.Tensor] = None,
        dosages: Optional[torch.Tensor] = None,
        degs: Optional[torch.Tensor] = None,
        covariates: Optional[List[torch.Tensor]] = None,
        clip_grad_norm: Optional[float] = None,
    ) -> dict:
        assert (perts is not None) or (perts_idx is not None and dosages is not None)

        g_pred, f_pred, cell_pert_embedding, (z_g, z_f, h, z_d, z_c, latent_treated) = self.predict(
            genes=genes, fluxes=fluxes,
            perts=perts, perts_idx=perts_idx, dosages=dosages,
            covariates=covariates, return_latents=True,
        )

        # Compute reconstruction losses based on available modalities
        reconstruction_loss = torch.tensor(0.0, device=self.device)
        stats = {}

        # In phase 2, skip main recon loss — only rotation_recon drives gradients
        _skip_main_recon = getattr(self, "_phase2_active", False)

        if g_pred is not None:
            if _skip_main_recon:
                with torch.no_grad():
                    mu_g = self._extract_mean(g_pred, "genes")
                    stats["loss_genes"] = 0.0
                    stats["mse_genes"] = F.mse_loss(mu_g, genes).item()
            else:
                w_g = self._loss_weights.get("genes", 0.0)
                if self.recon_loss_type_genes == "mse":
                    mu_g = g_pred
                    loss_genes = w_g * F.mse_loss(mu_g, genes)
                else:
                    G = g_pred.size(1) // 2
                    mu_g, var_g = g_pred[:, :G], g_pred[:, G:]
                    deg_up = self.hparams.get("deg_upweight", 1.0)
                    if deg_up != 1.0 and degs is not None:
                        gene_weights = 1.0 + (deg_up - 1.0) * degs
                        loss_genes = w_g * self.loss_genes(mu_g, genes, var_g, weights=gene_weights)
                    else:
                        loss_genes = w_g * self.loss_genes(mu_g, genes, var_g)
                reconstruction_loss = reconstruction_loss + loss_genes
                with torch.no_grad():
                    stats["loss_genes"] = loss_genes.item()
                    stats["mse_genes"] = F.mse_loss(mu_g, genes).item()

        if f_pred is not None:
            if _skip_main_recon:
                with torch.no_grad():
                    mu_f = self._extract_mean(f_pred, "fluxes")
                    stats["loss_fluxes"] = 0.0
                    stats["mse_fluxes"] = F.mse_loss(mu_f, fluxes).item()
            else:
                w_f = self._loss_weights.get("fluxes", 0.0)
                if self.recon_loss_type_fluxes == "mse":
                    mu_f = f_pred
                    loss_flux = w_f * F.mse_loss(mu_f, fluxes)
                else:
                    Fdim = f_pred.size(1) // 2
                    mu_f, var_f = f_pred[:, :Fdim], f_pred[:, Fdim:]
                    loss_flux = w_f * self.loss_fluxes(mu_f, fluxes, var_f)
                reconstruction_loss = reconstruction_loss + loss_flux
                with torch.no_grad():
                    stats["loss_fluxes"] = loss_flux.item()
                    stats["mse_fluxes"] = F.mse_loss(mu_f, fluxes).item()

        stats["loss_reconstruction"] = reconstruction_loss.item()

        # Manifold regularizers
        reg_cov_w = self.hparams.get("reg_cov_latent", 0.0)
        if reg_cov_w > 0:
            cov_loss = self._compute_cov_reg(h)
            reconstruction_loss = reconstruction_loss + reg_cov_w * cov_loss
            stats["loss_cov_reg"] = cov_loss.item()

        reg_lerank_w = self.hparams.get("reg_latent_erank", 0.0)
        if reg_lerank_w > 0:
            lerank_loss, lerank_val = self._compute_latent_erank_reg(h)
            reconstruction_loss = reconstruction_loss + reg_lerank_w * lerank_loss
            stats["loss_latent_erank"] = lerank_loss.item()
            stats["latent_erank"] = lerank_val

        reg_jrank_w = self.hparams.get("reg_jacobian_rank", 0.0)
        jrank_every = int(self.hparams.get("jacobian_rank_every", 50))
        if reg_jrank_w > 0 and self.iteration % jrank_every == 0:
            jrank_loss, erank, cond = self._compute_jacobian_rank_reg(h)
            reconstruction_loss = reconstruction_loss + reg_jrank_w * jrank_loss
            stats["loss_jrank_reg"] = jrank_loss.item()
            stats["erank"] = erank
            stats["cond_num"] = cond

        reg_curv_w = self.hparams.get("reg_curvature", 0.0)
        if reg_curv_w > 0:
            curv_loss, curv_val = self._compute_curvature_reg(h)
            reconstruction_loss = reconstruction_loss + reg_curv_w * curv_loss
            stats["loss_curvature"] = curv_loss.item()
            stats["curvature"] = curv_val

        reg_contrast_w = self.hparams.get("reg_contrastive_pert", 0.0)
        if reg_contrast_w > 0:
            contrast_loss, contrast_val = self._compute_contrastive_pert_reg(h, z_c, perts, genes)
            reconstruction_loss = reconstruction_loss + reg_contrast_w * contrast_loss
            stats["loss_contrast"] = contrast_val

        reg_rot_recon_w = self.hparams.get("reg_rotation_recon", 0.0)
        if reg_rot_recon_w > 0:
            rot_recon_loss, rot_recon_val = self._compute_rotation_recon_reg(h, z_c, perts, genes, degs)
            reconstruction_loss = reconstruction_loss + reg_rot_recon_w * rot_recon_loss
            stats["loss_rot_recon"] = rot_recon_val

        reg_min_rot_w = self.hparams.get("reg_min_rotation", 0.0)
        if reg_min_rot_w > 0:
            min_rot_loss, min_rot_val = self._compute_min_rotation_reg(perts, genes)
            reconstruction_loss = reconstruction_loss + reg_min_rot_w * min_rot_loss
            stats["loss_min_rot"] = min_rot_val

        reg_div_w = self.hparams.get("reg_contrastive_div", 0.0)
        if reg_div_w > 0 and self.pert_composition == "rotation":
            div_loss, div_val = self._compute_contrastive_div_reg(perts)
            reconstruction_loss = reconstruction_loss + reg_div_w * div_loss
            stats["loss_div"] = div_val

        reg_synth_rot_w = self.hparams.get("reg_synthetic_rotation", 0.0)
        if reg_synth_rot_w > 0 and self.gene_rotation is not None:
            synth_loss, synth_val = self._compute_synthetic_rotation_reg(h, z_c, perts, genes, degs)
            reconstruction_loss = reconstruction_loss + reg_synth_rot_w * synth_loss
            stats["loss_synth_rot"] = synth_val

        reg_janchor_w = self.hparams.get("reg_jacobian_anchor", 0.0)
        if reg_janchor_w > 0 and hasattr(self, "_jacobian_anchor_dirs"):
            anchor_every = int(self.hparams.get("jacobian_rank_every", 50))
            if self.iteration % anchor_every == 0:
                anchor_loss, anchor_cos = self._compute_jacobian_anchor_reg(h)
                reconstruction_loss = reconstruction_loss + reg_janchor_w * anchor_loss
                stats["loss_janchor"] = anchor_loss.item()
                stats["janchor_cos"] = anchor_cos

        reg_jconsist_w = self.hparams.get("reg_jacobian_consist", 0.0)
        if reg_jconsist_w > 0:
            jc_loss, jc_val = self._compute_jacobian_consist_reg(h, z_c, perts)
            reconstruction_loss = reconstruction_loss + reg_jconsist_w * jc_loss
            stats["loss_jconsist"] = jc_val

        # Multi-task
        multi_task_loss = torch.tensor(0.0, device=self.device)
        if self.multi_task and self.degs_predictor is not None and degs is not None:
            degs_pred = self.degs_predictor(cell_pert_embedding)
            multi_task_loss = self.loss_degs(degs_pred, degs)
        stats["loss_multi_task"] = multi_task_loss.item()

        # --- Adversarial training ---
        adv_mode = self.hparams.get("adversary_mode", "grl")
        lam = getattr(self, "lambda_adv", self.hparams.get("reg_adversary", 0.0))
        lam_cov = getattr(self, "lambda_adv_cov", self.hparams.get("reg_adversary_cov", 0.0))

        # Derive pert target index (used by both modes)
        pert_idx = None
        if self.num_perts > 0:
            if self.use_perts_idx:
                pert_idx = perts_idx
            else:
                pert_idx = perts.gt(0).float().argmax(dim=1)

        adv_perts_loss = torch.tensor(0.0, device=self.device)
        adv_covs_loss = torch.tensor(0.0, device=self.device)

        if adv_mode == "alternating":
            adv_steps = self.hparams.get("adversary_steps", 3)
            pen_adv = self.hparams.get("penalty_adversary", 3.0)

            if self.iteration % adv_steps != 0:
                # --- ADVERSARY STEP: train adversary on detached latent ---
                self.optimizer_adversaries.zero_grad()
                h_det = h.detach().requires_grad_(True)

                if self.num_perts > 0:
                    pred_perts = self.adversary_perts(h_det)
                    adv_perts_loss = self.loss_adversary_perts(pred_perts, pert_idx)
                    # Gradient penalty: ||dL/dz||^2
                    penalty = torch.autograd.grad(
                        pred_perts.sum(), h_det, create_graph=True, retain_graph=True
                    )[0].pow(2).mean()
                    total_adv = adv_perts_loss + pen_adv * penalty
                else:
                    total_adv = torch.tensor(0.0, device=self.device)

                # Covariate adversary losses + penalties
                if len(self.adversary_covariates) > 0 and covariates is not None:
                    h_det_cov = h.detach().requires_grad_(True)
                    for i, adv in enumerate(self.adversary_covariates):
                        pred = adv(h_det_cov)
                        cov_loss = self.loss_adversary_covariates[i](pred, covariates[i].argmax(1))
                        cov_pen = torch.autograd.grad(
                            pred.sum(), h_det_cov, create_graph=True, retain_graph=True
                        )[0].pow(2).mean()
                        adv_covs_loss = adv_covs_loss + cov_loss + pen_adv * cov_pen

                (total_adv + adv_covs_loss).backward()

                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.optimizer_adversaries.param_groups[0]["params"], clip_grad_norm)

                self.optimizer_adversaries.step()
            else:
                # --- AE STEP: recon_loss MINUS lam*adv_loss (confuse encoder) ---
                self.optimizer_autoencoder.zero_grad()
                if self.optimizer_dosers is not None:
                    self.optimizer_dosers.zero_grad()

                if self.num_perts > 0:
                    pred_perts = self.adversary_perts(h)
                    adv_perts_loss = self.loss_adversary_perts(pred_perts, pert_idx)

                if len(self.adversary_covariates) > 0 and covariates is not None:
                    for i, adv in enumerate(self.adversary_covariates):
                        pred = adv(h)
                        adv_covs_loss = adv_covs_loss + self.loss_adversary_covariates[i](
                            pred, covariates[i].argmax(1))

                total_loss = (
                    reconstruction_loss
                    - lam * adv_perts_loss
                    - lam_cov * adv_covs_loss
                    + self.hparams.get("reg_multi_task", 0.0) * multi_task_loss
                )
                total_loss.backward()

                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.optimizer_autoencoder.param_groups[0]["params"], clip_grad_norm)
                    if self.optimizer_dosers is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.optimizer_dosers.param_groups[0]["params"], clip_grad_norm)

                self.optimizer_autoencoder.step()
                if self.optimizer_dosers is not None:
                    self.optimizer_dosers.step()

        else:
            # --- GRL mode (default, unchanged) ---
            from oppert.layers import gradient_reversal

            h_rev = gradient_reversal(h, lam)

            if self.num_perts > 0:
                pred_perts = self.adversary_perts(h_rev)
                adv_perts_loss = self.loss_adversary_perts(pred_perts, pert_idx)

            if len(self.adversary_covariates) > 0 and covariates is not None:
                h_rev_cov = gradient_reversal(h, lam_cov)
                for i, adv in enumerate(self.adversary_covariates):
                    pred = adv(h_rev_cov)
                    adv_covs_loss = adv_covs_loss + self.loss_adversary_covariates[i](
                        pred, covariates[i].argmax(1)
                    )

            self.optimizer_autoencoder.zero_grad()
            self.optimizer_adversaries.zero_grad()
            if self.optimizer_dosers is not None:
                self.optimizer_dosers.zero_grad()

            total_loss = (
                reconstruction_loss
                + adv_perts_loss
                + adv_covs_loss
                + self.hparams.get("reg_multi_task", 0.0) * multi_task_loss
            )
            total_loss.backward()

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.optimizer_autoencoder.param_groups[0]["params"], clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.optimizer_adversaries.param_groups[0]["params"], clip_grad_norm)
                if self.optimizer_dosers is not None:
                    torch.nn.utils.clip_grad_norm_(self.optimizer_dosers.param_groups[0]["params"], clip_grad_norm)

            self.optimizer_autoencoder.step()
            self.optimizer_adversaries.step()
            if self.optimizer_dosers is not None:
                self.optimizer_dosers.step()

        self.iteration += 1

        stats.update({
            "loss_adv_perts": adv_perts_loss.item(),
            "loss_adv_covariates": adv_covs_loss.item(),
            "lambda_adv": lam,
        })
        return stats

    # Perturbation embedding computation (shared)

    def _raw_pert_effect_matrix(self) -> torch.Tensor:
        base = self.pert_embeddings.weight
        if self.enable_cpa_mode:
            return base
        else:
            return self.pert_encoder(base)

    def compute_pert_embeddings_(
        self,
        perts: Optional[torch.Tensor] = None,
        perts_idx: Optional[torch.Tensor] = None,
        dosages: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (perts is not None) or (perts_idx is not None and dosages is not None)
        all_effects = self._raw_pert_effect_matrix()

        if perts is not None:
            if self.doser_type == "mlp":
                scaled_rows = []
                for d in range(self.num_perts):
                    dose_d = perts[:, d].unsqueeze(1)
                    gate = (dose_d > 0).float()
                    scaled_rows.append(gate * self.dosers[d](dose_d).sigmoid())
                scaled = torch.cat(scaled_rows, dim=1)
            elif self.doser_type == "amortized":
                scaled = perts
            else:
                scaled = self.dosers(perts)
            return scaled @ all_effects

        if self.doser_type == "mlp":
            scaled = []
            for idx, dose in zip(perts_idx, dosages):
                scaled.append(self.dosers[idx.item()](dose.view(1, 1)).sigmoid().squeeze(0))
            scaled = torch.stack(scaled, dim=0).squeeze(-1)
        elif self.doser_type == "amortized":
            emb_batch = all_effects[perts_idx]
            cat = torch.cat([emb_batch, dosages.unsqueeze(1)], dim=1)
            scaled = self.dosers(cat).squeeze(-1)
        else:
            scaled = self.dosers(dosages, perts_idx)

        effects = all_effects[perts_idx]
        return effects * scaled.unsqueeze(1)

    # Early stopping + schedulers

    def early_stopping(self, score: Optional[float]) -> bool:
        if score is None:
            logging.warning("Early stopping score was None!")
        elif score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1
        return self.patience_trials > self.patience

    def step_schedulers(self):
        self.scheduler_autoencoder.step()
        self.scheduler_adversary.step()
        if self.scheduler_dosers is not None:
            self.scheduler_dosers.step()

    # Latent access (for disentanglement metrics)

    @torch.no_grad()
    def get_latent(self, genes: torch.Tensor, fluxes: torch.Tensor,
                   covariates: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Return basal latent h + z_c for a batch."""
        h = self._encode(genes=genes, fluxes=fluxes)
        if len(self.covariates_embeddings) > 0 and covariates is not None:
            cov_vecs = []
            for k, emb in enumerate(self.covariates_embeddings):
                cov_idx = covariates[k].argmax(1)
                cov_vecs.append(emb(cov_idx))
            z_c = torch.stack(cov_vecs, 1).sum(1)
        else:
            z_c = torch.zeros_like(h)
        return h + z_c

    # Manifold regularizers

    def _compute_cov_reg(self, h: torch.Tensor) -> torch.Tensor:
        """Penalize off-diagonal correlations in latent space (VICReg-style)."""
        h_centered = h - h.mean(dim=0)
        cov = (h_centered.T @ h_centered) / max(h.size(0) - 1, 1)
        off_diag = cov - torch.diag(cov.diag())
        return off_diag.pow(2).sum() / h.size(1)

    def _compute_latent_erank_reg(self, h: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Push latent effective rank up. Returns (loss, erank)."""
        h_centered = h - h.mean(dim=0)
        sv = torch.linalg.svdvals(h_centered)
        p = sv / (sv.sum() + 1e-12)
        log_p = torch.log(p + 1e-12)
        erank = torch.exp(-(p * log_p).sum())
        max_rank = float(min(h.size(0), h.size(1)))
        loss = -torch.log(erank / max_rank + 1e-12)
        return loss, float(erank.item())

    def _compute_jacobian_rank_reg(self, h: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Penalize low effective rank of decoder Jacobian at batch centroid.

        Returns (loss, effective_rank, condition_number).
        """
        z = h.mean(dim=0).detach().requires_grad_(True)
        decoder = self._perturbation_decoder
        scale = self._perturbation_output_scale
        D = self._perturbation_output_dim

        # Compute full Jacobian via row-by-row backward (cheaper than jacrev for single sample)
        out = decoder(z.unsqueeze(0))
        if self.recon_loss_type_genes != "mse":
            out = out[:, :D]
        mu = scale * out.squeeze(0)  # (G,)

        # Use random projection to reduce cost: project G → k
        k = min(64, mu.size(0))
        proj = torch.randn(mu.size(0), k, device=mu.device)
        proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-12)
        mu_proj = mu @ proj  # (k,)

        # Jacobian of projected output: (k, E)
        J_rows = []
        for i in range(k):
            g = torch.autograd.grad(mu_proj[i], z, retain_graph=True, create_graph=True)[0]
            J_rows.append(g)
        J = torch.stack(J_rows, dim=0)  # (k, E)

        sv = torch.linalg.svdvals(J)
        p = sv / (sv.sum() + 1e-12)
        log_p = torch.log(p + 1e-12)
        erank = torch.exp(-(p * log_p).sum())
        max_rank = float(min(k, h.size(1)))
        loss = -torch.log(erank / max_rank + 1e-12)
        return loss, float(erank.item()), float((sv[0] / (sv[-1] + 1e-12)).item())

    def _compute_contrastive_pert_reg(
        self, h: torch.Tensor, z_c: torch.Tensor,
        perts: torch.Tensor, genes: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Contrastive rotation loss: decoded difference between two perturbations
        applied to the same basal point should match their actual expression difference.

        Forces the decoder to be nonlinear — a linear decoder maps all rotation
        differences through the same linear map, unable to match pert-specific patterns.
        """
        if perts is None:
            return torch.tensor(0.0, device=self.device), 0.0

        pert_idx = perts.gt(0).float().argmax(dim=1)
        unique_perts = pert_idx.unique()
        if len(unique_perts) < 2:
            return torch.tensor(0.0, device=self.device), 0.0

        z0 = (h + z_c).mean(0).detach()
        decoder = self._perturbation_decoder
        scale = self._perturbation_output_scale
        D = self._perturbation_output_dim

        # Sample up to K random pairs
        K = min(8, len(unique_perts) // 2)
        perm = torch.randperm(len(unique_perts), device=h.device)

        loss = torch.tensor(0.0, device=h.device)
        n_valid = 0

        for i in range(K):
            p_a = unique_perts[perm[2 * i]]
            p_b = unique_perts[perm[2 * i + 1]]
            mask_a = pert_idx == p_a
            mask_b = pert_idx == p_b
            if mask_a.sum() < 2 or mask_b.sum() < 2:
                continue

            actual_diff = (genes[mask_a].mean(0) - genes[mask_b].mean(0)).detach()

            ohe_a = torch.zeros(1, self.num_perts, device=h.device)
            ohe_a[0, p_a] = 1.0
            ohe_b = torch.zeros(1, self.num_perts, device=h.device)
            ohe_b[0, p_b] = 1.0

            if self.pert_composition == "rotation":
                z_a = self.rotation(z0.unsqueeze(0), ohe_a, dosers=self.dosers)
                z_b = self.rotation(z0.unsqueeze(0), ohe_b, dosers=self.dosers)
            else:
                z_d_a = self.compute_pert_embeddings_(perts=ohe_a)
                z_d_b = self.compute_pert_embeddings_(perts=ohe_b)
                z_a = z0.unsqueeze(0) + z_d_a
                z_b = z0.unsqueeze(0) + z_d_b

            g_a = decoder(z_a)
            g_b = decoder(z_b)
            if self.recon_loss_type_genes != "mse":
                mu_a = scale * g_a[:, :D]
                mu_b = scale * g_b[:, :D]
            else:
                mu_a = scale * g_a
                mu_b = scale * g_b

            pred_diff = (mu_a - mu_b).squeeze(0)
            loss = loss + F.mse_loss(pred_diff, actual_diff)
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=self.device), 0.0
        loss = loss / n_valid
        return loss, float(loss.detach().item())

    def _compute_rotation_recon_reg(
        self, h: torch.Tensor, z_c: torch.Tensor,
        perts: torch.Tensor, genes: torch.Tensor,
        degs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Direct supervision on rotations: decode(R_p · mean_z_control) ≈ treated expression.

        Forces R_p to be non-identity by giving rotations explicit gradient pressure.
        Without this, the encoder encodes pert info into h, making R≈I optimal.
        DE genes are upweighted by deg_upweight_rotation so the loss isn't dominated
        by ~2000 non-DE genes where R=I is already correct.
        """
        if perts is None or self.pert_composition != "rotation":
            return torch.tensor(0.0, device=self.device), 0.0

        pert_idx = perts.gt(0).float().argmax(dim=1)
        is_control = perts.sum(dim=1) == 0
        is_treated = ~is_control

        if is_control.sum() < 2 or is_treated.sum() < 2:
            return torch.tensor(0.0, device=self.device), 0.0

        # Mean basal latent from control cells (detached — don't backprop into encoder)
        z_basal = (h[is_control] + z_c[is_control]).mean(0).detach()

        decoder = self._perturbation_decoder
        scale = self._perturbation_output_scale
        D = self._perturbation_output_dim

        # DEG mask: only compute loss on DE genes (ignore ~2200 non-DE genes where R=I is correct)
        use_deg_mask = degs is not None

        # Sample up to K unique treated perturbations
        treated_perts = pert_idx[is_treated]
        unique_perts = treated_perts.unique()
        K = min(int(self.hparams.get("rotation_recon_k", 64)), len(unique_perts))
        perm = torch.randperm(len(unique_perts), device=h.device)[:K]

        loss = torch.tensor(0.0, device=h.device)
        n_valid = 0

        for i in range(K):
            p = unique_perts[perm[i]]
            mask = is_treated & (pert_idx == p)
            if mask.sum() < 2:
                continue

            # Actual treated expression (detached target)
            target = genes[mask].mean(0).detach()

            # Apply rotation R_p to mean basal latent
            ohe = torch.zeros(1, self.num_perts, device=h.device)
            ohe[0, p] = 1.0
            z_rot = self.rotation(z_basal.unsqueeze(0), ohe, dosers=self.dosers)

            # Decode
            g_out = decoder(z_rot)
            if self.recon_loss_type_genes != "mse":
                mu = scale * g_out[:, :D]
            else:
                mu = scale * g_out
            pred = mu.squeeze(0)

            # MSE on DE genes only — non-DE genes are noise where R=I is trivially correct
            if use_deg_mask:
                deg_mask = degs[mask].mean(0).detach() > 0.5 if degs.dim() == 2 else degs > 0.5
                if deg_mask.any():
                    loss = loss + F.mse_loss(pred[deg_mask], target[deg_mask])
                else:
                    loss = loss + F.mse_loss(pred, target)
            else:
                loss = loss + F.mse_loss(pred, target)
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=self.device), 0.0
        loss = loss / n_valid
        return loss, float(loss.detach().item())

    def _compute_synthetic_rotation_reg(
        self, h: torch.Tensor, z_c: torch.Tensor,
        perts: torch.Tensor, genes: torch.Tensor,
        degs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Train gene-indexed rotations for ALL genes using real + synthetic targets.

        Seen perts (genes with real treated data in batch): train against actual
        treated expression, same as rotation_recon.
        Unseen genes: compute Jacobian row v_g, generate synthetic target
        decode(z_basal + alpha * v_g), train gene_rotation against that.
        """
        if self.gene_rotation is None or perts is None:
            return torch.tensor(0.0, device=self.device), 0.0

        is_control = perts.sum(dim=1) == 0
        if is_control.sum() < 2:
            return torch.tensor(0.0, device=self.device), 0.0

        z_basal = (h[is_control] + z_c[is_control]).mean(0).detach()

        K = int(self.hparams.get("synthetic_rotation_k", 32))
        alpha = float(self.hparams.get("synthetic_rotation_alpha", 0.1))
        mode = self.hparams.get("synthetic_rotation_mode", "i")

        decoder = self._perturbation_decoder
        scale = self._perturbation_output_scale
        D = self._perturbation_output_dim

        # Identify seen genes in this batch (perts that map to gene indices)
        pert_to_gene = getattr(self, "_pert_to_gene_map", {})
        pert_idx = perts.gt(0).float().argmax(dim=1)
        is_treated = ~is_control
        seen_gene_to_pert = {}
        if is_treated.any():
            for p in pert_idx[is_treated].unique().tolist():
                if p in pert_to_gene:
                    seen_gene_to_pert[pert_to_gene[p]] = p

        # Sample K genes (mix of seen and unseen)
        gene_indices = torch.randperm(self.num_genes, device=h.device)[:K]
        gene_list = gene_indices.tolist()

        # Separate seen vs unseen genes
        seen_list = [(i, g) for i, g in enumerate(gene_list) if g in seen_gene_to_pert]
        unseen_list = [(i, g) for i, g in enumerate(gene_list) if g not in seen_gene_to_pert]

        loss = torch.tensor(0.0, device=h.device)
        has_loss = False

        # --- Unseen genes: latent-space loss ---
        # Train gene_rotation so R_g * z_basal ≈ z_basal + alpha * v_g
        if unseen_list:
            z_for_grad = z_basal.detach().clone().requires_grad_(True)
            g_grad = decoder(z_for_grad.unsqueeze(0).expand(2, -1))  # batch>1 for BN
            if self.recon_loss_type_genes != "mse":
                mu_grad = scale * g_grad[:, :D]
            else:
                mu_grad = scale * g_grad

            z_targets = []
            unseen_gene_ids = []
            for _, g in unseen_list:
                target_val = mu_grad[0, g]
                v_g = torch.autograd.grad(target_val, z_for_grad, retain_graph=True)[0]
                if mode.lower() == "i":
                    v_g = -v_g
                v_g = v_g / (v_g.norm() + 1e-12)
                z_targets.append((z_basal + alpha * v_g).detach())
                unseen_gene_ids.append(g)

            K_u = len(unseen_gene_ids)
            ohe_u = torch.zeros(K_u, self.num_genes, device=h.device)
            for i, g in enumerate(unseen_gene_ids):
                ohe_u[i, g] = 1.0
            z_rot_u = self.gene_rotation(z_basal.unsqueeze(0).expand(K_u, -1), ohe_u)
            z_target_batch = torch.stack(z_targets)

            loss = loss + F.mse_loss(z_rot_u, z_target_batch)
            has_loss = True

        # --- Seen genes: gene-space loss (real treated expression) ---
        if seen_list:
            K_s = len(seen_list)
            ohe_s = torch.zeros(K_s, self.num_genes, device=h.device)
            valid_seen = []
            gene_targets = []
            for idx, (_, g) in enumerate(seen_list):
                p = seen_gene_to_pert[g]
                mask = is_treated & (pert_idx == p)
                if mask.sum() < 2:
                    continue
                ohe_s[len(valid_seen), g] = 1.0
                gene_targets.append(genes[mask].mean(0).detach())
                valid_seen.append(idx)

            if valid_seen:
                K_vs = len(valid_seen)
                ohe_s = ohe_s[:K_vs]
                z_rot_s = self.gene_rotation(z_basal.unsqueeze(0).expand(K_vs, -1), ohe_s)
                g_out_s = decoder(z_rot_s)
                if self.recon_loss_type_genes != "mse":
                    pred_s = scale * g_out_s[:, :D]
                else:
                    pred_s = scale * g_out_s
                target_s = torch.stack(gene_targets)
                loss = loss + F.mse_loss(pred_s, target_s)
                has_loss = True

        if not has_loss:
            return torch.tensor(0.0, device=self.device), 0.0
        return loss, float(loss.detach().item())

    def _warm_start_gene_rotation(self):
        """Copy learned rotation generators to gene_rotation for seen perts."""
        pert_to_gene = getattr(self, "_pert_to_gene_map", {})
        if self.gene_rotation is None or self.rotation is None or not pert_to_gene:
            return
        n = 0
        with torch.no_grad():
            for p_idx, g_idx in pert_to_gene.items():
                if p_idx < self.rotation.generator_params.size(0) and \
                   g_idx < self.gene_rotation.generator_params.size(0):
                    self.gene_rotation.generator_params.data[g_idx] = \
                        self.rotation.generator_params.data[p_idx]
                    n += 1
        return n

    def _compute_min_rotation_reg(
        self, perts: torch.Tensor, genes: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Hinge loss pushing generator norms above effect-proportional threshold.

        L = mean_p max(0, scale * ||Δ_p|| - ||A_p||_F)²
        Forces rotations to be non-trivial for perts with large observed effects.
        """
        if perts is None or self.pert_composition != "rotation":
            return torch.tensor(0.0, device=self.device), 0.0

        scale = self.hparams.get("min_rotation_scale", 0.5)

        pert_idx = perts.gt(0).float().argmax(dim=1)
        is_control = perts.sum(dim=1) == 0
        is_treated = ~is_control

        if is_control.sum() < 2 or is_treated.sum() < 2:
            return torch.tensor(0.0, device=self.device), 0.0

        ctrl_mean = genes[is_control].mean(0).detach()
        treated_perts = pert_idx[is_treated]
        unique_perts = treated_perts.unique()

        loss = torch.tensor(0.0, device=self.device)
        n = 0
        for p in unique_perts:
            mask = is_treated & (pert_idx == p)
            if mask.sum() < 2:
                continue

            effect_norm = (genes[mask].mean(0).detach() - ctrl_mean).norm()
            theta_min = scale * effect_norm

            A_p = self.rotation.get_generators(p.unsqueeze(0))
            gen_norm = A_p.reshape(-1).norm()

            loss = loss + F.relu(theta_min - gen_norm) ** 2
            n += 1

        if n == 0:
            return torch.tensor(0.0, device=self.device), 0.0
        loss = loss / n
        return loss, float(loss.detach().item())

    def _compute_contrastive_div_reg(
        self, perts: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Push generators of different perturbations apart in the Lie algebra.

        Penalizes cosine similarity above a margin between generator vectors
        of different perturbations in the batch.
        """
        if perts is None or self.rotation is None:
            return torch.tensor(0.0, device=self.device), 0.0

        margin = float(self.hparams.get("contrastive_div_margin", 0.3))
        K = int(self.hparams.get("contrastive_div_k", 32))

        pert_idx = perts.gt(0).float().argmax(dim=1)
        is_treated = perts.sum(dim=1) > 0
        unique_perts = pert_idx[is_treated].unique()

        if len(unique_perts) < 2:
            return torch.tensor(0.0, device=self.device), 0.0

        if len(unique_perts) > K:
            perm = torch.randperm(len(unique_perts), device=perts.device)[:K]
            unique_perts = unique_perts[perm]

        # Get flattened generators for selected perts: (K, 192)
        A = self.rotation.get_generators(unique_perts)  # (K, num_blocks, bs, bs)
        A_flat = A.reshape(len(unique_perts), -1)  # (K, 192)

        # Normalize
        norms = A_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        A_norm = A_flat / norms

        # Pairwise cosine similarity: (K, K)
        cos_sim = A_norm @ A_norm.T

        # Mask diagonal
        mask = ~torch.eye(len(unique_perts), dtype=torch.bool, device=perts.device)
        cos_vals = cos_sim[mask]

        # Hinge loss: penalize similarity above margin
        loss = (F.relu(cos_vals - margin) ** 2).mean()
        return loss, float(loss.detach().item())

    def _compute_curvature_reg(self, h: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Riemannian sectional curvature of the decoder manifold.

        Computes d²f_g/dv² for random genes g and random latent directions v.
        Non-zero second derivative = the Jacobian rotates as you move along v = curvature.
        Penalizes flat decoders where all perturbation directions produce parallel shifts.
        """
        z = h.mean(0).detach().clone().requires_grad_(True)
        decoder = self._perturbation_decoder
        scale = self._perturbation_output_scale
        D = self._perturbation_output_dim

        out = decoder(z.unsqueeze(0))
        if self.recon_loss_type_genes != "mse":
            out = out[:, :D]
        mu = (scale * out).squeeze(0)  # (G,)

        n_genes = 8
        n_dirs = 2
        G = mu.size(0)
        gene_idx = torch.randperm(G, device=z.device)[:n_genes]

        total_curv = torch.tensor(0.0, device=z.device)
        for _ in range(n_dirs):
            v = torch.randn_like(z)
            v = v / (v.norm() + 1e-12)
            for g_i in gene_idx:
                g1 = torch.autograd.grad(mu[g_i], z, create_graph=True, retain_graph=True)[0]
                dfdv = (g1 * v).sum()
                g2 = torch.autograd.grad(dfdv, z, create_graph=True, retain_graph=True)[0]
                d2fdv2 = (g2 * v).sum()
                total_curv = total_curv + d2fdv2.pow(2)

        mean_curv = total_curv / (n_genes * n_dirs)
        loss = -torch.log(mean_curv + 1e-8)
        return loss, float(mean_curv.detach().item())

    # Helpers

    def _extract_mean(self, out: torch.Tensor, modality: str = "genes") -> torch.Tensor:
        """Extract mean from decoder output (skip variance half when NLL)."""
        loss_type = self.recon_loss_type_genes if modality == "genes" else self.recon_loss_type_fluxes
        if loss_type == "mse":
            return out
        return out[:, : out.size(1) // 2]

    # Jacobian anchor (phase 3 directional preservation)

    def _snapshot_jacobian_directions(self, h: torch.Tensor, k: int = 32):
        centroid = h.mean(0).detach()
        with torch.no_grad():
            G = self._perturbation_decoder(centroid.unsqueeze(0)).size(1)
            if self.recon_loss_type_genes != "mse":
                G = G // 2
        indices = torch.randperm(G, device=h.device)[:k].tolist()
        dirs = []
        for gi in indices:
            v = self._oov_grad_direction(centroid, gi, mode="a")
            dirs.append(v)
        self.register_buffer("_jacobian_anchor_dirs", torch.stack(dirs, 0).detach())
        self._jacobian_anchor_genes = indices

    def _compute_jacobian_anchor_reg(self, h: torch.Tensor):
        centroid = h.mean(0).detach()
        anchors = self._jacobian_anchor_dirs
        cos_sims = []
        for i, gi in enumerate(self._jacobian_anchor_genes):
            v = self._oov_grad_direction(centroid, gi, mode="a")
            cos_sims.append(F.cosine_similarity(v.unsqueeze(0), anchors[i].unsqueeze(0)))
        cos_sims = torch.stack(cos_sims)
        loss = 1.0 - cos_sims.mean()
        return loss, cos_sims.mean().item()

    # Jacobian-rotation consistency (phase 3)

    def _compute_jacobian_consist_reg(
        self, h: torch.Tensor, z_c: torch.Tensor,
        perts: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        if perts is None or self.pert_composition != "rotation":
            return torch.tensor(0.0, device=self.device), 0.0

        pert_to_gene = getattr(self, "_pert_to_gene_map", {})
        if not pert_to_gene:
            return torch.tensor(0.0, device=self.device), 0.0

        pert_idx = perts.gt(0).float().argmax(dim=1)
        is_control = perts.sum(dim=1) == 0
        if is_control.sum() < 2:
            return torch.tensor(0.0, device=self.device), 0.0

        z_basal = (h[is_control] + z_c[is_control]).mean(0).detach()

        decoder = self._perturbation_decoder
        scale = self._perturbation_output_scale
        D = self._perturbation_output_dim
        mode = self.hparams.get("synthetic_rotation_mode", "i")
        alpha = float(self.hparams.get("jacobian_consist_alpha", 1.0))

        treated_perts = pert_idx[~is_control].unique()
        eligible = [p.item() for p in treated_perts if p.item() in pert_to_gene]
        if not eligible:
            return torch.tensor(0.0, device=self.device), 0.0

        K = min(int(self.hparams.get("jacobian_consist_k", 8)), len(eligible))
        sampled = [eligible[i] for i in torch.randperm(len(eligible))[:K].tolist()]
        gene_indices = [pert_to_gene[p] for p in sampled]

        # Jacobian directions: loop backward (one per gene), create_graph for 2nd-order
        z_g = z_basal.detach().clone().requires_grad_(True)
        g_out = decoder(z_g.unsqueeze(0).expand(2, -1))
        mu_g = scale * (g_out[0, :D] if self.recon_loss_type_genes != "mse" else g_out[0])

        v_ps = []
        for g in gene_indices:
            v = torch.autograd.grad(mu_g[g], z_g, create_graph=True, retain_graph=True)[0]
            if mode.lower() == "i":
                v = -v
            v_ps.append(v / (v.norm() + 1e-12))
        v_stack = torch.stack(v_ps)  # (K, E)

        # Batched Jacobian prediction: decode all K stepped points at once
        z_stepped = z_basal.unsqueeze(0) + alpha * v_stack  # (K, E)
        jac_out = decoder(z_stepped)
        jac_mu = scale * (jac_out[:, :D] if self.recon_loss_type_genes != "mse" else jac_out)

        # Basal + rotation predictions (detached)
        with torch.no_grad():
            basal_out = decoder(z_basal.unsqueeze(0).expand(2, -1))
            basal_mu = scale * (basal_out[0, :D] if self.recon_loss_type_genes != "mse" else basal_out[0])

            ohe = torch.zeros(K, self.num_perts, device=h.device)
            for i, p in enumerate(sampled):
                ohe[i, p] = 1.0
            z_rot = self.rotation(z_basal.unsqueeze(0).expand(K, -1), ohe, dosers=self.dosers)
            rot_out = decoder(z_rot)
            rot_mu = scale * (rot_out[:, :D] if self.recon_loss_type_genes != "mse" else rot_out)
            delta_rot = rot_mu - basal_mu.unsqueeze(0)

        delta_jac = jac_mu - basal_mu.unsqueeze(0)  # (K, G), basal detached
        cos = F.cosine_similarity(delta_jac, delta_rot, dim=1)  # (K,)
        loss = (1.0 - cos).mean()
        return loss, float(loss.detach().item())

    # CRISPR OOV methods (shared — use _perturbation_decoder property)

    def _oov_grad_direction(self, base_latent: torch.Tensor, gene_index: int, mode: str = "i") -> torch.Tensor:
        b = base_latent.mean(dim=0) if base_latent.dim() == 2 else base_latent
        b = b.detach().clone().requires_grad_(True)
        g_out = self._perturbation_decoder(b.unsqueeze(0))
        if self.recon_loss_type_genes == "mse":
            mu = self._perturbation_output_scale * g_out
        else:
            D = g_out.size(1) // 2
            mu = self._perturbation_output_scale * g_out[:, :D]
        target = mu[0, gene_index]
        grad = torch.autograd.grad(target, b, retain_graph=False, create_graph=False)[0]
        if mode.lower() == "i":
            grad = -grad
        return grad / (grad.norm() + 1e-12)

    def _oov_grad_direction_batch(self, base_latents: torch.Tensor, gene_index: int, mode: str = "i") -> torch.Tensor:
        x = base_latents.detach().clone().requires_grad_(True)
        g_out = self._perturbation_decoder(x)
        if self.recon_loss_type_genes == "mse":
            mu = self._perturbation_output_scale * g_out
        else:
            D = g_out.size(1) // 2
            mu = self._perturbation_output_scale * g_out[:, :D]
        target = mu[:, gene_index].sum()
        grads = torch.autograd.grad(target, x, retain_graph=False, create_graph=False)[0]
        if mode.lower() == "i":
            grads = -grads
        return F.normalize(grads, dim=1)

    def _raw_grad_direction(self, z: torch.Tensor, gene_index: int, mode: str = "i") -> torch.Tensor:
        b = z.mean(dim=0) if z.dim() == 2 else z
        b = b.detach().clone().requires_grad_(True)
        g_out = self._perturbation_decoder(b.unsqueeze(0))
        if self.recon_loss_type_genes == "mse":
            mu = self._perturbation_output_scale * g_out
        else:
            D = g_out.size(1) // 2
            mu = self._perturbation_output_scale * g_out[:, :D]
        target = mu[0, gene_index]
        grad = torch.autograd.grad(target, b, retain_graph=False, create_graph=False)[0]
        if mode.lower() == "i":
            grad = -grad
        return grad

    def _lie_bracket(
        self, z: torch.Tensor, gene_i: int, gene_j: int, mode: str = "i",
        alpha: float = 0.1, eps: float = 0.01,
    ) -> tuple:
        vi_raw = self._raw_grad_direction(z, gene_i, mode=mode)
        vj_raw = self._raw_grad_direction(z, gene_j, mode=mode)
        vi = alpha * vi_raw / (vi_raw.norm() + 1e-12)
        vj = alpha * vj_raw / (vj_raw.norm() + 1e-12)
        vj_shifted = self._raw_grad_direction(z + eps * vi, gene_j, mode=mode)
        vj_shifted = alpha * vj_shifted / (vj_shifted.norm() + 1e-12)
        vi_shifted = self._raw_grad_direction(z + eps * vj, gene_i, mode=mode)
        vi_shifted = alpha * vi_shifted / (vi_shifted.norm() + 1e-12)
        bracket = (vj_shifted - vj) / eps - (vi_shifted - vi) / eps
        return vi, vj, bracket

    def _calibrate_alpha_to_lfc(
        self, h, z_c, gene_index, v_dir, target_abs_lfc=1.0, mode="i", iters=24, hi_init=5.0
    ) -> float:
        eps = 1e-6
        with torch.no_grad():
            base = h + z_c
            g_pred, f_pred = self._decode(base)
            out = g_pred if "genes" in self.modalities else f_pred
            mu0 = self._extract_mean(out)[:, gene_index].mean().item()

            sgn = -1.0 if mode.lower() == "i" else 1.0
            lo, hi = 0.0, hi_init

            for _ in range(iters):
                mid = (lo + hi) / 2.0
                lat = base + (mid * sgn) * v_dir.unsqueeze(0).expand_as(base)
                g1, f1 = self._decode(lat)
                out1 = g1 if "genes" in self.modalities else f1
                mu1 = self._extract_mean(out1)[:, gene_index].mean().item()
                lfc = math.log2((mu1 + eps) / (mu0 + eps))

                if (lfc * sgn) >= target_abs_lfc:
                    hi = mid
                else:
                    lo = mid

            return hi

    def _calibrate_alpha_quantile(
        self, base_latents, gene_index, v_directions, mode="i",
        target_abs_lfc=1.0, q=0.7, iters=20, hi_init=6.0,
    ) -> float:
        B, E = base_latents.shape
        eps = 1e-6

        g0, f0 = self._decode(base_latents)
        out0 = g0 if "genes" in self.modalities else f0
        mu0 = self._extract_mean(out0)[:, gene_index]

        lo = torch.zeros(B, device=base_latents.device)
        hi = torch.full((B,), hi_init, device=base_latents.device)
        v = v_directions

        for _ in range(iters):
            mid = (lo + hi) / 2
            Zmid = base_latents + mid.unsqueeze(1) * v
            gm, fm = self._decode(Zmid)
            outm = gm if "genes" in self.modalities else fm
            mu_mid = self._extract_mean(outm)[:, gene_index]
            lfc_mid = torch.log2((mu_mid + eps) / (mu0 + eps))

            if mode.lower() == "i":
                achieved = lfc_mid <= -target_abs_lfc
            else:
                achieved = lfc_mid >= target_abs_lfc
            hi = torch.where(achieved, mid, hi)
            lo = torch.where(achieved, lo, mid)

        return float(torch.quantile(hi, q))

    def _project_to_cone(self, v_i: torch.Tensor, v_ref: torch.Tensor, cos_min: float = 0.5) -> torch.Tensor:
        vr = F.normalize(v_ref, dim=0)
        vi = F.normalize(v_i, dim=1)
        cos = vi @ vr
        mask = cos < cos_min
        if not mask.any():
            return vi
        a = (vi @ vr).unsqueeze(1) * vr.unsqueeze(0)
        u = vi - a
        u_norm = u.norm(dim=1, keepdim=True) + 1e-12
        u = u / u_norm
        a_norm = a.norm(dim=1, keepdim=True) + 1e-12
        cos_min_t = torch.tensor(cos_min, device=v_i.device, dtype=v_i.dtype)
        tan_th = torch.sqrt(1 - cos_min_t**2) / (cos_min_t + 1e-12)
        u_new = u * (a_norm * tan_th)
        v_new = F.normalize(a + u_new, dim=1)
        vi[mask] = v_new[mask]
        return vi

    @torch.no_grad()
    def _avg_effect_norm(self) -> float:
        if self._avg_in_vocab_effect_norm is not None:
            return self._avg_in_vocab_effect_norm
        if getattr(self, "num_perts", 0) <= 0:
            self._avg_in_vocab_effect_norm = 1.0
            return 1.0
        E_all = self._raw_pert_effect_matrix()
        self._avg_in_vocab_effect_norm = float(E_all.norm(dim=1).mean().item())
        return self._avg_in_vocab_effect_norm

    def _synthesize_knock_vector(
        self, base_latent, gene_index, mode="i", strength=1.0, calibrate_to_known=True,
    ) -> torch.Tensor:
        x = base_latent.detach().clone().requires_grad_(True)
        g_out = self._perturbation_decoder(x)
        D = g_out.size(1) // 2
        mu = self._perturbation_output_scale * g_out[:, :D]
        loss = mu[0, int(gene_index)]
        grad = torch.autograd.grad(loss, x, create_graph=False, retain_graph=False)[0]
        vec = grad.squeeze(0)
        sign = -1.0 if mode.lower() == "i" else +1.0
        vec = sign * vec
        vec = vec / (vec.norm() + 1e-8)
        if calibrate_to_known:
            vec = vec * self._avg_effect_norm()
        vec = vec * float(strength)
        return vec.detach()

    def _get_kd_vector(self, gene_index, base_latent, mode, strength, calibrate_to_known) -> torch.Tensor:
        key = (int(gene_index), mode.lower(), float(strength), bool(calibrate_to_known))
        if key in self._crispr_cache:
            return self._crispr_cache[key]
        v = self._synthesize_knock_vector(
            base_latent=base_latent, gene_index=int(gene_index),
            mode=mode, strength=strength, calibrate_to_known=calibrate_to_known,
        )
        self._crispr_cache[key] = v
        return v

    @torch.no_grad()
    def set_crispr_base_from_controls(self, genes_ctrl, flux_ctrl, covariates=None):
        h = self._encode(genes=genes_ctrl, fluxes=flux_ctrl)
        self._crispr_base_latents = h.detach().clone()
        self._crispr_base_latent = h.mean(dim=0).detach().clone()

    # Additional decoder helpers

    def _global_grad_dir(self, base_latent: torch.Tensor, gene_index: int, mode: str = "i") -> torch.Tensor:
        x = base_latent.detach().clone().requires_grad_(True)
        g_out = self._perturbation_decoder(x.unsqueeze(0))
        D = g_out.size(1) // 2
        mu = self._perturbation_output_scale * g_out[:, :D]
        g = torch.autograd.grad(mu[0, int(gene_index)], x, retain_graph=False)[0]
        if mode.lower() == "i":
            g = -g
        return g / (g.norm() + 1e-12)

    def hvp_along(self, base_latent: torch.Tensor, gene_index: int, v: torch.Tensor, mode: str = "i") -> float:
        x = base_latent.detach().clone().requires_grad_(True)
        g_out = self._perturbation_decoder(x.unsqueeze(0))
        D = g_out.size(1) // 2
        mu = self._perturbation_output_scale * g_out[:, :D]
        y = mu[0, int(gene_index)]
        if mode.lower() == "i":
            y = -y
        from torch.autograd.functional import hvp as torch_hvp
        scale = self._perturbation_output_scale
        decoder = self._perturbation_decoder
        val, hv = torch_hvp(
            lambda z: (scale * decoder(z.unsqueeze(0))[:, :D])[0, int(gene_index)], x, v
        )
        return float(hv.dot(v).item() * (-1.0 if mode.lower() == "i" else 1.0))

    def power_eig_jtj(
        self, base_latent: torch.Tensor, gene_subset: Optional[torch.Tensor] = None, iters: int = 10, seed: int = 0
    ):
        from torch.autograd.functional import jvp
        torch.manual_seed(seed)
        z = base_latent.detach().requires_grad_(True)
        E = z.numel()
        v = torch.randn(E, device=self.device)
        v = v / (v.norm() + 1e-12)

        scale = self._perturbation_output_scale
        decoder = self._perturbation_decoder

        def mu_subset(x):
            out = decoder(x.unsqueeze(0))
            D = out.size(1) // 2
            mu = scale * out[:, :D]
            if gene_subset is not None:
                return mu[0, gene_subset]
            return mu.squeeze(0)

        def Jv(x, vec):
            y, _ = jvp(mu_subset, (x,), (vec,), strict=False)
            return y

        def JT_y(x, y_vec):
            out = mu_subset(x)
            u = torch.autograd.grad(out, x, grad_outputs=y_vec, retain_graph=True)[0]
            return u

        for _ in range(iters):
            w = Jv(z, v)
            u = JT_y(z, w)
            v = u / (u.norm() + 1e-12)

        w = Jv(z, v)
        lam = float(w.pow(2).sum().item())
        return lam, v

    def _decoder_mean_genes(self, latent: torch.Tensor) -> torch.Tensor:
        out = self._perturbation_decoder(latent)
        if self.recon_loss_type_genes == "mse":
            return self._perturbation_output_scale * out
        D = out.size(1) // 2
        return self._perturbation_output_scale * out[:, :D]

    # Checkpoint migration

    @classmethod
    def _migrate_state_dict(cls, state_dict: dict) -> dict:
        """Remap legacy drug_* state_dict keys to pert_* naming."""
        remap_prefixes = [
            ("drug_embeddings.",        "pert_embeddings."),
            ("drug_embedding_encoder.", "pert_encoder."),
            ("adversary_drugs.",        "adversary_perts."),
        ]
        new_sd = {}
        for key, val in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in remap_prefixes:
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    break
            new_sd[new_key] = val
        return new_sd

    @staticmethod
    def _migrate_init_args(init_args: dict) -> dict:
        """Remap legacy init_args keys (num_drugs→num_perts, etc.)."""
        remap = {
            "num_drugs": "num_perts",
            "use_drugs_idx": "use_perts_idx",
        }
        return {remap.get(k, k): v for k, v in init_args.items()}
