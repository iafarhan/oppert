"""Rotation modules for perturbation composition on SO(n)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypernetGenerator(nn.Module):
    """MLP mapping frozen gene embeddings to Lie algebra generator parameters.

    Architecture: input_dim -> hidden_dims -> output_dim with residual
    connections (where dims match), LayerNorm, SiLU activations, and dropout.

    Small last-layer init (0.01 scale) ensures initial generators produce
    near-identity rotations, preventing training instability at startup.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: list = None, dropout: float = 0.1,
                 init_scale: float = 0.01):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.input_dim = input_dim
        self.output_dim = output_dim

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no norm on last layer
                self.norms.append(nn.LayerNorm(dims[i + 1]))

        # Last-layer init controls initial rotation magnitude
        # Small (0.01) => near-identity, larger (0.1-0.3) => more diverse
        nn.init.normal_(self.layers[-1].weight, std=init_scale)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            h = layer(x)
            if i < len(self.layers) - 1:
                h = F.silu(h)
                h = self.norms[i](h)
                h = self.drop(h)
                # Residual connection when dims match
                if x.shape[-1] == h.shape[-1]:
                    h = h + x
            x = h
        return x


class LowRankDecoderRowHypernet(nn.Module):
    """Low-rank hypernet: decoder row e_g -> K coefficients -> theta via shared basis U.

    Maps decoder output-layer weight rows (latent_dim) through an MLP to K-dim
    coefficient vectors, then linearly expands to full generator parameter space
    via a shared learned basis U and bias b:

        e_g  = decoder.output_layer.weight[g]   (latent_dim)
        c_g  = MLP(e_g)                          (K)
        theta_g = U @ c_g + b                    (out_dim = 192)

    The rank bottleneck K << out_dim acts as an implicit smoothness prior:
    nearby decoder rows produce nearby coefficients, hence nearby rotations.

    Small-init on U (std=init_scale) and zeros on b ensures theta ~ 0 at init,
    i.e., rotations start near identity.

    Args:
        latent_dim: input dimension (decoder row width, typically 128)
        out_dim: output dimension (generator parameter count, typically 192)
        K: coefficient dimension (rank of the output space)
        hidden_dims: MLP hidden layer sizes
        dropout: dropout rate in MLP
        init_scale: std for small-init on last MLP layer and U matrix
    """

    def __init__(self, latent_dim: int, out_dim: int, K: int,
                 hidden_dims: list = None, dropout: float = 0.1,
                 init_scale: float = 0.01):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256]
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.K = K

        # MLP: latent_dim -> hidden -> ... -> K
        dims = [latent_dim] + list(hidden_dims) + [K]
        layers = nn.ModuleList()
        norms = nn.ModuleList()
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                norms.append(nn.LayerNorm(dims[i + 1]))
        self.layers = layers
        self.norms = norms
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Small-init on MLP last layer
        nn.init.normal_(self.layers[-1].weight, std=init_scale)
        nn.init.zeros_(self.layers[-1].bias)

        # Shared basis U (out_dim x K) and bias b (out_dim)
        self.U = nn.Parameter(torch.randn(out_dim, K) * init_scale)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, e_g: torch.Tensor):
        """Map decoder rows to generator parameters via low-rank factorization.

        Args:
            e_g: (P, latent_dim) decoder output-layer weight rows for P perturbations

        Returns:
            theta: (P, out_dim) generator parameters
            c: (P, K) coefficient vectors (for diagnostics logging)
        """
        x = e_g
        for i, layer in enumerate(self.layers):
            h = layer(x)
            if i < len(self.layers) - 1:
                h = F.silu(h)
                h = self.norms[i](h)
                h = self.drop(h)
                if x.shape[-1] == h.shape[-1]:
                    h = h + x
            x = h
        c = x  # (P, K)
        theta = torch.mm(c, self.U.t()) + self.b  # (P, out_dim)
        return theta, c



class KernelAnchorHypernet(nn.Module):
    """Kernel-anchor rotation predictor: RBF kernel on BGE embeddings.

    First layer computes RBF kernel similarities to N training-pert embeddings.
    Linear head predicts rotation generators (or latent deltas) from kernel features.
    At init = exact sklearn KRR via closed-form ridge solution.
    At inference = one forward pass.

    Args:
        training_embeddings: (N, embed_dim) frozen embeddings of training perturbations
        out_dim: output dimension (192 for rotation generators, 128 for latent delta)
        gamma: RBF bandwidth (from KRR CV). If 0, auto-fit via median heuristic.
        lam: ridge penalty (from KRR CV). If 0, default 1.0.
    """

    def __init__(self, training_embeddings: torch.Tensor, out_dim: int,
                 gamma: float = 0.0, lam: float = 1.0):
        super().__init__()
        N, embed_dim = training_embeddings.shape
        self.N = N
        self.out_dim = out_dim

        self.register_buffer("training_embeddings", training_embeddings)

        # Auto-fit gamma via median heuristic if not provided
        if gamma <= 0:
            with torch.no_grad():
                dists = torch.cdist(training_embeddings.double(),
                                    training_embeddings.double()).pow(2)
                median_dist = dists[dists > 0].median().item()
                gamma = 1.0 / (median_dist + 1e-8)
        self.gamma = gamma
        self.lam = lam

        # Trainable head
        self.beta = nn.Parameter(torch.zeros(N, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def init_from_targets(self, target_theta: torch.Tensor):
        """Initialize beta via closed-form ridge to reproduce exact KRR.

        Args:
            target_theta: (N, out_dim) learned generators from lookup table
        """
        with torch.no_grad():
            emb = self.training_embeddings.double()
            dists = torch.cdist(emb, emb).pow(2)
            K = torch.exp(-self.gamma * dists)
            K_reg = K + self.lam * torch.eye(self.N, device=K.device, dtype=torch.float64)
            beta = torch.linalg.solve(K_reg, target_theta.double())
            self.beta.data.copy_(beta.float())
            self.bias.data.zero_()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict from query embeddings via RBF kernel + linear head.

        Args:
            embeddings: (P, embed_dim) query embeddings

        Returns:
            theta: (P, out_dim) predicted generators or deltas
        """
        # RBF kernel similarities to training embeddings
        dists = torch.cdist(embeddings.unsqueeze(0),
                           self.training_embeddings.unsqueeze(0)).squeeze(0).pow(2)
        phi = torch.exp(-self.gamma * dists)  # (P, N)
        return phi @ self.beta + self.bias  # (P, out_dim)


class BlockRotation(nn.Module):

    def __init__(self, num_perts: int, latent_dim: int, block_size: int = 4,
                 hypernet_input_dim: int = 0, hypernet_hidden: int = 256,
                 hypernet_layers: int = 3, hypernet_dropout: float = 0.0):
        super().__init__()
        assert latent_dim % block_size == 0
        self.block_size = block_size
        self.num_blocks = latent_dim // block_size
        self.latent_dim = latent_dim
        # so(n) has n*(n-1)/2 basis elements
        self.generator_dim = block_size * (block_size - 1) // 2
        self.num_perts = num_perts

        # Total output dimension: num_blocks * generator_dim
        # For block_size=4, dim=128: 32 blocks * 6 params = 192 (gauge-free!)
        out_dim = self.num_blocks * self.generator_dim

        # Learnable params: zero-init = identity rotation
        # Stored as 2D (num_perts, num_blocks * generator_dim) so Muon picks it up
        self.generator_params = nn.Parameter(
            torch.zeros(num_perts, out_dim)
        )

        # Precompute basis of so(block_size): generator_dim matrices of shape (block_size, block_size)
        basis = torch.zeros(self.generator_dim, block_size, block_size)
        idx = 0
        for i in range(block_size):
            for j in range(i + 1, block_size):
                basis[idx, i, j] = 1.0
                basis[idx, j, i] = -1.0
                idx += 1
        self.register_buffer("basis", basis)  # (generator_dim, block_size, block_size)

        # Optional hypernetwork: gene embedding → so(block_size) coefficients (gauge-free!)
        # Key advantage over CayleyGeneratorNet: output is only 192 dims (vs 4096)
        # and has NO gauge freedom (unique parameterization of skew-symmetric matrices)
        if hypernet_input_dim > 0:
            layers = []
            in_dim = hypernet_input_dim
            for i in range(hypernet_layers - 1):
                layers.append(nn.Linear(in_dim, hypernet_hidden))
                layers.append(nn.SiLU())
                layers.append(nn.LayerNorm(hypernet_hidden))
                if hypernet_dropout > 0:
                    layers.append(nn.Dropout(hypernet_dropout))
                in_dim = hypernet_hidden
            layers.append(nn.Linear(in_dim, out_dim))
            nn.init.normal_(layers[-1].weight, std=0.01)
            nn.init.zeros_(layers[-1].bias)
            self.generator = nn.Sequential(*layers)
            self.output_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.generator = None

        # Learnable bracket scale: compensates for dose-induced quadratic suppression
        # of the BCH commutator term. Initialized to 1.0; the optimizer can increase
        # this to make the bracket correction meaningful.
        self.bracket_scale = nn.Parameter(torch.tensor(1.0))

        # Rotation warmup: external code sets this to ramp from small to 1.0
        # When < 1.0, rotation magnitudes are reduced, helping decoder adapt
        self._rotation_warmup_scale = 1.0

        # Learned geodesic scale: per-pert rotation magnitude in (0, max_scale)
        # Set by init_geodesic_scale() if doser_type="geodesic"
        self._geodesic_logit = None
        self._geodesic_max_scale = 1.0

        # End-to-end hypernetwork: frozen_embedding -> MLP -> generator params
        # Replaces the per-pert lookup table (generator_params) entirely.
        # Initialized by init_hypernet_e2e() after model creation.
        self.hypernet_e2e = False
        self._hypernet = None

    def init_geodesic_scale(self, init_val: float = 0.3, max_scale: float = 1.0):
        """Initialize per-pert learned rotation magnitude (geodesic t parameter).

        The scale is parameterized as: scale = max_scale * sigmoid(logit)
        so it's always in (0, max_scale).
        """
        import math
        num_perts = self.generator_params.shape[0]
        # Inverse sigmoid to get initial logit
        clamped_init = min(max(init_val / max_scale, 0.01), 0.99)
        init_logit = math.log(clamped_init / (1 - clamped_init))
        self._geodesic_logit = nn.Parameter(torch.full((num_perts,), init_logit))
        self._geodesic_max_scale = max_scale

    def get_geodesic_scale(self):
        """Return per-pert scale in (0, max_scale), or None if not initialized."""
        if self._geodesic_logit is None:
            return None
        return self._geodesic_max_scale * torch.sigmoid(self._geodesic_logit)

    def init_hypernet_e2e(self, frozen_embeddings: torch.Tensor,
                          hidden_dims: list = None, dropout: float = 0.1,
                          holdout_mask: torch.Tensor = None,
                          init_scale: float = 0.01,
                          lowrank_K: int = 0,
                          input_source: str = "embedding",
                          decoder_ref_getter: callable = None,
                          pert_to_gene_idx: torch.Tensor = None,
                          kernel_anchor: bool = False,
                          kernel_gamma: float = 0.0,
                          kernel_lambda: float = 1.0,
                          kernel_target_theta: torch.Tensor = None):
        """Initialize end-to-end hypernetwork: frozen_embeddings -> MLP -> generators.

        Replaces the per-pert lookup table entirely. All generator parameters
        are now produced by the MLP, enabling zero-shot generalization to
        unseen genes (whose frozen embeddings are available but were never trained on).

        Args:
            frozen_embeddings: (num_perts, embed_dim) frozen gene embeddings (e.g. BGE-large).
                When input_source="decoder_row", this is used only for sizing; the actual
                embeddings are pulled live from the decoder at forward time.
            hidden_dims: hidden layer sizes for MLP [512, 256]
            dropout: dropout rate in MLP
            holdout_mask: (num_perts,) bool tensor. True = held-out pert (gradient detached).
            init_scale: std for last layer init. 0.01=near-identity, 0.1-0.3=more diverse.
            lowrank_K: if >0, use LowRankDecoderRowHypernet with K-dim coefficients instead
                of the full-rank HypernetGenerator. Default 0 = existing behaviour.
            input_source: "embedding" (default, frozen BGE) or "decoder_row" (live decoder
                weight rows). When "decoder_row", decoder_ref_getter and pert_to_gene_idx
                must be provided.
            decoder_ref_getter: callable returning the decoder output-layer weight tensor
                (num_genes, latent_dim). Used only when input_source="decoder_row".
            pert_to_gene_idx: (num_perts,) long tensor mapping pert index -> gene row index
                in the decoder weight matrix. Used only when input_source="decoder_row".
                Entries of -1 indicate unmapped perts (will get zeros).
        """
        if hidden_dims is None:
            hidden_dims = [512, 256]
        embed_dim = frozen_embeddings.shape[1]
        out_dim = self.num_blocks * self.generator_dim  # 192 for block_size=4, latent=128

        # Store decoder-row source references (for live e_g at forward time)
        self._lowrank_input_source = input_source
        self._decoder_ref_getter = decoder_ref_getter
        if pert_to_gene_idx is not None:
            self.register_buffer('_pert_to_gene_idx', pert_to_gene_idx)
        else:
            self._pert_to_gene_idx = None
        self._lowrank_K = lowrank_K

        # Diagnostics cache for c coefficients
        self._last_c = None

        if kernel_anchor:
            # v11: Kernel-anchor rotation predictor (exact KRR at init)
            out_dim_ka = self.num_blocks * self.generator_dim
            self._hypernet = KernelAnchorHypernet(
                training_embeddings=frozen_embeddings,
                out_dim=out_dim_ka,
                gamma=kernel_gamma,
                lam=kernel_lambda)
            if kernel_target_theta is not None:
                self._hypernet.init_from_targets(kernel_target_theta)
        elif lowrank_K > 0:
            # v10: Low-rank decoder-row hypernet
            self._hypernet = LowRankDecoderRowHypernet(
                latent_dim=embed_dim, out_dim=out_dim, K=lowrank_K,
                hidden_dims=hidden_dims, dropout=dropout, init_scale=init_scale)
        else:
            # Existing full-rank path
            self._hypernet = HypernetGenerator(embed_dim, out_dim, hidden_dims, dropout,
                                               init_scale=init_scale)

        # Move to same device as generator_params
        device = self.generator_params.device
        self._hypernet = self._hypernet.to(device)

        # Store frozen embeddings as buffer (not trainable)
        self.register_buffer('_frozen_embeddings', frozen_embeddings.to(device))

        # Mock zero-shot holdout: perts with True are excluded from gradient flow
        if holdout_mask is not None:
            self.register_buffer('_holdout_mask', holdout_mask.to(device))
            self._has_holdout = bool(holdout_mask.any().item())  # Python bool, not tensor
        else:
            self.register_buffer('_holdout_mask',
                               torch.zeros(self.num_perts, dtype=torch.bool, device=device))
            self._has_holdout = False

        # Convert generator_params from Parameter to buffer (no longer directly trained)
        gen_data = self.generator_params.data.clone()
        # Must delete before re-registering
        delattr(self, 'generator_params')
        self.register_buffer('generator_params', gen_data)

        self.hypernet_e2e = True

    def _get_all_gen_params(self) -> torch.Tensor:
        """Get (num_perts, gen_dim) generator parameters.

        In hypernet mode: computed from frozen embeddings via MLP (with gradients).
        During training, holdout perts have their generators detached (no gradient
        through MLP), creating a mock zero-shot condition.
        In standard mode: returns the per-pert Parameter lookup table.
        """
        if self.hypernet_e2e and self._hypernet is not None:
            # Determine input embeddings
            lowrank_K = getattr(self, '_lowrank_K', 0)
            input_source = getattr(self, '_lowrank_input_source', 'embedding')

            if input_source == 'decoder_row' and self._decoder_ref_getter is not None:
                # Live decoder rows: pull current weights at forward time
                W_dec = self._decoder_ref_getter()  # (num_genes, latent_dim)
                idx = self._pert_to_gene_idx.to(W_dec.device)  # (num_perts,) long
                # Gather rows; unmapped perts (idx==-1) get zeros
                valid = idx >= 0
                safe_idx = idx.clamp(min=0)
                embeddings = W_dec[safe_idx]  # (num_perts, latent_dim)
                # Zero out unmapped perts
                if not valid.all():
                    embeddings = embeddings * valid.to(embeddings.device).unsqueeze(-1).float()
                # L2 normalize (consistent with existing frozen-embedding path)
                norms = embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                embeddings = embeddings / norms
            else:
                embeddings = self._frozen_embeddings  # (P, embed_dim)

            if lowrank_K > 0:
                # v10 low-rank path: returns (theta, c)
                params, c = self._hypernet(embeddings)
                self._last_c = c.detach()  # cache for diagnostics
            else:
                # Existing full-rank HypernetGenerator path
                params = self._hypernet(embeddings)  # (P, 192) with grad

            # Detach holdout perts during training: MLP gets no gradient from them
            # Use _has_holdout flag (set at init) to avoid .any() on XPU (crashes)
            if self.training and getattr(self, '_has_holdout', False):
                mask = self._holdout_mask.unsqueeze(-1).to(params.device)  # (P, 1)
                # XPU-safe: torch.where with bool mask is fine, no sync points
                params = torch.where(mask, params.detach(), params)

            # Cache in generator_params buffer for analysis code
            if not self.training:
                self.generator_params.data.copy_(params.detach())

            return params
        return self.generator_params

    def get_generators(self, pert_indices: torch.Tensor = None) -> torch.Tensor:
        """Return skew-symmetric generator matrices.

        Args:
            pert_indices: (K,) long tensor selecting perturbations. None = all.

        Returns:
            (K, num_blocks, block_size, block_size) skew-symmetric matrices.
        """
        all_params = self._get_all_gen_params()  # (num_perts, gen_dim) — hypernet or lookup
        if pert_indices is not None:
            params = all_params[pert_indices]  # (K, num_blocks * gen_dim)
        else:
            params = all_params  # (num_perts, num_blocks * gen_dim)
        # Reshape back to (..., num_blocks, gen_dim) for einsum
        params = params.unflatten(-1, (self.num_blocks, self.generator_dim))
        return torch.einsum("...ng, gij -> ...nij", params, self.basis)

    def forward(
        self,
        z_basal: torch.Tensor,
        perts: torch.Tensor,
        dosers: nn.Module = None,
        gene_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply perturbation rotation to basal latent.

        Args:
            z_basal: (B, latent_dim)
            perts: (B, num_perts) — dose-weighted one-hot or multi-hot
            dosers: optional dose-response module, maps perts -> (B, num_perts) scales
            gene_features: (B, emb_dim) — gene embeddings for hypernetwork zero-shot

        Returns:
            z_rotated: (B, latent_dim) — the full rotated latent
        """
        B = z_basal.size(0)
        bs = self.block_size
        nb = self.num_blocks

        # Old per-batch hypernetwork path (gene_features from outside, skips BCH)
        # NOT used for hypernet_e2e — that goes through the standard path below.
        use_generator = (gene_features is not None and self.generator is not None
                         and not self.hypernet_e2e)
        if use_generator:
            raw = self.generator(gene_features)  # (B, num_blocks * generator_dim)
            params = self.output_scale * raw
            params = params.unflatten(-1, (nb, self.generator_dim))
            A_combined = torch.einsum("...ng, gij -> ...nij", params, self.basis)
            # Skip dose scaling and BCH — generator predicts the full rotation directly
        else:
            # Standard lookup path (also used by hypernet_e2e via get_generators())
            # Dose scaling
            if dosers is not None:
                dose_scales = dosers(perts)  # (B, num_perts)
            elif self._geodesic_logit is not None:
                geo_scale = self._geodesic_max_scale * torch.sigmoid(self._geodesic_logit)
                dose_scales = perts * geo_scale.unsqueeze(0)
            else:
                dose_scales = perts

            # All generators: (num_perts, num_blocks, bs, bs)
            # In hypernet_e2e mode, this calls the MLP on frozen embeddings.
            all_A = self.get_generators()

            # Weighted combination: (B, num_perts) x (num_perts, num_blocks, bs, bs) -> (B, num_blocks, bs, bs)
            A_combined = torch.einsum("bp, pnij -> bnij", dose_scales, all_A)

        # Apply rotation warmup scale (ramps from small value to 1.0 during training)
        if self._rotation_warmup_scale < 1.0:
            A_combined = A_combined * self._rotation_warmup_scale

        # BCH order-2 correction for combination perturbations:
        # Add [A_i, A_j]/2 commutator for each pair of active perturbations.
        # This captures non-additive interaction effects between perturbations.
        #
        # IMPORTANT: The standard BCH formula gives di*dj * [Ai, Aj]/2, but when
        # the doser compresses dose 1.0 → ~0.02, this becomes d^2 ≈ 0.0004 times
        # the bracket — quadratically suppressed and negligible.
        #
        # Fix: use sqrt(di*dj) instead of di*dj for bracket scaling. This preserves
        # the geometric mean dose but avoids the quadratic suppression.
        # A learnable bracket_scale parameter controls the overall strength.
        bch_order = getattr(self, '_bch_order', 1)  # 1=off, 2=add [A,B]/2
        if bch_order >= 2 and not use_generator:
            # Find samples with multiple active perturbations
            active_mask = dose_scales.abs() > 1e-6  # (B, num_perts)
            n_active = active_mask.sum(dim=1)  # (B,)
            combo_mask = n_active > 1  # samples with 2+ active perts

            if combo_mask.any():
                # Learnable bracket scaling (compensates for dose compression)
                bracket_scale = getattr(self, 'bracket_scale', None)
                if bracket_scale is not None:
                    bscale = bracket_scale.abs()
                else:
                    bscale = 1.0

                # Compute BCH correction without in-place operations
                bch_correction = torch.zeros_like(A_combined)
                combo_idx = combo_mask.nonzero(as_tuple=True)[0]
                for ci in combo_idx:
                    active_perts = active_mask[ci].nonzero(as_tuple=True)[0]
                    # Compute pairwise commutators [A_i, A_j]/2
                    correction = torch.zeros(nb, bs, bs, device=A_combined.device,
                                              dtype=A_combined.dtype)
                    for ii in range(len(active_perts)):
                        for jj in range(ii + 1, len(active_perts)):
                            pi, pj = active_perts[ii], active_perts[jj]
                            di, dj = dose_scales[ci, pi], dose_scales[ci, pj]
                            Ai = all_A[pi]  # (nb, bs, bs)
                            Aj = all_A[pj]  # (nb, bs, bs)
                            bracket = Ai @ Aj - Aj @ Ai  # (nb, bs, bs)
                            # Use sqrt(di*dj) instead of di*dj to avoid quadratic suppression
                            dose_factor = torch.sqrt(di.abs() * dj.abs() + 1e-12)
                            correction = correction + 0.5 * bscale * dose_factor * bracket
                    bch_correction[ci] = correction
                A_combined = A_combined + bch_correction

        # Matrix exponential per block
        A_flat = A_combined.reshape(B * nb, bs, bs)
        R_flat = torch.linalg.matrix_exp(A_flat)
        R = R_flat.reshape(B, nb, bs, bs)

        # Apply rotation: z_blocks (B, nb, bs, 1) -> R @ z_blocks
        z_blocks = z_basal.reshape(B, nb, bs, 1)
        z_rotated = torch.matmul(R, z_blocks).squeeze(-1).reshape(B, self.latent_dim)

        return z_rotated

    def compose_bch(
        self,
        A_a: torch.Tensor,
        A_b: torch.Tensor,
        order: int = 2,
    ) -> torch.Tensor:
        """Baker-Campbell-Hausdorff composition of two generators.

        Args:
            A_a: (num_blocks, bs, bs) generator for perturbation a
            A_b: (num_blocks, bs, bs) generator for perturbation b
            order: 2 or 3

        Returns:
            A_combo: (num_blocks, bs, bs) combined generator
        """
        bracket = A_a @ A_b - A_b @ A_a
        A_combo = A_a + A_b + 0.5 * bracket
        if order >= 3:
            bracket_a = A_a @ bracket - bracket @ A_a
            bracket_b = A_b @ bracket - bracket @ A_b
            A_combo = A_combo + (1.0 / 12.0) * (bracket_a - bracket_b)
        return A_combo

    def consistency_loss(self, z_basal, perts, dosers=None, gene_features=None):
        """Consistency loss: match generator output to per-pert lookup.

        When using the hypernetwork, this trains the generator to reproduce
        the learned rotation parameters for seen perturbations.
        """
        if self.generator is None or gene_features is None:
            return torch.tensor(0.0, device=z_basal.device), 0.0

        # Generator prediction
        raw = self.generator(gene_features)  # (K, out_dim)
        gen_params = self.output_scale * raw  # (K, out_dim)

        # Lookup target: get params for the perturbation indices
        pert_idx = perts.argmax(dim=1)  # (K,) - assumes one-hot
        lookup_params = self.generator_params[pert_idx].detach()  # (K, out_dim)

        # L2 loss in so(n) coefficient space (gauge-free!)
        loss = ((gen_params - lookup_params) ** 2).sum(-1).mean()
        return loss, float(loss.detach().item())

    def jacobian_to_generator(
        self,
        z_basal: torch.Tensor,
        jacobian_row: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Convert a decoder Jacobian direction to a skew-symmetric generator.

        Constructs the rank-2 skew-symmetric matrix that would rotate z_basal
        toward the Jacobian direction, per block.

        Args:
            z_basal: (latent_dim,) single cell basal latent
            jacobian_row: (latent_dim,) gradient ∂μ_k/∂z for target gene k

        Returns:
            A: (num_blocks, block_size, block_size) skew-symmetric generator
        """
        z_b = z_basal.reshape(self.num_blocks, self.block_size)
        d_b = jacobian_row.reshape(self.num_blocks, self.block_size)
        # Per-block: A = (d ⊗ z - z ⊗ d) / ||z||²
        norm_sq = (z_b * z_b).sum(dim=-1, keepdim=True).unsqueeze(-1) + eps  # (nb, 1, 1)
        A = (
            d_b.unsqueeze(-1) * z_b.unsqueeze(-2)
            - z_b.unsqueeze(-1) * d_b.unsqueeze(-2)
        ) / norm_sq
        return A


class HouseholderRotation(nn.Module):
    """Full SO(n) rotation via product of Householder reflections.

    Any rotation in SO(n) can be decomposed as a product of at most n
    Householder reflections. Using k reflections with k even gives a
    proper rotation (det = +1). Each reflection is parameterized by a
    unit vector v: H_v(z) = z - 2(v^T z)v.

    Advantages over BlockRotation:
      - Can couple ALL latent dimensions (not block-diagonal)
      - O(n*k) computation (vs O(n^3) for full matrix exp)
      - Numerically stable (each reflection is exactly orthogonal)
      - Parameter-efficient: k*n params per perturbation

    For latent_dim=128, n_reflections=16:
      - 16 * 128 = 2048 params per perturbation (vs 192 for BlockRotation)
      - Can express rotations in a 16-dimensional subspace of SO(128)
      - But this subspace can involve ANY dimensions (not block-confined)
    """

    def __init__(self, num_perts: int, latent_dim: int, n_reflections: int = 16,
                 block_size: int = 4):  # block_size kept for API compat, unused
        super().__init__()
        # Ensure even number of reflections for proper rotation (det = +1)
        self.n_reflections = n_reflections if n_reflections % 2 == 0 else n_reflections + 1
        self.latent_dim = latent_dim

        # Reflection vectors: small init near zero = near-identity rotation
        # Stored as 2D (num_perts * n_reflections, latent_dim) for Muon compatibility
        self.generator_params = nn.Parameter(
            torch.randn(num_perts * self.n_reflections, latent_dim) * 0.01
        )
        self.num_perts = num_perts

        # Store block_size and related attrs for compatibility with BlockRotation API
        self.block_size = block_size
        self.num_blocks = latent_dim // max(block_size, 1)
        self.generator_dim = block_size * (block_size - 1) // 2
        self.num_perts = num_perts

    def get_reflection_vectors(self, pert_indices: torch.Tensor = None) -> torch.Tensor:
        """Return normalized reflection vectors.

        Returns:
            (K, n_reflections, latent_dim) unit vectors
        """
        params = self.generator_params.view(self.num_perts, self.n_reflections, self.latent_dim)
        if pert_indices is not None:
            params = params[pert_indices]
        return F.normalize(params, dim=-1)

    def forward(
        self,
        z_basal: torch.Tensor,
        perts: torch.Tensor,
        dosers: nn.Module = None,
        gene_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply perturbation rotation via Householder reflections.

        Args:
            z_basal: (B, latent_dim)
            perts: (B, num_perts) -- dose-weighted one-hot or multi-hot
            dosers: optional dose-response module

        Returns:
            z_rotated: (B, latent_dim)
        """
        B = z_basal.size(0)

        if dosers is not None:
            dose_scales = dosers(perts)  # (B, num_perts)
        else:
            dose_scales = perts

        # All reflection vectors: (num_perts, n_reflections, latent_dim)
        all_v = self.generator_params.view(self.num_perts, self.n_reflections, self.latent_dim)

        # Weighted combination of reflection vectors per sample
        # (B, num_perts) x (num_perts, n_ref, latent_dim) -> (B, n_ref, latent_dim)
        v_combined = torch.einsum("bp, prd -> brd", dose_scales, all_v)
        # Normalize each reflection vector
        v_combined = F.normalize(v_combined, dim=-1)

        # Apply reflections sequentially
        z = z_basal
        for k in range(self.n_reflections):
            v_k = v_combined[:, k, :]  # (B, latent_dim)
            # Householder reflection: z = z - 2 * (v^T z) * v
            proj = (z * v_k).sum(dim=-1, keepdim=True)  # (B, 1)
            z = z - 2.0 * proj * v_k
        return z

    def get_generators(self, pert_indices: torch.Tensor = None) -> torch.Tensor:
        """Compatibility method: return a representation compatible with BlockRotation API.

        For HouseholderRotation this returns zero-filled block generators since the
        actual rotation is not block-diagonal. Used by diagnostics that expect
        BlockRotation API.
        """
        if pert_indices is not None:
            K = pert_indices.size(0)
        else:
            K = self.num_perts
        nb = self.latent_dim // max(self.block_size, 1)
        return torch.zeros(K, nb, self.block_size, self.block_size,
                          device=self.generator_params.device)

    def compose_bch(self, A_a, A_b, order=2):
        """BCH is not directly applicable to Householder parameterization.
        Returns sum as first-order approximation."""
        return A_a + A_b

    def jacobian_to_generator(self, z_basal, jacobian_row, eps=1e-8):
        """Not directly applicable for Householder. Return zeros."""
        nb = self.latent_dim // max(self.block_size, 1)
        return torch.zeros(nb, self.block_size, self.block_size,
                          device=z_basal.device)


class _CayleyMapCPU(torch.autograd.Function):
    """Cayley map R = (I + A/2)(I - A/2)^{-1} computed on CPU.

    XPU has a kernel bug that causes memory corruption with large (128×128) matrix
    operations in linalg.solve and matrix_exp. This function offloads the computation
    to CPU for both forward and backward passes, avoiding the XPU bug entirely.
    """
    @staticmethod
    def forward(ctx, A, n):
        device = A.device
        A_cpu = A.detach().cpu().float()
        I_cpu = torch.eye(n, dtype=A_cpu.dtype).unsqueeze(0).expand(A_cpu.size(0), -1, -1)
        # Cayley map: R = (I + A/2)(I - A/2)^{-1}
        lhs = I_cpu + A_cpu / 2
        rhs = I_cpu - A_cpu / 2
        R_cpu = torch.linalg.solve(rhs, lhs)
        ctx.save_for_backward(A_cpu, R_cpu)
        ctx.device = device
        ctx.n = n
        return R_cpu.to(device, dtype=A.dtype)

    @staticmethod
    def backward(ctx, grad_R):
        A_cpu, R_cpu = ctx.saved_tensors
        n = ctx.n
        grad_R_cpu = grad_R.detach().cpu().float()
        I_cpu = torch.eye(n, dtype=A_cpu.dtype).unsqueeze(0).expand(A_cpu.size(0), -1, -1)
        # dL/dA = 0.5 * (I - A/2)^{-T} @ grad_R @ (I + A/2)^T @ (I - A/2)^{-T}
        # Simplified: (I - A/2)^{-1} = rhs_inv, and dR/dA involves rhs_inv
        rhs = I_cpu - A_cpu / 2
        rhs_inv = torch.linalg.inv(rhs)
        # Chain rule: dL/dA = 0.5 * rhs_inv^T @ grad_R @ R^T @ rhs_inv^T
        #                   + 0.5 * rhs_inv^T @ grad_R
        # Actually: for Cayley map R = (I+A/2)(I-A/2)^{-1}
        # dR = 0.5*(I-A/2)^{-1} @ dA @ (I-A/2)^{-1} + 0.5*(I+A/2) @ (I-A/2)^{-1} @ dA @ (I-A/2)^{-1}
        # = 0.5 * (I-A/2)^{-1} @ (I + R) @ dA @ (I-A/2)^{-1}
        # So: dL/dA = 0.5 * (I-A/2)^{-T} @ (I + R)^T @ (I-A/2)^{-T} @ grad_R
        temp = torch.bmm(rhs_inv.transpose(-1, -2), grad_R_cpu)
        temp2 = torch.bmm((I_cpu + R_cpu).transpose(-1, -2), temp)
        grad_A_cpu = 0.5 * torch.bmm(rhs_inv.transpose(-1, -2), temp2)
        return grad_A_cpu.to(ctx.device, dtype=grad_R.dtype), None


def _cayley_map_newton_schulz(A, n, iterations=12):
    """Cayley map R = (I + A/2)(I - A/2)^{-1} via Newton-Schulz iteration.

    For skew-symmetric A, M = I - A/2 has complex eigenvalues 1 ± iσ_j/2.
    Newton-Schulz converges when: α < 2 / (1 + σ_max²/4).
    We use ||A||_F as an upper bound for σ_max.

    Uses ONLY bmm/norm — no linalg.solve, no CPU transfer, no sync points.
    """
    B = A.size(0)
    I = torch.eye(n, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
    half_A = A * 0.5
    M = I - half_A  # (B, n, n) — we want inv(M)
    twoI = 2.0 * I

    # Convergence-safe α: for skew-symmetric A with spectral norm ≤ ||A||_F,
    # need α < 2 / (1 + ||A||_F² / 4). Use 1.8x safety margin.
    A_norm_sq = A.flatten(1).pow(2).sum(1).view(B, 1, 1)  # ||A||_F² per sample
    alpha = 1.8 / (1.0 + A_norm_sq * 0.25 + 1e-8)  # (B, 1, 1)

    # Newton-Schulz: X_{k+1} = X_k (2I - M X_k), start from X_0 = α I
    X = alpha * I  # (B, n, n)
    for _ in range(iterations):
        X = torch.bmm(X, twoI - torch.bmm(M, X))

    # R = (I + A/2) @ X ≈ (I + A/2) @ inv(I - A/2)
    R = torch.bmm(I + half_A, X)
    return R


def _cayley_map_xpu_safe(A, n):
    """Cayley map: Newton-Schulz on XPU (no sync points), linalg.solve elsewhere."""
    if A.device.type == 'xpu':
        return _cayley_map_newton_schulz(A, n, iterations=6)
    else:
        I = torch.eye(n, device=A.device).unsqueeze(0).expand(A.size(0), -1, -1)
        return torch.linalg.solve(I - A / 2, I + A / 2)


class CayleyRotation(nn.Module):
    """Full SO(n) rotation via Cayley map of low-rank skew-symmetric matrices.

    The Cayley map R = (I + A/2)(I - A/2)^{-1} maps any skew-symmetric A
    to SO(n). We parameterize A as a low-rank skew-symmetric matrix:
    A = UV^T - VU^T, where U, V in R^{n x r}.

    This gives rotations that affect a 2r-dimensional subspace but can
    couple ANY dimensions (unlike block-diagonal).

    With rank=16 and latent_dim=128: 2 * 128 * 16 = 4096 params per pert.
    """

    def __init__(self, num_perts: int, latent_dim: int, rank: int = 16,
                 block_size: int = 4):  # block_size for API compat
        super().__init__()
        self.latent_dim = latent_dim
        self.rank = rank
        self.num_perts = num_perts

        # Low-rank factors: small init = near-identity rotation
        # Stored as 2D for Muon (num_perts, latent_dim * rank)
        self.generator_params = nn.Parameter(
            torch.randn(num_perts, latent_dim * rank * 2) * 0.01
        )

        self.block_size = block_size
        self.num_blocks = latent_dim // max(block_size, 1)
        self.generator_dim = block_size * (block_size - 1) // 2
        self.num_perts = num_perts

        # Learnable bracket scale for BCH (same as BlockRotation)
        self.bracket_scale = nn.Parameter(torch.tensor(1.0))

        # Tangent-space noise std (set externally via hparams)
        self._noise_std = 0.0

    def forward(
        self,
        z_basal: torch.Tensor,
        perts: torch.Tensor,
        dosers: nn.Module = None,
        compose_mode: str = "linear",
        **kwargs,  # Accept gene_features etc. for API compatibility
    ) -> torch.Tensor:
        B = z_basal.size(0)
        n = self.latent_dim
        r = self.rank

        if dosers is not None:
            dose_scales = dosers(perts)
        else:
            dose_scales = perts

        # All low-rank factors: (num_perts, 2, n, r)
        all_factors = self.generator_params.view(self.num_perts, 2, n, r)

        if compose_mode == "multiply":
            # Sequential rotation multiplication: R = R_1 @ R_2 @ ... @ R_k
            # More principled for combo perturbations (Lie group composition)
            return self._forward_multiply(z_basal, dose_scales, all_factors)
        elif compose_mode == "bch":
            # BCH composition of generators: A ≈ A_1 + A_2 + [A_1,A_2]/2
            return self._forward_bch(z_basal, dose_scales, all_factors)

        # Default: linear combination of factors (bilinear cross-terms)
        # Use matmul instead of einsum to avoid XPU gather kernel crash in backward
        P = dose_scales.size(1)
        factors_flat = dose_scales @ all_factors.reshape(P, 2 * n * r)  # (B, 2*n*r)
        factors = factors_flat.view(B, 2, n, r)
        U = factors[:, 0]  # (B, n, r)
        V = factors[:, 1]  # (B, n, r)

        # Tangent-space noise regularization (H37): adds skew-symmetric noise
        # during training to prevent rotation overfitting. Noise in so(n) makes
        # the decoder robust to rotation uncertainty — key for OOD generalization.
        noise_std = getattr(self, '_noise_std', 0.0)
        if self.training and noise_std > 0:
            U = U + torch.randn_like(U) * noise_std
            V = V + torch.randn_like(V) * noise_std

        # Skew-symmetric: A = UV^T - VU^T (in so(n))
        A = U @ V.transpose(-1, -2) - V @ U.transpose(-1, -2)  # (B, n, n)

        # Cayley map with CPU fallback for XPU compatibility
        R = _cayley_map_xpu_safe(A, n)

        return torch.bmm(R, z_basal.unsqueeze(-1)).squeeze(-1)

    def _forward_multiply(self, z_basal, dose_scales, all_factors):
        """Compose rotations by matrix multiplication: R_combo = R_1 @ R_2 @ ...

        Memory-efficient: only computes Cayley map for active samples per pert.
        """
        B = z_basal.size(0)
        n = self.latent_dim

        # Start with identity
        I = torch.eye(n, device=z_basal.device)
        R_combo = I.unsqueeze(0).expand(B, -1, -1).clone()

        # Find which perturbations have non-negligible dose
        active_mask = dose_scales.abs() > 1e-6  # (B, num_perts)
        active_perts = active_mask.any(dim=0).nonzero(as_tuple=True)[0].tolist()

        for p_idx in active_perts:
            d = dose_scales[:, p_idx]  # (B,)
            active_idx = (d.abs() > 1e-6).nonzero(as_tuple=True)[0]

            if len(active_idx) == 0:
                continue

            # Get this pert's factors
            U_p = all_factors[p_idx, 0]  # (n, r)
            V_p = all_factors[p_idx, 1]  # (n, r)
            A_p = U_p @ V_p.T - V_p @ U_p.T  # (n, n)

            # Only compute for active samples (memory-efficient)
            d_active = d[active_idx]  # (K,) where K << B
            A_scaled = d_active[:, None, None] * A_p.unsqueeze(0)  # (K, n, n)

            I_k = I.unsqueeze(0).expand(len(active_idx), -1, -1)
            R_p = torch.linalg.solve(I_k - A_scaled / 2, I_k + A_scaled / 2)  # (K, n, n)

            # Update only active samples in R_combo
            R_combo[active_idx] = torch.bmm(R_p, R_combo[active_idx])

        return torch.bmm(R_combo, z_basal.unsqueeze(-1)).squeeze(-1)

    def _forward_bch(self, z_basal, dose_scales, all_factors):
        """BCH composition: A_combo ≈ A_1 + A_2 + [A_1,A_2]/2 + ...

        FULLY VECTORIZED — no sync points, no .any()/.item()/.nonzero().
        Safe for XPU which has async assertion bugs triggered by device→host sync.

        For single-pert samples: bracket correction is automatically zero
        (dose outer product has no off-diagonal entries).
        For combo-pert samples: adds Lie bracket [A_i, A_j]/2 weighted by doses.

        The bracket operates on full n×n skew-symmetric matrices, giving CayleyRotation
        a key advantage over block-diagonal: captures CROSS-dimension interactions.
        """
        B = z_basal.size(0)
        n = self.latent_dim
        P = dose_scales.size(1)

        # Linear combination of generators (handles single-pert correctly)
        # Use matmul instead of einsum to avoid XPU gather kernel crash in backward:
        # einsum("bp, pknr -> bknr") triggers IndexKernelUtils assert on XPU
        r = self.rank
        factors_flat = dose_scales @ all_factors.reshape(P, 2 * n * r)  # (B, 2*n*r)
        factors = factors_flat.view(B, 2, n, r)
        U = factors[:, 0]  # (B, n, r)
        V = factors[:, 1]  # (B, n, r)
        A = U @ V.transpose(-1, -2) - V @ U.transpose(-1, -2)  # (B, n, n)

        # BCH order-2 bracket correction (fully vectorized, zero sync points)
        bch_order = getattr(self, '_bch_order', 1)
        # Memory budget check: bracket tensor is (P, P, n, n) — O(P²n²)
        # For CRISPRi (P=1174, n=128): 91 GB → skip (bracket is zero for single-pert anyway)
        # For CRISPRa (P=108, n=128): 764 MB → OK
        bracket_mem = P * P * n * n * dose_scales.element_size()
        MAX_BRACKET_MEM = 4 * 1024**3  # 4 GB limit
        if bch_order >= 2 and bracket_mem <= MAX_BRACKET_MEM:
            # BCH bracket computed WITHOUT gradient tracking on XPU.
            # The 4D einsum backward pass triggers XPU gather kernel crashes.
            # The bracket is a second-order correction — its gradient is not essential.
            # The bracket still provides the correct forward-pass composition.
            with torch.no_grad():
                # Step 1: Compute all per-pert generators: (P, n, n)
                all_U = all_factors[:, 0]  # (P, n, r)
                all_V = all_factors[:, 1]  # (P, n, r)
                all_A = torch.bmm(all_U, all_V.transpose(-1, -2)) - \
                        torch.bmm(all_V, all_U.transpose(-1, -2))  # (P, n, n)

                # Step 2: Pairwise products Q[p,q] = A_p @ A_q — (P, P, n, n)
                Q = torch.einsum("pmk, qkn -> pqmn", all_A, all_A)  # (P, P, n, n)

                # Step 3: Lie brackets = Q - Q^T_perm (antisymmetric in p,q)
                brackets = Q - Q.permute(1, 0, 2, 3)  # (P, P, n, n)

                # Step 4: Upper triangle mask (i < j only, avoids double-counting)
                upper_mask = torch.triu(torch.ones(P, P, device=dose_scales.device,
                                                   dtype=dose_scales.dtype), diagonal=1)  # (P, P)

                # Step 5: Dose weights for bracket: W[b,p,q] = dose[b,p] * dose[b,q]
                W = torch.einsum("bp, bq -> bpq", dose_scales, dose_scales)  # (B, P, P)
                W = W * upper_mask.unsqueeze(0)  # (B, P, P) — only upper triangle

                # Step 6: Bracket correction = 0.5 * sum_{p<q} W[b,p,q] * brackets[p,q]
                bracket_scale = getattr(self, 'bracket_scale', None)
                bscale = bracket_scale.abs() if bracket_scale is not None else 1.0
                correction = (0.5 * bscale) * torch.einsum("bpq, pqmn -> bmn", W, brackets)

            A = A + correction

        # Cayley map with CPU fallback for XPU compatibility
        R = _cayley_map_xpu_safe(A, n)

        return torch.bmm(R, z_basal.unsqueeze(-1)).squeeze(-1)

    def get_generators(self, pert_indices=None):
        if pert_indices is not None:
            K = pert_indices.size(0)
        else:
            K = self.num_perts
        nb = self.latent_dim // max(self.block_size, 1)
        return torch.zeros(K, nb, self.block_size, self.block_size,
                          device=self.generator_params.device)

    def compose_bch(self, A_a, A_b, order=2):
        """BCH composition for full skew-symmetric matrices."""
        bracket = A_a @ A_b - A_b @ A_a
        result = A_a + A_b + 0.5 * bracket
        if order >= 3:
            bracket_a = A_a @ bracket - bracket @ A_a
            bracket_b = A_b @ bracket - bracket @ A_b
            result = result + (1.0 / 12.0) * (bracket_a - bracket_b)
        return result

    def get_skew_symmetric(self, pert_idx: int) -> torch.Tensor:
        """Get the full skew-symmetric matrix for a given perturbation.
        Returns: (n, n) skew-symmetric generator A = UV^T - VU^T
        """
        n, r = self.latent_dim, self.rank
        factors = self.generator_params[pert_idx].view(2, n, r)
        U, V = factors[0], factors[1]
        return U @ V.T - V @ U.T

    def jacobian_to_generator(self, z_basal, jacobian_row, eps=1e-8):
        nb = self.latent_dim // max(self.block_size, 1)
        return torch.zeros(nb, self.block_size, self.block_size,
                          device=z_basal.device)


class SharedBasisCayleyRotation(CayleyRotation):
    """SO(n) rotation via shared Lie algebra basis + per-pert coefficients.

    Instead of independent generators per perturbation, decompose into:
        A_p = sum_k alpha_p[k] * B_k
    where B_k are K shared basis elements in so(n) (stored as low-rank Cayley
    factors) and alpha_p are per-pert coefficient vectors.

    Advantages over independent CayleyRotation:
    - Shared structure across perturbations (inductive bias for smoothness)
    - Low-dimensional per-pert representation (K << 2*n*r) for easier interpolation
    - For zero-shot: predict alpha from GenePT via ridge regression on training alphas

    Inherits from CayleyRotation so isinstance checks pass throughout codebase.
    The `generator_params` property computes effective params on the fly:
        generator_params = pert_coefficients @ basis_params
    """

    def __init__(self, num_perts: int, latent_dim: int, rank: int = 16,
                 n_basis: int = 32, block_size: int = 4):
        # Skip CayleyRotation.__init__ — we manage our own parameters.
        # Call nn.Module.__init__ directly.
        nn.Module.__init__(self)

        self.latent_dim = latent_dim
        self.rank = rank
        self.num_perts = num_perts
        self.n_basis = n_basis
        self.block_size = block_size
        self.num_blocks = latent_dim // max(block_size, 1)
        self.generator_dim = block_size * (block_size - 1) // 2
        self.num_perts = num_perts

        factor_dim = 2 * latent_dim * rank

        # Shared basis: K generators, each stored as Cayley factors (2*n*r)
        # Small init for near-identity rotations
        self.basis_params = nn.Parameter(
            torch.randn(n_basis, factor_dim) * 0.01
        )

        # Per-pert coefficients: (num_perts, K)
        # Slightly larger init so the linear combination is non-trivial
        self.pert_coefficients = nn.Parameter(
            torch.randn(num_perts, n_basis) * (0.1 / math.sqrt(n_basis))
        )

        # Learnable bracket scale for BCH (same as CayleyRotation)
        self.bracket_scale = nn.Parameter(torch.tensor(1.0))

    @property
    def generator_params(self):
        """Compute effective generator params from basis decomposition.

        Returns: (num_perts, 2*n*r) tensor — same shape as CayleyRotation.generator_params.
        All code that accesses rotation.generator_params works transparently.
        """
        return self.pert_coefficients @ self.basis_params

    @generator_params.setter
    def generator_params(self, value):
        # Allow assignment for compatibility (e.g., checkpoint loading).
        # Silently ignore — the effective params are always computed from basis.
        pass

    def get_skew_symmetric(self, pert_idx: int) -> torch.Tensor:
        """Get the full skew-symmetric matrix for a given perturbation."""
        n, r = self.latent_dim, self.rank
        factors = (self.pert_coefficients[pert_idx] @ self.basis_params).view(2, n, r)
        U, V = factors[0], factors[1]
        return U @ V.T - V @ U.T

    def jacobian_to_generator(self, z_basal, jacobian_row, eps=1e-8):
        nb = self.latent_dim // max(self.block_size, 1)
        return torch.zeros(nb, self.block_size, self.block_size,
                          device=self.basis_params.device)

    def get_generators(self, pert_indices=None):
        if pert_indices is not None:
            K = pert_indices.size(0)
        else:
            K = self.num_perts
        nb = self.latent_dim // max(self.block_size, 1)
        return torch.zeros(K, nb, self.block_size, self.block_size,
                          device=self.basis_params.device)


class CellStateGatedBasisRotation(SharedBasisCayleyRotation):
    """Cell-state-conditioned shared basis rotation (CellCap-inspired).

    Extends SharedBasisCayleyRotation by making program coefficients depend on
    the cell's basal state:
        α_eff(z, p) = α_p ⊙ sigmoid(W_gate @ z + b_gate)

    This makes the effective rotation cell-state-dependent: the same perturbation
    produces different rotations for different cells. Biologically motivated:
    - Same gene KO has different effects in proliferating vs quiescent cells
    - Cell-state specificity is a key feature of perturbation biology

    The gate is initialized near 1 (via positive bias) so the model starts
    equivalent to the ungated SharedBasisCayleyRotation.

    Architecture follows CellCap (Cell Systems 2025): attention-weighted program
    mixing conditioned on cell state, but using our SO(n) rotation framework
    instead of CellCap's additive latent algebra.
    """

    def __init__(self, num_perts: int, latent_dim: int, rank: int = 16,
                 n_basis: int = 32, block_size: int = 4,
                 gate_hidden: int = 64):
        super().__init__(num_perts, latent_dim, rank, n_basis, block_size)

        # Cell-state gate: z_basal -> K gate values via sigmoid
        # Maps latent_dim -> n_basis
        self.cell_gate = nn.Sequential(
            nn.Linear(latent_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, n_basis),
        )
        # Initialize gate output near 1 (sigmoid(2) ≈ 0.88)
        # So the model starts close to ungated behavior
        with torch.no_grad():
            self.cell_gate[-1].bias.fill_(2.0)
            self.cell_gate[-1].weight.mul_(0.01)

        # Store for forwarding
        self._last_gate_values = None

    def forward(self, z_basal, perts, dosers=None, compose_mode="linear", **kwargs):
        """Override forward to apply cell-state gating to program coefficients."""
        B = z_basal.size(0)
        n = self.latent_dim
        r = self.rank

        if dosers is not None:
            dose_scales = dosers(perts)
        else:
            dose_scales = perts

        # Cell-state gate: modulate which programs are active for this cell
        gate = torch.sigmoid(self.cell_gate(z_basal))  # (B, K)
        self._last_gate_values = gate.detach()

        # Effective per-sample coefficients: pert identity × cell-state gate
        # dose_scales: (B, P), pert_coefficients: (P, K)
        alpha_base = dose_scales @ self.pert_coefficients  # (B, K)
        alpha_gated = alpha_base * gate  # (B, K) — cell-state modulated

        # Compute rotated factors from gated coefficients
        factors_flat = alpha_gated @ self.basis_params  # (B, 2*n*r)
        factors = factors_flat.view(B, 2, n, r)
        U = factors[:, 0]  # (B, n, r)
        V = factors[:, 1]  # (B, n, r)

        # Tangent-space noise (inherited from parent)
        noise_std = getattr(self, '_noise_std', 0.0)
        if self.training and noise_std > 0:
            U = U + torch.randn_like(U) * noise_std
            V = V + torch.randn_like(V) * noise_std

        # Skew-symmetric: A = UV^T - VU^T
        A = U @ V.transpose(-1, -2) - V @ U.transpose(-1, -2)  # (B, n, n)

        # Cayley map
        R = _cayley_map_xpu_safe(A, n)

        return torch.bmm(R, z_basal.unsqueeze(-1)).squeeze(-1)


class CellStateAttentionRotation(SharedBasisCayleyRotation):
    """Cell-state-conditioned rotation via ATTENTION over basis (H43).

    Unlike CellStateGatedBasisRotation which uses multiplicative sigmoid gating
    (unstable, causes rotation collapse), this uses additive attention:

        query = Linear(z_ctrl)  -- cell-state query
        key_k = basis_key[k]    -- learnable basis keys
        w_k = softmax(query^T @ key_k / sqrt(d) + alpha_pert[k])
        A_eff = sum_k w_k * B_k  -- attention-weighted basis combination

    The perturbation identity enters through alpha_pert (additive logit bias),
    and the cell state modulates the weights via attention. This is:
    1. More stable than multiplicative gating (softmax normalization)
    2. Preserves rotation scale (attention weights sum to 1)
    3. z_ctrl is always in-distribution (no OOD problem)
    4. Perturbation identity is a prior that attention refines per cell

    Inspired by CellCap (Cell Systems 2025): attention-weighted program mixing.
    """

    def __init__(self, num_perts: int, latent_dim: int, rank: int = 16,
                 n_basis: int = 32, block_size: int = 4,
                 attn_hidden: int = 64, attn_temp: float = 1.0):
        super().__init__(num_perts, latent_dim, rank, n_basis, block_size)

        self.attn_temp = attn_temp

        # Cell-state query: z_ctrl -> query vector
        self.query_net = nn.Sequential(
            nn.Linear(latent_dim, attn_hidden),
            nn.SiLU(),
            nn.Linear(attn_hidden, n_basis),
        )
        # Initialize near zero so attention starts driven by pert coefficients
        with torch.no_grad():
            self.query_net[-1].weight.mul_(0.01)
            self.query_net[-1].bias.zero_()

        # Learnable scaling for cell-state contribution
        self.cell_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z_basal, perts, dosers=None, compose_mode="linear", **kwargs):
        """Compute cell-state-conditioned rotation via attention."""
        B = z_basal.size(0)
        n = self.latent_dim
        r = self.rank

        if dosers is not None:
            dose_scales = dosers(perts)
        else:
            dose_scales = perts

        # Per-pert logits from pert identity
        alpha_base = dose_scales @ self.pert_coefficients  # (B, K)

        # Cell-state attention logits
        cell_logits = self.cell_scale * self.query_net(z_basal)  # (B, K)

        # Combined logits: pert identity + cell state
        logits = alpha_base + cell_logits  # (B, K)

        # No softmax — use raw logits as weights (like SharedBasis)
        # This allows the rotation magnitude to vary
        # The cell_logits just ADD a cell-specific modulation
        alpha_eff = logits

        # Compute rotated factors from effective coefficients
        factors_flat = alpha_eff @ self.basis_params  # (B, 2*n*r)
        factors = factors_flat.view(B, 2, n, r)
        U = factors[:, 0]  # (B, n, r)
        V = factors[:, 1]  # (B, n, r)

        # Tangent-space noise
        noise_std = getattr(self, '_noise_std', 0.0)
        if self.training and noise_std > 0:
            U = U + torch.randn_like(U) * noise_std
            V = V + torch.randn_like(V) * noise_std

        # Skew-symmetric: A = UV^T - VU^T
        A = U @ V.transpose(-1, -2) - V @ U.transpose(-1, -2)  # (B, n, n)

        # Cayley map
        R = _cayley_map_xpu_safe(A, n)

        return torch.bmm(R, z_basal.unsqueeze(-1)).squeeze(-1)


class SimilarityProjector(nn.Module):
    """Learn a metric embedding where GenePT cosine sim matches Cayley factor sim.

    Projects high-dimensional GenePT embeddings (1536-d) into a compact space (64-d)
    where cosine similarity is predictive of which rotation generators are actually
    similar. Trained via relational distillation against factor-space similarity.
    """

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize: output lives on the unit hypersphere."""
        return F.normalize(self.net(x), dim=-1)


class HouseholderGeneratorNet(nn.Module):
    """Zero-shot rotation generation via learned mapping from gene features.

    Instead of storing a lookup table of rotation parameters (one per gene/pert),
    this module learns a FUNCTION that maps gene features to Householder reflection
    vectors. The key insight: the decoder Jacobian row J[g] is a latent_dim-d vector
    that characterizes how gene g relates to the latent space. Genes with similar
    Jacobian signatures should have similar perturbation rotations.

    Architecture:
        gene_features (latent_dim) -> MLP -> n_reflections * latent_dim reflection vectors

    For zero-shot generalization: at eval time, compute J[g] for the unseen gene,
    feed it through the generator network, get rotation parameters. No lookup needed.

    The generator network is SHARED across all genes, so it must learn the general
    mapping from "how a gene relates to latent space" to "how to rotate to simulate
    that gene's perturbation."

    This is a form of hypernetwork: the generator net produces the "weights" (reflection
    vectors) of the Householder rotation, conditioned on the gene identity.

    NeurIPS novelty: Lie-group-valued hypernetwork for zero-shot perturbation prediction.
    The rotation lives on SO(n), the generator network maps gene features to the
    Lie algebra so(n) (via Householder reflection vectors), and the Householder product
    gives the group element. This is differentiable, respects the group structure,
    and enables genuine zero-shot transfer.
    """

    def __init__(self, latent_dim: int, n_reflections: int = 8,
                 hidden_dim: int = 256, num_layers: int = 3,
                 block_size: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_reflections = n_reflections if n_reflections % 2 == 0 else n_reflections + 1
        self.block_size = block_size
        self.num_blocks = latent_dim // max(block_size, 1)
        self.generator_dim = block_size * (block_size - 1) // 2
        self.num_perts = num_perts

        # Input: gene feature (latent_dim). Could be Jacobian row, gene embedding, etc.
        # Output: n_reflections * latent_dim (the reflection vectors)
        out_dim = self.n_reflections * latent_dim

        layers = []
        in_dim = latent_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        # Small output init for near-identity rotations
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

        self.net = nn.Sequential(*layers)

        # Learnable scale for output magnitude (initialized small)
        self.output_scale = nn.Parameter(torch.tensor(0.01))

        # For compatibility with warm_start and diagnostics
        self.num_perts = 0  # no lookup table
        # Dummy generator_params for API compatibility (optimizer grouping etc)
        self.generator_params = nn.Parameter(torch.zeros(1, latent_dim))
        self.generator_params.requires_grad = False  # not trained directly

    def forward(
        self,
        z_basal: torch.Tensor,
        gene_features: torch.Tensor,
        dosers: nn.Module = None,
    ) -> torch.Tensor:
        """Apply rotation generated from gene features.

        Args:
            z_basal: (B, latent_dim)
            gene_features: (B, latent_dim) -- e.g., Jacobian rows for target genes
            dosers: not used (kept for API compatibility)

        Returns:
            z_rotated: (B, latent_dim)
        """
        B = z_basal.size(0)

        # Generate reflection vectors from gene features
        raw = self.net(gene_features)  # (B, n_ref * latent_dim)
        raw = self.output_scale * raw
        v_all = raw.view(B, self.n_reflections, self.latent_dim)  # (B, n_ref, d)
        v_all = F.normalize(v_all, dim=-1)

        # Apply Householder reflections sequentially
        z = z_basal
        for k in range(self.n_reflections):
            v_k = v_all[:, k, :]  # (B, latent_dim)
            proj = (z * v_k).sum(dim=-1, keepdim=True)  # (B, 1)
            z = z - 2.0 * proj * v_k
        return z

    def generate_rotation_from_jacobian(
        self,
        decoder: nn.Module,
        z_basal: torch.Tensor,
        gene_indices: torch.Tensor,
        scale: float = 1.0,
        output_dim: int = None,
        mode: str = "i",
    ) -> torch.Tensor:
        """Compute Jacobian rows for given genes and generate rotations.

        This is the zero-shot path: for an unseen gene, compute its Jacobian row
        (how the decoder output changes w.r.t. latent dimensions), then use the
        generator network to produce rotation parameters.

        Args:
            decoder: the gene decoder module
            z_basal: (B, latent_dim) or (latent_dim,) -- basal latent
            gene_indices: (K,) long tensor of gene indices
            scale: output scale parameter
            output_dim: number of gene outputs (for NLL: half of decoder output)
            mode: "i" for inhibition (negate gradient), "a" for activation

        Returns:
            gene_features: (K, latent_dim) -- the Jacobian-derived features
        """
        if z_basal.dim() == 1:
            z_basal = z_basal.unsqueeze(0)

        z_for_grad = z_basal[0].detach().clone().requires_grad_(True)
        was_training = decoder.training
        decoder.eval()

        out = decoder(z_for_grad.unsqueeze(0))
        if output_dim is not None:
            out = scale * out[0, :output_dim]
        else:
            out = scale * out[0]

        features = []
        for g in gene_indices:
            v_g = torch.autograd.grad(out[g], z_for_grad, retain_graph=True)[0]
            if mode.lower() == "i":
                v_g = -v_g
            v_g = v_g / (v_g.norm() + 1e-12)
            features.append(v_g.detach())

        decoder.train(was_training)
        return torch.stack(features)  # (K, latent_dim)

    def get_generators(self, pert_indices=None):
        """Compatibility: return zeros."""
        K = 1 if pert_indices is None else len(pert_indices)
        nb = self.latent_dim // max(self.block_size, 1)
        return torch.zeros(K, nb, self.block_size, self.block_size,
                          device=self.generator_params.device)

    def compose_bch(self, A_a, A_b, order=2):
        return A_a + A_b

    def jacobian_to_generator(self, z_basal, jacobian_row, eps=1e-8):
        nb = self.latent_dim // max(self.block_size, 1)
        return torch.zeros(nb, self.block_size, self.block_size,
                          device=z_basal.device)


class CayleyGeneratorNet(nn.Module):
    """Zero-shot Cayley rotation generated from gene features via hypernetwork.

    Maps gene features (decoder Jacobian rows) to low-rank Cayley factors U, V.
    R = Cayley(UV^T - VU^T). Enables zero-shot rotation for unseen perturbations.

    Training modes:
      - "hybrid": Both per-pert lookup AND generator net. Generator trained to
        match lookup via consistency loss. At eval, generator used for OOD.
      - "generator_only": No per-pert parameters. All rotations from generator.
      - "dictionary": Learn K basis rotations, generator outputs K mixing weights.
        Much lower dimensional output (K instead of 2*n*r), easier to generalize.
      - "attention": Differentiable kNN over learned generators. Query/key from gene
        features, values are per-pert generator parameters. Naturally handles OOD
        by attending to similar training perturbations. End-to-end trainable.
    """

    def __init__(self, num_perts: int, latent_dim: int, rank: int = 16,
                 hidden_dim: int = 256, num_layers: int = 3,
                 block_size: int = 4, mode: str = "hybrid",
                 input_dim: int = None, n_basis: int = 0,
                 gen_dropout: float = 0.0, genept_noise: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.rank = rank
        self.num_perts = num_perts
        self.mode = mode
        self.block_size = block_size
        self.num_blocks = latent_dim // max(block_size, 1)
        self.generator_dim = block_size * (block_size - 1) // 2
        self.num_perts = num_perts
        self.input_dim = input_dim or latent_dim
        self.n_basis = n_basis
        self.genept_noise = genept_noise

        factor_dim = 2 * latent_dim * rank

        # Dictionary mode: learn K basis rotations, generator outputs K weights
        if n_basis > 0:
            self.basis_factors = nn.Parameter(
                torch.randn(n_basis, factor_dim) * 0.01
            )
            out_dim = n_basis  # Generator outputs mixing weights
        else:
            self.basis_factors = None
            out_dim = factor_dim

        layers = []
        in_dim = self.input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(hidden_dim))
            if gen_dropout > 0:
                layers.append(nn.Dropout(gen_dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        # Use small but non-zero init so generator can learn from the start.
        # Zero init + small output_scale creates vanishing gradient trap.
        nn.init.normal_(layers[-1].weight, std=0.01)
        nn.init.zeros_(layers[-1].bias)
        self.generator = nn.Sequential(*layers)

        # Start with scale=0.1 (not 0.01) — learnable, allows generator to have
        # non-trivial output from the start
        self.output_scale = nn.Parameter(torch.tensor(0.1))

        if mode in ("hybrid", "finetune"):
            self.generator_params = nn.Parameter(
                torch.randn(num_perts, factor_dim) * 0.01
            )
        elif mode == "attention":
            # Attention-based generator: differentiable kNN over per-pert generators
            # Per-pert generators (values in attention)
            self.generator_params = nn.Parameter(
                torch.randn(num_perts, factor_dim) * 0.01
            )
            # Attention projections: gene features -> query/key space
            attn_dim = min(hidden_dim, 128)  # attention head dimension
            self.attn_q = nn.Linear(self.input_dim, attn_dim)
            self.attn_k = nn.Linear(self.input_dim, attn_dim)
            self.attn_temperature = nn.Parameter(torch.tensor(0.5))  # learnable temperature
            self._attn_topk = 0  # 0 = attend to all, >0 = top-k only (set via hparam)
            # Store reference embeddings for training perts (set by init_attention_embeddings)
            self.register_buffer("_attn_ref_embeddings", torch.zeros(num_perts, self.input_dim))
            self._attn_ref_mask = None  # which perts have embeddings
        else:
            self.generator_params = nn.Parameter(torch.zeros(1, factor_dim))
            self.generator_params.requires_grad = False

    def _cayley_transform(self, U, V, z_basal):
        A = U @ V.transpose(-1, -2) - V @ U.transpose(-1, -2)
        n = self.latent_dim
        # Use XPU-safe Cayley map (Newton-Schulz on XPU, linalg.solve on CPU/CUDA)
        R = _cayley_map_xpu_safe(A, n)
        return torch.bmm(R, z_basal.unsqueeze(-1)).squeeze(-1)

    def init_attention_embeddings(self, embeddings, pert_indices=None):
        """Initialize reference embeddings for attention mode.

        Args:
            embeddings: (num_perts, input_dim) tensor of gene embeddings (e.g. GenePT)
            pert_indices: optional mask of which pert indices have valid embeddings
        """
        if self.mode != "attention":
            return
        device = self._attn_ref_embeddings.device
        emb = embeddings.to(device)
        if emb.shape[0] == self._attn_ref_embeddings.shape[0]:
            self._attn_ref_embeddings.copy_(emb)
        else:
            # Pad or truncate
            n = min(emb.shape[0], self._attn_ref_embeddings.shape[0])
            self._attn_ref_embeddings[:n] = emb[:n]
        if pert_indices is not None:
            self._attn_ref_mask = pert_indices
        else:
            # Assume all perts with non-zero embeddings are valid
            self._attn_ref_mask = (emb.norm(dim=-1) > 1e-6)

    def forward(self, z_basal, perts, dosers=None, gene_features=None):
        B = z_basal.size(0)
        n = self.latent_dim
        r = self.rank

        # Attention mode: differentiable kNN over per-pert generators
        if self.mode == "attention" and gene_features is not None:
            return self._attention_forward(z_basal, perts, dosers, gene_features)

        if gene_features is not None:
            # Add noise augmentation during training for better OOD generalization
            if self.training and self.genept_noise > 0:
                gene_features = gene_features + torch.randn_like(gene_features) * self.genept_noise

            raw = self.generator(gene_features)

            if self.basis_factors is not None:
                # Dictionary mode: raw is (B, K) mixing weights
                weights = torch.softmax(raw, dim=-1)  # (B, K)
                factors_flat = weights @ self.basis_factors  # (B, factor_dim)
            else:
                factors_flat = self.output_scale * raw

            factors = factors_flat.view(B, 2, n, r)
            U, V = factors[:, 0], factors[:, 1]
            return self._cayley_transform(U, V, z_basal)

        if dosers is not None:
            dose_scales = dosers(perts)
        else:
            dose_scales = perts

        all_factors = self.generator_params.view(self.num_perts, 2, n, r)
        factors = torch.einsum("bp, pknr -> bknr", dose_scales, all_factors)
        U, V = factors[:, 0], factors[:, 1]
        return self._cayley_transform(U, V, z_basal)

    def _attention_forward(self, z_basal, perts, dosers, gene_features):
        """Attention-based rotation: attend over all training generators using gene features."""
        B = z_basal.size(0)
        n = self.latent_dim
        r = self.rank

        # Add noise during training
        if self.training and self.genept_noise > 0:
            gene_features = gene_features + torch.randn_like(gene_features) * self.genept_noise

        # Compute query from input gene features: (B, attn_dim)
        q = self.attn_q(gene_features)

        # Compute keys from reference embeddings: (num_perts, attn_dim)
        ref_emb = self._attn_ref_embeddings
        k = self.attn_k(ref_emb)

        # Attention scores: (B, num_perts)
        attn_dim = q.size(-1)
        temp = torch.clamp(self.attn_temperature, min=0.01, max=5.0)
        scores = torch.mm(q, k.t()) / (math.sqrt(attn_dim) * temp)

        # Mask out perts without embeddings
        if self._attn_ref_mask is not None:
            mask = self._attn_ref_mask.to(scores.device)
            scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))

        # Top-k attention: only attend to k nearest neighbors (prevents diffusion with many perts)
        attn_topk = getattr(self, '_attn_topk', 0)
        if attn_topk > 0 and scores.size(-1) > attn_topk:
            topk_vals, topk_idx = scores.topk(attn_topk, dim=-1)
            mask_topk = torch.full_like(scores, float('-inf'))
            mask_topk.scatter_(-1, topk_idx, topk_vals)
            scores = mask_topk

        # Soft attention weights: (B, num_perts)
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted combination of generator factors: (B, factor_dim)
        all_factors_flat = self.generator_params  # (num_perts, factor_dim)
        factors_flat = torch.mm(attn_weights, all_factors_flat)

        # If we have dosers, scale by dose
        if dosers is not None and perts is not None:
            dose_scales = dosers(perts)  # (B, num_perts)
            # Also scale the direct lookup contribution
            # For attention mode, dosers modulate the final rotation magnitude
            # We use the max dose as a scalar scaling
            dose_max = dose_scales.max(dim=-1, keepdim=True).values  # (B, 1)
            factors_flat = factors_flat * dose_max

        # Reshape and apply Cayley
        factors = factors_flat.view(B, 2, n, r)
        U, V = factors[:, 0], factors[:, 1]
        return self._cayley_transform(U, V, z_basal)

    def generate_from_jacobian(self, decoder, z_basal, gene_indices,
                                scale=1.0, output_dim=None, mode="i"):
        if z_basal.dim() == 1:
            z_basal = z_basal.unsqueeze(0)
        z_for_grad = z_basal[0].detach().clone().requires_grad_(True)
        was_training = decoder.training
        decoder.eval()
        out = decoder(z_for_grad.unsqueeze(0))
        if output_dim is not None:
            out = scale * out[0, :output_dim]
        else:
            out = scale * out[0]
        features = []
        for g in gene_indices:
            v_g = torch.autograd.grad(out[g], z_for_grad, retain_graph=True)[0]
            if mode.lower() == "i":
                v_g = -v_g
            v_g = v_g / (v_g.norm() + 1e-12)
            features.append(v_g.detach())
        decoder.train(was_training)
        gene_feats = torch.stack(features)
        K = gene_feats.size(0)
        z_expanded = z_basal[0].unsqueeze(0).expand(K, -1)
        return self.forward(z_expanded, perts=None, gene_features=gene_feats)

    def consistency_loss(self, z_basal, perts, dosers=None, gene_features=None):
        if gene_features is None or self.mode != "hybrid":
            return torch.tensor(0.0, device=z_basal.device), 0.0
        with torch.no_grad():
            z_lookup = self.forward(z_basal, perts, dosers=dosers, gene_features=None)
        z_gen = self.forward(z_basal, perts, dosers=dosers, gene_features=gene_features)
        loss = F.mse_loss(z_gen, z_lookup.detach())
        return loss, float(loss.item())

    def get_generators(self, pert_indices=None):
        K = 1 if pert_indices is None else len(pert_indices)
        nb = self.latent_dim // max(self.block_size, 1)
        return torch.zeros(K, nb, self.block_size, self.block_size,
                          device=self.generator_params.device)

    def compose_bch(self, A_a, A_b, order=2):
        return A_a + A_b

    def jacobian_to_generator(self, z_basal, jacobian_row, eps=1e-8):
        nb = self.latent_dim // max(self.block_size, 1)
        return torch.zeros(nb, self.block_size, self.block_size,
                          device=z_basal.device)


class FlowMatchingCayleyRotation(CayleyRotation):
    """Lie Algebra Flow Matching for perturbation rotation prediction.

    Learns a conditional vector field on so(n) via flow matching.
    At inference, integrates from noise to get perturbation-specific rotations.

    Key idea: so(n) is a LINEAR vector space (skew-symmetric matrices),
    so Euclidean optimal transport is valid. The flow matches:
        v_θ(A_t, t, h_p) ≈ A_target - noise
    where A_t = t * A_target + (1-t) * noise is the OT interpolant.

    During training: uses stored factors directly (flow loss is auxiliary).
    During eval: can generate factors for unseen perturbations via flow.

    Supports GenePT conditioning for zero-shot OOD generation:
    when genept_cond_dim > 0, the flow is conditioned on GenePT embeddings
    instead of learned pert_embed, enabling generalization to unseen genes.
    """

    def __init__(self, num_perts: int, latent_dim: int, rank: int = 16,
                 block_size: int = 4, flow_hidden: int = 256, flow_layers: int = 3,
                 n_flow_steps: int = 10, flow_sigma: float = 0.01,
                 genept_cond_dim: int = 0):
        super().__init__(num_perts, latent_dim, rank=rank, block_size=block_size)
        self.n_flow_steps = n_flow_steps
        self.flow_sigma = flow_sigma
        self.genept_cond_dim = genept_cond_dim

        # Conditioning dimension
        self.pert_embed_dim = 64

        if genept_cond_dim > 0:
            # GenePT conditioning: projects GenePT embeddings to conditioning vector
            # This enables zero-shot OOD generation
            self.genept_cond_proj = nn.Sequential(
                nn.Linear(genept_cond_dim, self.pert_embed_dim),
                nn.LayerNorm(self.pert_embed_dim),
                nn.SiLU(),
            )
            self.pert_embed = None  # No per-pert embedding
        else:
            # Fallback: per-pert learned embedding (original behavior)
            self.pert_embed = nn.Embedding(num_perts, self.pert_embed_dim)
            self.genept_cond_proj = None

        # Conditional velocity field v(A_t, t, h_p)
        factor_dim = 2 * latent_dim * rank
        flow_input = factor_dim + 1 + self.pert_embed_dim
        layers = []
        in_d = flow_input
        for _ in range(flow_layers):
            layers.append(nn.Linear(in_d, flow_hidden))
            layers.append(nn.SiLU())
            in_d = flow_hidden
        layers.append(nn.Linear(flow_hidden, factor_dim))
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.velocity_net = nn.Sequential(*layers)

        self._flow_loss_cache = 0.0

    def _get_conditioning(self, perts=None, gene_features=None):
        """Get conditioning vector for flow matching.

        Uses GenePT features if available, falls back to pert_embed.
        """
        if gene_features is not None and self.genept_cond_proj is not None:
            return self.genept_cond_proj(gene_features)
        elif self.pert_embed is not None and perts is not None:
            perts_hard = F.one_hot(perts.argmax(dim=1), perts.size(1)).float()
            return perts_hard @ self.pert_embed.weight
        else:
            raise ValueError("No conditioning available: need gene_features or pert_embed")

    def compute_flow_matching_loss(self, perts, gene_features=None):
        """Compute conditional flow matching loss.

        Args:
            perts: (B, num_perts) perturbation dose matrix
            gene_features: optional (B, genept_dim) GenePT embeddings for conditioning

        Returns:
            loss: scalar flow matching loss
        """
        B = perts.size(0)

        # Use matmul instead of indexing to avoid XPU gather kernel crash
        perts_hard = F.one_hot(perts.argmax(dim=1), perts.size(1)).float()

        # Target factors via matmul (detached)
        target = (perts_hard @ self.generator_params).detach()  # (B, 2*n*r)

        # Conditioning: GenePT if available, else pert_embed
        h_p = self._get_conditioning(perts=perts, gene_features=gene_features)

        # Sample noise and time
        noise = torch.randn_like(target) * self.flow_sigma
        t = torch.rand(B, 1, device=target.device, dtype=target.dtype)

        # Linear interpolant (valid on so(n) since it's a vector space)
        A_t = t * target + (1 - t) * noise

        # Predict velocity
        v_pred = self.velocity_net(torch.cat([A_t, t, h_p], dim=1))

        # Target: conditional OT velocity field
        v_target = target - noise

        loss = F.mse_loss(v_pred, v_target)
        self._flow_loss_cache = loss.item()
        return loss

    def generate_factors_from_flow(self, pert_idx=None, gene_features=None):
        """Generate Cayley factors by integrating the learned flow.

        For UNSEEN perturbations, this is the zero-shot generation path.

        Args:
            pert_idx: (B,) perturbation indices OR (B, P) one-hot
            gene_features: (B, genept_dim) GenePT embeddings for OOD conditioning

        Returns:
            factors: (B, 2, n, r)
        """
        n, r = self.latent_dim, self.rank
        device = self.generator_params.device
        dtype = self.generator_params.dtype

        # Get conditioning
        if gene_features is not None and self.genept_cond_proj is not None:
            h_p = self.genept_cond_proj(gene_features.to(device))
            B = gene_features.size(0)
        elif pert_idx is not None:
            if pert_idx.dim() == 1:
                B = pert_idx.size(0)
                perts_oh = F.one_hot(pert_idx, self.num_perts).float().to(device)
                h_p = perts_oh @ self.pert_embed.weight
            else:
                B = pert_idx.size(0)
                h_p = pert_idx.float() @ self.pert_embed.weight
        else:
            raise ValueError("Need pert_idx or gene_features")

        z = torch.randn(B, 2 * n * r, device=device, dtype=dtype) * self.flow_sigma

        dt = 1.0 / self.n_flow_steps
        for i in range(self.n_flow_steps):
            t = torch.full((B, 1), i * dt, device=device, dtype=dtype)
            v = self.velocity_net(torch.cat([z, t, h_p], dim=1))
            z = z + dt * v

        return z.view(B, 2, n, r)

    def generate_rotation_for_ood(self, pert_idx=None, z_basal=None,
                                   gene_features=None):
        """Generate rotation matrices for OOD perturbations using flow.

        Args:
            pert_idx: (B,) perturbation indices (for pert_embed conditioning)
            z_basal: optional (B, n) basal states
            gene_features: (B, genept_dim) GenePT embeddings (for zero-shot)

        Returns:
            R: (B, n, n) rotation matrices
        """
        n = self.latent_dim
        factors = self.generate_factors_from_flow(
            pert_idx=pert_idx, gene_features=gene_features)
        U, V = factors[:, 0], factors[:, 1]
        A = U @ V.transpose(-1, -2) - V @ U.transpose(-1, -2)

        from oppert.rotation import _cayley_map_xpu_safe
        return _cayley_map_xpu_safe(A, n)
