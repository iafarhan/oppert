"""Conditional flow matching on the Lie algebra for rotation prediction."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F




def _build_so4_basis(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Basis of so(4) in canonical OpPert ordering: pairs (0,1),(0,2),(0,3),(1,2),(1,3),(2,3).
    Returns [6, 4, 4] with B[g, i, j]=1, B[g, j, i]=-1 at the g-th above-diagonal entry.
    """
    B = torch.zeros(6, 4, 4, device=device, dtype=dtype)
    idx = 0
    for i in range(4):
        for j in range(i + 1, 4):
            B[idx, i, j] = 1.0
            B[idx, j, i] = -1.0
            idx += 1
    return B


def block_bracket_norm_sq(
    theta_a: torch.Tensor,
    theta_b: torch.Tensor,
    basis: Optional[torch.Tensor] = None,
    num_blocks: int = 32,
    block_dim: int = 4,
    gen_dim: int = 6,
) -> torch.Tensor:
    """Sum of squared block-wise Lie-bracket matrix entries per sample.

    theta_a, theta_b : [..., num_blocks*gen_dim] flat so(4) generator vectors.
    Returns          : [...] sum over (block, i, j) of ([M_a, M_b])^2.
    Zero iff generators block-wise commute (flow tangent aligned with A itself).
    """
    assert theta_a.shape == theta_b.shape
    *lead, D = theta_a.shape
    assert D == num_blocks * gen_dim, f"theta dim {D} != {num_blocks*gen_dim}"
    if basis is None:
        basis = _build_so4_basis(theta_a.device, theta_a.dtype)
    coef_a = theta_a.view(*lead, num_blocks, gen_dim)
    coef_b = theta_b.view(*lead, num_blocks, gen_dim)
    M_a = torch.einsum("...bg,gij->...bij", coef_a, basis)
    M_b = torch.einsum("...bg,gij->...bij", coef_b, basis)
    bracket = torch.matmul(M_a, M_b) - torch.matmul(M_b, M_a)
    return (bracket ** 2).flatten(start_dim=-3).sum(dim=-1)


def bch_compose(
    theta_a: torch.Tensor,
    theta_b: torch.Tensor,
    basis: Optional[torch.Tensor] = None,
    num_blocks: int = 32,
    block_dim: int = 4,
    gen_dim: int = 6,
    scale: float = 0.5,
) -> torch.Tensor:
    """Compose two so(n) generator vectors via second-order BCH per so(4) block.

    theta_a, theta_b: [..., num_blocks * gen_dim] flat generator vectors.
    Returns:          [..., num_blocks * gen_dim] theta_combo = theta_a + theta_b + scale*[A, B]
    where [A, B] is computed block-wise and re-flattened in the OpPert canonical ordering.
    """
    assert theta_a.shape == theta_b.shape
    *lead, D = theta_a.shape
    assert D == num_blocks * gen_dim, f"theta dim {D} != {num_blocks*gen_dim}"
    if basis is None:
        basis = _build_so4_basis(theta_a.device, theta_a.dtype)
    coef_a = theta_a.view(*lead, num_blocks, gen_dim)
    coef_b = theta_b.view(*lead, num_blocks, gen_dim)
    # Skew matrices: coef @ basis -> [..., num_blocks, 4, 4]
    M_a = torch.einsum("...bg,gij->...bij", coef_a, basis)
    M_b = torch.einsum("...bg,gij->...bij", coef_b, basis)
    bracket = torch.matmul(M_a, M_b) - torch.matmul(M_b, M_a)  # [..., num_blocks, 4, 4]
    # Extract 6 above-diagonal entries in canonical order.
    # bracket is skew by construction: (i,j)th entry = -(j,i)th.
    i_idx = [0, 0, 0, 1, 1, 2]
    j_idx = [1, 2, 3, 2, 3, 3]
    bracket_coef = bracket[..., i_idx, j_idx]  # [..., num_blocks, 6]
    bracket_flat = bracket_coef.reshape(*lead, D)
    return theta_a + theta_b + scale * bracket_flat




def sinusoidal_time_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """t: [B] in [0,1] → [B, dim] sinusoidal embedding (a la transformer)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None, :] * 2.0 * math.pi
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # [B, dim]


class AdaLNBlock(nn.Module):
    """
    Linear → SiLU → Linear residual with AdaLN: scale & shift from conditioning vector.
    (DiT-style: gamma, beta from cond; modulate LayerNorm output.)
    """

    def __init__(self, d: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.cond_proj = nn.Linear(d_cond, 2 * d)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1.0 + gamma) + beta
        h = self.fc2(F.silu(self.fc1(h)))
        return x + h




class SkewBlockHead(nn.Module):
    """
    Block-structured output head: d_hidden → num_blocks × gen_dim.
    Factors through a shared per-block refinement (LN + SiLU + shared linear),
    encoding the fact that theta is 32 independent so(4) tangent coefficients.
    """

    def __init__(self, d_hidden: int, num_blocks: int, gen_dim: int, d_block: int = 16):
        super().__init__()
        self.num_blocks = num_blocks
        self.gen_dim = gen_dim
        self.d_block = d_block
        self.split = nn.Linear(d_hidden, num_blocks * d_block)
        self.norm = nn.LayerNorm(d_block, elementwise_affine=False)
        self.to_coef = nn.Linear(d_block, gen_dim)
        nn.init.zeros_(self.to_coef.weight)
        nn.init.zeros_(self.to_coef.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        x = self.split(h).view(B, self.num_blocks, self.d_block)
        x = F.silu(self.norm(x))
        return self.to_coef(x).reshape(B, self.num_blocks * self.gen_dim)


class DiTBlockOverTokens(nn.Module):
    """DiT-style transformer block over `num_blocks` tokens.

    Self-attention across block tokens + MLP, each modulated by (gamma, beta, gate)
    produced from the conditioning vector (shared across tokens). Zero-init on the
    condition projection so at init the block is an identity residual (Peebles & Xie 2023).
    """

    def __init__(self, d_token: int, d_cond: int, n_heads: int = 4, mlp_mult: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token, elementwise_affine=False)
        d_mlp = int(mlp_mult * d_token)
        self.mlp = nn.Sequential(
            nn.Linear(d_token, d_mlp), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_mlp, d_token),
        )
        # 6 modulation params per block (attn gamma/beta/gate + mlp gamma/beta/gate)
        self.cond_proj = nn.Linear(d_cond, 6 * d_token)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]   cond: [B, d_cond]
        mods = self.cond_proj(cond)
        g_a, b_a, gate_a, g_m, b_m, gate_m = mods.chunk(6, dim=-1)
        # Broadcast modulation across tokens: [B, 1, D]
        g_a, b_a, gate_a = g_a[:, None, :], b_a[:, None, :], gate_a[:, None, :]
        g_m, b_m, gate_m = g_m[:, None, :], b_m[:, None, :], gate_m[:, None, :]
        h = self.norm1(x) * (1.0 + g_a) + b_a
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a * h
        h = self.norm2(x) * (1.0 + g_m) + b_m
        h = self.mlp(h)
        x = x + gate_m * h
        return x


class BlockTransformerVelocityNet(nn.Module):
    """Transformer velocity net over so(4) block tokens (lever #2).

    Reshapes theta [B, theta_dim] → [B, num_blocks, gen_dim] and treats each block's
    6 coefficients as a token. Self-attention lets the v-net model inter-block
    correlations that the flat AdaLN-MLP (ConditionalVelocityNet) could not express.
    """

    def __init__(
        self,
        theta_dim: int = 192,
        d_embed: int = 1800,
        num_blocks: int = 32,
        gen_dim: int = 6,
        d_token: int = 128,
        d_cond: int = 512,
        n_layers: int = 4,
        n_heads: int = 4,
        mlp_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_blocks * gen_dim == theta_dim, \
            f"num_blocks*gen_dim ({num_blocks*gen_dim}) != theta_dim ({theta_dim})"
        self.theta_dim = theta_dim
        self.d_embed = d_embed
        self.num_blocks = num_blocks
        self.gen_dim = gen_dim
        self.d_token = d_token

        self.in_proj = nn.Linear(gen_dim, d_token)
        self.pos_embed = nn.Parameter(torch.zeros(num_blocks, d_token))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.t_proj = nn.Sequential(
            nn.Linear(64, d_cond), nn.SiLU(), nn.Linear(d_cond, d_cond),
        )
        self.e_proj = nn.Sequential(
            nn.Linear(d_embed, d_cond), nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_cond, d_cond),
        )

        self.layers = nn.ModuleList([
            DiTBlockOverTokens(d_token, d_cond, n_heads=n_heads, mlp_mult=mlp_mult,
                               dropout=dropout)
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d_token, elementwise_affine=False)
        self.out_proj = nn.Linear(d_token, gen_dim)
        # Zero init so initial velocity ≈ 0 (identity residual at init).
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, A: torch.Tensor, t: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        B = A.shape[0]
        t_emb = sinusoidal_time_embedding(t, dim=64)
        cond = self.t_proj(t_emb) + self.e_proj(e)  # [B, d_cond]
        x = A.view(B, self.num_blocks, self.gen_dim)
        x = self.in_proj(x) + self.pos_embed[None, :, :]  # [B, T, d_token]
        for layer in self.layers:
            x = layer(x, cond)
        x = self.out_norm(x)
        out = self.out_proj(x)  # [B, T, gen_dim]
        return out.reshape(B, self.theta_dim)


class ConditionalVelocityNet(nn.Module):
    """
    v_psi(A, t, e) → R^{theta_dim}
      A: [B, theta_dim]  current generator
      t: [B]             in [0, 1]
      e: [B, d_embed]    perturbation descriptor (gene or drug)
    Output: velocity field dA/dt at (A, t, e).
    head_kind: "flat" (default, linear to theta_dim) or
               "skew_block" (per-block shared head with d_block intermediate).
    """

    def __init__(
        self,
        theta_dim: int = 192,
        d_embed: int = 1800,
        d_hidden: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.0,
        head_kind: str = "flat",
        num_blocks_head: int = 32,
        gen_dim_head: int = 6,
        d_block_head: int = 16,
    ):
        super().__init__()
        self.theta_dim = theta_dim
        self.d_embed = d_embed
        d_cond = d_hidden

        # A projection
        self.in_proj = nn.Linear(theta_dim, d_hidden)
        # t embedding
        self.t_proj = nn.Sequential(
            nn.Linear(64, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )
        # e embedding (with optional dropout to prevent overfitting on sparse data)
        self.e_proj = nn.Sequential(
            nn.Linear(d_embed, d_cond),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_cond, d_cond),
        )
        # Stack of AdaLN MLP blocks
        self.blocks = nn.ModuleList([AdaLNBlock(d_hidden, d_cond) for _ in range(n_blocks)])
        # Output
        self.out_norm = nn.LayerNorm(d_hidden, elementwise_affine=False)
        self.head_kind = head_kind
        if head_kind == "flat":
            self.out_proj = nn.Linear(d_hidden, theta_dim)
            nn.init.zeros_(self.out_proj.weight)  # start from zero velocity ≈ identity
            nn.init.zeros_(self.out_proj.bias)
        elif head_kind == "skew_block":
            assert num_blocks_head * gen_dim_head == theta_dim, \
                f"num_blocks_head*gen_dim_head ({num_blocks_head*gen_dim_head}) != theta_dim ({theta_dim})"
            self.out_proj = SkewBlockHead(
                d_hidden=d_hidden,
                num_blocks=num_blocks_head,
                gen_dim=gen_dim_head,
                d_block=d_block_head,
            )
        else:
            raise ValueError(f"unknown head_kind={head_kind!r}")

    def forward(self, A: torch.Tensor, t: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        A: [B, theta_dim],  t: [B], e: [B, d_embed]
        """
        t_emb = sinusoidal_time_embedding(t, dim=64)
        cond = self.t_proj(t_emb) + self.e_proj(e)
        h = self.in_proj(A)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out_proj(self.out_norm(h))




def flow_matching_loss(
    v_net: ConditionalVelocityNet,
    theta_train: torch.Tensor,      # [N, theta_dim] true generators of training perts
    embed_train: torch.Tensor,      # [N, d_embed]   descriptors of training perts
    batch_idx: torch.Tensor,        # [B] indices into training set
    sigma_noise: float = 1.0,
    prior_train: Optional[torch.Tensor] = None,  # [N, theta_dim] A_0 base per pert
    interpolant: str = "ot",
    t_schedule: str = "uniform",
    t_logit_m: float = 0.0,
    t_logit_s: float = 1.0,
    bch_aug_rate: float = 0.0,
    bch_basis: Optional[torch.Tensor] = None,
    bch_embed_mode: str = "mean",
    cfg_drop_prob: float = 0.0,
) -> torch.Tensor:
    """
    Conditional flow matching loss on a stochastic interpolant.
      A_0 = prior_i + sigma * eps    (prior_i = 0 if prior_train is None)
      A_1 = theta_i
      A_t = alpha(t) A_0 + beta(t) A_1
      u_t = alpha'(t) A_0 + beta'(t) A_1
      L = E || v_net(A_t, t, e_i) - u_t ||^2
    Interpolants:
      "ot"     — alpha=1-t, beta=t (OT / straight paths; default, back-compat).
      "cosine" — alpha=cos(pi t/2), beta=sin(pi t/2) (variance-preserving trig interpolant).
    """
    device = theta_train.device
    B = batch_idx.shape[0]
    theta_i = theta_train[batch_idx]
    e_i = embed_train[batch_idx]
    A0_base_i = prior_train[batch_idx] if prior_train is not None else None

    # BCH data augmentation: replace a fraction of the batch with pseudo-combos
    # theta_combo = theta_a + theta_b + 0.5*[A_a, A_b] from singles (a, b).
    if bch_aug_rate > 0.0:
        n_aug = int(round(bch_aug_rate * B))
        if n_aug > 0:
            N_all = theta_train.shape[0]
            # Sample two distinct single indices for each of the n_aug slots.
            a_idx = torch.randint(0, N_all, (n_aug,), device=device)
            b_idx = torch.randint(0, N_all, (n_aug,), device=device)
            eq = a_idx == b_idx
            if eq.any():
                # Nudge duplicates by +1 modulo N to guarantee distinct pairs.
                b_idx = torch.where(eq, (b_idx + 1) % N_all, b_idx)
            theta_a, theta_b = theta_train[a_idx], theta_train[b_idx]
            theta_combo = bch_compose(theta_a, theta_b, basis=bch_basis)
            e_a, e_b = embed_train[a_idx], embed_train[b_idx]
            if bch_embed_mode == "mean":
                e_combo = 0.5 * (e_a + e_b)
            elif bch_embed_mode == "sum":
                e_combo = e_a + e_b
            else:
                raise ValueError(f"unknown bch_embed_mode={bch_embed_mode!r}")
            # Overwrite the first n_aug positions.
            theta_i = torch.cat([theta_combo, theta_i[n_aug:]], dim=0)
            e_i = torch.cat([e_combo, e_i[n_aug:]], dim=0)
            if A0_base_i is not None:
                prior_a, prior_b = prior_train[a_idx], prior_train[b_idx]
                prior_combo = bch_compose(prior_a, prior_b, basis=bch_basis)
                A0_base_i = torch.cat([prior_combo, A0_base_i[n_aug:]], dim=0)

    if A0_base_i is not None:
        A0 = A0_base_i + sigma_noise * torch.randn_like(theta_i)
    else:
        A0 = sigma_noise * torch.randn_like(theta_i)
    if t_schedule == "uniform":
        t = torch.rand(B, device=device)
    elif t_schedule == "logit_normal":
        # SD3-style: u ~ N(m, s^2); t = sigmoid(u) concentrates mass near t=0.5 for m=0.
        u = t_logit_m + t_logit_s * torch.randn(B, device=device)
        t = torch.sigmoid(u)
    else:
        raise ValueError(f"unknown t_schedule={t_schedule!r}; expected 'uniform' or 'logit_normal'")
    if interpolant == "ot":
        alpha = (1.0 - t)[:, None]
        beta = t[:, None]
        alpha_dot = -torch.ones_like(alpha)
        beta_dot = torch.ones_like(beta)
    elif interpolant == "cosine":
        half_pi = math.pi * 0.5
        u = half_pi * t
        alpha = torch.cos(u)[:, None]
        beta = torch.sin(u)[:, None]
        alpha_dot = (-half_pi * torch.sin(u))[:, None]
        beta_dot = (half_pi * torch.cos(u))[:, None]
    else:
        raise ValueError(f"unknown interpolant={interpolant!r}; expected 'ot' or 'cosine'")
    At = alpha * A0 + beta * theta_i
    u_target = alpha_dot * A0 + beta_dot * theta_i
    # Classifier-free guidance: randomly drop conditioning per-sample so the net
    # learns both p(v|e) and p(v|∅). At inference, we scale v_cond - v_uncond.
    if cfg_drop_prob > 0.0:
        drop_mask = (torch.rand(e_i.shape[0], device=device) < cfg_drop_prob).float().unsqueeze(1)
        e_in = e_i * (1.0 - drop_mask)
    else:
        e_in = e_i
    v_pred = v_net(At, t, e_in)
    return F.mse_loss(v_pred, u_target)


def _time_grid(n_steps: int, schedule: str, device: torch.device) -> torch.Tensor:
    """Return tensor of n_steps+1 time points from 0 to 1 under the chosen schedule.
    'uniform'    : t_i = i/n_steps
    'cosine_end' : concentrated near t=1:   t_i = 1 - cos(pi/2 * i/n_steps)
    'cosine_start' : concentrated near t=0:  t_i = sin(pi/2 * i/n_steps)
    'poly2_end'  : concentrated near t=1:   t_i = (i/n_steps)**2
    'poly2_start': concentrated near t=0:    t_i = 1 - (1 - i/n_steps)**2
    """
    u = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    if schedule == "uniform":
        return u
    if schedule == "cosine_end":
        return 1.0 - torch.cos(u * (math.pi / 2.0))
    if schedule == "cosine_start":
        return torch.sin(u * (math.pi / 2.0))
    if schedule == "poly2_end":
        return u.pow(2)
    if schedule == "poly2_start":
        return 1.0 - (1.0 - u).pow(2)
    raise ValueError(f"unknown time_schedule={schedule!r}")


@torch.no_grad()
def sample_theta(
    v_net: ConditionalVelocityNet,
    e: torch.Tensor,                 # [B, d_embed]
    n_steps: int = 20,
    sigma_noise: float = 1.0,
    seed: Optional[int] = None,
    prior: Optional[torch.Tensor] = None,   # [B, theta_dim] A_0 base (e.g. KRR prediction)
    integrator: str = "euler",
    cfg_scale: float = 1.0,
    time_schedule: str = "uniform",
    antithetic_sign: float = 1.0,
) -> torch.Tensor:
    """
    Integrate dA/dt = v_net(A, t, e) from t=0 to t=1.
    A_0 = prior + sigma * (antithetic_sign * noise) (prior=0 if not given).
    integrator: "euler" (1st order, default) or "heun" (2nd order predictor-corrector).
    cfg_scale: classifier-free guidance scale w. 1.0 = no guidance (default).
      v_guided = v_uncond + w*(v_cond - v_uncond). Requires the model to have been
      trained with cfg_drop_prob>0 so the ∅-conditioned branch is meaningful.
    time_schedule: 'uniform' (default), 'cosine_end', 'cosine_start', 'poly2_end',
      'poly2_start'. Non-uniform schedules change step spacing to concentrate compute
      near t=0 or t=1. The reverse-flow vector field tends to vary most near t=1 where
      the trajectory approaches the teacher generator, so 'cosine_end' may reduce
      discretization error at fixed n_steps.
    antithetic_sign: ±1.0 multiplier applied to the initial noise term. Used by
      ensemble sampling for antithetic variates (variance reduction).
    Returns sampled theta ∈ R^{theta_dim}.
    """
    device = e.device
    B = e.shape[0]
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    noise = torch.randn(B, v_net.theta_dim, device=device, generator=gen)
    if antithetic_sign != 1.0:
        noise = antithetic_sign * noise
    if prior is not None:
        A = prior + sigma_noise * noise
    else:
        A = sigma_noise * noise
    ts = _time_grid(n_steps, time_schedule, device)
    use_cfg = abs(cfg_scale - 1.0) > 1e-8
    e_null = torch.zeros_like(e) if use_cfg else None

    def _v(A_in, t_in):
        v_c = v_net(A_in, t_in, e)
        if not use_cfg:
            return v_c
        v_u = v_net(A_in, t_in, e_null)
        return v_u + cfg_scale * (v_c - v_u)

    if integrator == "euler":
        for step in range(n_steps):
            t_val = float(ts[step].item())
            dt_step = float((ts[step + 1] - ts[step]).item())
            t = torch.full((B,), t_val, device=device)
            v = _v(A, t)
            A = A + dt_step * v
    elif integrator == "heun":
        for step in range(n_steps):
            t0_val = float(ts[step].item())
            t1_val = float(ts[step + 1].item())
            dt_step = t1_val - t0_val
            t0 = torch.full((B,), t0_val, device=device)
            t1 = torch.full((B,), min(t1_val, 1.0), device=device)
            v0 = _v(A, t0)
            A_pred = A + dt_step * v0
            v1 = _v(A_pred, t1)
            A = A + 0.5 * dt_step * (v0 + v1)
    else:
        raise ValueError(f"unknown integrator={integrator!r}; expected 'euler' or 'heun'")
    return A


@torch.no_grad()
def sample_theta_ensemble(
    v_net: ConditionalVelocityNet,
    e: torch.Tensor,                 # [B, d_embed]
    n_samples: int = 16,
    n_steps: int = 20,
    sigma_noise: float = 1.0,
    prior: Optional[torch.Tensor] = None,
    integrator: str = "euler",
    cfg_scale: float = 1.0,
    time_schedule: str = "uniform",
    antithetic: bool = False,
) -> torch.Tensor:
    """
    Draw multiple samples per pert; return [n_samples, B, theta_dim].
    Mean can be used as point estimate; variance = uncertainty.
    cfg_scale>1.0 activates classifier-free guidance (see sample_theta).
    antithetic: pair samples as (noise, -noise) for variance reduction. When True
    and n_samples is even, samples 2k and 2k+1 share the same seed but use
    opposite noise signs. When False, each sample uses an independent seed.
    """
    samples = []
    for s in range(n_samples):
        if antithetic:
            seed = s // 2
            sign = 1.0 if (s % 2 == 0) else -1.0
        else:
            seed = s
            sign = 1.0
        samples.append(sample_theta(
            v_net, e, n_steps=n_steps, sigma_noise=sigma_noise, seed=seed, prior=prior,
            integrator=integrator, cfg_scale=cfg_scale,
            time_schedule=time_schedule, antithetic_sign=sign,
        ))
    return torch.stack(samples, dim=0)
