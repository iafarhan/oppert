"""Neural network building blocks."""

import math
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


class _GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def gradient_reversal(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal: identity forward, negate+scale backward."""
    return _GradientReversalFn.apply(x, scale)


class MLP(nn.Module):
    """
    Generic MLP with optional BatchNorm and optional extra adapter layer:
      - append_layer_position="first" adds a front adapter (henc).
      - append_layer_position="last"  adds a tail adapter (hdec).

    Special case: if last_layer_act == "ReLU", only the *first half* of outputs is ReLU'ed
    (useful when output is [mean, var] and var is post-softplus).
    """

    def __init__(
        self,
        sizes: Sequence[int],
        batch_norm: bool = True,
        last_layer_act: str = "linear",
        append_layer_width: Optional[int] = None,
        append_layer_position: Optional[str] = None,
    ):
        super().__init__()
        assert last_layer_act in ("linear", "ReLU")
        layers: List[nn.Module] = []
        for s in range(len(sizes) - 1):
            layers += [
                nn.Linear(sizes[s], sizes[s + 1]),
                nn.BatchNorm1d(sizes[s + 1]) if batch_norm and s < len(sizes) - 2 else None,
                nn.ReLU(),
            ]
        layers = [l for l in layers if l is not None][:-1]

        if append_layer_width is not None:
            assert append_layer_position in ("first", "last")
            layers_dict = OrderedDict()
            if append_layer_position == "first":
                layers_dict["henc_linear"] = nn.Linear(append_layer_width, sizes[0])
                layers_dict["henc_bn1d"] = nn.BatchNorm1d(sizes[0])
                layers_dict["henc_relu"] = nn.ReLU()
                for i, module in enumerate(layers):
                    layers_dict[str(i)] = module
            else:
                for i, module in enumerate(layers):
                    layers_dict[str(i)] = module
                layers_dict["hdec_bn1d"] = nn.BatchNorm1d(sizes[-1])
                layers_dict["hdec_relu"] = nn.ReLU()
                layers_dict["hdec_linear"] = nn.Linear(sizes[-1], append_layer_width)
        else:
            layers_dict = OrderedDict({str(i): m for i, m in enumerate(layers)})

        self.network = nn.Sequential(layers_dict)
        self.last_layer_act = last_layer_act
        if last_layer_act == "ReLU":
            self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        if self.last_layer_act == "ReLU":
            dim = x.size(1) // 2
            return torch.cat([self.relu(x[:, :dim]), x[:, dim:]], dim=1)
        return x


class GeneralizedSigmoid(nn.Module):
    """Per-drug dose-response warper: 'sigm', 'logsigm', or None (linear)."""

    def __init__(self, dim: int, device: str, nonlin: Optional[str] = "sigm", doser_min: float = 0.0):
        super().__init__()
        assert nonlin in ("sigm", "logsigm", None)
        self.nonlin = nonlin
        self.doser_min = doser_min
        self.beta = nn.Parameter(torch.ones(1, dim, device=device), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, dim, device=device), requires_grad=True)

    def forward(self, x: torch.Tensor, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.nonlin is None:
            return x
        if self.nonlin == "logsigm":
            if idx is None:
                c0 = self.bias.sigmoid()
                return (torch.log1p(x) * self.beta + self.bias).sigmoid() - c0
            bias = self.bias[0][idx]
            beta = self.beta[0][idx]
            c0 = bias.sigmoid()
            return (torch.log1p(x) * beta + bias).sigmoid() - c0

        if idx is None:
            c0 = self.bias.sigmoid()
            return (x * self.beta + self.bias).sigmoid() - c0
        bias = self.bias[0][idx]
        beta = self.beta[0][idx]
        c0 = bias.sigmoid()
        return (x * beta + bias).sigmoid() - c0

    def one_drug(self, x: torch.Tensor, i: int) -> torch.Tensor:
        if self.nonlin == "logsigm":
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        if self.nonlin == "sigm":
            c0 = self.bias[0][i].sigmoid()
            return (x * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        return x


class AttentionBlock(nn.Module):
    """Transformer-style MHA block (batch_first=True), with residuals and FFN."""

    def __init__(self, embed_dim: int, width: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, width),
            nn.GELU(),
            nn.Linear(width, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x = q
        y, _ = self.attn(self.ln1(x), self.ln1(k), self.ln1(v), need_weights=False)
        x = x + self.drop(y)
        y = self.ff(self.ln2(x))
        x = x + self.drop(y)
        return x

    def forward_with_weights(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        y, attn_w = self.attn(self.ln1(q), self.ln1(k), self.ln1(v), need_weights=True)
        x = q + self.drop(y)
        y = self.ff(self.ln2(x))
        x = x + self.drop(y)
        return x, attn_w


class CapsuleNetwork(nn.Module):
    """
    Capsule-inspired projection: maps (B, E) -> (B, E).
    """

    def __init__(
        self,
        embed_dim: int,
        num_route_nodes: int = 8,
        num_capsules: int = 8,
        caps_in: Optional[int] = None,
        caps_out: Optional[int] = None,
        routing_iters: int = 3,
    ):
        super().__init__()
        self.E = embed_dim
        self.R = num_route_nodes
        self.N = num_capsules
        self.C_in = caps_in if caps_in is not None else embed_dim // num_route_nodes
        self.C_out = caps_out if caps_out is not None else embed_dim // num_capsules
        self.routing_iters = routing_iters

        self.W_primary = nn.Parameter(torch.randn(self.R, self.E, self.C_in) * (1.0 / math.sqrt(self.E)))
        self.b_primary = nn.Parameter(torch.zeros(self.R, self.C_in))
        self.W_route = nn.Parameter(torch.randn(self.R, self.N, self.C_in, self.C_out) * (1.0 / math.sqrt(self.C_in)))
        self.out_proj = nn.Linear(self.N * self.C_out, self.E)

    @staticmethod
    def _squash(t: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        sq = (t**2).sum(dim=dim, keepdim=True)
        scale = sq / (1.0 + sq)
        return scale * t / (torch.sqrt(sq + eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        U = torch.einsum("be,rec->brc", x, self.W_primary) + self.b_primary
        U = self._squash(U, dim=-1)
        priors = torch.einsum("brc,rnck->brnk", U, self.W_route)
        logits = torch.zeros(B, self.R, self.N, device=x.device)
        for it in range(self.routing_iters):
            c = torch.softmax(logits, dim=-1)
            s = (c.unsqueeze(-1) * priors).sum(dim=1)
            v = self._squash(s, dim=-1)
            if it < self.routing_iters - 1:
                agree = torch.einsum("brnk,bnk->brn", priors, v)
                logits = logits + agree
        out = v.reshape(B, self.N * self.C_out)
        out = self.out_proj(out)
        return out

    def analyze(self, x: torch.Tensor):
        B = x.size(0)
        U = torch.einsum("be,rec->brc", x, self.W_primary) + self.b_primary
        U = self._squash(U, dim=-1)
        priors = torch.einsum("brc,rnck->brnk", U, self.W_route)
        logits = torch.zeros(B, self.R, self.N, device=x.device)
        couplings, caps_outs = [], []
        for it in range(self.routing_iters):
            c = torch.softmax(logits, dim=-1)
            s = (c.unsqueeze(-1) * priors).sum(dim=1)
            v = self._squash(s, dim=-1)
            couplings.append(c)
            caps_outs.append(v)
            if it < self.routing_iters - 1:
                agree = torch.einsum("brnk,bnk->brn", priors, v)
                logits = logits + agree
        flat = caps_outs[-1].reshape(B, self.N * self.C_out)
        projected = self.out_proj(flat)
        return {
            "primary": U, "priors": priors, "couplings": couplings,
            "caps_out": caps_outs, "flat": flat, "projected": projected,
        }


class UnifiedRepresentation(nn.Module):
    """Capsule stacks + cross-attention fusion → (B, E). [Legacy: was used for gene+flux fusion]"""

    def __init__(
        self, embed_dim: int, width: int, num_heads: int = 4, dropout: float = 0.1,
        num_route_nodes: int = 8, num_capsules: int = 8, routing_iters: int = 3,
    ):
        super().__init__()
        self.caps_g = CapsuleNetwork(embed_dim, num_route_nodes, num_capsules, routing_iters=routing_iters)
        self.caps_f = CapsuleNetwork(embed_dim, num_route_nodes, num_capsules, routing_iters=routing_iters)
        self.attn = AttentionBlock(embed_dim, width, num_heads=num_heads, dropout=dropout)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, z_g: torch.Tensor, z_f: torch.Tensor) -> torch.Tensor:
        q = self.caps_g(z_g)
        k = self.caps_f(z_f)
        v = k
        q_ = q.unsqueeze(1)
        k_ = k.unsqueeze(1)
        v_ = v.unsqueeze(1)
        out = self.attn(q_, k_, v_)
        out = out.squeeze(1)
        return self.ln(out)

    def inspect(self, z_g: torch.Tensor, z_f: torch.Tensor):
        cg = self.caps_g.analyze(z_g)
        cf = self.caps_f.analyze(z_f)
        q = cg["projected"].unsqueeze(1)
        k = cf["projected"].unsqueeze(1)
        v = k
        out, attn_w = self.attn.forward_with_weights(q, k, v)
        out = out.squeeze(1)
        return self.ln(out), {"attn": attn_w, "caps_g": cg, "caps_f": cf}


class CrossGate(nn.Module):
    """Simple, stable cross-gating block; often competitive with attention for 2 vectors."""

    def __init__(self, dim: int):
        super().__init__()
        self.f2g = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.g2f = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.ln = nn.LayerNorm(dim)

    def forward(self, g: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        g_hat = self.ln(g + f * self.f2g(f))
        f_hat = self.ln(f + g * self.g2f(g))
        return self.ln(g_hat + f_hat)


class SwiGLUBlock(nn.Module):
    """SwiGLU activation block: out = (xW1 * silu(xW_gate)) @ W2"""

    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x) * F.silu(self.w_gate(x)))


class ResidualMLP(nn.Module):
    """input_proj -> [SwiGLU + residual + LayerNorm] x depth -> output_proj

    Takes same ``sizes`` list as MLP: [input_dim, hidden, ..., output_dim].
    Uses sizes[1] as the residual block dimension; depth = len(sizes) - 2.
    """

    def __init__(self, sizes: Sequence[int], hidden_mult: int = 2):
        super().__init__()
        assert len(sizes) >= 3, "Need at least [input, hidden, output]"
        self.input_proj = nn.Linear(sizes[0], sizes[1])
        depth = len(sizes) - 2
        self.blocks = nn.ModuleList([
            nn.Sequential(SwiGLUBlock(sizes[1], hidden_mult), nn.LayerNorm(sizes[1]))
            for _ in range(depth)
        ])
        self.output_proj = nn.Linear(sizes[1], sizes[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)
        return self.output_proj(x)


class SplitDecoder(nn.Module):
    """Shared trunk -> separate mu and var heads.

    Output is [mu, var] concatenated along dim=1, same as a standard decoder MLP.
    """

    def __init__(
        self,
        trunk_sizes: Sequence[int],
        head_dim: int,
        head_depth: int = 2,
        head_width: int = 256,
        encoder_type: str = "mlp",
    ):
        super().__init__()
        if encoder_type == "swiglu":
            self.trunk = ResidualMLP(list(trunk_sizes))
        else:
            self.trunk = MLP(list(trunk_sizes), last_layer_act="linear")
        trunk_out = trunk_sizes[-1]
        self.mu_head = MLP(
            [trunk_out] + [head_width] * head_depth + [head_dim],
            last_layer_act="linear",
        )
        self.var_head = MLP(
            [trunk_out] + [head_width] * head_depth + [head_dim],
            last_layer_act="linear",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.cat([self.mu_head(h), self.var_head(h)], dim=1)


class SkipLinearDecoder(nn.Module):
    """Decoder with direct linear skip connection for guaranteed full-rank Jacobian.

    Architecture: linear_skip(z) + alpha * residual_mlp(z)

    The linear skip connection (latent_dim -> output_dim) guarantees that the
    Jacobian has rank >= min(latent_dim, output_dim), regardless of the
    residual network's rank. The residual MLP adds nonlinear expressiveness
    for gene-specific effects. The mixing coefficient alpha is learnable but
    initialized small (0.1) so the linear path dominates early, ensuring
    full-rank Jacobian from the start.

    This fixes the structural low-rank issue where SwiGLU ResidualMLP decoders
    with residual connections + LayerNorm collapse the Jacobian erank to ~19.
    """

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 512,
                 depth: int = 2, hidden_mult: int = 2, fixed_alpha: float = -1.0):
        super().__init__()
        # Direct linear path: guaranteed rank = min(latent_dim, output_dim)
        self.linear_skip = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear_skip.weight)  # maximize rank

        # Nonlinear residual path: adds expressiveness
        self.residual = ResidualMLP(
            [latent_dim] + [hidden_dim] * depth + [output_dim],
            hidden_mult=hidden_mult,
        )

        # Mixing coefficient: fixed or learnable
        # When fixed_alpha >= 0, alpha is constant (not a parameter).
        # When fixed_alpha < 0, alpha is learnable via log_alpha.
        self._fixed_alpha = fixed_alpha
        if fixed_alpha < 0:
            # Learnable mixing coefficient (initialized small -> linear dominates)
            self.log_alpha = nn.Parameter(torch.tensor(-2.3))  # alpha ~ 0.1
        else:
            self.register_buffer("_alpha_val", torch.tensor(fixed_alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fixed_alpha >= 0:
            alpha = self._fixed_alpha
        else:
            alpha = torch.sigmoid(self.log_alpha)  # bounded (0, 1)
        return self.linear_skip(x) + alpha * self.residual(x)


class ReLUBottleneckDecoder(nn.Module):
    """Piecewise-linear decoder with locally constant but high-rank Jacobian.

    Architecture: z -> Linear(latent, hidden) -> ReLU -> Linear(hidden, output)

    Key insight: Within each ReLU linear region, J(z) is a constant matrix with
    rank up to min(latent_dim, hidden_dim). Since we use hidden_dim >> latent_dim
    (e.g., 512 vs 128), the effective rank can reach 128 — much higher than the
    ~19 erank of trained skip-linear decoders.

    The piecewise linearity means:
    - Jacobian is constant within regions (clean zero-shot like linear decoder)
    - But different cells CAN have different Jacobians (if in different regions)
    - Rotation should stay in same region for small perturbations (local)

    Optional: skip connection preserves gradient flow through the linear path.
    """

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 512,
                 n_layers: int = 1, skip: bool = True):
        super().__init__()
        self.skip = skip

        layers = []
        in_dim = latent_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

        if skip:
            self.linear_skip = nn.Linear(latent_dim, output_dim)
            nn.init.orthogonal_(self.linear_skip.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        if self.skip:
            out = out + self.linear_skip(x)
        return out


class MultiHeadLinearDecoder(nn.Module):
    """Decoder with K independent linear heads, each predicting a subset of genes.

    Instead of one Linear(latent, output) with erank ~19, this uses K heads:
    head_k: Linear(latent, output/K) — each head predicts ~output/K genes.

    Key advantage: per-head spectral regularization ensures each head uses
    ALL latent dimensions for its gene subset. The overall decoder weight
    matrix is block-diagonal by initialization, and per-head spectral reg
    keeps it diverse.

    Also maintains a linear_skip attribute pointing to the concatenated weight
    for compatibility with the spectral regularizer API.
    """

    def __init__(self, latent_dim: int, output_dim: int, n_heads: int = 10):
        super().__init__()
        self.n_heads = n_heads
        self.latent_dim = latent_dim

        genes_per_head = output_dim // n_heads
        remainder = output_dim % n_heads

        self.heads = nn.ModuleList()
        self.head_sizes = []
        for i in range(n_heads):
            size = genes_per_head + (1 if i < remainder else 0)
            self.head_sizes.append(size)
            head = nn.Linear(latent_dim, size)
            nn.init.orthogonal_(head.weight)
            self.heads.append(head)

        # Dummy linear_skip for compatibility with spectral reg API
        # (spectral reg will check hasattr(decoder, 'linear_skip'))
        # We provide the first head's weight as proxy; actual reg is per-head
        self.linear_skip = self.heads[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([h(x) for h in self.heads], dim=1)

    def per_head_spectral_reg(self) -> Tuple[torch.Tensor, float, float]:
        """Compute spectral regularization per head and average.

        Returns (loss, avg_erank, avg_cond) across all heads.
        """
        total_loss = torch.tensor(0.0, device=self.heads[0].weight.device)
        total_erank = 0.0
        total_cond = 0.0

        for head in self.heads:
            sv = torch.linalg.svdvals(head.weight)  # sorted descending
            p = sv / (sv.sum() + 1e-12)
            log_p = torch.log(p + 1e-12)
            erank = torch.exp(-(p * log_p).sum())
            max_rank = float(sv.size(0))
            loss = -torch.log(erank / max_rank + 1e-12)
            cond = float((sv[0] / (sv[-1] + 1e-12)).item())
            total_loss = total_loss + loss
            total_erank += float(erank.item())
            total_cond += cond

        n = len(self.heads)
        return total_loss / n, total_erank / n, total_cond / n


class StiefelDecoder(nn.Module):
    """Decoder with orthogonal weight directions and learnable per-dimension scales.

    Architecture: diag(exp(log_scale)) @ U^T @ z + bias, where U is orthogonal.

    This decouples the decoder into:
    - U (orthogonal directions): prevents column correlation, fixes up-regulation bias
    - exp(log_scale) (learnable per-dimension scales): preserves reconstruction quality

    The key insight: with orthogonal U, the Jacobian J = diag(s) @ U^T has
    no correlated columns, so the predicted sign pattern sign(J J^T) depends
    on the scale-weighted angular structure, not spurious correlations.

    For linear perturbation prediction: delta_j = -alpha * sum_k s_k * U[j,k] * U[g,k]
    With orthogonal U, this sum has both positive and negative terms (unlike correlated W).
    """

    def __init__(self, latent_dim: int, output_dim: int, **kwargs):
        super().__init__()
        # Orthogonal weight: output_dim x latent_dim
        self.linear = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear.weight)

        # Apply PyTorch's orthogonal parametrization
        # This constrains the weight to stay on the Stiefel manifold during training
        import torch.nn.utils.parametrizations as P
        P.orthogonal(self.linear, name="weight", orthogonal_map="householder")

        # Learnable per-dimension scale (in log space for positivity)
        # Initialize to match SVD of a random orthogonal init (all 1s)
        self.log_scale = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W_eff = diag(scale) @ W_orth, but we compute as: scale * (W_orth @ x)
        h = self.linear(x)  # (B, output_dim), with orthogonal weight
        scale = torch.exp(self.log_scale)  # (output_dim,)
        return h * scale.unsqueeze(0)


class PertCondStiefelDecoder(nn.Module):
    """Stiefel decoder with perturbation-conditioned per-gene scale modulation.

    Architecture: (scale + pert_modulation) * (W_orth @ z), where:
    - W_orth is on the Stiefel manifold (orthogonal columns via Householder parametrization)
    - scale = exp(log_scale) provides base per-gene scaling
    - pert_modulation = strength * tanh(MLP(pert_encoding)) adjusts scale per gene per perturbation

    The modulation is multiplicative: scale * (1 + mod), where mod is bounded by tanh.
    For unseen perturbations, the pert_encoding is a one-hot over perturbation genes,
    which generalizes naturally (just set the one-hot for the target gene).

    This decouples:
    - W_orth: high-rank directional structure (good for directional accuracy)
    - pert_modulation: per-gene magnitude adjustment (good for expression prediction)
    """

    def __init__(self, latent_dim: int, output_dim: int, pert_embed_dim: int = 128,
                 proj_hidden: int = 256, **kwargs):
        super().__init__()
        # Orthogonal weight: output_dim x latent_dim
        self.linear = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear.weight)

        # Apply PyTorch's orthogonal parametrization (Stiefel manifold)
        import torch.nn.utils.parametrizations as P
        P.orthogonal(self.linear, name="weight", orthogonal_map="householder")

        # Learnable per-dimension base scale (in log space for positivity)
        self.log_scale = nn.Parameter(torch.zeros(output_dim))

        # Perturbation modulation network: pert_encoding -> per-gene scale adjustment
        self.pert_proj = nn.Sequential(
            nn.Linear(pert_embed_dim, proj_hidden),
            nn.SiLU(),
            nn.Linear(proj_hidden, output_dim),
            nn.Tanh(),  # bounded modulation in [-1, 1]
        )
        # Modulation strength: sigmoid(-3) ~ 0.05, so initial modulation is very small
        self.modulation_strength = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x: torch.Tensor, pert_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.linear(x)  # (B, output_dim), orthogonal weight
        scale = torch.exp(self.log_scale)  # (output_dim,)
        if pert_embed is not None:
            strength = torch.sigmoid(self.modulation_strength)
            mod = strength * self.pert_proj(pert_embed)  # (B, output_dim)
            return h * (scale.unsqueeze(0) * (1.0 + mod))
        return h * scale.unsqueeze(0)


class LightCrossAttention(nn.Module):
    """Bidirectional cross-attention for two vectors, output averaged."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.g2f = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.f2g = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, g: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        g_, f_ = g.unsqueeze(1), f.unsqueeze(1)
        g_att, _ = self.g2f(g_, f_, f_)  # cross-attention A
        f_att, _ = self.f2g(f_, g_, g_)  # cross-attention B
        return self.ln((g_att + f_att).squeeze(1))


# Nonlinear decoder architectures that preserve rotation/geodesic structure


class GatedResidualDecoder(nn.Module):
    """Nonlinear decoder: W@z + sigmoid(gate(z)) ⊙ V@z.

    The linear part (W@z) carries the rotation structure for geodesic kNN.
    The gated part adds cell-state-dependent nonlinearity: different cells
    activate different gene subsets depending on their latent position.

    This is genuinely nonlinear (gate depends on z) but preserves the linear
    backbone that rotations rely on. The Jacobian is:
        J(z) = W + diag(σ(g(z))) @ V + diag(V@z) @ diag(σ'(g(z))) @ dg/dz
    which is input-dependent → perturbation-specific gene responses.
    """

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 256,
                 gate_hidden: int = 128, init_gate_bias: float = -2.0):
        super().__init__()
        # Primary linear path (carries rotation structure)
        self.linear_skip = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear_skip.weight)

        # Gated nonlinear path
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, output_dim),
        )
        # Initialize gate to be mostly closed (sigmoid(-2) ≈ 0.12)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, init_gate_bias)

        # Secondary linear path (modulated by gate)
        self.value_linear = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.value_linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear_skip(x)
        gate = torch.sigmoid(self.gate_net(x))
        gated_out = gate * self.value_linear(x)
        return linear_out + gated_out


class QuadraticResidualDecoder(nn.Module):
    """Decoder with quadratic nonlinearity: W@z + α * (z^T M z) per output dim.

    Architecture: linear(z) + α * quadratic(z)
    where quadratic captures pairwise latent interactions.

    For efficiency, we use a low-rank factorization of the quadratic form:
    q_g(z) = ||P_g @ z||^2 - ||N_g @ z||^2  (difference of squared projections)
    where P_g, N_g are rank-r matrices. This gives signed quadratic terms.

    Actually simplified: q(z) = (U@z)^2 summed with learnable weights
    i.e., for each output dim g: q_g = w_g^T (Fz ⊙ Fz) where F is shared.

    Jacobian: J(z) = W + 2α * diag(w_g) @ F^T @ diag(Fz) @ ... (input-dependent)
    """

    def __init__(self, latent_dim: int, output_dim: int, quad_rank: int = 32,
                 alpha: float = 0.1):
        super().__init__()
        # Linear backbone
        self.linear_skip = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear_skip.weight)

        # Shared quadratic feature extractor: z -> (Fz)^2
        self.quad_proj = nn.Linear(latent_dim, quad_rank, bias=False)
        nn.init.orthogonal_(self.quad_proj.weight)

        # Per-output weights on quadratic features
        self.quad_out = nn.Linear(quad_rank, output_dim, bias=False)
        nn.init.normal_(self.quad_out.weight, std=0.01)

        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear_skip(x)
        # Quadratic features: element-wise square of projected z
        qf = self.quad_proj(x)  # (batch, quad_rank)
        qf_sq = qf * qf  # element-wise square -> quadratic in z
        quad_out = self.quad_out(qf_sq)  # (batch, output_dim)
        return linear_out + self.alpha * quad_out


class SpectralGateDecoder(nn.Module):
    """Multiplicative gating decoder: (1 + α*tanh(v_g^T z)) * (W@z)_g.

    Each gene g has a gating direction v_g in latent space. The gate
    modulates the linear output multiplicatively based on cell state.

    This is genuinely nonlinear: the Jacobian has z-dependent scaling per gene.
    But the structure is simple enough that rotations approximately compose:
    if R is small, tanh(v^T R z) ≈ tanh(v^T z) + sech²(v^T z) * v^T(R-I)z

    The gating is interpretable: v_g defines the cell-state direction along which
    gene g's response is amplified or suppressed.
    """

    def __init__(self, latent_dim: int, output_dim: int, alpha: float = 0.3):
        super().__init__()
        self.linear_skip = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear_skip.weight)

        # Per-gene gating direction (output_dim x latent_dim)
        self.gate_dirs = nn.Linear(latent_dim, output_dim, bias=False)
        nn.init.normal_(self.gate_dirs.weight, std=0.01)  # start near identity gate

        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear_skip(x)
        gate = 1.0 + self.alpha * torch.tanh(self.gate_dirs(x))  # (batch, output)
        return gate * linear_out


class NormPreservingDecoder(nn.Module):
    """Decoder using GroupSort activation (Lipschitz-1, norm-preserving).

    Architecture: W2 @ GroupSort(W1 @ z + b1) + b2

    GroupSort sorts within groups of 2, which is a Lipschitz-1 activation.
    Combined with spectral-normalized weights, this gives a Lipschitz decoder.

    Key property: if ||R@z - z|| is small (rotation is near identity),
    then ||f(R@z) - f(z)|| is proportionally small. Rotations "pass through"
    the decoder approximately.

    This is genuinely nonlinear (GroupSort is not linear) but preserves
    the norm structure that rotations rely on.
    """

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 512,
                 n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for i in range(n_layers):
            linear = nn.Linear(in_dim, hidden_dim)
            nn.init.orthogonal_(linear.weight)
            layers.append(linear)
            layers.append(GroupSort(group_size=2))
            in_dim = hidden_dim
        self.body = nn.Sequential(*layers)

        self.out_linear = nn.Linear(hidden_dim, output_dim)
        nn.init.orthogonal_(self.out_linear.weight)

        # Skip connection for gradient flow
        self.linear_skip = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear_skip.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_skip(x) + self.out_linear(self.body(x))


class GroupSort(nn.Module):
    """GroupSort activation: sort within groups. Lipschitz-1 and norm-preserving."""

    def __init__(self, group_size: int = 2):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape into groups and sort
        *batch_dims, d = x.shape
        g = self.group_size
        assert d % g == 0, f"Dim {d} not divisible by group_size {g}"
        x_grouped = x.view(*batch_dims, d // g, g)
        x_sorted, _ = torch.sort(x_grouped, dim=-1, descending=True)
        return x_sorted.view(*batch_dims, d)


# Neural ODE Decoder — deep nonlinear decoder via ODE integration

class ODEFunc(nn.Module):
    """Velocity field v(z, t) for the Neural ODE.

    Uses time-conditioned MLP: dz/dt = v(z, t) = MLP([z; t])
    Tanh activations for Lipschitz control (bounded gradients).
    """

    def __init__(self, dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = dim + 1  # +1 for time
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, dim))
        # Initialize last layer small so ODE starts near identity
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # t is scalar, z is (B, dim)
        t_expand = t.expand(z.shape[0], 1)
        return self.net(torch.cat([z, t_expand], dim=-1))


class NeuralODEDecoder(nn.Module):
    """Neural ODE decoder: integrates dz/dt = v(z,t) from t=0 to t=1.

    Architecture:
    1. Linear projection: latent_dim -> ode_dim (optional, if ode_dim != latent_dim)
    2. ODE integration: z(0) -> z(1) via Euler/midpoint steps
    3. Linear readout: ode_dim -> output_dim

    Key properties:
    - Jacobian is INPUT-DEPENDENT (unlike linear decoder where J is constant)
    - Provides deep nonlinearity through the ODE flow
    - Skip connection for stability: output = W_skip @ z + W_read @ ODE(z)
    - Time integration provides smooth, well-conditioned mapping

    For perturbation prediction:
    - Rotation R acts on z_0 (initial condition)
    - ODE flow transforms R@z_0 differently from z_0
    - delta = decode(R@z) - decode(z) is input-dependent and perturbation-specific
    """

    def __init__(self, latent_dim: int, output_dim: int,
                 ode_dim: int = 128, hidden_dim: int = 256,
                 n_layers: int = 2, n_steps: int = 5,
                 solver: str = "midpoint", alpha: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.ode_dim = ode_dim
        self.n_steps = n_steps
        self.solver = solver
        self.alpha = alpha  # Weight of ODE branch vs skip

        # Project to ODE space if needed
        if latent_dim != ode_dim:
            self.proj_in = nn.Linear(latent_dim, ode_dim)
            nn.init.orthogonal_(self.proj_in.weight)
        else:
            self.proj_in = nn.Identity()

        # ODE velocity field
        self.ode_func = ODEFunc(ode_dim, hidden_dim, n_layers)

        # Readout from ODE state
        self.readout = nn.Linear(ode_dim, output_dim)
        nn.init.orthogonal_(self.readout.weight)

        # Skip connection (linear baseline) — named linear_skip for SVD clamping compat
        self.linear_skip = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear_skip.weight)

    def _integrate(self, z0: torch.Tensor) -> torch.Tensor:
        """Integrate ODE from t=0 to t=1 using fixed-step solver."""
        dt = 1.0 / self.n_steps
        z = z0

        for i in range(self.n_steps):
            t = torch.tensor(i * dt, device=z.device, dtype=z.dtype)

            if self.solver == "euler":
                z = z + dt * self.ode_func(t, z)
            elif self.solver == "midpoint":
                # Midpoint method (2nd order)
                k1 = self.ode_func(t, z)
                t_mid = torch.tensor((i + 0.5) * dt, device=z.device, dtype=z.dtype)
                z_mid = z + 0.5 * dt * k1
                k2 = self.ode_func(t_mid, z_mid)
                z = z + dt * k2
            elif self.solver == "rk4":
                # Classic RK4 (4th order)
                k1 = self.ode_func(t, z)
                t2 = torch.tensor((i + 0.5) * dt, device=z.device, dtype=z.dtype)
                k2 = self.ode_func(t2, z + 0.5 * dt * k1)
                k3 = self.ode_func(t2, z + 0.5 * dt * k2)
                t3 = torch.tensor((i + 1) * dt, device=z.device, dtype=z.dtype)
                k4 = self.ode_func(t3, z + dt * k3)
                z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: skip + alpha * ODE flow."""
        # Project to ODE space
        z0 = self.proj_in(x)

        # Integrate ODE
        z1 = self._integrate(z0)

        # Combine skip and ODE branches
        return self.linear_skip(x) + self.alpha * self.readout(z1)


# FiLM Hyper-Decoder: perturbation-conditioned via program coefficients

class FiLMHyperDecoder(nn.Module):
    """Decoder whose output is modulated by perturbation program coefficients α.

    Architecture:
        base_output = W @ z + b                  (standard linear decoder)
        gamma, beta = MLP_film(alpha)             (FiLM modulation from programs)
        output = (1 + gamma) * base_output + beta (element-wise affine transform)

    This makes the decoder genuinely nonlinear: different perturbations
    (with different alpha vectors) produce different decoding functions.
    When alpha is None (e.g. control cells), falls back to the linear decoder.

    The FiLM modulation is initialized near identity (gamma≈0, beta≈0) so
    training starts from the linear decoder and gradually learns perturbation-
    specific corrections. This is critical for stable autoencoder pre-training.

    Biology: Each program coefficient α_k represents activation of a gene program
    (e.g. apoptosis, UPR, cell cycle). The FiLM layer modulates gene-level
    predictions based on which programs are active, implementing pathway-specific
    dose-response curves.
    """

    def __init__(self, latent_dim: int, output_dim: int,
                 n_programs: int = 32, film_hidden: int = 128,
                 film_layers: int = 1, film_dropout: float = 0.0,
                 alpha_dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_programs = n_programs
        self._film_dropout_p = alpha_dropout  # prob of zeroing alpha per sample

        # Base linear decoder (same as skip_linear's linear path)
        self.linear = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear.weight)

        # FiLM generator: alpha -> (gamma, beta)
        # Multi-layer for expressiveness, but initialized near zero
        if film_layers <= 1:
            self.film_net = nn.Sequential(
                nn.Linear(n_programs, 2 * output_dim),
            )
        else:
            layers = [nn.Linear(n_programs, film_hidden), nn.SiLU()]
            if film_dropout > 0:
                layers.append(nn.Dropout(film_dropout))
            for _ in range(film_layers - 2):
                layers.extend([nn.Linear(film_hidden, film_hidden), nn.SiLU()])
                if film_dropout > 0:
                    layers.append(nn.Dropout(film_dropout))
            layers.append(nn.Linear(film_hidden, 2 * output_dim))
            self.film_net = nn.Sequential(*layers)

        # Initialize FiLM output near identity: gamma≈0, beta≈0
        with torch.no_grad():
            self.film_net[-1].weight.zero_()
            self.film_net[-1].bias.zero_()

    def forward(self, z: torch.Tensor, alpha: torch.Tensor = None) -> torch.Tensor:
        base = self.linear(z)  # (B, output_dim)
        if alpha is not None:
            # Alpha dropout: with probability p, zero out alpha to prevent
            # the FiLM from encoding perturbation identity directly
            if self.training and self._film_dropout_p > 0:
                mask = torch.rand(alpha.size(0), 1, device=alpha.device) > self._film_dropout_p
                alpha = alpha * mask.float()
            film_out = self.film_net(alpha)  # (B, 2*output_dim)
            gamma = film_out[:, :self.output_dim]   # scale modulation
            beta = film_out[:, self.output_dim:]     # shift modulation
            return (1 + gamma) * base + beta
        return base


class ProgramAttentionDecoder(nn.Module):
    """Decoder with K program-specific linear heads weighted by α.

    Architecture:
        head_k(z) = W_k @ z + b_k    for k = 1..K
        output = Σ_k softmax(alpha)_k * head_k(z)

    Each head specializes in one gene program (pathway). The program
    coefficients alpha determine which heads contribute to the output.
    This is a mixture-of-experts with the mixture weights determined
    by the perturbation's program decomposition.

    More parameter-efficient than full hyper-network: only K*output_dim*latent_dim
    extra parameters (vs output_dim^2 for full weight generation).
    Uses low-rank heads (W_k = U_k @ V_k^T) to reduce parameters further.
    """

    def __init__(self, latent_dim: int, output_dim: int,
                 n_programs: int = 32, head_rank: int = 8,
                 temperature: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_programs = n_programs
        self.temperature = temperature

        # Base linear decoder (shared backbone)
        self.linear = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear.weight)

        # Low-rank program heads: W_k = U_k @ V_k^T
        # U: (K, output_dim, head_rank), V: (K, head_rank, latent_dim)
        self.head_U = nn.Parameter(torch.randn(n_programs, output_dim, head_rank) * 0.01)
        self.head_V = nn.Parameter(torch.randn(n_programs, head_rank, latent_dim) * 0.01)
        # Per-program bias
        self.head_bias = nn.Parameter(torch.zeros(n_programs, output_dim))

    def forward(self, z: torch.Tensor, alpha: torch.Tensor = None) -> torch.Tensor:
        base = self.linear(z)  # (B, output_dim)
        if alpha is not None:
            # Compute program-specific corrections: delta_k = U_k @ V_k @ z + b_k
            # z: (B, latent_dim) -> (B, 1, latent_dim)
            z_exp = z.unsqueeze(1)  # (B, 1, D)
            # V @ z: (K, R, D) @ (B, 1, D)^T -> use einsum
            Vz = torch.einsum('krd,bd->bkr', self.head_V, z)  # (B, K, R)
            delta = torch.einsum('kor,bkr->bko', self.head_U, Vz)  # (B, K, O)
            delta = delta + self.head_bias.unsqueeze(0)  # (B, K, O)

            # Weight by softmax(alpha/temperature)
            weights = torch.softmax(alpha / self.temperature, dim=-1)  # (B, K)
            correction = torch.einsum('bk,bko->bo', weights, delta)  # (B, O)
            return base + correction
        return base


class SimplexFiLMDecoder(nn.Module):
    """FiLM decoder with simplex-constrained alpha for OOD robustness.

    Key insight: Raw alpha coefficients go OOD when interpolated for unseen perts,
    causing catastrophic FiLM divergence (R²=-300). Solution: pass alpha through
    softmax BEFORE FiLM, keeping inputs on the probability simplex regardless of
    interpolation. Also clip gamma to [-max_gamma, +max_gamma] for safety.

    Architecture:
        alpha_simplex = softmax(alpha / temp)
        gamma, beta = MLP(alpha_simplex)
        gamma = clip(gamma, -max_gamma, max_gamma)
        output = (1 + gamma) * (W @ z) + beta
    """

    def __init__(self, latent_dim: int, output_dim: int,
                 n_programs: int = 32, film_hidden: int = 128,
                 film_layers: int = 1, simplex_temp: float = 1.0,
                 max_gamma: float = 0.5, alpha_dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_programs = n_programs
        self.simplex_temp = simplex_temp
        self.max_gamma = max_gamma
        self._alpha_dropout_p = alpha_dropout

        # Base linear decoder
        self.linear = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear.weight)

        # FiLM generator from simplex-constrained alpha
        if film_layers <= 1:
            self.film_net = nn.Sequential(
                nn.Linear(n_programs, 2 * output_dim),
            )
        else:
            layers = [nn.Linear(n_programs, film_hidden), nn.SiLU()]
            for _ in range(film_layers - 2):
                layers.extend([nn.Linear(film_hidden, film_hidden), nn.SiLU()])
            layers.append(nn.Linear(film_hidden, 2 * output_dim))
            self.film_net = nn.Sequential(*layers)

        # Initialize near identity
        with torch.no_grad():
            self.film_net[-1].weight.zero_()
            self.film_net[-1].bias.zero_()

    def forward(self, z: torch.Tensor, alpha: torch.Tensor = None) -> torch.Tensor:
        base = self.linear(z)
        if alpha is not None:
            # Alpha dropout
            if self.training and self._alpha_dropout_p > 0:
                mask = torch.rand(alpha.size(0), 1, device=alpha.device) > self._alpha_dropout_p
                alpha = alpha * mask.float()
            # Project to simplex — guarantees bounded, in-distribution input
            alpha_simplex = torch.softmax(alpha / self.simplex_temp, dim=-1)
            film_out = self.film_net(alpha_simplex)
            gamma = film_out[:, :self.output_dim]
            beta = film_out[:, self.output_dim:]
            # Clip gamma for additional safety
            gamma = gamma.clamp(-self.max_gamma, self.max_gamma)
            return (1 + gamma) * base + beta
        return base


class ProgramGatedResidualDecoder(nn.Module):
    """Linear decoder with a small, gated nonlinear correction.

    Architecture:
        linear_out = W @ z                          (main path)
        correction = MLP(z || alpha) * gate_scale   (small nonlinear correction)
        gate = sigmoid(linear_gate(alpha))           (perturbation-dependent gate)
        output = linear_out + gate * correction

    The gate and gate_scale ensure the correction stays small and perturbation-specific.
    Initialized with gate_scale=0.01 so training starts essentially linear.
    """

    def __init__(self, latent_dim: int, output_dim: int,
                 n_programs: int = 32, hidden_dim: int = 128,
                 gate_scale_init: float = 0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_programs = n_programs

        # Main linear decoder
        self.linear = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear.weight)

        # Nonlinear correction: f(z, alpha) -> correction
        self.correction_net = nn.Sequential(
            nn.Linear(latent_dim + n_programs, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Initialize small
        with torch.no_grad():
            self.correction_net[-1].weight.mul_(0.01)
            self.correction_net[-1].bias.zero_()

        # Per-gene gate from alpha (which genes to correct)
        self.gate_net = nn.Sequential(
            nn.Linear(n_programs, output_dim),
            nn.Sigmoid(),
        )
        # Initialize gate near 0.5 (neutral)
        with torch.no_grad():
            self.gate_net[0].weight.zero_()
            self.gate_net[0].bias.zero_()

        # Learnable scale parameter (starts very small)
        self.gate_scale = nn.Parameter(torch.tensor(gate_scale_init))

    def forward(self, z: torch.Tensor, alpha: torch.Tensor = None) -> torch.Tensor:
        base = self.linear(z)
        if alpha is not None:
            alpha_simplex = torch.softmax(alpha, dim=-1)  # normalize
            concat = torch.cat([z, alpha_simplex], dim=-1)
            correction = self.correction_net(concat)
            gate = self.gate_net(alpha_simplex)
            return base + self.gate_scale * gate * correction
        return base


class EffectFiLMDecoder(nn.Module):
    """FiLM decoder conditioned on rotation EFFECT (z_d = R@z - z) instead of alpha.

    KEY INSIGHT: Previous FiLM decoders condition on alpha (basis coefficients), which
    are abstract and don't generalize to OOD. This decoder conditions on the EFFECT of
    the rotation in latent space, which is:
    1. Naturally bounded (same scale as z)
    2. Cell-state dependent (biologically meaningful: same KO → different effect in different cells)
    3. Smooth under rotation interpolation (no abstract coefficient extrapolation)
    4. Works with ANY rotation type (per-pert, shared basis, etc.)

    Architecture:
        base = W @ z_pert
        effect = z_pert - z_ctrl  (= R@z - z ≈ A@z for small rotations)
        (gamma, beta) = FiLM_net(effect)
        output = (1 + clamp(gamma)) * base + beta

    For control cells: effect = 0 → gamma ≈ 0, beta ≈ 0 → output ≈ base (linear)
    """

    def __init__(self, latent_dim: int, output_dim: int,
                 film_hidden: int = 128, film_layers: int = 2,
                 max_gamma: float = 0.3, effect_dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.max_gamma = max_gamma
        self.effect_dropout = effect_dropout

        # Main linear decoder
        self.linear = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear.weight)

        # FiLM network: effect (latent_dim) -> (gamma, beta) (2 * output_dim)
        layers = []
        in_dim = latent_dim
        for i in range(film_layers):
            layers.append(nn.Linear(in_dim, film_hidden))
            layers.append(nn.LayerNorm(film_hidden))
            layers.append(nn.SiLU())
            in_dim = film_hidden
        layers.append(nn.Linear(film_hidden, 2 * output_dim))
        self.film_net = nn.Sequential(*layers)

        # Initialize near identity: zero output → gamma=0, beta=0
        with torch.no_grad():
            self.film_net[-1].weight.zero_()
            self.film_net[-1].bias.zero_()

    def forward(self, z: torch.Tensor, pert_delta: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            z: perturbed latent (B, latent_dim)
            pert_delta: rotation effect z_d = z_pert - z_ctrl (B, latent_dim)
                        If None, returns linear decoder output (for controls).
        """
        base = self.linear(z)
        if pert_delta is not None:
            # Optional dropout on effect during training
            if self.training and self.effect_dropout > 0:
                pert_delta = F.dropout(pert_delta, p=self.effect_dropout, training=True)

            film_out = self.film_net(pert_delta)
            gamma = film_out[:, :self.output_dim]
            beta = film_out[:, self.output_dim:]
            gamma = gamma.clamp(-self.max_gamma, self.max_gamma)
            return (1 + gamma) * base + beta
        return base


class SpectralNormedFiLMDecoder(nn.Module):
    """FiLM decoder with spectral normalization for Lipschitz-bounded conditioning.

    KEY INSIGHT: FiLM decoders fail on OOD because the alpha→(gamma,beta) mapping
    is unconstrained, amplifying interpolation errors. Spectral normalization bounds
    the Lipschitz constant of the FiLM network, ensuring small changes in alpha
    produce small changes in (gamma, beta).

    Architecture:
        base = W @ z
        alpha_norm = alpha / (||alpha|| + eps)  (unit-norm input)
        (gamma, beta) = SpectralNormed_FiLM(alpha_norm)
        gamma = max_gamma * tanh(gamma)  (bounded output)
        output = (1 + gamma) * base + beta

    Spectral norm ensures: ||d(gamma,beta)/dalpha|| ≤ product(sigma_max_i)
    """

    def __init__(self, latent_dim: int, output_dim: int,
                 n_programs: int = 32, film_hidden: int = 128,
                 film_layers: int = 2, max_gamma: float = 0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.max_gamma = max_gamma

        # Main linear decoder
        self.linear = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear.weight)

        # Spectrally-normalized FiLM network
        layers = []
        in_dim = n_programs
        for i in range(film_layers):
            lin = nn.utils.parametrizations.spectral_norm(nn.Linear(in_dim, film_hidden))
            layers.append(lin)
            layers.append(nn.SiLU())
            in_dim = film_hidden
        # Final layer also spectral-normed
        layers.append(nn.utils.parametrizations.spectral_norm(
            nn.Linear(film_hidden, 2 * output_dim)))
        self.film_net = nn.Sequential(*layers)

        # Initialize near zero — access underlying weight through parametrization
        with torch.no_grad():
            last_layer = self.film_net[-1]
            if hasattr(last_layer, 'weight_orig'):
                last_layer.weight_orig.zero_()
            elif hasattr(last_layer, 'parametrizations'):
                # torch.nn.utils.parametrizations.spectral_norm stores in 'weight'
                last_layer.weight.zero_()
            else:
                last_layer.weight.zero_()
            last_layer.bias.zero_()

    def forward(self, z: torch.Tensor, alpha: torch.Tensor = None) -> torch.Tensor:
        base = self.linear(z)
        if alpha is not None:
            # Normalize alpha to unit norm for consistent scale
            alpha_norm = alpha / (alpha.norm(dim=-1, keepdim=True) + 1e-8)
            film_out = self.film_net(alpha_norm)
            gamma = film_out[:, :self.output_dim]
            beta = film_out[:, self.output_dim:]
            # Bounded gamma via tanh
            gamma = self.max_gamma * torch.tanh(gamma)
            return (1 + gamma) * base + beta
        return base


class RotationAwareFiLMDecoder(nn.Module):
    """FiLM decoder that uses BOTH alpha AND rotation effect for conditioning.

    Combines the structural information from alpha (program coefficients) with
    the cell-state-dependent rotation effect (z_d). This gives the decoder
    access to both WHAT program is active and HOW it affects this particular cell.

    Architecture:
        base = W @ z_pert
        h_alpha = MLP(simplex(alpha))  (program identity)
        h_effect = MLP(z_d)           (cell-specific effect)
        combined = h_alpha + h_effect
        (gamma, beta) = linear(combined)
        output = (1 + clamp(gamma)) * base + beta
    """

    def __init__(self, latent_dim: int, output_dim: int,
                 n_programs: int = 32, film_hidden: int = 64,
                 max_gamma: float = 0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.max_gamma = max_gamma

        # Main linear decoder
        self.linear = nn.Linear(latent_dim, output_dim)
        nn.init.orthogonal_(self.linear.weight)

        # Alpha branch (program identity)
        self.alpha_net = nn.Sequential(
            nn.Linear(n_programs, film_hidden),
            nn.SiLU(),
        )

        # Effect branch (cell-specific)
        self.effect_net = nn.Sequential(
            nn.Linear(latent_dim, film_hidden),
            nn.SiLU(),
        )

        # Combined → FiLM params
        self.film_head = nn.Linear(film_hidden, 2 * output_dim)

        # Initialize near zero
        with torch.no_grad():
            self.film_head.weight.zero_()
            self.film_head.bias.zero_()

    def forward(self, z: torch.Tensor, alpha: torch.Tensor = None,
                pert_delta: torch.Tensor = None) -> torch.Tensor:
        base = self.linear(z)
        if alpha is not None or pert_delta is not None:
            h = torch.zeros(z.size(0), self.film_head.in_features, device=z.device)
            if alpha is not None:
                alpha_simplex = torch.softmax(alpha, dim=-1)
                h = h + self.alpha_net(alpha_simplex)
            if pert_delta is not None:
                h = h + self.effect_net(pert_delta)
            film_out = self.film_head(h)
            gamma = film_out[:, :self.output_dim].clamp(-self.max_gamma, self.max_gamma)
            beta = film_out[:, self.output_dim:]
            return (1 + gamma) * base + beta
        return base
