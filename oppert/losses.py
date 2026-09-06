"""Loss functions."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def _nan2inf(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class NBLoss(nn.Module):
    """Negative binomial log-likelihood (for count-like targets)."""

    def __init__(self):
        super().__init__()

    def forward(self, yhat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        dim = yhat.size(1) // 2
        mu = yhat[:, :dim]
        theta = yhat[:, dim:]
        if theta.ndimension() == 1:
            theta = theta.view(1, theta.size(0))
        t1 = torch.lgamma(theta + eps) + torch.lgamma(y + 1.0) - torch.lgamma(y + theta + eps)
        t2 = (theta + y) * torch.log1p(mu / (theta + eps)) + y * (torch.log(theta + eps) - torch.log(mu + eps))
        final = _nan2inf(t1 + t2)
        return final.mean()


class StableGaussianNLL(nn.Module):
    def __init__(self, min_var: float = 1e-3, add_const: bool = True):
        super().__init__()
        self.min_var = min_var
        self.add_const = add_const
        self.const = 0.5 * math.log(2 * math.pi)

    def forward(self, mean: torch.Tensor, target: torch.Tensor, var: torch.Tensor,
                weights: torch.Tensor | None = None):
        var = var.clamp_min(self.min_var)
        nll = 0.5 * ((target - mean).pow(2) / var + var.log())
        if self.add_const:
            nll = nll + self.const
        if weights is not None:
            nll = nll * weights
        return nll.mean()


class GaussianLoss(nn.Module):
    """Gaussian NLL. yhat = [mean, var], var will be softplus'ed upstream."""

    def __init__(self):
        super().__init__()

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dim = yhat.size(1) // 2
        mean = yhat[:, :dim]
        variance = yhat[:, dim:]
        term1 = variance.log().div(2)
        term2 = (y - mean).pow(2).div(variance.mul(2))
        return (term1 + term2).mean()


class FocalLoss(nn.Module):
    """Binary focal loss (uses torchvision.ops implementation)."""

    def __init__(self, alpha=0.3, gamma=3.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        from torchvision.ops import focal_loss
        return focal_loss.sigmoid_focal_loss(
            inputs, target, reduction=self.reduction, gamma=self.gamma, alpha=self.alpha
        )
