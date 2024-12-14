"""VAE loss implementation."""

from typing import Tuple

import torch
from torch import Tensor, nn


def reversed_kl_div(mu: Tensor, logvar: Tensor) -> Tensor:
    """Reverse KL divergence.

    Args:
        mu (Tensor): mean tensor
        logvar (Tensor): log variance tensor

    Returns:
        Tensor: reverse KL divergence
    """
    return 0.5 * (mu**2 + torch.exp(logvar) - logvar - 1)


class VAELoss(nn.Module):
    """A PyTorch nn.Module that encapsulates Variational Autoencoder loss calculation."""

    def __init__(self, kl_weight: float = 1.0, reduction: str = "mean"):
        """Initialize the VAE loss module.

        Args:
            kl_weight (float): Weighting factor for the KL divergence term. Default is 1.0.
            reduction (str): Specifies the reduction to apply to the reconstruction loss. One of ['mean', 'sum'].
        """
        super().__init__()
        self.reconstr_loss = nn.MSELoss(reduction=reduction)
        self.kl_weight = kl_weight
        self.reduction = reduction

    def forward(
        self,
        target: Tensor,
        mu: Tensor,
        logvar: Tensor,
        reconstruction: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for the VAE loss computation.

        Args:
            target (Tensor): The ground truth target tensor. Shape: (B, C, H, W)
            mu (Tensor): The mean tensor from the encoder. Shape: (B, latent_dim)
            logvar (Tensor): The log-variance tensor from the encoder. Shape: (B, latent_dim)
            reconstruction (Tensor): The reconstructed output. Shape: (B, C, H, W)

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - total_loss: The combined VAE loss.
                - kl_loss: The KL divergence component of the loss.
                - recon_loss: The reconstruction loss component.
        """
        kl_div = reversed_kl_div(mu, logvar).sum(dim=1).mean()
        recon_loss = self.reconstr_loss(reconstruction, target)
        total_loss = recon_loss + kl_div * self.kl_weight
        return total_loss, kl_div, recon_loss
