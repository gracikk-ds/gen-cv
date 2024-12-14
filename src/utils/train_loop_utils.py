"""Utility functions for training loops."""

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter  # type: ignore

SCALE_PIX: int = 255


def log_all_reconstructions(
    writer: SummaryWriter,
    epoch: int,
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Log the entire batch of reconstructions, with reconstructed images below real ones, transposed.

    Args:
        writer (SummaryWriter): TensorBoard writer.
        epoch (int): Current epoch.
        reconstruction (torch.Tensor): Reconstructed images.
        target (torch.Tensor): Target images.
    """
    # Stack real and reconstructed images along the vertical axis (dimension 0)
    images_to_log = torch.stack([target, reconstruction], dim=1)

    # Reshape the images to arrange them in a grid-like structure
    images_to_log = images_to_log.view(-1, *target.shape[1:])  # Flatten along the batch dimension

    # Create a grid with nrow=batch_size to ensure images are side-by-side
    grid = torchvision.utils.make_grid(images_to_log, nrow=2)

    # Log the grid of images to TensorBoard
    writer.add_image("Reconstructions", grid, epoch)  # type: ignore

