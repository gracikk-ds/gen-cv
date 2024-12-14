"""Logging utilities."""

from torch.utils.tensorboard import SummaryWriter


def log_vae_scalars(
    writer: SummaryWriter,
    epoch: int,
    stage: str,
    loss: float,
    log_kl_loss: float,
    log_mse_loss: float,
) -> None:
    """Log the scalars to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer.
        epoch (int): Current epoch.
        stage (str): Stage of the training loop ("train" or "val").
        loss (float): Loss value.
        log_kl_loss (float): KL divergence loss value.
        log_mse_loss (float): MSE loss value.
    """
    writer.add_scalar(f"Loss/{stage}", loss, epoch)  # type: ignore
    writer.add_scalar(f"KL_Loss/{stage}", log_kl_loss, epoch)  # type: ignore
    writer.add_scalar(f"MSE_Loss/{stage}", log_mse_loss, epoch)  # type: ignore
