"""Train a VAE model."""

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

from src.models.tiny_vae import TinyVAE
from src.utils.logging import log_vae_scalars
from src.utils.scheduler import OptimizerSchedulerHandler
from src.utils.train_loop_utils import log_all_reconstructions


def run_epoch(
    model: TinyVAE,
    losses: nn.Module,
    loader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    split: str,
    optimizer_scheduler: Optional[OptimizerSchedulerHandler] = None,
    log_reconstructions: bool = False,
) -> Tuple[float, float]:
    """Run one epoch of training, validation, or testing.

    Args:
        model (TinyVAE): The VAE model to train or evaluate.
        losses (nn.Module): The losses to use for training.
        loader (DataLoader): The data loader for the current split (train/val/test).
        device (torch.device): The device on which computations run.
        writer (SummaryWriter): TensorBoard writer for logging.
        epoch (int): Current epoch number.
        split (str): One of 'train', 'val', or 'test'.
        optimizer_scheduler (Optional[OptimizerSchedulerHandler]): Optimizer and scheduler handler for training.
        log_reconstructions (bool): Whether to log a reconstruction at the end of the loop.

    Returns:
        Tuple[float, float]: Mean MSE and KL losses for the epoch.
    """
    is_train = split == "train"
    if is_train:
        model.train()
    else:
        model.eval()

    loader_len = len(loader)
    mse_loss = 0
    kl_loss = 0
    new_epoch = epoch == 0
    with torch.set_grad_enabled(is_train):
        for idx, batch in tqdm(enumerate(loader), total=loader_len, leave=False, desc=split):
            batch = batch.to(device)
            reconstruction, mu, logvar = model(batch)
            loss, log_kl_loss, log_mse_loss = losses(batch, mu, logvar, reconstruction)

            if is_train and optimizer_scheduler is not None:
                optimizer_scheduler.zero_grad()
                loss.backward()
                optimizer_scheduler.step(new_epoch=new_epoch)
                new_epoch = False

            step = epoch * loader_len + idx
            mse_loss += log_mse_loss.item()
            kl_loss += log_kl_loss.item()
            log_vae_scalars(writer, step, split, loss.item(), log_kl_loss.item(), log_mse_loss.item())

            # Log reconstruction images at the last batch of the loader
            if log_reconstructions and idx == loader_len - 1:
                log_all_reconstructions(writer, epoch, reconstruction, batch)

    return mse_loss / loader_len, kl_loss / loader_len


def train_vae(  # noqa: WPS216,WPS213,WPS210
    epochs: int,
    device: str,
    model: TinyVAE,
    losses: nn.Module,
    schopt_handler: OptimizerSchedulerHandler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    log_dir: str = "runs/vae",
) -> Dict[str, Dict[str, float]]:
    """Train a Variational Autoencoder (VAE) model.

    Args:
        epochs (int): Number of training epochs.
        device (str): Device on which the model will be trained ('cuda' or 'cpu').
        model (TinyVAE): The VAE model instance.
        losses (nn.Module): The losses to use for training.
        schopt_handler (OptimizerSchedulerHandler): Combined optimizer & scheduler handler.
        train_loader (DataLoader): Data loader for the training split.
        val_loader (DataLoader): Data loader for the validation split.
        test_loader (Optional[DataLoader]): Data loader for the test split.
        log_dir (str): Directory to save TensorBoard logs.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing the last losses for train, val, and test.
    """
    # Set device to cuda or cpu
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)  # type: ignore
    last_losses: Dict[str, Dict[str, float]] = {"train": {}, "val": {}, "test": {}}

    for epoch in tqdm(range(epochs)):
        # Run one epoch of training
        train_mse_loss, train_kl_loss = run_epoch(
            model=model,
            losses=losses,
            loader=train_loader,
            device=torch_device,
            writer=writer,
            epoch=epoch,
            split="train",
            optimizer_scheduler=schopt_handler,
            log_reconstructions=False,
        )
        last_losses["train"] = {"mse": train_mse_loss, "kl": train_kl_loss}

        # Run one epoch of validation
        val_mse_loss, val_kl_loss = run_epoch(
            model=model,
            losses=losses,
            loader=val_loader,
            device=torch_device,
            writer=writer,
            epoch=epoch,
            split="val",
            optimizer_scheduler=None,
            log_reconstructions=True,  # Log reconstruction at the end of val epoch
        )
        last_losses["val"] = {"mse": val_mse_loss, "kl": val_kl_loss}
    # Run one epoch of testing (if provided)
    if test_loader is not None:
        test_mse_loss, test_kl_loss = run_epoch(
            model=model,
            losses=losses,
            loader=test_loader,
            device=torch_device,
            writer=writer,
            epoch=epochs,
            split="test",
            optimizer_scheduler=None,
            log_reconstructions=True,  # Log reconstruction at the end of test epoch
        )
        last_losses["test"] = {"mse": test_mse_loss, "kl": test_kl_loss}
    writer.close()  # type: ignore
    return last_losses
