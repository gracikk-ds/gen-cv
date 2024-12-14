"""Train loops entrypoint."""

from typing import Dict, Optional

from torch import nn
from torch.utils.data import DataLoader

from src.train_loops.train_vae import train_vae
from src.utils.scheduler import OptimizerSchedulerHandler


def train_loop(
    model_type: str,
    epochs: int,
    device: str,
    model: nn.Module,
    losses: nn.Module,
    schopt_handler: OptimizerSchedulerHandler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: Optional[DataLoader] = None,
    log_dir: str = "logs",
) -> Dict[str, Dict[str, float]]:
    """Train a model based on the provided configuration.

    Args:
        model_type (str): The type of model to train.
        epochs (int): The number of epochs to train the model.
        device (str): The device to train the model on.
        model (nn.Module): The model to train.
        losses (nn.Module): The losses to use for training.
        schopt_handler (OptimizerSchedulerHandler): The optimizer and scheduler handler.
        train_dataloader (DataLoader): The data loader for the training split.
        val_dataloader (DataLoader): The data loader for the validation split.
        test_dataloader (Optional[DataLoader]): The data loader for the test split.
        log_dir (str): The directory to save the logs.

    Raises:
        ValueError: If the model type is not supported.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing the metrics for train, val, and test.
    """
    if model_type == "vae":
        return train_vae(
            epochs,
            device,
            model,
            losses,
            schopt_handler,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            log_dir,
        )
    raise ValueError(f"Unsupported model type: {model_type}")
