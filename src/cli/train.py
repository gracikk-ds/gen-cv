# pylint: disable=protected-access
"""Command-line script for training models using PyTorch Lightning."""

from typing import Any, Dict, Optional

import hydra
import pytorch_lightning as pylight
from clearml import Task
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from src.train_loops.main_loop import train_loop
from src.utils.basic_utils import save_metrics


def setup_clearml_task(config: DictConfig) -> None:
    """
    Initialize a logger task using given configuration.

    Args:
        config (DictConfig): The configuration object with project and experiment names.
    """
    task = Task.init(
        project_name=config.project_name,
        task_name=config.task_name,
        reuse_last_task_id=False,
        tags=config.tags,
        auto_connect_frameworks={
            "tensorboard": {"report_hparams": True},
            "matplotlib": True,
            "hydra": True,
            "pytorch": "*.ckpt",
            "detect_repository": True,
            "jsonargparse": True,
        },
    )
    task.set_comment(config.description)


def train(config: DictConfig) -> Dict[str, Any]:
    """
    Train and test a model based on the provided configuration and trainer parameters.

    Args:
        config (DictConfig): Configuration object containing details for the data module and model runner.

    Returns:
        Dict[str, Any]: A dictionary containing the training and testing metrics.

    Raises:
        RuntimeError: If no seed is found in the configuration.

    Note:
        The function will set global seeds for reproducibility and will also run the test phase on
        the best checkpoint after the training phase is complete.
    """
    # Set reproducibility
    if config.get("seed", False):
        pylight.seed_everything(config.seed, workers=True)
    else:
        raise RuntimeError("No seed found! Unable to ensure reproducibility.")

    logger.info("Instantiating data entities")  # noqa: WPS237
    train_dataloader = hydra.utils.instantiate(config.data.train_dataloader)
    val_dataloader = hydra.utils.instantiate(config.data.val_dataloader)

    if config.data.get("test_dataloader"):
        test_dataloader = hydra.utils.instantiate(config.data.test_dataloader)
    else:
        test_dataloader = None

    model = hydra.utils.instantiate(config.model.model)
    schopt_handler = hydra.utils.instantiate(config.optimizer.schopt_handler, model_parameters=model.parameters())
    losses = hydra.utils.instantiate(config.losses)

    if not config.is_local_run:
        load_dotenv(config.dotenv_path)
        setup_clearml_task(config)

    logger.info("Starting training!")
    return train_loop(
        model_type=config.model.model_type,
        epochs=config.optimizer.epochs,
        device=config.device,
        model=model,
        losses=losses,
        schopt_handler=schopt_handler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        log_dir=config.paths.output_dir,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")  # type: ignore
def main(config: DictConfig) -> Optional[float]:
    """
    Initialize training based on the provided configuration.

    Args:
        config (DictConfig): Configuration object containing details for training.

    Returns:
        Optional[float]: The value of the optimized metric if available, otherwise None.
    """
    metrics = train(config=config)
    return save_metrics(metrics, config.paths.output_dir, config.get("optimized_metric"))


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
