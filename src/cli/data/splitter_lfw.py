"""Split the LFW dataset into train, val, and test sets."""

import os
import random

import click
from loguru import logger


@click.command()
@click.option("--path_to_folders", type=str, default="data/lfw")
@click.option("--path_to_save", type=str, default="data/lfw_split")
def split_dataset(path_to_folders: str, path_to_save: str):
    """Split the LFW dataset into train, val, and test sets.

    Args:
        path_to_folders: The path to the LFW dataset.
        path_to_save: The path to save the splits to.
    """
    os.makedirs(path_to_save, exist_ok=True)
    # get all folders in the path
    folders = [
        file
        for file in os.listdir(path_to_folders)
        if os.path.isdir(os.path.join(path_to_folders, file))  # noqa:WPS221
    ]
    num_folders = len(folders)
    logger.info(f"Found {num_folders} folders in {path_to_folders}")

    # split the folders into train, val, and test sets
    random.shuffle(folders)
    train_folders = folders[: int(num_folders * 0.8)]
    val_folders = folders[int(num_folders * 0.8) : int(num_folders * 0.9)]
    test_folders = folders[int(num_folders * 0.9) :]

    split_sizes = f"{len(train_folders)}/{len(val_folders)}/{len(test_folders)}"  # noqa: WPS237,WPS221
    logger.info(f"Splitting into train, val, and test sets: {split_sizes}")

    # save the splits to files
    with open(os.path.join(path_to_save, "train.txt"), "w") as f:
        for folder in train_folders:
            # iterate over all files in the folder
            for file in os.listdir(os.path.join(path_to_folders, folder)):
                f.write(f"{os.path.join(path_to_folders, folder, file)}\n")

    with open(os.path.join(path_to_save, "val.txt"), "w") as f:
        for folder in val_folders:
            # iterate over all files in the folder
            for file in os.listdir(os.path.join(path_to_folders, folder)):
                f.write(f"{os.path.join(path_to_folders, folder, file)}\n")

    with open(os.path.join(path_to_save, "test.txt"), "w") as f:
        for folder in test_folders:
            # iterate over all files in the folder
            for file in os.listdir(os.path.join(path_to_folders, folder)):
                f.write(f"{os.path.join(path_to_folders, folder, file)}\n")


if __name__ == "__main__":
    split_dataset()
