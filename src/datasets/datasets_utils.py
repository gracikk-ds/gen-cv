"""Utility functions for datasets."""

import os
from random import sample
from typing import List

from loguru import logger


def load_image_paths(images_path: str, n_samples: int = -1) -> List[str]:
    """Load paths of .jpg images from the directory. Optionally sample n_samples of them.

    Args:
        images_path (str): Path to the directory containing the images.
        n_samples (int): Number of samples to load.

    Returns:
        List[str]: List of image file paths.
    """
    paths = []
    for dirpath, _, filenames in os.walk(images_path):
        for fname in filenames:
            if fname.endswith(".jpg"):
                paths.append(os.path.join(dirpath, fname))
    logger.info(f"Found {len(paths)} images in {images_path}")

    if n_samples != -1 and n_samples < len(paths):
        logger.info(f"Sampling {n_samples} images")
        paths = sample(paths, n_samples)
    return paths


def load_paths_from_split(split_path: str, n_samples: int) -> List[str]:
    """Load paths from a split file.

    Args:
        split_path (str): Path to the split file.
        n_samples (int): Number of samples to load.

    Returns:
        List[str]: List of image file paths.
    """
    with open(split_path, "r") as f:
        paths = [line.strip() for line in f.readlines()]
    if n_samples != -1 and n_samples < len(paths):
        logger.info(f"Sampling {n_samples} images")
        paths = sample(paths, n_samples)
    return paths
