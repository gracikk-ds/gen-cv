"""VAE dataset."""

from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.datasets_utils import load_paths_from_split


class VAEDataset(Dataset):
    """VAE dataset."""

    def __init__(self, split_path: str, n_samples: int, center_crop: int, resize: Tuple[int, int]):
        """Initialize the dataset.

        Args:
            split_path (str): Path to the split file.
            n_samples (int): Number of samples to load.
            center_crop (int): Size of the center crop.
            resize (Tuple[int, int]): Size of the resized image.
        """
        self.file_paths = load_paths_from_split(split_path, n_samples)
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(center_crop),
                transforms.Resize(resize, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),
            ],
        )

    def __len__(self) -> int:
        """Return the total number of image file paths.

        Returns:
            int: Number of image file paths.
        """
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieve an image by index, applies transformations if provided.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: The retrieved image, possibly transformed.
        """
        img_path = self.file_paths[idx]
        with open(img_path, "rb") as f:
            image = Image.open(f).convert("RGB")

        # Apply transformations
        return self.transform(image)
