"""Download and extract the LFW dataset."""

import os
import tarfile

import click
import requests
from loguru import logger
from tqdm import tqdm


@click.command()
@click.option("--output_dir", type=str, default="data/lfw")
def download_dataset(output_dir: str):
    """Download and extract the LFW dataset.

    Args:
        output_dir: The directory to download the dataset to.
    """
    # download the data archive and extract it if it does not exist
    if not os.path.exists(output_dir):
        logger.info("Images not found, donwloading...")

        with requests.get("http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz", stream=True, timeout=10) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            with (  # noqa: WPS316
                open("tmp.tgz", "wb") as file,
                tqdm(total=total_size, unit="iB", unit_scale=True) as progress_bar,
            ):
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

        logger.info("extracting...")
        with tarfile.open("tmp.tgz", "r:gz") as tar:
            tar.extractall(path=output_dir)  # noqa: S202
        os.remove("tmp.tgz")
        logger.info("done")
        assert os.path.exists(output_dir)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    download_dataset()
