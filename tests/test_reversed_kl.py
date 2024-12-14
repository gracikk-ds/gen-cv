"""Test reversed KL."""

import pytest
import torch
from torch import Tensor

from src.losses.vae_loss import reversed_kl_div

EPS: float = 1e-4
MAGIC_CONST: float = 2.8069


@pytest.mark.parametrize(  # type: ignore
    "mu, logvar, expected_output",
    [
        (
            torch.zeros((1, 24, 8, 8)),
            torch.log(torch.ones((1, 24, 8, 8)) ** 2),
            torch.zeros((1, 24, 8, 8)),
        ),
        (
            torch.ones((1, 24, 8, 8)),
            torch.log(torch.ones((1, 24, 8, 8)) ** 2),
            torch.zeros((1, 24, 8, 8)) + 0.5,
        ),
        (
            torch.ones((1, 24, 8, 8)) * 2,
            torch.log(torch.ones((1, 24, 8, 8)).mul(2) ** 2),  # noqa: WPS221
            torch.zeros((1, 24, 8, 8)) + MAGIC_CONST,
        ),
    ],
)
def test_reversed_kl(mu: Tensor, logvar: Tensor, expected_output: Tensor):
    """Test reversed KL.

    Args:
        mu (Tensor): mean tensor
        logvar (Tensor): log variance tensor
        expected_output (Tensor): expected output tensor
    """
    assert torch.allclose(reversed_kl_div(mu, logvar), expected_output, rtol=EPS)
