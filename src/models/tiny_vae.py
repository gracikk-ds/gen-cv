"""TinyVAE model."""

from typing import Tuple

import torch
from torch import Tensor, nn


class TinyVAE(nn.Module):  # noqa: WPS230
    """Tiny VAE model with no skip connections."""

    def __init__(self, tmp_ch: int = 6, latent_dim: int = 128, input_size: int = 64, num_blocks: int = 2):
        """
        Initialize TinyVAE.

        Args:
            tmp_ch (int): Number of channels in the first layer. Defaults to 6.
            latent_dim (int): Dimension of the latent space. Defaults to 128.
            input_size (int): Size of the input images (assumed square). Defaults to 64.
            num_blocks (int): Number of encoder and decoder blocks. Defaults to 2.
        """
        super().__init__()
        self.tmp_ch = tmp_ch
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.num_blocks = num_blocks

        # Encoder
        self.encoder = nn.ModuleDict()
        self.encoder["in_conv"] = nn.Conv2d(3, tmp_ch, 1)
        stride_total = 1

        # Build encoder blocks
        encoder_channels = tmp_ch
        for idx in range(num_blocks):
            self.encoder[f"block_{idx}"] = nn.Sequential(
                nn.Conv2d(encoder_channels, encoder_channels * 2, 3, padding=1, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(encoder_channels * 2),
            )
            encoder_channels = encoder_channels * 2
            stride_total *= 2

        self.encoder["out_conv"] = nn.Conv2d(encoder_channels, encoder_channels * 2, 1)
        encoder_channels = encoder_channels * 2

        self.stride_total = stride_total
        assert self.input_size % self.stride_total == 0, "Input size must be divisible by stride total"

        # Calculate final feature map size after downsampling
        self.feature_map_size = self.input_size // self.stride_total

        # Flatten and FC for latent space
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(encoder_channels * self.feature_map_size * self.feature_map_size, latent_dim)
        self.fc_logvar = nn.Linear(encoder_channels * self.feature_map_size * self.feature_map_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, encoder_channels * self.feature_map_size * self.feature_map_size)

        # Decoder
        self.decoder = nn.ModuleDict()
        # First layer in decoder (reverse of out_conv)
        self.decoder["in_conv"] = nn.Conv2d(encoder_channels, encoder_channels // 2, 1)
        decoder_channels = encoder_channels // 2

        # Decoder blocks (no skip connections)
        for idx in range(num_blocks):
            self.decoder[f"block_{idx}"] = nn.Sequential(
                nn.ConvTranspose2d(decoder_channels, decoder_channels // 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(decoder_channels // 2),
            )
            decoder_channels = decoder_channels // 2

        self.decoder["out_conv"] = nn.Conv2d(decoder_channels, 3, 1)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self) -> None:  # noqa: WPS231
        """Initialize weights of all layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick.

        Args:
            mu (Tensor): Mean tensor.
            logvar (Tensor): Log variance tensor.

        Returns:
            Tensor: Latent tensor.
        """
        std = torch.exp(logvar.mul(0.5))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            images (Tensor): Input images.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Reconstruction, mean, logvar.
        """
        # Encoder forward pass
        embs = self.encoder["in_conv"](images)
        for name, layer in self.encoder.items():
            if name not in {"in_conv", "out_conv"}:
                embs = layer(embs)
        embs = self.encoder["out_conv"](embs)

        # Flatten and calculate mu and logvar
        embs = self.flatten(embs)
        mu = self.fc_mu(embs)
        logvar = self.fc_logvar(embs)

        # Reparameterization
        latent = self.reparameterize(mu, logvar)

        # Decoder forward pass
        embs = self.fc_decode(latent)
        # Reshape into decoder input
        ch = self.decoder["in_conv"].in_channels
        embs = embs.view(-1, ch, self.feature_map_size, self.feature_map_size)
        embs = self.decoder["in_conv"](embs)

        for idx in range(self.num_blocks):
            embs = self.decoder[f"block_{idx}"](embs)

        # Final output layer
        reconstruction = self.decoder["out_conv"](embs)
        return reconstruction, mu, logvar
