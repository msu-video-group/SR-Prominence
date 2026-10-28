"""The baseline fusion model: a per-pixel MLP over the feature heatmaps."""

from pathlib import Path

import torch
from torch import nn


class ArtifactMetric(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()

        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        pad = (kernel_size // 2) * 2
        if pad > 0:
            pad = nn.ZeroPad2d(pad)
        else:
            pad = nn.Identity()

        self.fusion_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            pad,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map (B, C, H, W) feature heatmaps to (B, 1, H, W) predicted prominence."""
        return self.fusion_head(features)


def load_checkpoint(checkpoint: Path, device: torch.device) -> tuple[ArtifactMetric, list[str]]:
    """Load a trained Lightning checkpoint, returning the model and its input feature names."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = ckpt["state_dict"]
    hparams = ckpt.get("hyper_parameters", {})
    features = list(hparams.get("features", ("dists", "ssm_jup", "bd_jup")))

    model = ArtifactMetric(
        in_channels=len(features),
        hidden_channels=hparams.get("hidden_channels", 64),
        kernel_size=hparams.get("kernel_size", 3),
    )
    model.load_state_dict({k.removeprefix("model."): v for k, v in state_dict.items()})
    model.to(device)
    model.eval()
    return model, features
