"""dists heatmap: DISTS computed block-wise in 16x16-pixel blocks."""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from kornia.contrib import combine_tensor_patches, extract_tensor_patches
from PIL import Image

from third_party.dists.DISTS_pt import DISTS


class DISTSHeatmap:
    BLOCK_SIZE = 16
    BATCH_SIZE = 1000

    def __init__(self, device: torch.device):
        self.device = device
        self.metric = DISTS().to(device).eval()

    def __call__(self, *, gt: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_patches = extract_tensor_patches(
            target,
            window_size=self.BLOCK_SIZE,
            stride=self.BLOCK_SIZE,
            allow_auto_padding=True,
        )[0]
        gt_patches = extract_tensor_patches(
            gt,
            window_size=self.BLOCK_SIZE,
            stride=self.BLOCK_SIZE,
            allow_auto_padding=True,
        )[0]

        results = []
        for i in range(0, len(target_patches), self.BATCH_SIZE):
            result = self.metric(
                target_patches[i : i + self.BATCH_SIZE],
                gt_patches[i : i + self.BATCH_SIZE],
                batch_average=False,
            ).reshape(-1, 1)
            results.append(
                result.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(1, -1, 1, self.BLOCK_SIZE, self.BLOCK_SIZE)
            )

        result = torch.cat(results, dim=1)
        result = combine_tensor_patches(
            result,
            target.shape[-2:],
            window_size=self.BLOCK_SIZE,
            stride=self.BLOCK_SIZE,
            allow_auto_unpadding=True,
        )[0, 0]
        return result


class DistsFeature:
    """Compute the dists heatmap of an SR image against a reference."""

    def __init__(self, device: torch.device):
        # Batched grouped convolutions in DISTS' L2 pooling produce NaNs on ROCm.
        # Keep NVIDIA CUDA acceleration, but use the exact same implementation on CPU for ROCm.
        if device.type == "cuda" and torch.version.hip:
            print("Computing DISTS on CPU because its grouped pooling is unreliable on ROCm")
            device = torch.device("cpu")
        self.device = device
        self.metric = DISTSHeatmap(device)

    def __call__(self, *, gt: Image.Image, sr: Image.Image, lr: Image.Image) -> np.ndarray:
        target = TF.to_tensor(sr).unsqueeze(0).to(self.device)
        reference = TF.to_tensor(gt).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return self.metric(gt=reference, target=target).cpu().numpy()
