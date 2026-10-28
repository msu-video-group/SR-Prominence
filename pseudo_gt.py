"""RLFN pseudo-GT: a reference for full-reference features when there is no HR ground truth."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from third_party.rlfn.rlfn import RLFN

WEIGHTS = Path(__file__).resolve().parent / "third_party/rlfn/rlfn_x4_finetuned.pth"


class PseudoGT:
    """Upscale an LR image ×4 with a finetuned RLFN, the dataset's RLFN_x4 reference."""

    def __init__(self, device: torch.device):
        self.device = device
        self.model = RLFN()
        self.model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
        self.model.to(device).eval()

    def __call__(self, lr: Image.Image) -> Image.Image:
        x = torch.from_numpy(np.array(lr)).permute(2, 0, 1).float().div(255)
        with torch.inference_mode():
            y = self.model(x.unsqueeze(0).to(self.device))[0]
        y = y.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(y)
