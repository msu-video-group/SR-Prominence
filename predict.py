#!/usr/bin/env python3
"""Predict the artifact-prominence map of a ×4 super-resolved image.

The reference is either the original HR image (--gt) or, without one, an RLFN upscale of
the LR image (--lr), the paper's pseudo-GT protocol. The map has the SR image's resolution.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from compute_features import FEATURES
from model import load_checkpoint
from pseudo_gt import PseudoGT

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints/nn-20250421-gtgt-e30.ckpt"
SCALE = 4
# The smallest SR image the features support.
MIN_SR_SIZE = 64

ImageInput = Image.Image | str | Path


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("I", "I;16", "I;16B", "I;16L"):
        # PIL clips 16-bit values to 255 when converting to RGB, so scale them to 8 bits first.
        image = Image.fromarray((np.asarray(image) >> 8).astype(np.uint8))
    return image.convert("RGB")


def load_rgb(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return to_rgb(image)
    with Image.open(image) as f:
        return to_rgb(f)


class ProminenceModel:
    """The baseline prominence detector: three full-reference features fused by an MLP."""

    def __init__(
        self, device: torch.device | str | None = None, checkpoint: Path = DEFAULT_CHECKPOINT
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, feature_names = load_checkpoint(checkpoint, self.device)
        self.features = [FEATURES[name](self.device) for name in feature_names]
        self.pseudo_gt = None

    def predict(
        self, sr: ImageInput, *, lr: ImageInput | None = None, gt: ImageInput | None = None
    ) -> np.ndarray:
        """Return the (H, W) prominence map of sr, against gt if given, else against lr."""
        sr = load_rgb(sr)
        if min(sr.size) < MIN_SR_SIZE:
            raise ValueError(f"SR size {sr.size} must be at least {MIN_SR_SIZE}x{MIN_SR_SIZE}")
        lr = None if lr is None else load_rgb(lr)
        if lr is not None and sr.size != (lr.width * SCALE, lr.height * SCALE):
            raise ValueError(f"SR size {sr.size} must be {SCALE}x the LR size {lr.size}")

        if gt is not None:
            gt = load_rgb(gt)
            if gt.size != sr.size:
                raise ValueError(f"GT size {gt.size} must match the SR size {sr.size}")
            if lr is None:
                if gt.width % SCALE or gt.height % SCALE:
                    raise ValueError(
                        f"GT size {gt.size} must be a multiple of {SCALE} to derive the LR image"
                    )
                lr = gt.resize((gt.width // SCALE, gt.height // SCALE), Image.Resampling.BICUBIC)
        elif lr is not None:
            if self.pseudo_gt is None:
                self.pseudo_gt = PseudoGT(self.device)
            gt = self.pseudo_gt(lr)
        else:
            raise ValueError("Pass lr, gt, or both")

        x = np.stack([feature(gt=gt, sr=sr, lr=lr) for feature in self.features])
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        # The paper checkpoint was trained as an nn.Linear MLP and converted to 1x1
        # convolutions. Disable TF32 convolutions to match its full FP32 precision.
        with (
            torch.inference_mode(),
            torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled, allow_tf32=False),
        ):
            return self.model(x)[0, 0].cpu().numpy()


def colorize(prominence: np.ndarray) -> np.ndarray:
    """Map prominence in [0, 1] (clipped) to an RGB inferno colormap."""
    gray = np.round(np.clip(prominence, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sr", type=Path, required=True, help="super-resolved image")
    parser.add_argument("--lr", type=Path, help="low-resolution input of the SR method")
    parser.add_argument("--gt", type=Path, help="original high-resolution image")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output path without extension; writes <output>.npy and <output>.png "
        "(default: <SR name>-prominence in the current directory)",
    )
    parser.add_argument(
        "--overlay", action="store_true", help="blend the PNG map over the SR image"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="trained model checkpoint (default: the paper's)",
    )
    args = parser.parse_args()
    if args.lr is None and args.gt is None:
        parser.error("pass --lr, --gt, or both")
    return args


def main() -> None:
    args = parse_args()
    output = args.output or Path(f"{args.sr.stem}-prominence")
    output.parent.mkdir(parents=True, exist_ok=True)

    model = ProminenceModel(checkpoint=args.checkpoint)
    print(f"Running on {model.device}")
    prominence = model.predict(args.sr, lr=args.lr, gt=args.gt)

    np.save(output.with_name(f"{output.name}.npy"), prominence)
    image = colorize(prominence)
    if args.overlay:
        sr = np.asarray(load_rgb(args.sr), dtype=np.float32)
        image = np.round(0.5 * sr + 0.5 * image).astype(np.uint8)
    Image.fromarray(image).save(output.with_name(f"{output.name}.png"))
    print(f"Wrote {output}.npy and {output}.png")


if __name__ == "__main__":
    main()
