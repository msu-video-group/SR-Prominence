#!/usr/bin/env python3
"""Compute baseline heatmaps from a trained Lightning checkpoint."""

import argparse
from gzip import GzipFile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data import AnnotationRecord, check_features, load_annotations, load_features
from model import load_checkpoint

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output"

SUBSETS = ("open_images", "urban100")


def checkpoint_run_name(checkpoint: Path) -> str:
    """Name a training run checkpoint <run>-<version>[-<stem>], or a standalone one <stem>."""
    version = checkpoint.parent.parent
    if checkpoint.parent.name != "checkpoints" or not version.name.startswith("version_"):
        return checkpoint.stem

    name = f"{version.parent.name}-{version.name}"
    if checkpoint.stem != "last":
        name += f"-{checkpoint.stem}"
    return name


def load_method(
    checkpoint: Path,
    *,
    device: torch.device,
    records: list[AnnotationRecord],
    heatmaps_root: Path,
):
    model, features = load_checkpoint(checkpoint, device)
    check_features(records, features, heatmaps_root)

    def method(record: AnnotationRecord) -> np.ndarray:
        x = torch.as_tensor(load_features(record, features, heatmaps_root), dtype=torch.float32)
        with torch.inference_mode():
            pred = model(x.unsqueeze(0).to(device))
        return pred.squeeze((0, 1)).cpu().numpy(force=True)

    return method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained baseline checkpoint")
    parser.add_argument("--subset", choices=SUBSETS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gt", choices=("GT", "RLFN_x4"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    checkpoint = args.checkpoint.resolve()

    # Covers every annotated SR image of the subset; open_images_uncurated is a subset of it.
    annotations = dataset_root / f"annotations/{args.subset}.csv"
    images_root = dataset_root / f"images/{args.subset}"
    records = load_annotations(annotations, images_root)

    heatmaps = Path(f"heatmaps/{args.subset}/gt_{args.gt}")
    # The paper checkpoint was trained as an nn.Linear MLP and converted to 1x1 convolutions.
    # Disable TF32 convolutions to match its full FP32 precision.
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = load_method(
        checkpoint, device=device, records=records, heatmaps_root=dataset_root / heatmaps
    )

    predictions_root = output_root / heatmaps / checkpoint_run_name(checkpoint)
    predictions_root.mkdir(parents=True, exist_ok=True)
    print(f"Writing heatmaps to: {predictions_root}")

    unique_records = {record.sr_path: record for record in records}
    for record in tqdm(unique_records.values(), desc="evaluate", ncols=80):
        output = predictions_root / f"{record.sr_path.stem}.npy.gz"
        if output.exists():
            continue
        result = method(record)
        with GzipFile(output, "w") as f:
            np.save(f, result)


if __name__ == "__main__":
    main()
