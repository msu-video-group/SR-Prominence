#!/usr/bin/env python3
"""Compute the dists, ssm_jup, and bd_jup feature heatmaps for a dataset subset.

Heatmaps are written next to the ones shipped with the dataset, to
<dataset-root>/heatmaps/<subset>/gt_<gt>/<feature>/<sr_stem>.npy.gz.
Heatmaps that already exist, whether shipped or computed earlier, are skipped.
"""

import argparse
from gzip import GzipFile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from bd_jup import BdJupFeature
from data import load_annotations, metric_input_paths
from dists import DistsFeature
from ssm_jup import SsmJupFeature

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR.parent
SUBSETS = ("open_images", "urban100")
FEATURES = {
    "dists": DistsFeature,
    "ssm_jup": SsmJupFeature,
    "bd_jup": BdJupFeature,
}


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subset", choices=SUBSETS, required=True)
    parser.add_argument("--gt", choices=("GT", "RLFN_x4"), required=True)
    parser.add_argument(
        "--features", nargs="+", choices=list(FEATURES), default=list(FEATURES)
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    records = load_annotations(
        dataset_root / f"annotations/{args.subset}.csv", dataset_root / f"images/{args.subset}"
    )
    heatmaps_root = dataset_root / f"heatmaps/{args.subset}/gt_{args.gt}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for feature in args.features:
        output_dir = heatmaps_root / feature
        output_dir.mkdir(parents=True, exist_ok=True)
        pending = {
            record.sr_path: metric_input_paths(record, args.gt)
            for record in records
            if not (output_dir / f"{record.sr_path.stem}.npy.gz").exists()
        }
        if not pending:
            continue

        metric = FEATURES[feature](device)
        for gt_path, sr_path, lr_path in tqdm(
            pending.values(), desc=f"compute {feature}", ncols=80
        ):
            result = metric(gt=load_rgb(gt_path), sr=load_rgb(sr_path), lr=load_rgb(lr_path))
            with GzipFile(output_dir / f"{sr_path.stem}.npy.gz", "w") as f:
                np.save(f, result)
        del metric


if __name__ == "__main__":
    main()
