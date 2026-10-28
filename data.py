"""
Data-loading helpers for SR artifact experiments.
"""

import csv
from dataclasses import dataclass
from gzip import GzipFile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnnotationRecord:
    mask_path: Path
    sr_path: Path
    ratio_bi: float
    dilated: bool  # whether the mask was dilated and should be eroded


def parse_bool(value: str) -> bool:
    if value == "False":
        return False
    if value == "True":
        return True
    raise ValueError(f'invalid boolean value, can be "True" or "False": {value!r}')


def load_annotations(path: Path, images_root: Path) -> list[AnnotationRecord]:
    records = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            root = images_root / row["folder"]
            records.append(
                AnnotationRecord(
                    mask_path=root / row["mask_fn"],
                    sr_path=root / row["sr_fn"],
                    ratio_bi=float(row["ratio_bi"]),
                    dilated=parse_bool(row.get("dilated", "False")),
                )
            )
    return records


def metric_input_paths(record: AnnotationRecord, gt: str) -> tuple[Path, Path, Path]:
    """Return (reference, target SR, bicubic LR) for a reference gt (GT or an SR method)."""
    root = record.mask_path.parent
    name = root.name

    if gt == "GT":
        gt_path = root / f"{name}.png"
    else:
        gt_path = root / f"{name}@SR@{gt}.png"

    lr_path = root / f"{name}@LR@bicubic_x4.png"
    return gt_path, record.sr_path, lr_path


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def load_mask(mask_path: Path) -> np.ndarray:
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise FileNotFoundError(mask_path)
    return mask_gray == 255


def gz_npy_load(path: Path) -> np.ndarray:
    try:
        with GzipFile(path) as f:
            return np.load(f)
    except Exception:
        raise RuntimeError(f"Error loading {path}")


def feature_path(record: AnnotationRecord, feature: str, heatmaps_root: Path) -> Path:
    return heatmaps_root / feature / f"{record.sr_path.stem}.npy.gz"


def check_features(
    records: Sequence[AnnotationRecord], features: Sequence[str], heatmaps_root: Path
) -> None:
    """Fail early, pointing to compute_features.py, if any feature heatmap is missing."""
    missing = {
        path
        for feature in features
        for record in records
        if not (path := feature_path(record, feature, heatmaps_root)).exists()
    }
    if missing:
        subset = heatmaps_root.parent.name
        gt = heatmaps_root.name.removeprefix("gt_")
        raise FileNotFoundError(
            f"{len(missing)} feature heatmaps are missing, e.g. {min(missing)}; compute them with "
            f"`uv run compute_features.py --subset {subset} --gt {gt}`"
        )


def load_features(
    record: AnnotationRecord, features: Sequence[str], heatmaps_root: Path
) -> np.ndarray:
    """Load and stack the feature heatmaps (gzip-compressed (H, W) arrays) into (C, H, W)."""
    heatmaps = []
    for feature in features:
        heatmap_path = feature_path(record, feature, heatmaps_root)
        heatmap = gz_npy_load(heatmap_path)
        if heatmap.ndim != 2:
            raise ValueError(
                f"Heatmap file {heatmap_path} must have shape (H, W), got {heatmap.shape}"
            )
        heatmaps.append(heatmap)
    return np.stack(heatmaps)


# ---------------------------------------------------------------------------
# Mask processing
# ---------------------------------------------------------------------------


def erode_ann(ann: np.ndarray) -> np.ndarray:
    if ann.dtype != bool:
        raise TypeError("Mask must be boolean")
    ann = np.where(ann, np.uint8(255), np.uint8(0))

    # Cut out the border pixels that were zeroed for subjective masks.
    CUT = 12
    ann = ann[CUT:-CUT, CUT:-CUT]

    kds = max(64, min(ann.shape[0], ann.shape[1]) // 12)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [kds] * 2)
    ann_eroded = cv2.morphologyEx(ann, cv2.MORPH_ERODE, kd, borderValue=[255])

    # Erosion can get rid of the entire annotation, in this case just use the original.
    if np.count_nonzero(ann_eroded) > 0:
        ann = ann_eroded

    ann = np.pad(ann, (CUT,))
    return ann == 255
