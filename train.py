#!/usr/bin/env python3
"""Train the SR artifact-prominence baseline."""

import argparse
import datetime
from pathlib import Path
from typing import Literal, Sequence

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from scipy.stats import spearmanr
from torch.nn import functional as F
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset

from data import (
    AnnotationRecord,
    check_features,
    erode_ann,
    load_annotations,
    load_features,
    load_mask,
)
from model import ArtifactMetric

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output"
DEFAULT_TRAIN_CSV = SCRIPT_DIR / "splits/open_images_train_half_curated.csv"
DEFAULT_VAL_CSV = SCRIPT_DIR / "splits/open_images_val_half_curated.csv"
FEATURES = ["dists", "ssm_jup", "bd_jup"]

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class AnnotationDataset(Dataset):
    def __init__(
        self,
        records: Sequence[AnnotationRecord],
        features: Sequence[str],
        heatmaps_root: Path,
        *,
        erode: bool,
    ):
        """
        Parameters
        ----------
        erode : bool
            Whether to erode GT masks that have been dilated
            (use for training, to undo the subjective postprocessing dilation).
        """
        self.records = list(records)
        self.features = list(features)
        self.heatmaps_root = heatmaps_root
        self.erode = erode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Batch:
        record = self.records[idx]
        mask = load_mask(record.mask_path)
        if self.erode and record.dilated:
            mask = erode_ann(mask)

        features = torch.as_tensor(
            load_features(record, self.features, self.heatmaps_root), dtype=torch.float32
        )
        mask = torch.as_tensor(mask)
        ratio_bi = torch.as_tensor(record.ratio_bi, dtype=torch.float32)
        return features, mask, ratio_bi


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of x over mask per image, NaN for an empty mask."""
    return torch.nanmean(torch.where(mask, x, torch.nan), dim=(1, 2))


class LitMetric(pl.LightningModule):
    def __init__(
        self,
        *,
        train_gt: Literal["GT", "RLFN_x4"],
        features: list[str],
        dataset_root: Path,
        output_root: Path,
        train_csv: Path,
        val_csv: Path,
        batch_size: int,
        hidden_channels: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset_root", "output_root", "train_csv", "val_csv"])
        self.hparams.update(
            {
                "dataset_root": str(dataset_root),
                "output_root": str(output_root),
                "train_csv": str(train_csv),
                "val_csv": str(val_csv),
            }
        )
        self.train_gt = train_gt
        self.dataset_root = Path(dataset_root)
        self.images_root = self.dataset_root / "images/open_images"
        self.train_csv = Path(train_csv)
        self.val_csv = Path(val_csv)
        self.heatmaps_root = self.dataset_root / f"heatmaps/open_images/gt_{self.train_gt}"
        self.batch_size = batch_size
        self.features = list(features)

        self.model = ArtifactMetric(
            in_channels=len(self.features),
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        self.val_mean_gtarea: list[torch.Tensor] = []
        self.val_y: list[torch.Tensor] = []

    def setup(self, stage: str):
        if stage != "fit":
            return

        self.train_ds, self.val_ds = (
            AnnotationDataset(
                load_annotations(csv, self.images_root),
                self.features,
                self.heatmaps_root,
                erode=True,
            )
            for csv in (self.train_csv, self.val_csv)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def compute_loss(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Regress the mean prediction to ratio_bi inside the mask and to 0 outside it."""
        x, ann, y = batch
        pred = self.model(x).squeeze(1)

        mean_gtarea = masked_mean(pred, ann)
        # Images with an empty mask are NaN; replace them with zeros to avoid corrupting the loss.
        mean_gtarea_zero = torch.where(mean_gtarea.isnan(), 0, mean_gtarea)
        loss_gtarea = F.mse_loss(mean_gtarea_zero, y)

        mean_outside = masked_mean(pred, ~ann)
        mean_outside_zero = torch.where(mean_outside.isnan(), 0, mean_outside)
        loss_outside = F.mse_loss(mean_outside_zero, torch.zeros_like(mean_outside_zero))

        loss = loss_gtarea + loss_outside
        return mean_gtarea, {
            "loss_gtarea": loss_gtarea,
            "loss_outside": loss_outside,
            "loss": loss,
        }

    def on_train_epoch_start(self):
        torch.cuda.empty_cache()

    def training_step(self, batch: Batch, batch_idx: int):
        _, losses = self.compute_loss(batch)
        self.log_dict({f"train/{name}": value for name, value in losses.items()})
        return losses["loss"]

    def validation_step(self, batch: Batch, batch_idx: int):
        mean_gtarea, losses = self.compute_loss(batch)
        self.log_dict({f"val/{name}": value for name, value in losses.items()})
        self.val_mean_gtarea.append(mean_gtarea)
        self.val_y.append(batch[2])

    def on_validation_epoch_end(self):
        mean_gtarea = torch.cat(self.val_mean_gtarea)
        y = torch.cat(self.val_y)
        self.val_mean_gtarea.clear()
        self.val_y.clear()

        plcc_gtarea = torch.corrcoef(torch.stack((mean_gtarea, y)))[0, 1].item()
        srcc_gtarea = spearmanr(mean_gtarea.numpy(force=True), y.numpy(force=True)).statistic
        self.log_dict({"val/plcc_gtarea": plcc_gtarea, "val/srcc_gtarea": srcc_gtarea})

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.001)
        return {"optimizer": optimizer}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SR artifact-prominence baseline")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--gt", choices=("GT", "RLFN_x4"), default="GT")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--accumulate-grad-batches", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    images_root = dataset_root / "images/open_images"
    train_gt = args.gt
    heatmaps_root = dataset_root / f"heatmaps/open_images/gt_{train_gt}"
    train_csv = args.train_csv.absolute()
    val_csv = args.val_csv.absolute()

    check_features(
        load_annotations(train_csv, images_root) + load_annotations(val_csv, images_root),
        FEATURES,
        heatmaps_root,
    )

    seed = 42
    pl.seed_everything(seed, workers=True)

    hidden_channels = 128
    kernel_size = 1
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    components = [f"gt{train_gt}", f"c{hidden_channels}k{kernel_size}"]
    save_dir = output_root / "logdir" / (date_str + "-" + "-".join(components))
    logger = TensorBoardLogger(save_dir, name=None)

    model = LitMetric(
        train_gt=train_gt,
        features=list(FEATURES),
        dataset_root=dataset_root,
        output_root=output_root,
        train_csv=train_csv,
        val_csv=val_csv,
        batch_size=args.batch_size,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
    )

    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            filename="epoch={epoch}-step={step}",
            auto_insert_metric_name=False,
            save_top_k=-1,
            save_last=True,
            every_n_epochs=1,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=30,
        log_every_n_steps=1,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        deterministic=True,
        accumulate_grad_batches=args.accumulate_grad_batches,
        profiler="simple",
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
