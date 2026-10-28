"""bd_jup heatmap: a weighted sum of block-wise LPIPS and inverted ERQA."""

from functools import partial

import kornia as K
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


# ERQA is adapted from https://github.com/msu-video-group/erqa
# (Copyright (c) 2021 Anastasia Kirillova, Eugene Lyapustin; MIT, see third_party/erqa/LICENSE),
# ported to kornia and trimmed to the configuration bd_jup uses.
class ERQA:
    """Edge Restoration Quality Assessment (ERQA): F1 score of matched Canny edges."""

    def __init__(self, *, shift_compensation, penalize_wider_edges, threshold1, threshold2):
        self.shift_compensation = shift_compensation
        self.penalize_wider_edges = penalize_wider_edges
        self.edge_detector = partial(
            K.filters.canny, low_threshold=threshold1, high_threshold=threshold2
        )

    def __call__(self, img, gt):
        assert gt.shape == img.shape

        edge = self.edge_detector(img)[1]
        gt_edge = self.edge_detector(gt)[1]
        true_positive, false_negative = self.match_edges(edge, gt_edge)
        return self.f1_matches(edge, true_positive, false_negative)

    def match_edges(self, edge, gt_edge):
        assert gt_edge.shape == edge.shape

        true_positive = torch.zeros_like(edge)
        false_negative = torch.clone(gt_edge)
        window_range = 1 if self.shift_compensation else 0
        window = sorted(range(-window_range, window_range + 1), key=abs)

        for i in window:
            for j in window:
                gt_ = torch.roll(false_negative, i, dims=1)
                gt_ = torch.roll(gt_, j, dims=0)
                ad = edge * gt_ * torch.logical_not(true_positive)
                true_positive = torch.logical_or(true_positive, ad)
                if self.penalize_wider_edges:
                    ad = torch.roll(ad, -j, dims=1)
                    ad = torch.roll(ad, -i, dims=0)
                    false_negative = torch.logical_and(false_negative, torch.logical_not(ad))

        if not self.penalize_wider_edges:
            false_negative = gt_edge * torch.logical_not(true_positive)

        assert not torch.logical_and(true_positive, false_negative).any()
        return true_positive.to(dtype=torch.uint8), false_negative.to(dtype=torch.uint8)

    def f1_matches(self, edge, true_positive, false_negative):
        tp = torch.sum(true_positive, dim=(1, 2, 3))
        fp = torch.sum(edge, dim=(1, 2, 3)) - tp
        fn = torch.sum(false_negative, dim=(1, 2, 3))

        prec = torch.div(tp, tp + fp)
        recall = torch.div(tp, tp + fn)
        f1 = torch.div(2 * prec * recall, prec + recall)
        f1[tp + fp == 0] = 0
        f1[tp == 0] = 0
        f1[tp + fn == 0] = 1
        return f1


def get_padding(side, patch_size, stride):
    subd = side - patch_size
    return int(np.ceil(subd / stride)) * stride - subd


def smart_unfold(image, patch_size, stride=None):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if stride is None:
        stride = patch_size
    *_, height, width = image.shape
    padding = (
        get_padding(height, patch_size, stride) // 2,
        get_padding(width, patch_size, stride) // 2,
    )
    image = F.unfold(image, patch_size, stride=stride, padding=padding)
    return image.view(1, 3, patch_size, patch_size, -1)


def get_fold_divisor(image: torch.Tensor, patch_size, stride, padding):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image[:1, :1]
    input_ones = torch.ones(image.shape, dtype=image.dtype)
    return F.fold(
        F.unfold(input_ones, patch_size, stride=stride, padding=padding),
        image.shape[-2:],
        patch_size,
        stride=stride,
        padding=padding,
    )


def probs_fold(probs, output_size, patch_size, stride, device):
    probs_repeated = (
        probs[:, None, None]
        .repeat(1, patch_size, patch_size, 1)
        .view(probs.shape[0], -1, probs.shape[1])
    )
    height, width = output_size
    padding = (
        get_padding(height, patch_size, stride) // 2,
        get_padding(width, patch_size, stride) // 2,
    )
    probs_folded = F.fold(
        probs_repeated, output_size, patch_size, stride=stride, padding=padding
    )
    divisor = get_fold_divisor(probs_folded, patch_size, stride, padding).to(device)
    return probs_folded / divisor


def patch_lpips(gt, sample, patch_size, metric, device, stride=None, batch_size=1024):
    if stride is None:
        stride = patch_size

    gt_blocks = torch.movedim(smart_unfold(gt, patch_size, stride), -1, 1)[0]
    sample_blocks = torch.movedim(smart_unfold(sample, patch_size, stride), -1, 1)[0]

    sample_batches = torch.split(sample_blocks, split_size_or_sections=batch_size, dim=0)
    gt_batches = torch.split(gt_blocks, split_size_or_sections=batch_size, dim=0)
    probs = [
        metric(gt_batch, sample_batch).view(-1)
        for gt_batch, sample_batch in zip(gt_batches, sample_batches)
    ]
    probs = torch.cat(probs).view(1, -1)
    probs = probs_fold(probs, gt.shape[-2:], patch_size, stride=stride, device=device)
    return probs[:, 0].squeeze(0)


def patch_erqa(gt, sample, patch_size, metric, device, stride=None, batch_size=1024):
    if stride is None:
        stride = patch_size

    gt_blocks = torch.movedim(smart_unfold(gt, patch_size, stride), -1, 1)[0]
    sample_blocks = torch.movedim(smart_unfold(sample, patch_size, stride), -1, 1)[0]
    n_blocks, *shape = gt_blocks.shape
    gt_blocks = gt_blocks.view(n_blocks, *shape)
    sample_blocks = sample_blocks.view(n_blocks, *shape)

    sample_batches = torch.split(sample_blocks, split_size_or_sections=batch_size, dim=0)
    gt_batches = torch.split(gt_blocks, split_size_or_sections=batch_size, dim=0)
    probs = [
        metric(sample_batch, gt_batch)
        for sample_batch, gt_batch in zip(sample_batches, gt_batches)
    ]
    probs = torch.cat(probs).float().to(device).view(1, -1)
    probs = probs_fold(probs, gt.shape[-2:], patch_size, stride=stride, device=device)
    return probs[:, 0].squeeze(0)


def get_heatmap(
    *,
    sample,
    gt,
    device: torch.device,
    lpips_metric,
    erqa_block_size,
    lpips_block_size,
    threshold1,
    threshold2,
    erqa_stride,
    lpips_stride,
    erqa_weight,
    lpips_weight,
):
    lpips_blocks = patch_lpips(
        gt, sample, lpips_block_size, lpips_metric, stride=lpips_stride, device=device
    )
    # ERQA 1.1 (penalize_wider_edges) with shift compensation.
    erqa = ERQA(
        shift_compensation=True,
        penalize_wider_edges=True,
        threshold1=threshold1,
        threshold2=threshold2,
    )
    erqa_blocks = patch_erqa(
        gt, sample, erqa_block_size, metric=erqa, stride=erqa_stride, device=device
    )
    final_map = erqa_weight * (1 - erqa_blocks) + lpips_weight * lpips_blocks
    final_map[torch.isnan(final_map)] = torch.mean(final_map[~torch.isnan(final_map)])
    return final_map


class CombDet:
    def __init__(self, device: torch.device):
        self.threshold1 = 0.9
        self.threshold2 = 0.99
        self.erqa_block_size = 8
        self.lpips_block_size = 32
        self.erqa_stride = 8
        self.lpips_stride = 16
        self.erqa_weight = 0.4
        self.lpips_weight = 0.6
        self.device = device
        self.lpips_metric = lpips.LPIPS(verbose=False).to(device)
        self.lpips_metric.eval()

    def __call__(self, *, gt: torch.Tensor, target: torch.Tensor):
        return get_heatmap(
            gt=gt,
            sample=target,
            lpips_metric=self.lpips_metric,
            threshold1=self.threshold1,
            threshold2=self.threshold2,
            erqa_block_size=self.erqa_block_size,
            lpips_block_size=self.lpips_block_size,
            erqa_stride=self.erqa_stride,
            lpips_stride=self.lpips_stride,
            erqa_weight=self.erqa_weight,
            lpips_weight=self.lpips_weight,
            device=self.device,
        )


class BdJupFeature:
    """Compute the bd_jup heatmap of an SR image against a reference."""

    def __init__(self, device: torch.device):
        self.metric = CombDet(device)
        self.device = device

    def __call__(self, *, gt: Image.Image, sr: Image.Image, lr: Image.Image) -> np.ndarray:
        target = TF.to_tensor(sr).mul_(2).sub_(1).to(self.device)
        reference = TF.to_tensor(gt).mul_(2).sub_(1).to(self.device)
        with torch.inference_mode():
            return self.metric(gt=reference, target=target).cpu().numpy()
