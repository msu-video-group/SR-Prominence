"""ssm_jup heatmap: a structure-similarity map from the JPEG AI small-color-artifact detector."""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# get_local_weights and get_refined_artifact_map are adapted from LDL,
# https://github.com/csjliang/LDL (Apache License 2.0, see third_party/ldl/LICENSE):
# made methods with a configurable dimension for the patch-level variance, and without
# zeroing the pixels where the EMA output has a larger residual.
class Structure_Similarity:
    def __init__(self, LocalWeights_ksize=33, GaussianBlur_ksize=103, GaussianBlur_sigma=33):
        self.GaussianBlur_ksize = [GaussianBlur_ksize, GaussianBlur_ksize]
        self.GaussianBlur_sigma = [GaussianBlur_sigma, GaussianBlur_sigma]
        self.LocalWeights_ksize = LocalWeights_ksize

    def get_local_weights(self, residual):
        pad = (self.LocalWeights_ksize - 1) // 2
        residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode="reflect")
        unfolded_residual = residual_pad.unfold(2, self.LocalWeights_ksize, 1).unfold(
            3, self.LocalWeights_ksize, 1
        )
        pixel_level_weight = (
            torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True)
            .squeeze(-1)
            .squeeze(-1)
        )
        return pixel_level_weight

    def get_refined_artifact_map(self, img_gt, img_output, dim=(-1, -2, -3)):
        residual_SR = torch.sum(torch.abs(img_gt - img_output), 1, keepdim=True)
        patch_level_weight = torch.var(residual_SR.clone(), dim=dim, keepdim=True) ** (1 / 5)
        pixel_level_weight = self.get_local_weights(residual_SR.clone())
        overall_weight = patch_level_weight * pixel_level_weight
        return overall_weight

    def get_heatmap(self, gt_image: torch.Tensor, ai_image: torch.Tensor, tc_image: torch.Tensor):
        dim = (-1, -2, -3)
        ai_heatmap = self.get_refined_artifact_map(ai_image, gt_image, dim)
        tc_heatmap = self.get_refined_artifact_map(tc_image, gt_image, dim)

        ai_heatmap_blurred = TF.gaussian_blur(
            ai_heatmap, kernel_size=self.GaussianBlur_ksize, sigma=self.GaussianBlur_sigma
        )
        tc_heatmap_blurred = TF.gaussian_blur(
            tc_heatmap, kernel_size=self.GaussianBlur_ksize, sigma=self.GaussianBlur_sigma
        )

        return ai_heatmap_blurred - tc_heatmap_blurred


def bicubic_upscale(lr: Image.Image) -> Image.Image:
    """Upscale the LR image ×4 with PIL's bicubic, as the dataset's bicubic references are."""
    return lr.resize((lr.width * 4, lr.height * 4), Image.Resampling.BICUBIC)


class SsmJupFeature:
    """Compute the ssm_jup heatmap of an SR image against a reference and the bicubic upscale."""

    def __init__(self, device: torch.device):
        self.device = device
        self.metric = Structure_Similarity()
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, *, gt: Image.Image, sr: Image.Image, lr: Image.Image) -> np.ndarray:
        target = self.transform(sr).unsqueeze(0).to(self.device)
        reference = self.transform(gt).unsqueeze(0).to(self.device)
        bicubic = self.transform(bicubic_upscale(lr)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return self.metric.get_heatmap(reference, target, bicubic)[0, 0].cpu().numpy()
