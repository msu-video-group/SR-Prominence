<h1 align="center">SR-Prominence</h1>
<h3 align="center">A Crowdsourced Protocol and Dataset Suite for Perceptually-Weighted Super-Resolution Artifact Evaluation</h3>
<p align="center">Ivan Molodetskikh, Kirill Malyshev, Mark Mirgaleev, Nikita Zagainov, Evgeney Bogatyrev, Dmitriy Vatolin</p>
<p align="center"><em>NeurIPS 2026 Evaluations &amp; Datasets Track</em></p>
<p align="center">
<a href="https://arxiv.org/abs/2605.14847"><img src="https://img.shields.io/badge/arXiv-2605.14847-b31b1b.svg" alt="arXiv"></a>
<a href="https://huggingface.co/datasets/imolodetskikh/sr-artifact-prominence"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow.svg" alt="Hugging Face dataset"></a>
</p>

Modern image super-resolution methods often generate visual artifacts: unnatural textures, smeared faces or garbled text.
These artifacts vary in how prominent they are to viewers: some are barely noticeable, others are highly disturbing.
The [SR-Prominence dataset](https://huggingface.co/datasets/imolodetskikh/sr-artifact-prominence) measures this with crowdsourced prominence scores for artifacts from 15 SR methods.

This repository contains the paper's baseline artifact detector.
Given a ×4 super-resolved image and either its low-resolution input or the original high-resolution image, it predicts a map of how prominent the SR artifacts are, at the resolution of the SR image.
See the [paper](https://arxiv.org/abs/2605.14847) for how it works.

![Prominent and non-prominent SR artifact examples](assets/prominence_examples.jpg)

*Artifact examples from Prominence-OpenImages (top) and Prominence-DeSRA (bottom).
Left: prominent artifacts; RealSR blurred out holes on the radio panel, and LDL reconstructed an incorrect linear pattern on the bag.
Right: non-prominent artifacts; GFPGAN incorrectly restored a natural water surface, and SwinIR generated a dotted texture artifact on a non-salient floor region.*

## Running the detector

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/msu-video-group/SR-Prominence
cd SR-Prominence
uv run predict.py --lr examples/flute/lr.jpg --sr examples/flute/sr.jpg -o flute
```

This writes `flute.npy` and `flute.png`, which should match `examples/flute/prominence.png`.
The first run downloads the VGG16 and AlexNet weights used by DISTS and LPIPS from torchvision.

There are two modes, depending on the reference you have:

```bash
# Low-resolution input and SR output
uv run predict.py --lr lr.png --sr sr.png -o out

# Original high-resolution image and SR output
uv run predict.py --gt gt.png --sr sr.png -o out
```

- `--lr` is the realistic case: you upscaled an image and want to find the artifacts in the result.
  The reference is then an upscale of the LR image by a finetuned [RLFN](https://github.com/bytedance/RLFN), the paper's pseudo-GT protocol.
  The SR image must be exactly 4× the size of the LR image.
- With `--gt`, the original image is the reference, which gives more accurate maps.
  The GT and SR images must have the same size.
  The method also uses a bicubic upscale of the LR image; by default the LR image is the bicubic ×4 downscale of the GT, which you can override with `--lr`.

The outputs are:

- `out.npy`: the float32 prominence map with the SR image's height and width.
  Values are roughly the predicted fraction of viewers who would notice an artifact there: about 0 without artifacts, and up to about 1 for the most prominent ones.
  They can go slightly outside [0, 1].
  For binary artifact masks, the paper thresholds the map at 0.3, the value that best detects prominent artifacts on the DeSRA dataset.
- `out.png`: the map in the inferno colormap, clipped to [0, 1].
  Pass `--overlay` to blend it over the SR image.

Without `-o`, the outputs are named after the SR image, in the current directory.
A CUDA GPU is used when available; set `CUDA_VISIBLE_DEVICES=` to force the CPU.

### Python

From the repository directory:

```python
from predict import ProminenceModel

model = ProminenceModel()
prominence = model.predict("sr.png", lr="lr.png")  # or gt="gt.png"
```

`predict` accepts paths or PIL images and returns the map as a NumPy array.
Keep the model around to process several images without reloading it.

## Examples

`examples/` holds three Open Images photos upscaled by LDL, SUPIR, and SwinIR (top to bottom), with the maps predicted from the LR input (`--lr`).

| SR image | Predicted prominence |
|:-:|:-:|
| ![](examples/flute/sr.jpg) | ![](examples/flute/prominence.png) |
| ![](examples/goal/sr.jpg) | ![](examples/goal/prominence.png) |
| ![](examples/shower/sr.jpg) | ![](examples/shower/prominence.png) |

To reproduce a map, run e.g. `uv run predict.py --lr examples/goal/lr.jpg --sr examples/goal/sr.jpg -o goal`.
Each example also has the original image, `gt.jpg`, for the `--gt` mode.

## Dataset, training, and evaluation

The dataset, with the images, artifact masks, prominence annotations, and precomputed heatmaps, is on [Hugging Face](https://huggingface.co/datasets/imolodetskikh/sr-artifact-prominence).

The training, feature computation, and evaluation scripts (`train.py`, `compute_features.py`, `eval.py`) are also here, and need the dataset.
The same code is in the dataset's [`code/`](https://huggingface.co/datasets/imolodetskikh/sr-artifact-prominence/tree/main/code) directory, which documents how to reproduce the paper's results.
When running them from this repository, pass `--dataset-root` with the path to the downloaded dataset.

## License

The code is licensed under the [MIT License](LICENSE).
It includes code adapted from third-party projects, under their own licenses in `third_party/`:

- `third_party/dists/`: the official [DISTS](https://github.com/dingkeyan93/DISTS) implementation and weights (MIT).
- `third_party/rlfn/`: the [RLFN](https://github.com/bytedance/RLFN) architecture (Apache License 2.0), with finetuned weights.
- `bd_jup.py`: ERQA, adapted from [ERQA](https://github.com/msu-video-group/erqa) (MIT, `third_party/erqa/LICENSE`).
- `ssm_jup.py`: the artifact map, adapted from [LDL](https://github.com/csjliang/LDL) (Apache License 2.0, `third_party/ldl/LICENSE`).

The images in `examples/` are derived from Open Images photos under CC BY 2.0, and are shared under the same license.
The figure in `assets/` is from the paper.

## Citation

```bibtex
@misc{molodetskikh2026srprominence,
  title         = {{SR-Prominence}: A Crowdsourced Protocol and Dataset Suite for Perceptually-Weighted Super-Resolution Artifact Evaluation},
  author        = {Molodetskikh, Ivan and Malyshev, Kirill and Mirgaleev, Mark and Zagainov, Nikita and Bogatyrev, Evgeney and Vatolin, Dmitriy},
  year          = {2026},
  eprint        = {2605.14847},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2605.14847}
}
```
